"""
E7-B1c: Control Smoothing Ablation

B1b 가설 확인: FULL+RF가 망가진 이유 = Δa clamp 하에서 스무딩 부족 → 오실레이션

실험군 (3개만):
- FULL: goal EMA + risk hysteresis (B1b 최고)
- FULL+RF: risk filter만 (B1b에서 실패)
- FULL+RF+LPF: risk filter + 짧은 action low-pass (β=0.5)

예상:
- 가설이 맞으면: FULL+RF+LPF의 RT(p95)가 30→10, near-miss가 FULL 수준으로 회복

Usage:
    python test_e7_b1c.py
"""

import numpy as np
import sys
import os
from typing import Dict
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e7_popup_path import PathPopupNavEnv, PathPopupConfig


# ============================================================================
# Agents
# ============================================================================

class FullAgent:
    """FULL: goal EMA + risk hysteresis (B1b BEST)"""
    def __init__(
        self,
        d_safe: float = 0.3,
        ema_alpha: float = 0.3,
        risk_on: float = 0.4,
        risk_off: float = 0.2,
        risk_ema_alpha: float = 0.5,
    ):
        self.d_safe = d_safe
        self.ema_alpha = ema_alpha
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha

        self.goal_ema = None
        self.risk_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        closest_obs_dx, closest_obs_dy = 0.0, 0.0
        min_dist = float('inf')
        for i in range(3):
            base = 7 + i * 3
            if base + 2 < len(obs):
                obs_dist = obs[base + 2]
                if obs_dist < min_dist:
                    min_dist = obs_dist
                    closest_obs_dx = obs[base]
                    closest_obs_dy = obs[base + 1]

        # Goal EMA
        current_goal = np.array([goal_dx, goal_dy])
        if self.goal_ema is None:
            self.goal_ema = current_goal
        else:
            self.goal_ema = self.ema_alpha * current_goal + (1 - self.ema_alpha) * self.goal_ema

        goal_dx, goal_dy = self.goal_ema[0], self.goal_ema[1]

        t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
        raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)
        self.current_risk = raw_risk

        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        if self.in_defense_mode:
            if self.risk_ema < self.risk_off:
                self.in_defense_mode = False
        else:
            if self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        t_risk = min(self.risk_ema * 1.2, 1.0) if self.in_defense_mode else self.risk_ema

        gain = 0.8 + t_goal * 0.4 - t_risk * 0.3
        gain = np.clip(gain, 0.4, 1.2)
        damp = 0.3 + t_risk * 0.3
        damp = np.clip(damp, 0.2, 0.6)
        avoid_strength = t_risk * 0.5

        goal_dir = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        action = goal_dir * gain

        if avoid_strength > 0 and (abs(closest_obs_dx) > 1e-6 or abs(closest_obs_dy) > 1e-6):
            obs_dir = np.array([closest_obs_dx, closest_obs_dy])
            obs_norm = np.linalg.norm(obs_dir)
            if obs_norm > 1e-6:
                obs_dir = obs_dir / obs_norm
                action -= obs_dir * avoid_strength

        action -= np.array([vel_x, vel_y]) * damp
        return action

    def reset(self):
        self.goal_ema = None
        self.risk_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0

    def on_goal_switch(self):
        if self.goal_ema is not None:
            self.goal_ema = self.goal_ema * 0.3

    def get_defense_mode(self):
        return self.in_defense_mode

    def get_current_risk(self):
        return self.current_risk


class RiskFilterAgent:
    """FULL+RF: risk filter만 (NO action smoothing) - B1b에서 실패"""
    def __init__(
        self,
        d_safe: float = 0.3,
        risk_on: float = 0.4,
        risk_off: float = 0.2,
        risk_ema_alpha: float = 0.5,
    ):
        self.d_safe = d_safe
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha

        self.risk_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        closest_obs_dx, closest_obs_dy = 0.0, 0.0
        min_dist = float('inf')
        for i in range(3):
            base = 7 + i * 3
            if base + 2 < len(obs):
                obs_dist = obs[base + 2]
                if obs_dist < min_dist:
                    min_dist = obs_dist
                    closest_obs_dx = obs[base]
                    closest_obs_dy = obs[base + 1]

        # NO goal EMA - raw with noise
        noise = np.random.randn(2) * 0.15
        goal_dx += noise[0]
        goal_dy += noise[1]

        t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
        raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)
        self.current_risk = raw_risk

        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        if self.in_defense_mode:
            if self.risk_ema < self.risk_off:
                self.in_defense_mode = False
        else:
            if self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        t_risk = min(self.risk_ema * 1.2, 1.0) if self.in_defense_mode else self.risk_ema

        gain = 0.8 + t_goal * 0.4 - t_risk * 0.3
        gain = np.clip(gain, 0.4, 1.2)
        damp = 0.3 + t_risk * 0.3
        damp = np.clip(damp, 0.2, 0.6)
        avoid_strength = t_risk * 0.5

        goal_dir = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        action = goal_dir * gain

        if avoid_strength > 0 and (abs(closest_obs_dx) > 1e-6 or abs(closest_obs_dy) > 1e-6):
            obs_dir = np.array([closest_obs_dx, closest_obs_dy])
            obs_norm = np.linalg.norm(obs_dir)
            if obs_norm > 1e-6:
                obs_dir = obs_dir / obs_norm
                action -= obs_dir * avoid_strength

        action -= np.array([vel_x, vel_y]) * damp
        return action

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0

    def on_goal_switch(self):
        pass

    def get_defense_mode(self):
        return self.in_defense_mode

    def get_current_risk(self):
        return self.current_risk


class RiskFilterLPFAgent:
    """FULL+RF+LPF: risk filter + 짧은 action low-pass (β=0.5)"""
    def __init__(
        self,
        d_safe: float = 0.3,
        risk_on: float = 0.4,
        risk_off: float = 0.2,
        risk_ema_alpha: float = 0.5,
        action_lpf_beta: float = 0.5,  # 짧은 time constant
    ):
        self.d_safe = d_safe
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.action_lpf_beta = action_lpf_beta

        self.risk_ema = None
        self.action_ema = None  # Action low-pass filter
        self.in_defense_mode = False
        self.current_risk = 0.0

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        closest_obs_dx, closest_obs_dy = 0.0, 0.0
        min_dist = float('inf')
        for i in range(3):
            base = 7 + i * 3
            if base + 2 < len(obs):
                obs_dist = obs[base + 2]
                if obs_dist < min_dist:
                    min_dist = obs_dist
                    closest_obs_dx = obs[base]
                    closest_obs_dy = obs[base + 1]

        # NO goal EMA - raw with noise (RF 방식)
        noise = np.random.randn(2) * 0.15
        goal_dx += noise[0]
        goal_dy += noise[1]

        t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
        raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)
        self.current_risk = raw_risk

        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        if self.in_defense_mode:
            if self.risk_ema < self.risk_off:
                self.in_defense_mode = False
        else:
            if self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        t_risk = min(self.risk_ema * 1.2, 1.0) if self.in_defense_mode else self.risk_ema

        gain = 0.8 + t_goal * 0.4 - t_risk * 0.3
        gain = np.clip(gain, 0.4, 1.2)
        damp = 0.3 + t_risk * 0.3
        damp = np.clip(damp, 0.2, 0.6)
        avoid_strength = t_risk * 0.5

        goal_dir = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        raw_action = goal_dir * gain

        if avoid_strength > 0 and (abs(closest_obs_dx) > 1e-6 or abs(closest_obs_dy) > 1e-6):
            obs_dir = np.array([closest_obs_dx, closest_obs_dy])
            obs_norm = np.linalg.norm(obs_dir)
            if obs_norm > 1e-6:
                obs_dir = obs_dir / obs_norm
                raw_action -= obs_dir * avoid_strength

        raw_action -= np.array([vel_x, vel_y]) * damp

        # Action low-pass filter (핵심 추가!)
        if self.action_ema is None:
            self.action_ema = raw_action
        else:
            self.action_ema = self.action_lpf_beta * raw_action + (1 - self.action_lpf_beta) * self.action_ema

        return self.action_ema

    def reset(self):
        self.risk_ema = None
        self.action_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0

    def on_goal_switch(self):
        pass

    def get_defense_mode(self):
        return self.in_defense_mode

    def get_current_risk(self):
        return self.current_risk


# ============================================================================
# Test Runner
# ============================================================================

def run_single_config_test(
    agent_class,
    agent_name: str,
    n_seeds: int = 100,
    base_seed: int = 42,
) -> Dict:
    """단일 에이전트 테스트"""
    print(f"  Testing {agent_name}...")

    all_collisions = []
    all_event_collisions = []
    all_near_misses = []
    all_reaction_times = []

    for s in range(n_seeds):
        seed = base_seed + s
        config = PathPopupConfig()
        env = PathPopupNavEnv(config, seed=seed)
        agent = agent_class()
        agent.reset()

        obs = env.reset(seed=seed)

        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

            env.record_defense_state(
                agent.get_defense_mode(),
                agent.get_current_risk()
            )

            if info.get('goal_switch'):
                agent.on_goal_switch()

            if done:
                break

        all_collisions.append(1 if env.state.collision_this_episode else 0)
        all_event_collisions.append(1 if env.state.event_window_collision else 0)

        had_near_miss = env.state.near_miss_count > 0 or \
            (env.state.min_popup_distance < config.popup_radius * 0.5 and
             not env.state.event_window_collision)
        all_near_misses.append(1 if had_near_miss else 0)

        if env.state.reaction_time >= 0:
            all_reaction_times.append(env.state.reaction_time)

    mean_collision = np.mean(all_collisions)
    event_collision_rate = np.mean(all_event_collisions)
    near_miss_rate = np.mean(all_near_misses)

    if len(all_reaction_times) > 0:
        mean_rt = np.mean(all_reaction_times)
        sorted_rt = np.sort(all_reaction_times)
        p95_idx = min(int(len(sorted_rt) * 0.95), len(sorted_rt) - 1)
        p95_rt = sorted_rt[p95_idx]
    else:
        mean_rt = 0.0
        p95_rt = 0.0

    print(f"    Mean: {mean_collision:.1%} | Event: {event_collision_rate:.1%} | "
          f"Near-miss: {near_miss_rate:.1%} | RT(p95): {p95_rt:.1f}")

    return {
        'mean_collision': mean_collision,
        'event_collision_rate': event_collision_rate,
        'near_miss_rate': near_miss_rate,
        'mean_reaction_time': mean_rt,
        'p95_reaction_time': p95_rt,
    }


def run_e7_b1c_test(n_seeds: int = 100, base_seed: int = 42) -> Dict:
    """E7-B1c 테스트"""
    print("\n" + "=" * 70)
    print("  E7-B1c: Control Smoothing Ablation")
    print("  가설: FULL+RF 실패 = Δa clamp 하 오실레이션")
    print("=" * 70)

    start_time = time.time()

    configs = {
        'FULL': FullAgent,
        'FULL+RF': RiskFilterAgent,
        'FULL+RF+LPF': RiskFilterLPFAgent,
    }

    results = {}
    for name, agent_class in configs.items():
        results[name] = run_single_config_test(agent_class, name, n_seeds, base_seed)

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print("  Summary Table")
    print(f"{'='*70}")
    print(f"\n  {'Config':<15} | {'Mean Coll':>10} | {'Event Coll':>11} | {'Near-miss':>10} | {'RT(p95)':>8}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*8}")
    for name, r in results.items():
        print(f"  {name:<15} | {r['mean_collision']:>9.1%} | {r['event_collision_rate']:>10.1%} | "
              f"{r['near_miss_rate']:>9.1%} | {r['p95_reaction_time']:>7.1f}")

    # Hypothesis verification
    print(f"\n{'='*70}")
    print("  Hypothesis Verification")
    print(f"{'='*70}")

    full = results['FULL']
    rf = results['FULL+RF']
    lpf = results['FULL+RF+LPF']

    print(f"\n  B1b 결과 (참고):")
    print(f"    FULL: event=10%, near-miss=37%, RT(p95)=9")
    print(f"    FULL+RF: event=18%, near-miss=46%, RT(p95)=30")

    print(f"\n  B1c 결과:")
    print(f"    FULL: event={full['event_collision_rate']:.0%}, near-miss={full['near_miss_rate']:.0%}, RT(p95)={full['p95_reaction_time']:.0f}")
    print(f"    FULL+RF: event={rf['event_collision_rate']:.0%}, near-miss={rf['near_miss_rate']:.0%}, RT(p95)={rf['p95_reaction_time']:.0f}")
    print(f"    FULL+RF+LPF: event={lpf['event_collision_rate']:.0%}, near-miss={lpf['near_miss_rate']:.0%}, RT(p95)={lpf['p95_reaction_time']:.0f}")

    # 가설 검증
    hypothesis_confirmed = False

    # 조건 1: LPF가 RF보다 RT를 낮춤
    rt_improved = lpf['p95_reaction_time'] < rf['p95_reaction_time']

    # 조건 2: LPF가 RF보다 event_collision 또는 near_miss 낮춤
    event_improved = lpf['event_collision_rate'] < rf['event_collision_rate']
    near_miss_improved = lpf['near_miss_rate'] < rf['near_miss_rate']

    # 조건 3: LPF가 FULL에 근접
    event_close_to_full = abs(lpf['event_collision_rate'] - full['event_collision_rate']) <= 0.05
    near_miss_close_to_full = abs(lpf['near_miss_rate'] - full['near_miss_rate']) <= 0.10

    print(f"\n  검증:")
    print(f"    RT 개선 (LPF < RF): {'YES' if rt_improved else 'NO'} "
          f"({lpf['p95_reaction_time']:.0f} vs {rf['p95_reaction_time']:.0f})")
    print(f"    Event collision 개선: {'YES' if event_improved else 'NO'} "
          f"({lpf['event_collision_rate']:.0%} vs {rf['event_collision_rate']:.0%})")
    print(f"    Near-miss 개선: {'YES' if near_miss_improved else 'NO'} "
          f"({lpf['near_miss_rate']:.0%} vs {rf['near_miss_rate']:.0%})")
    print(f"    FULL에 근접 (event): {'YES' if event_close_to_full else 'NO'}")
    print(f"    FULL에 근접 (near-miss): {'YES' if near_miss_close_to_full else 'NO'}")

    if rt_improved and (event_improved or near_miss_improved):
        hypothesis_confirmed = True
        print(f"\n  [CONFIRMED] 가설 확인: 짧은 action LPF가 오실레이션 완화")
    else:
        print(f"\n  [NOT CONFIRMED] 가설 미확인 또는 다른 원인 존재")

    print(f"\n{'='*70}")
    print("  Final Conclusion")
    print(f"{'='*70}")

    if hypothesis_confirmed:
        print(f"""
  E7-B1c 결론:
  ├─ FULL+RF가 B1b에서 실패한 이유 = Δa clamp 하 오실레이션
  ├─ 짧은 action low-pass (β=0.5)로 오실레이션 완화 가능
  └─ 최적 설계: Risk filter + 조건부 action smoothing
        """)
    else:
        print(f"""
  E7-B1c 결론:
  ├─ 가설이 완전히 확인되지 않음
  ├─ FULL의 우위는 goal EMA + defense 결합 효과일 수 있음
  └─ 추가 분석 필요
        """)

    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'hypothesis_confirmed': hypothesis_confirmed,
        'results': results,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    result = run_e7_b1c_test(args.seeds, args.seed)
    exit(0 if result['hypothesis_confirmed'] else 1)
