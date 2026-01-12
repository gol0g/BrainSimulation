"""
E7-B1b: Path-biased Pop-up Obstacle Test

진행 방향 앞에 pop-up 생성 → +MEM lag 병리 극대화

핵심 지표:
1. Mean Collision: 전체 충돌률
2. Event Collision: 이벤트 후 10 step 내 충돌률
3. Near-miss rate: 충돌은 안 했지만 가까웠던 비율
4. RT(p95): 반응 시간 95퍼센타일

예상:
- +MEM: event_coll ↑, near_miss ↑ (lag 병리)
- FULL+RF: 최저 event_coll, 최저 near_miss

Usage:
    python test_e7_popup_path.py
    python test_e7_popup_path.py --seeds 100
"""

import numpy as np
import sys
import os
from typing import Dict, List
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e7_popup_path import (
    PathPopupNavEnv, PathPopupConfig,
    E7_B1bGate, E7PathEventGateResult,
)


# ============================================================================
# Agents (B1과 동일)
# ============================================================================

class BaseAgent:
    """BASE: 랜덤 노이즈"""
    def __init__(self):
        self.in_defense_mode = False
        self.current_risk = 0.0

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        min_obs_dist = obs[16]

        self.current_risk = np.clip((0.3 - min_obs_dist) / 0.3, 0, 1)

        noise = np.random.randn(2) * 0.15
        goal_dx += noise[0]
        goal_dy += noise[1]

        goal_dir = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        action = goal_dir * 1.0 - np.array([vel_x, vel_y]) * 0.3
        return action

    def reset(self):
        self.in_defense_mode = False
        self.current_risk = 0.0

    def on_goal_switch(self):
        pass

    def get_defense_mode(self):
        return False

    def get_current_risk(self):
        return self.current_risk


class MemoryAgent:
    """+MEM: EMA 스무딩 (lag 병리 대상)"""
    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.goal_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        min_obs_dist = obs[16]

        self.current_risk = np.clip((0.3 - min_obs_dist) / 0.3, 0, 1)

        current_goal = np.array([goal_dx, goal_dy])

        if self.goal_ema is None:
            self.goal_ema = current_goal
        else:
            self.goal_ema = self.ema_alpha * current_goal + (1 - self.ema_alpha) * self.goal_ema

        goal_dir = self.goal_ema
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        action = goal_dir * 1.0 - np.array([vel_x, vel_y]) * 0.3
        return action

    def reset(self):
        self.goal_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0

    def on_goal_switch(self):
        if self.goal_ema is not None:
            self.goal_ema = self.goal_ema * 0.3

    def get_defense_mode(self):
        return False

    def get_current_risk(self):
        return self.current_risk


class FullAgent:
    """FULL: t_goal + t_risk + hysteresis + action EMA"""
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
    """FULL+RF: Risk filter only (NO action EMA)"""
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

        # NO EMA - raw with noise
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


# ============================================================================
# Test Runners
# ============================================================================

def run_single_config_test(
    agent_class,
    agent_name: str,
    n_seeds: int = 100,
    base_seed: int = 42,
) -> Dict:
    """단일 에이전트 설정 테스트"""
    print(f"  Testing {agent_name}...")

    all_collisions = []
    all_goals = []
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

        # Episode stats
        all_collisions.append(1 if env.state.collision_this_episode else 0)
        all_goals.append(env.state.goals_completed)
        all_event_collisions.append(1 if env.state.event_window_collision else 0)

        # Near-miss
        had_near_miss = env.state.near_miss_count > 0 or \
            (env.state.min_popup_distance < config.popup_radius * 0.5 and
             not env.state.event_window_collision)
        all_near_misses.append(1 if had_near_miss else 0)

        if env.state.reaction_time >= 0:
            all_reaction_times.append(env.state.reaction_time)

    # Compute statistics
    mean_collision = np.mean(all_collisions)
    event_collision_rate = np.mean(all_event_collisions)
    near_miss_rate = np.mean(all_near_misses)
    triple_rate = sum(1 for g in all_goals if g == 3) / len(all_goals)

    if len(all_reaction_times) > 0:
        mean_rt = np.mean(all_reaction_times)
        sorted_rt = np.sort(all_reaction_times)
        p95_idx = min(int(len(sorted_rt) * 0.95), len(sorted_rt) - 1)
        p95_rt = sorted_rt[p95_idx]
    else:
        mean_rt = 0.0
        p95_rt = 0.0

    print(f"    Mean Coll: {mean_collision:.1%} | Event Coll: {event_collision_rate:.1%} | Near-miss: {near_miss_rate:.1%}")
    print(f"    RT: mean={mean_rt:.1f}, p95={p95_rt:.1f}")

    return {
        'mean_collision': mean_collision,
        'event_collision_rate': event_collision_rate,
        'near_miss_rate': near_miss_rate,
        'mean_reaction_time': mean_rt,
        'p95_reaction_time': p95_rt,
        'triple_success_rate': triple_rate,
        'n_episodes': n_seeds,
    }


def run_full_e7_b1b_test(n_seeds: int = 100, base_seed: int = 42) -> Dict:
    """E7-B1b 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E7-B1b: Path-biased Pop-up - Lag Pathology Test")
    print("=" * 70)

    start_time = time.time()

    configs = {
        'BASE': BaseAgent,
        '+MEM': MemoryAgent,
        'FULL': FullAgent,
        'FULL+RF': RiskFilterAgent,
    }

    results = {}
    for name, agent_class in configs.items():
        results[name] = run_single_config_test(agent_class, name, n_seeds, base_seed)

    elapsed = time.time() - start_time

    # Gate evaluation
    gate = E7_B1bGate()

    print(f"\n{'='*70}")
    print("  E7-B1b Path Event Gate Evaluation")
    print(f"{'='*70}")

    gate_results = {}
    for name, r in results.items():
        gr = gate.evaluate_path_event_response(
            r['event_collision_rate'],
            r['near_miss_rate'],
            r['mean_reaction_time'],
            r['p95_reaction_time'],
        )
        gate_results[name] = gr

        status = "[PASS]" if gr.passed else "[FAIL]"
        print(f"\n  {name}: {status}")
        print(f"    event_coll={r['event_collision_rate']:.1%} (gate<2%): "
              f"{'OK' if gr.event_collision_passed else 'FAIL'}")
        print(f"    near_miss={r['near_miss_rate']:.1%} (gate<10%): "
              f"{'OK' if gr.near_miss_passed else 'FAIL'}")

    # Summary table
    print(f"\n{'='*70}")
    print("  Summary Table")
    print(f"{'='*70}")
    print(f"\n  {'Config':<10} | {'Mean Coll':>10} | {'Event Coll':>11} | {'Near-miss':>10} | {'RT(p95)':>8} | {'Gate':>6}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")
    for name, r in results.items():
        gr = gate_results[name]
        status = "PASS" if gr.passed else "FAIL"
        print(f"  {name:<10} | {r['mean_collision']:>9.1%} | {r['event_collision_rate']:>10.1%} | "
              f"{r['near_miss_rate']:>9.1%} | {r['p95_reaction_time']:>7.1f} | {status:>6}")

    # Lag pathology analysis
    print(f"\n{'='*70}")
    print("  +MEM Lag Pathology Analysis")
    print(f"{'='*70}")

    base_ec = results['BASE']['event_collision_rate']
    mem_ec = results['+MEM']['event_collision_rate']
    full_ec = results['FULL']['event_collision_rate']
    rf_ec = results['FULL+RF']['event_collision_rate']

    base_nm = results['BASE']['near_miss_rate']
    mem_nm = results['+MEM']['near_miss_rate']
    full_nm = results['FULL']['near_miss_rate']
    rf_nm = results['FULL+RF']['near_miss_rate']

    print(f"\n  Event Collision: BASE={base_ec:.0%}, +MEM={mem_ec:.0%}, FULL={full_ec:.0%}, FULL+RF={rf_ec:.0%}")
    print(f"  Near-miss:       BASE={base_nm:.0%}, +MEM={mem_nm:.0%}, FULL={full_nm:.0%}, FULL+RF={rf_nm:.0%}")

    # Pathology detection
    lag_pathology_detected = False

    if mem_ec > base_ec:
        print(f"\n  [CONFIRMED] +MEM lag increases EVENT COLLISION (+{(mem_ec-base_ec)*100:.0f}%p vs BASE)")
        lag_pathology_detected = True
    elif mem_ec == base_ec and mem_nm > base_nm:
        print(f"\n  [CONFIRMED] +MEM lag increases NEAR-MISS (+{(mem_nm-base_nm)*100:.0f}%p vs BASE)")
        lag_pathology_detected = True
    else:
        print(f"\n  [NOTE] +MEM lag pathology not clearly detected in this run")

    if full_ec <= base_ec and rf_ec <= base_ec:
        print(f"  [CONFIRMED] Defense structure reduces event collision (FULL={full_ec:.0%}, RF={rf_ec:.0%})")

    if rf_ec <= full_ec and rf_nm <= full_nm:
        print(f"  [CONFIRMED] FULL+RF is best (event_coll={rf_ec:.0%}, near_miss={rf_nm:.0%})")

    # Final verdict
    full_passed = gate_results['FULL'].passed
    rf_passed = gate_results['FULL+RF'].passed
    overall_passed = full_passed or rf_passed

    print(f"\n{'='*70}")
    print("  E7-B1b Final Verdict")
    print(f"{'='*70}")
    print(f"\n  FULL gate: {'PASS' if full_passed else 'FAIL'}")
    print(f"  FULL+RF gate: {'PASS' if rf_passed else 'FAIL'}")
    print(f"  Lag pathology detected: {'YES' if lag_pathology_detected else 'NO'}")
    print(f"\n  Overall: [{'PASS' if overall_passed else 'FAIL'}]")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'passed': overall_passed,
        'lag_pathology_detected': lag_pathology_detected,
        'results': results,
        'gate_results': gate_results,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    result = run_full_e7_b1b_test(args.seeds, args.seed)
    exit(0 if result['passed'] else 1)
