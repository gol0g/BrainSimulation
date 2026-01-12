"""
E7-D1: TTC-Triggered Defense Ablation

B1b/B1c에서 확인된 가설:
- FULL+RF가 path-biased pop-up에서 취약한 이유 = "reactive" 특성
- TTC(Time-To-Collision)가 짧은 정면 위협에서 감지 후 반응은 물리적으로 늦음

실험:
- FULL: goal EMA + risk hysteresis (B1b 최고)
- FULL+RF: risk filter만 (B1b에서 실패)
- FULL+RF+TTC: risk filter + TTC 기반 선제 방어 트리거

TTC 계산:
    ttc = distance_along_velocity_to_obstacle / speed
    if ttc < τ_on AND approaching: force defensive mode

예상:
- 가설이 맞으면: FULL+RF+TTC의 event coll / near-miss가 FULL 수준으로 회복
- RT(p95)도 안정적으로 낮아져야 함

Usage:
    python test_e7_d1.py --seeds 100
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
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
    """FULL+RF: risk filter만 (NO goal EMA) - B1b에서 실패"""
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


class RiskFilterTTCAgent:
    """
    FULL+RF+TTC: risk filter + TTC 기반 선제 방어 트리거

    핵심 추가:
    - TTC(Time-To-Collision) 계산
    - TTC < τ_on이면 즉시 방어 모드 강제 진입
    - 기존 risk hysteresis와 OR 조건으로 결합
    """
    def __init__(
        self,
        d_safe: float = 0.3,
        risk_on: float = 0.4,
        risk_off: float = 0.2,
        risk_ema_alpha: float = 0.5,
        ttc_threshold: float = 10.0,  # TTC < 10 steps면 선제 방어
        ttc_off_threshold: float = 15.0,  # TTC > 15면 방어 해제 허용
    ):
        self.d_safe = d_safe
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.ttc_threshold = ttc_threshold
        self.ttc_off_threshold = ttc_off_threshold

        self.risk_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0
        self.current_ttc = float('inf')
        self.ttc_triggered = False  # TTC로 인한 방어 모드 여부

    def _compute_ttc(self, vel: np.ndarray, obs_positions: List[Tuple[float, float, float]]) -> float:
        """
        Time-To-Collision 계산

        TTC = distance_along_velocity / speed
        - 속도 방향으로 투영된 거리 / 속도 크기
        - 접근 중일 때만 유효 (양수 TTC)
        """
        speed = np.linalg.norm(vel)
        if speed < 0.05:  # 거의 정지 상태
            return float('inf')

        vel_dir = vel / speed
        min_ttc = float('inf')

        for obs_dx, obs_dy, obs_dist in obs_positions:
            if obs_dist > 3.0:  # 너무 먼 장애물 무시
                continue

            # 장애물 방향 벡터
            obs_vec = np.array([obs_dx, obs_dy])

            # 속도 방향으로 투영
            projection = np.dot(obs_vec, vel_dir)

            # 접근 중인 경우만 (projection > 0 = 속도 방향에 장애물이 있음)
            if projection > 0:
                # TTC = 투영 거리 / 속도
                # 단, 실제 거리는 obs_dist이므로, 투영 거리로 보정
                ttc = projection / speed

                # 측면 거리도 고려 (충돌 범위 안에 들어오는지)
                lateral_dist = np.sqrt(max(0, obs_dist**2 - projection**2))
                collision_margin = 0.5  # 충돌 반경 마진

                if lateral_dist < collision_margin:
                    min_ttc = min(min_ttc, ttc)

        return min_ttc

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        # 장애물 정보 수집
        obs_positions = []
        closest_obs_dx, closest_obs_dy = 0.0, 0.0
        min_dist = float('inf')

        for i in range(3):
            base = 7 + i * 3
            if base + 2 < len(obs):
                o_dx = obs[base]
                o_dy = obs[base + 1]
                o_dist = obs[base + 2]
                obs_positions.append((o_dx, o_dy, o_dist))

                if o_dist < min_dist:
                    min_dist = o_dist
                    closest_obs_dx = o_dx
                    closest_obs_dy = o_dy

        # TTC 계산 (핵심!)
        vel = np.array([vel_x, vel_y])
        self.current_ttc = self._compute_ttc(vel, obs_positions)

        # NO goal EMA - raw with noise (RF 방식 유지)
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

        # 방어 모드 결정: 기존 risk hysteresis + TTC trigger (OR 조건)

        # TTC 기반 선제 방어 (핵심 추가!)
        if self.current_ttc < self.ttc_threshold:
            self.ttc_triggered = True
            self.in_defense_mode = True
        elif self.current_ttc > self.ttc_off_threshold:
            self.ttc_triggered = False

        # 기존 risk hysteresis (TTC 트리거 아닐 때만)
        if not self.ttc_triggered:
            if self.in_defense_mode:
                if self.risk_ema < self.risk_off:
                    self.in_defense_mode = False
            else:
                if self.risk_ema > self.risk_on:
                    self.in_defense_mode = True

        t_risk = min(self.risk_ema * 1.2, 1.0) if self.in_defense_mode else self.risk_ema

        # TTC가 매우 짧으면 추가 회피 강화
        if self.current_ttc < 5.0:
            t_risk = max(t_risk, 0.8)

        gain = 0.8 + t_goal * 0.4 - t_risk * 0.3
        gain = np.clip(gain, 0.4, 1.2)
        damp = 0.3 + t_risk * 0.3
        damp = np.clip(damp, 0.2, 0.6)
        avoid_strength = t_risk * 0.5

        # TTC 짧을 때 회피 강화
        if self.current_ttc < 8.0:
            avoid_strength = max(avoid_strength, 0.6)

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
        self.current_ttc = float('inf')
        self.ttc_triggered = False

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


def run_e7_d1_test(n_seeds: int = 100, base_seed: int = 42) -> Dict:
    """E7-D1 테스트: TTC-Triggered Defense"""
    print("\n" + "=" * 70)
    print("  E7-D1: TTC-Triggered Defense Ablation")
    print("  가설: RF 취약점 = TTC 짧은 정면 위협에서 reactive가 늦음")
    print("=" * 70)

    start_time = time.time()

    configs = {
        'FULL': FullAgent,
        'FULL+RF': RiskFilterAgent,
        'FULL+RF+TTC': RiskFilterTTCAgent,
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
    ttc = results['FULL+RF+TTC']

    print(f"\n  기준값 (B1b 결과):")
    print(f"    FULL: event=10%, near-miss=37%")
    print(f"    FULL+RF: event=16-18%, near-miss=46-47%")

    print(f"\n  D1 결과:")
    print(f"    FULL: event={full['event_collision_rate']:.0%}, near-miss={full['near_miss_rate']:.0%}, RT(p95)={full['p95_reaction_time']:.0f}")
    print(f"    FULL+RF: event={rf['event_collision_rate']:.0%}, near-miss={rf['near_miss_rate']:.0%}, RT(p95)={rf['p95_reaction_time']:.0f}")
    print(f"    FULL+RF+TTC: event={ttc['event_collision_rate']:.0%}, near-miss={ttc['near_miss_rate']:.0%}, RT(p95)={ttc['p95_reaction_time']:.0f}")

    # 가설 검증
    hypothesis_confirmed = False

    # 조건 1: TTC가 RF보다 개선
    event_improved = ttc['event_collision_rate'] < rf['event_collision_rate']
    near_miss_improved = ttc['near_miss_rate'] < rf['near_miss_rate']

    # 조건 2: TTC가 FULL에 근접 또는 능가
    event_close_to_full = ttc['event_collision_rate'] <= full['event_collision_rate'] + 0.03
    near_miss_close_to_full = ttc['near_miss_rate'] <= full['near_miss_rate'] + 0.05

    # 조건 3: RT가 합리적 (RF 이하)
    rt_reasonable = ttc['p95_reaction_time'] <= rf['p95_reaction_time'] + 5

    print(f"\n  검증:")
    print(f"    Event coll 개선 (TTC < RF): {'YES' if event_improved else 'NO'} "
          f"({ttc['event_collision_rate']:.0%} vs {rf['event_collision_rate']:.0%})")
    print(f"    Near-miss 개선 (TTC < RF): {'YES' if near_miss_improved else 'NO'} "
          f"({ttc['near_miss_rate']:.0%} vs {rf['near_miss_rate']:.0%})")
    print(f"    FULL 수준 도달 (event): {'YES' if event_close_to_full else 'NO'} "
          f"({ttc['event_collision_rate']:.0%} vs {full['event_collision_rate']:.0%})")
    print(f"    FULL 수준 도달 (near-miss): {'YES' if near_miss_close_to_full else 'NO'} "
          f"({ttc['near_miss_rate']:.0%} vs {full['near_miss_rate']:.0%})")
    print(f"    RT 합리적: {'YES' if rt_reasonable else 'NO'} "
          f"({ttc['p95_reaction_time']:.0f} vs {rf['p95_reaction_time']:.0f})")

    # 종합 판정
    if event_improved and near_miss_improved and (event_close_to_full or near_miss_close_to_full):
        hypothesis_confirmed = True
        print(f"\n  [CONFIRMED] 가설 확인: TTC 선제 방어가 RF의 취약점 보완")
    elif event_improved or near_miss_improved:
        print(f"\n  [PARTIAL] 부분 확인: TTC가 일부 지표 개선")
    else:
        print(f"\n  [NOT CONFIRMED] 가설 미확인")

    print(f"\n{'='*70}")
    print("  Final Conclusion")
    print(f"{'='*70}")

    if hypothesis_confirmed:
        print(f"""
  E7-D1 결론:
  ├─ FULL+RF가 path-biased pop-up에서 취약한 이유 = reactive 특성
  ├─ TTC 기반 선제 방어로 취약점 보완 가능
  ├─ TTC+RF가 FULL 수준 성능 달성
  └─ 최적 설계: Risk filter + TTC-aware preemptive defense

  핵심 통찰:
  - "Reactive risk filtering"만으로는 TTC가 짧은 정면 위협에 취약
  - "TTC-aware preemptive trigger"가 이 문제를 해결
  - Goal EMA의 "관성(prior)"과 TTC의 "선제 방어"는 같은 효과를 다른 방식으로 달성
        """)
    else:
        print(f"""
  E7-D1 결론:
  ├─ TTC 선제 방어 효과가 예상보다 제한적
  ├─ FULL의 우위는 goal EMA의 복합 효과일 수 있음
  └─ 추가 분석: Goal EMA 제거 실험(D2) 필요
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

    result = run_e7_d1_test(args.seeds, args.seed)
    exit(0 if result['hypothesis_confirmed'] else 1)
