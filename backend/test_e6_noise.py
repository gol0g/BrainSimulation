"""
E6-4b: Observation Noise Test

Noise sweep: σ ∈ {0.01, 0.03, 0.06}
4 experimental groups: BASE, +MEM, FULL, FULL+risk_filter

핵심: "행동 EMA는 위험, 위험추정 필터는 유익" 분리 증명

Usage:
    python test_e6_noise.py
    python test_e6_noise.py --sigma 0.03
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e6_noise import (
    NoiseNavEnv, NoiseConfig,
    E6_4bGate, E6NoiseGateResult, E6FilterComparisonResult,
)


# ============================================================================
# Agents (4 variants)
# ============================================================================

class RandomAgent:
    """랜덤 정책"""
    def __init__(self, action_scale: float = 0.5):
        self.action_scale = action_scale

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        return np.random.randn(2) * self.action_scale

    def reset(self):
        pass

    def on_goal_switch(self):
        pass

    def is_in_defense_mode(self) -> bool:
        return False

    def get_perceived_risk(self) -> float:
        return 0.0


class NoiseAgent:
    """
    노이즈 환경용 에이전트

    Variants:
    - BASE: no memory, no hierarchy
    - +MEM: action EMA, no hierarchy
    - FULL: action EMA + t_goal + t_risk + hysteresis
    - FULL+risk_filter: NO action EMA, but risk EMA + hysteresis
    """

    def __init__(
        self,
        use_memory: bool = False,
        use_hierarchy: bool = False,
        use_risk_filter_only: bool = False,  # FULL+risk_filter variant
        d_safe: float = 0.3,
        ema_reset_factor: float = 0.3,
        risk_on_threshold: float = 0.4,
        risk_off_threshold: float = 0.2,
        risk_ema_alpha: float = 0.3,
    ):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy
        self.use_risk_filter_only = use_risk_filter_only
        self.d_safe = d_safe
        self.ema_reset_factor = ema_reset_factor

        self.risk_on_threshold = risk_on_threshold
        self.risk_off_threshold = risk_off_threshold
        self.risk_ema_alpha = risk_ema_alpha
        self.in_defense_mode = False

        self.goal_ema = None
        self.risk_ema = None
        self.ema_alpha = 0.3
        self.perceived_risk = 0.0

        self.prev_goal_dir = None
        self.switch_detected = False

    def reset(self):
        self.goal_ema = None
        self.risk_ema = None
        self.prev_goal_dir = None
        self.switch_detected = False
        self.in_defense_mode = False
        self.perceived_risk = 0.0

    def on_goal_switch(self):
        if self.goal_ema is not None and not self.use_risk_filter_only:
            self.goal_ema = self.goal_ema * self.ema_reset_factor
        self.switch_detected = True

    def is_in_defense_mode(self) -> bool:
        return self.in_defense_mode

    def get_perceived_risk(self) -> float:
        return self.perceived_risk

    def _detect_goal_switch(self, goal_dir: np.ndarray) -> bool:
        if self.prev_goal_dir is None:
            self.prev_goal_dir = goal_dir.copy()
            return False

        prev_norm = np.linalg.norm(self.prev_goal_dir)
        curr_norm = np.linalg.norm(goal_dir)

        if prev_norm > 0.01 and curr_norm > 0.01:
            cos_sim = np.dot(self.prev_goal_dir, goal_dir) / (prev_norm * curr_norm)
            if cos_sim < 0.3 or curr_norm > prev_norm * 1.5:
                self.prev_goal_dir = goal_dir.copy()
                return True

        self.prev_goal_dir = goal_dir.copy()
        return False

    def _update_risk_hysteresis(self, raw_risk: float) -> float:
        """Risk EMA + hysteresis"""
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        smoothed_risk = self.risk_ema
        self.perceived_risk = smoothed_risk

        if self.in_defense_mode:
            if smoothed_risk < self.risk_off_threshold:
                self.in_defense_mode = False
        else:
            if smoothed_risk > self.risk_on_threshold:
                self.in_defense_mode = True

        return smoothed_risk

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        goal_dir = np.array([goal_dx, goal_dy])

        if info is not None and info.get('goal_switch'):
            self.on_goal_switch()
        elif self._detect_goal_switch(goal_dir):
            self.on_goal_switch()

        # 장애물 정보
        closest_obs_dx, closest_obs_dy = 0.0, 0.0
        min_dist = float('inf')
        for i in range(3):
            base = 7 + i * 3
            obs_dist = obs[base + 2]
            if obs_dist < min_dist:
                min_dist = obs_dist
                closest_obs_dx = obs[base]
                closest_obs_dy = obs[base + 1]

        # FULL+risk_filter: NO action EMA (goal 방향 직접 사용)
        if self.use_risk_filter_only:
            # 노이즈 추가 (BASE와 동일)
            noise = np.random.randn(2) * 0.15
            goal_dx += noise[0]
            goal_dy += noise[1]
        elif not self.use_memory:
            # BASE: 노이즈 추가
            noise = np.random.randn(2) * 0.15
            goal_dx += noise[0]
            goal_dy += noise[1]
        else:
            # +MEM / FULL: action EMA
            current_goal = np.array([goal_dx, goal_dy])

            if self.switch_detected:
                effective_alpha = 0.7
                self.switch_detected = False
            else:
                effective_alpha = self.ema_alpha

            if self.goal_ema is None:
                self.goal_ema = current_goal
            else:
                self.goal_ema = effective_alpha * current_goal + (1 - effective_alpha) * self.goal_ema
            goal_dx, goal_dy = self.goal_ema[0], self.goal_ema[1]

        # Hierarchy (FULL, FULL+risk_filter 모두 해당)
        if self.use_hierarchy or self.use_risk_filter_only:
            t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
            raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)

            # Risk EMA + hysteresis (FULL, FULL+risk_filter 공통)
            smoothed_risk = self._update_risk_hysteresis(raw_risk)

            if self.in_defense_mode:
                t_risk = min(smoothed_risk * 1.2, 1.0)
            else:
                t_risk = smoothed_risk

            gain = 0.8 + t_goal * 0.4 - t_risk * 0.3
            gain = np.clip(gain, 0.4, 1.2)

            damp = 0.3 + t_risk * 0.3
            damp = np.clip(damp, 0.2, 0.6)

            avoid_strength = t_risk * 0.5
        else:
            gain = 1.0
            damp = 0.3
            avoid_strength = 0.0

        # 목표 방향
        goal_dir_norm = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir_norm)
        if goal_norm > 1e-6:
            goal_dir_norm = goal_dir_norm / goal_norm

        action = goal_dir_norm * gain

        # 장애물 회피
        if avoid_strength > 0 and (abs(closest_obs_dx) > 1e-6 or abs(closest_obs_dy) > 1e-6):
            obs_dir = np.array([closest_obs_dx, closest_obs_dy])
            obs_norm = np.linalg.norm(obs_dir)
            if obs_norm > 1e-6:
                obs_dir = obs_dir / obs_norm
                action -= obs_dir * avoid_strength

        action -= np.array([vel_x, vel_y]) * damp

        return action


# ============================================================================
# Test Runners
# ============================================================================

def run_single_noise_test(
    sigma: float,
    n_seeds: int = 30,
    seed: int = 42,
) -> Dict:
    """단일 노이즈 레벨 테스트"""
    print(f"\n  Testing σ = {sigma}...")

    configs = {
        'BASE': {'use_memory': False, 'use_hierarchy': False, 'use_risk_filter_only': False},
        '+MEM': {'use_memory': True, 'use_hierarchy': False, 'use_risk_filter_only': False},
        'FULL': {'use_memory': True, 'use_hierarchy': True, 'use_risk_filter_only': False},
        'FULL+RF': {'use_memory': False, 'use_hierarchy': False, 'use_risk_filter_only': True},
    }

    results = {}

    for config_name, config_opts in configs.items():
        total_rewards = []
        goals_completed_list = []
        triple_successes = 0
        collision_episodes = 0

        for s in range(n_seeds):
            env_config = NoiseConfig(
                noise_sigma=sigma,
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
            )
            env = NoiseNavEnv(env_config, seed=seed + s)
            agent = NoiseAgent(
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
                use_risk_filter_only=config_opts['use_risk_filter_only'],
            )
            agent.reset()

            obs = env.reset(seed=seed + s)

            for _ in range(env_config.max_steps):
                action = agent.act(obs)

                # 방어 모드 기록 (false defense 추적)
                env.record_defense_event(
                    agent.is_in_defense_mode(),
                    agent.get_perceived_risk()
                )

                obs, reward, done, info = env.step(action)
                if info.get('goal_switch'):
                    agent.on_goal_switch()
                if done:
                    break

            total_rewards.append(env.state.total_reward)
            goals_completed_list.append(env.state.goals_completed)

            if env.state.goals_completed == 3:
                triple_successes += 1
            if env.state.collision_this_episode:
                collision_episodes += 1

        noise_stats = env.get_noise_stats()

        results[config_name] = {
            'avg_reward': np.mean(total_rewards),
            'avg_goals': np.mean(goals_completed_list),
            'triple_rate': triple_successes / n_seeds,
            'collision_rate': collision_episodes / n_seeds,
            'false_defense_rate': noise_stats['false_defense_rate'],
            'near_miss_rate': noise_stats['near_miss_rate'],
        }

        print(f"    {config_name:>7}: Triple={triple_successes/n_seeds:.0%}, "
              f"Coll={collision_episodes/n_seeds:.0%}, "
              f"FalseDef={noise_stats['false_defense_rate']:.0%}, "
              f"NearMiss={noise_stats['near_miss_rate']:.1%}")

    return results


def run_noise_sweep(n_seeds: int = 30, seed: int = 42) -> Dict:
    """Noise sweep 테스트"""
    print("\n" + "=" * 70)
    print("  E6-4b: Observation Noise - Sweep Test")
    print("=" * 70)

    sigmas = [0.01, 0.03, 0.06]
    level_results = {}

    start_time = time.time()

    for sigma in sigmas:
        level_results[sigma] = run_single_noise_test(sigma, n_seeds, seed)

    elapsed = time.time() - start_time

    # Gate 평가
    gate = E6_4bGate()
    noise_result = gate.evaluate_noise_robustness(level_results)

    # FULL vs FULL+RF 비교 (σ=0.06에서)
    sigma_max = max(sigmas)
    full_results = level_results[sigma_max]['FULL']
    filter_results = level_results[sigma_max]['FULL+RF']
    filter_comparison = gate.compare_filter_variants(full_results, filter_results)

    # 결과 출력
    print(f"\n{'='*70}")
    print("  Noise Robustness Results")
    print(f"{'='*70}")

    print("\n  Collision rates by σ:")
    print(f"  {'σ':>6} | {'BASE':>6} | {'+MEM':>6} | {'FULL':>6} | {'FULL+RF':>7}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}")
    for sigma in sigmas:
        r = level_results[sigma]
        print(f"  {sigma:>6.2f} | {r['BASE']['collision_rate']:>5.0%} | "
              f"{r['+MEM']['collision_rate']:>5.0%} | "
              f"{r['FULL']['collision_rate']:>5.0%} | "
              f"{r['FULL+RF']['collision_rate']:>6.0%}")

    print(f"\n  False defense rates by σ:")
    print(f"  {'σ':>6} | {'FULL':>6} | {'FULL+RF':>7}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*7}")
    for sigma in sigmas:
        r = level_results[sigma]
        print(f"  {sigma:>6.2f} | {r['FULL']['false_defense_rate']:>5.0%} | "
              f"{r['FULL+RF']['false_defense_rate']:>6.0%}")

    print(f"\n  Robustness: [{'PASS' if noise_result.passed else 'FAIL'}] {noise_result.reason}")

    print(f"\n{'='*70}")
    print("  FULL vs FULL+risk_filter Comparison (at σ=0.06)")
    print(f"{'='*70}")
    print(f"\n  Collision: FULL={filter_comparison.full_collision:.0%}, "
          f"FULL+RF={filter_comparison.filter_collision:.0%} "
          f"[{'RF better' if filter_comparison.filter_better_collision else 'FULL better'}]")
    print(f"  False def: FULL={filter_comparison.full_false_defense:.0%}, "
          f"FULL+RF={filter_comparison.filter_false_defense:.0%} "
          f"[{'RF better' if filter_comparison.filter_better_false_defense else 'FULL better'}]")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'='*70}\n")

    return {
        'noise_result': noise_result,
        'filter_comparison': filter_comparison,
        'level_results': level_results,
        'elapsed_sec': elapsed,
    }


def run_full_e6_4b_test(n_seeds: int = 30) -> Dict:
    """E6-4b 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E6-4b: Observation Noise - Full Test")
    print("=" * 70)

    start_time = time.time()

    sweep = run_noise_sweep(n_seeds)

    elapsed = time.time() - start_time

    all_passed = sweep['noise_result'].passed

    print("\n" + "=" * 70)
    print("  E6-4b Final Summary")
    print("=" * 70)
    print(f"\n  Robustness Gate: [{'PASS' if sweep['noise_result'].passed else 'FAIL'}]")
    print(f"  FULL advantage in {sweep['noise_result'].full_advantage_levels}/3 levels")

    fc = sweep['filter_comparison']
    print(f"\n  Key insight (at max noise):")
    print(f"    FULL (action EMA): collision={fc.full_collision:.0%}, false_def={fc.full_false_defense:.0%}")
    print(f"    FULL+RF (risk filter only): collision={fc.filter_collision:.0%}, false_def={fc.filter_false_defense:.0%}")

    if fc.filter_better_collision and fc.filter_better_false_defense:
        print(f"\n  [CONFIRMED] FULL+RF proves: 'Action EMA is harmful, Risk filter is beneficial'")
    elif fc.filter_better_collision:
        print(f"\n  [PARTIAL] FULL+RF has lower collision but similar false_defense")
    else:
        print(f"\n  [UNCLEAR] FULL+RF not clearly better (needs investigation)")

    print(f"\n  Overall: [{'PASS' if all_passed else 'FAIL'}]")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'passed': all_passed,
        'sweep': sweep,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    if args.sigma is not None:
        result = run_single_noise_test(args.sigma, args.seeds)
    else:
        result = run_full_e6_4b_test(args.seeds)
        exit(0 if result['passed'] else 1)
