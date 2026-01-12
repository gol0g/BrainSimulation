"""
E6-4c: Sensor Latency Test

Latency sweep: k ∈ {0, 1, 3, 5} step delay
각 레벨에서 BASE / +MEM / FULL 비교

핵심 지표:
1. Lag-amplification slope: collision rate vs k
2. Risk reaction time: Δt

Usage:
    python test_e6_latency.py
    python test_e6_latency.py --k 3
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e6_latency import (
    LatencyNavEnv, LatencyConfig,
    E6_4cGate, E6LatencyGateResult, E6ReactionTimeResult,
)


# ============================================================================
# Agents (E6-4a와 동일, risk hysteresis 포함)
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


class LatencyAgent:
    """
    Latency 환경용 에이전트

    - Risk hysteresis
    - Goal switch EMA reset
    """

    def __init__(
        self,
        use_memory: bool = False,
        use_hierarchy: bool = False,
        d_safe: float = 0.3,
        ema_reset_factor: float = 0.3,
        risk_on_threshold: float = 0.4,
        risk_off_threshold: float = 0.2,
        risk_ema_alpha: float = 0.5,
    ):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy
        self.d_safe = d_safe
        self.ema_reset_factor = ema_reset_factor

        self.risk_on_threshold = risk_on_threshold
        self.risk_off_threshold = risk_off_threshold
        self.risk_ema_alpha = risk_ema_alpha
        self.in_defense_mode = False

        self.goal_ema = None
        self.risk_ema = None
        self.ema_alpha = 0.3

        self.prev_goal_dir = None
        self.switch_detected = False

    def reset(self):
        self.goal_ema = None
        self.risk_ema = None
        self.prev_goal_dir = None
        self.switch_detected = False
        self.in_defense_mode = False

    def on_goal_switch(self):
        if self.goal_ema is not None:
            self.goal_ema = self.goal_ema * self.ema_reset_factor
        self.switch_detected = True

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
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        smoothed_risk = self.risk_ema

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

        closest_obs_dx, closest_obs_dy = 0.0, 0.0
        min_dist = float('inf')
        for i in range(3):
            base = 7 + i * 3
            obs_dist = obs[base + 2]
            if obs_dist < min_dist:
                min_dist = obs_dist
                closest_obs_dx = obs[base]
                closest_obs_dy = obs[base + 1]

        if not self.use_memory:
            noise = np.random.randn(2) * 0.15
            goal_dx += noise[0]
            goal_dy += noise[1]
        else:
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

        if self.use_hierarchy:
            t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
            raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)
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

        goal_dir_norm = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir_norm)
        if goal_norm > 1e-6:
            goal_dir_norm = goal_dir_norm / goal_norm

        action = goal_dir_norm * gain

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

def run_single_latency_test(
    latency_k: int,
    n_seeds: int = 30,
    seed: int = 42,
) -> Dict:
    """단일 latency 레벨 테스트"""
    print(f"\n  Testing k = {latency_k} step delay...")

    configs = {
        'BASE': {'use_memory': False, 'use_hierarchy': False},
        '+MEM': {'use_memory': True, 'use_hierarchy': False},
        'FULL': {'use_memory': True, 'use_hierarchy': True},
    }

    results = {}

    for config_name, config_opts in configs.items():
        total_rewards = []
        goals_completed_list = []
        triple_successes = 0
        collision_episodes = 0
        all_reaction_times = []

        for s in range(n_seeds):
            env_config = LatencyConfig(
                latency_k=latency_k,
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
            )
            env = LatencyNavEnv(env_config, seed=seed + s)
            agent = LatencyAgent(
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
            )
            agent.reset()

            obs = env.reset(seed=seed + s)

            for _ in range(env_config.max_steps):
                action = agent.act(obs)
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

            # Reaction time
            rt_stats = env.get_reaction_time_stats()
            if rt_stats['n_events'] > 0:
                all_reaction_times.extend([rt_stats['mean_reaction_time']] * rt_stats['n_events'])

        mean_rt = np.mean(all_reaction_times) if all_reaction_times else 0.0

        results[config_name] = {
            'avg_reward': np.mean(total_rewards),
            'avg_goals': np.mean(goals_completed_list),
            'triple_rate': triple_successes / n_seeds,
            'collision_rate': collision_episodes / n_seeds,
            'mean_reaction_time': mean_rt,
        }

        print(f"    {config_name}: Triple={triple_successes/n_seeds:.0%}, "
              f"Coll={collision_episodes/n_seeds:.0%}, "
              f"RT={mean_rt:.1f}steps")

    return results


def run_latency_sweep(n_seeds: int = 30, seed: int = 42) -> Dict:
    """
    Latency sweep 테스트

    k ∈ {0, 1, 3, 5}
    """
    print("\n" + "=" * 70)
    print("  E6-4c: Sensor Latency - Sweep Test")
    print("=" * 70)

    latencies = [0, 1, 3, 5]
    level_results = {}

    start_time = time.time()

    for k in latencies:
        level_results[k] = run_single_latency_test(k, n_seeds, seed)

    elapsed = time.time() - start_time

    # Gate 평가
    gate = E6_4cGate()

    # Collision by k
    collision_by_k = {}
    for k, results in level_results.items():
        collision_by_k[k] = {
            'BASE': results['BASE']['collision_rate'],
            '+MEM': results['+MEM']['collision_rate'],
            'FULL': results['FULL']['collision_rate'],
        }

    latency_result = gate.evaluate_latency_robustness(collision_by_k)

    # Reaction time 분석 (k=5에서)
    k5_results = level_results.get(5, level_results[max(level_results.keys())])
    rt_result = gate.analyze_reaction_time(
        k5_results['BASE']['mean_reaction_time'],
        k5_results['+MEM']['mean_reaction_time'],
        k5_results['FULL']['mean_reaction_time'],
    )

    # 결과 출력
    print(f"\n{'='*70}")
    print("  Latency Robustness Results")
    print(f"{'='*70}")

    print("\n  Collision rates by latency k:")
    print(f"  {'k':>3} | {'BASE':>8} | {'+MEM':>8} | {'FULL':>8} | Threshold")
    print(f"  {'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")
    for k in latencies:
        threshold = gate.get_safety_threshold(k)
        base_c = collision_by_k[k]['BASE']
        mem_c = collision_by_k[k]['+MEM']
        full_c = collision_by_k[k]['FULL']
        print(f"  {k:>3} | {base_c:>7.0%} | {mem_c:>7.0%} | {full_c:>7.0%} | {threshold:.0%}")

    print(f"\n  Lag-amplification slope (collision increase per k):")
    print(f"    BASE:  {latency_result.base_slope:+.2%}/step")
    print(f"    +MEM:  {latency_result.mem_slope:+.2%}/step")
    print(f"    FULL:  {latency_result.full_slope:+.2%}/step")

    print(f"\n  FULL collision increase (k=0→5): {latency_result.full_increase:+.1%}")
    print(f"  Robustness: [{'PASS' if latency_result.passed else 'FAIL'}] {latency_result.reason}")

    print(f"\n{'='*70}")
    print("  Reaction Time Analysis (at k=5)")
    print(f"{'='*70}")
    print(f"\n  Mean reaction time (steps to avoidance after risk detected):")
    print(f"    BASE:  {rt_result.base_mean_rt:.1f} steps")
    print(f"    +MEM:  {rt_result.mem_mean_rt:.1f} steps")
    print(f"    FULL:  {rt_result.full_mean_rt:.1f} steps")
    print(f"\n  +MEM reaction time penalty: {rt_result.mem_rt_penalty:+.1f} steps")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'='*70}\n")

    return {
        'latency_result': latency_result,
        'rt_result': rt_result,
        'level_results': level_results,
        'elapsed_sec': elapsed,
    }


def run_full_e6_4c_test(n_seeds: int = 30) -> Dict:
    """E6-4c 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E6-4c: Sensor Latency - Full Test")
    print("=" * 70)

    start_time = time.time()

    # Latency sweep
    sweep = run_latency_sweep(n_seeds)

    elapsed = time.time() - start_time

    # 최종 판정
    all_passed = sweep['latency_result'].passed

    print("\n" + "=" * 70)
    print("  E6-4c Final Summary")
    print("=" * 70)
    print(f"\n  Robustness Gate: [{'PASS' if sweep['latency_result'].passed else 'FAIL'}]")
    print(f"  FULL increase (k=0→5): {sweep['latency_result'].full_increase:+.1%}")

    print(f"\n  Slope comparison:")
    print(f"    BASE:  {sweep['latency_result'].base_slope:+.2%}/step")
    print(f"    +MEM:  {sweep['latency_result'].mem_slope:+.2%}/step (expected steeper)")
    print(f"    FULL:  {sweep['latency_result'].full_slope:+.2%}/step (expected flattest)")

    print(f"\n  +MEM lag penalty: {sweep['rt_result'].mem_rt_penalty:+.1f} steps reaction time")

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
    parser.add_argument("--k", type=int, default=None, help="Test single latency")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    if args.k is not None:
        result = run_single_latency_test(args.k, args.seeds)
    else:
        result = run_full_e6_4c_test(args.seeds)
        exit(0 if result['passed'] else 1)
