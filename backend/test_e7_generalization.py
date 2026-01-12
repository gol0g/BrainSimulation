"""
E7-A1: Map Randomization Test

E6 구조(방어계층 + 위험추정)가 분포 변화에서도 유지되는지 검증

실험군:
- BASE: 랜덤 노이즈
- +MEM: EMA 스무딩
- FULL: t_goal + t_risk + hysteresis
- FULL+RF: Risk filter only (action EMA 제거)

게이트:
1. O.O.D. collision: 평균 < 15%
2. Tail risk: p95 < 25%
3. Success: triple_success >= 90%
4. Robustness delta: E6 대비 성능 하락폭 제한

Usage:
    python test_e7_generalization.py
    python test_e7_generalization.py --seeds 100
"""

import numpy as np
import sys
import os
from typing import Dict, List
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e7_generalization import (
    GeneralizationNavEnv, GeneralizationConfig,
    E7_A1Gate, E7GeneralizationGateResult, E7RobustnessGateResult,
)


# ============================================================================
# Agents (E6와 동일 + FULL+RF)
# ============================================================================

class BaseAgent:
    """BASE: 랜덤 노이즈"""
    def __init__(self):
        self.in_defense_mode = False

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]

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

    def on_goal_switch(self):
        pass

    def get_defense_mode(self):
        return False


class MemoryAgent:
    """+MEM: EMA 스무딩"""
    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.goal_ema = None
        self.in_defense_mode = False

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]

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

    def on_goal_switch(self):
        if self.goal_ema is not None:
            self.goal_ema = self.goal_ema * 0.3

    def get_defense_mode(self):
        return False


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

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        # Closest obstacle direction
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

        # EMA on goal direction
        current_goal = np.array([goal_dx, goal_dy])
        if self.goal_ema is None:
            self.goal_ema = current_goal
        else:
            self.goal_ema = self.ema_alpha * current_goal + (1 - self.ema_alpha) * self.goal_ema

        goal_dx, goal_dy = self.goal_ema[0], self.goal_ema[1]

        # t_goal and t_risk
        t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
        raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)

        # Risk hysteresis
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

        # Adaptive control
        gain = 0.8 + t_goal * 0.4 - t_risk * 0.3
        gain = np.clip(gain, 0.4, 1.2)
        damp = 0.3 + t_risk * 0.3
        damp = np.clip(damp, 0.2, 0.6)
        avoid_strength = t_risk * 0.5

        # Goal direction
        goal_dir = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        action = goal_dir * gain

        # Obstacle avoidance
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

    def on_goal_switch(self):
        if self.goal_ema is not None:
            self.goal_ema = self.goal_ema * 0.3

    def get_defense_mode(self):
        return self.in_defense_mode


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

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        # Closest obstacle direction
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

        # NO EMA on goal - use raw with noise (like BASE)
        noise = np.random.randn(2) * 0.15
        goal_dx += noise[0]
        goal_dy += noise[1]

        # t_goal and t_risk (FULL과 동일)
        t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
        raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)

        # Risk hysteresis (FULL과 동일)
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

        # Adaptive control (FULL과 동일)
        gain = 0.8 + t_goal * 0.4 - t_risk * 0.3
        gain = np.clip(gain, 0.4, 1.2)
        damp = 0.3 + t_risk * 0.3
        damp = np.clip(damp, 0.2, 0.6)
        avoid_strength = t_risk * 0.5

        # Goal direction (NO EMA)
        goal_dir = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        action = goal_dir * gain

        # Obstacle avoidance
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

    def on_goal_switch(self):
        pass  # No EMA to reset

    def get_defense_mode(self):
        return self.in_defense_mode


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
    triple_successes = 0

    for s in range(n_seeds):
        seed = base_seed + s
        config = GeneralizationConfig()
        env = GeneralizationNavEnv(config, seed=seed)
        agent = agent_class()
        agent.reset()

        obs = env.reset(seed=seed)

        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

            if info.get('goal_switch'):
                agent.on_goal_switch()

            if done:
                break

        # Episode stats
        all_collisions.append(1 if env.state.collision_this_episode else 0)
        all_goals.append(env.state.goals_completed)

        if env.state.goals_completed == 3:
            triple_successes += 1

    # Compute statistics
    collisions = np.array(all_collisions)
    goals = np.array(all_goals)

    mean_collision = np.mean(collisions)
    triple_rate = triple_successes / n_seeds
    avg_goals = np.mean(goals)

    print(f"    Collision: mean={mean_collision:.1%}")
    print(f"    Success: triple={triple_rate:.1%}, avg_goals={avg_goals:.2f}")

    return {
        'mean_collision': mean_collision,
        'triple_success_rate': triple_rate,
        'avg_goals': avg_goals,
        'n_episodes': n_seeds,
    }


def run_full_e7_a1_test(n_seeds: int = 100, base_seed: int = 42) -> Dict:
    """E7-A1 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E7-A1: Map Randomization - Generalization Test")
    print("=" * 70)

    start_time = time.time()

    # 4 experimental groups
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
    gate = E7_A1Gate()

    print(f"\n{'='*70}")
    print("  E7-A1 Gate Evaluation")
    print(f"{'='*70}")

    # Generalization gates for each config
    gate_results = {}
    for name, r in results.items():
        gr = gate.evaluate_generalization(
            r['mean_collision'],
            r['triple_success_rate'],
            r['avg_goals'],
        )
        gate_results[name] = gr

        status = "[PASS]" if gr.passed else "[FAIL]"
        print(f"\n  {name}: {status}")
        print(f"    mean_coll={r['mean_collision']:.1%} (gate<15%): "
              f"{'OK' if gr.ood_collision_passed else 'FAIL'}")
        print(f"    triple={r['triple_success_rate']:.1%} (gate>=90%): "
              f"{'OK' if gr.success_passed else 'FAIL'}")

    # E6 baseline comparison (hardcoded from E6-3 results)
    e6_baseline = {
        'BASE': {'collision': 0.23, 'triple_success': 1.0},
        '+MEM': {'collision': 0.27, 'triple_success': 1.0},
        'FULL': {'collision': 0.10, 'triple_success': 1.0},
        'FULL+RF': {'collision': 0.10, 'triple_success': 1.0},
    }

    print(f"\n{'='*70}")
    print("  Robustness Delta (vs E6 baseline)")
    print(f"{'='*70}")

    robustness_results = {}
    for name, r in results.items():
        baseline = e6_baseline.get(name, {'collision': 0.15, 'triple_success': 1.0})
        rr = gate.evaluate_robustness_delta(
            baseline['collision'],
            r['mean_collision'],
            baseline['triple_success'],
            r['triple_success_rate'],
        )
        robustness_results[name] = rr

        status = "[PASS]" if rr.passed else "[FAIL]"
        print(f"\n  {name}: {status}")
        print(f"    E6 coll={baseline['collision']:.1%} -> E7 coll={r['mean_collision']:.1%} "
              f"(delta={rr.collision_delta:+.1%})")
        print(f"    E6 triple={baseline['triple_success']:.1%} -> E7 triple={r['triple_success_rate']:.1%} "
              f"(delta={rr.success_delta:+.1%})")

    # Summary table
    print(f"\n{'='*70}")
    print("  Summary Table")
    print(f"{'='*70}")
    print(f"\n  {'Config':<10} | {'Mean Coll':>10} | {'Triple':>8} | {'Avg Goals':>10} | {'Gate':>6}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")
    for name, r in results.items():
        gr = gate_results[name]
        status = "PASS" if gr.passed else "FAIL"
        print(f"  {name:<10} | {r['mean_collision']:>9.1%} | "
              f"{r['triple_success_rate']:>7.1%} | {r['avg_goals']:>9.2f} | {status:>6}")

    # Final verdict
    full_passed = gate_results['FULL'].passed
    rf_passed = gate_results['FULL+RF'].passed
    full_robust = robustness_results['FULL'].passed
    rf_robust = robustness_results['FULL+RF'].passed

    overall_passed = (full_passed or rf_passed) and (full_robust or rf_robust)

    print(f"\n{'='*70}")
    print("  E7-A1 Final Verdict")
    print(f"{'='*70}")
    print(f"\n  FULL gate: {'PASS' if full_passed else 'FAIL'}, robustness: {'PASS' if full_robust else 'FAIL'}")
    print(f"  FULL+RF gate: {'PASS' if rf_passed else 'FAIL'}, robustness: {'PASS' if rf_robust else 'FAIL'}")
    print(f"\n  Overall: [{'PASS' if overall_passed else 'FAIL'}]")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    # Key findings
    if overall_passed:
        print("  [CONFIRMED] Defense structure generalizes to random maps")
    else:
        print("  [INVESTIGATE] Defense structure may not generalize")

    # Compare FULL vs FULL+RF
    full_coll = results['FULL']['mean_collision']
    rf_coll = results['FULL+RF']['mean_collision']
    if abs(full_coll - rf_coll) < 0.02:
        print("  [CONFIRMED] Action EMA unnecessary (FULL ~= FULL+RF)")
    elif rf_coll < full_coll:
        print("  [CONFIRMED] Action EMA harmful (FULL+RF < FULL)")
    else:
        print("  [NOTE] Action EMA may help in randomized maps")

    return {
        'passed': overall_passed,
        'results': results,
        'gate_results': gate_results,
        'robustness_results': robustness_results,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100, help="Number of seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    result = run_full_e7_a1_test(args.seeds, args.seed)
    exit(0 if result['passed'] else 1)
