"""
E7-B1: Pop-up Obstacle Test

동적 돌발 이벤트에서 방어 구조 검증

핵심 지표:
1. Event collision rate: 이벤트 후 10 step 내 충돌률
2. Reaction time: 방어 모드 전환까지 시간 (mean, p95)
3. False defense: 이벤트 전 과방어율

실험군: BASE / +MEM / FULL / FULL+RF

Usage:
    python test_e7_popup.py
    python test_e7_popup.py --seeds 100
"""

import numpy as np
import sys
import os
from typing import Dict, List
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e7_popup import (
    PopupNavEnv, PopupConfig,
    E7_B1Gate, E7EventGateResult, E7PopupSummary,
)


# ============================================================================
# Agents (defense state reporting 추가)
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

        # Risk 계산 (reporting용)
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
    """+MEM: EMA 스무딩"""
    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.goal_ema = None
        self.in_defense_mode = False
        self.current_risk = 0.0

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        min_obs_dist = obs[16]

        # Risk 계산 (reporting용)
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
        self.current_risk = raw_risk

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

        # t_goal and t_risk
        t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
        raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)
        self.current_risk = raw_risk

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
    all_reaction_times = []
    all_false_defense = []

    for s in range(n_seeds):
        seed = base_seed + s
        config = PopupConfig()
        env = PopupNavEnv(config, seed=seed)
        agent = agent_class()
        agent.reset()

        obs = env.reset(seed=seed)

        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)

            # 방어 상태 기록 (reaction time 추적용)
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

        if env.state.reaction_time >= 0:
            all_reaction_times.append(env.state.reaction_time)

        if env.state.pre_event_total_steps > 0:
            fd = env.state.pre_event_defense_steps / env.state.pre_event_total_steps
            all_false_defense.append(fd)

    # Compute statistics
    mean_collision = np.mean(all_collisions)
    event_collision_rate = np.mean(all_event_collisions)
    triple_rate = sum(1 for g in all_goals if g == 3) / len(all_goals)

    if len(all_reaction_times) > 0:
        mean_rt = np.mean(all_reaction_times)
        sorted_rt = np.sort(all_reaction_times)
        p95_idx = min(int(len(sorted_rt) * 0.95), len(sorted_rt) - 1)
        p95_rt = sorted_rt[p95_idx]
    else:
        mean_rt = 0.0
        p95_rt = 0.0

    false_defense_rate = np.mean(all_false_defense) if all_false_defense else 0.0

    print(f"    Mean Coll: {mean_collision:.1%}, Event Coll: {event_collision_rate:.1%}")
    print(f"    RT: mean={mean_rt:.1f}, p95={p95_rt:.1f} | False Def: {false_defense_rate:.1%}")

    return {
        'mean_collision': mean_collision,
        'event_collision_rate': event_collision_rate,
        'mean_reaction_time': mean_rt,
        'p95_reaction_time': p95_rt,
        'false_defense_rate': false_defense_rate,
        'triple_success_rate': triple_rate,
        'n_episodes': n_seeds,
    }


def run_full_e7_b1_test(n_seeds: int = 100, base_seed: int = 42) -> Dict:
    """E7-B1 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E7-B1: Pop-up Obstacle - Adversarial Event Test")
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
    gate = E7_B1Gate()

    print(f"\n{'='*70}")
    print("  E7-B1 Event Gate Evaluation")
    print(f"{'='*70}")

    gate_results = {}
    for name, r in results.items():
        gr = gate.evaluate_event_response(
            r['event_collision_rate'],
            r['mean_reaction_time'],
            r['p95_reaction_time'],
            r['false_defense_rate'],
        )
        gate_results[name] = gr

        status = "[PASS]" if gr.passed else "[FAIL]"
        print(f"\n  {name}: {status}")
        print(f"    event_coll={r['event_collision_rate']:.1%} (gate<5%): "
              f"{'OK' if gr.event_collision_passed else 'FAIL'}")
        print(f"    p95_rt={r['p95_reaction_time']:.1f} (gate<10): "
              f"{'OK' if gr.reaction_time_passed else 'FAIL'}")
        print(f"    false_defense={r['false_defense_rate']:.1%}")

    # Summary table
    print(f"\n{'='*70}")
    print("  Summary Table")
    print(f"{'='*70}")
    print(f"\n  {'Config':<10} | {'Mean Coll':>10} | {'Event Coll':>11} | {'RT(p95)':>8} | {'False Def':>10} | {'Gate':>6}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")
    for name, r in results.items():
        gr = gate_results[name]
        status = "PASS" if gr.passed else "FAIL"
        print(f"  {name:<10} | {r['mean_collision']:>9.1%} | {r['event_collision_rate']:>10.1%} | "
              f"{r['p95_reaction_time']:>7.1f} | {r['false_defense_rate']:>9.1%} | {status:>6}")

    # Final verdict
    full_passed = gate_results['FULL'].passed
    rf_passed = gate_results['FULL+RF'].passed

    overall_passed = full_passed or rf_passed

    print(f"\n{'='*70}")
    print("  E7-B1 Final Verdict")
    print(f"{'='*70}")
    print(f"\n  FULL event gate: {'PASS' if full_passed else 'FAIL'}")
    print(f"  FULL+RF event gate: {'PASS' if rf_passed else 'FAIL'}")

    # Key findings
    print(f"\n  Key Findings:")

    # Event collision comparison
    base_ec = results['BASE']['event_collision_rate']
    mem_ec = results['+MEM']['event_collision_rate']
    full_ec = results['FULL']['event_collision_rate']
    rf_ec = results['FULL+RF']['event_collision_rate']

    print(f"    Event Collision: BASE={base_ec:.0%}, +MEM={mem_ec:.0%}, FULL={full_ec:.0%}, FULL+RF={rf_ec:.0%}")

    if mem_ec > base_ec:
        print(f"    [CONFIRMED] +MEM lag makes event response WORSE (+{(mem_ec-base_ec)*100:.0f}%p)")
    if full_ec < base_ec:
        print(f"    [CONFIRMED] FULL defense reduces event collision (-{(base_ec-full_ec)*100:.0f}%p)")

    # False defense comparison
    full_fd = results['FULL']['false_defense_rate']
    rf_fd = results['FULL+RF']['false_defense_rate']

    if rf_fd < full_fd:
        print(f"    [CONFIRMED] FULL+RF has lower false defense ({rf_fd:.0%} vs {full_fd:.0%})")

    print(f"\n  Overall: [{'PASS' if overall_passed else 'FAIL'}]")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'passed': overall_passed,
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

    result = run_full_e7_b1_test(args.seeds, args.seed)
    exit(0 if result['passed'] else 1)
