"""
E6-4a: Partial Observability Test (Goal Dropout)

3 difficulty levels: p_drop = 0.1 / 0.3 / 0.5
각 레벨에서 BASE / +MEM / FULL 비교

핵심 게이트:
1. Stability / Safety
2. Robustness (FULL이 2+ 레벨에서 BASE 우위)
3. Lag sensitivity (event_collision_rate)

Usage:
    python test_e6_partial_obs.py
    python test_e6_partial_obs.py --level 0.3
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e6_partial_obs import (
    PartialObsNavEnv, PartialObsConfig,
    E6_4Gate, E6RobustnessGateResult, E6LagSensitivityResult,
)


# ============================================================================
# Agents for E6-4a
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


class PartialObsAgent:
    """
    부분 관측 환경용 에이전트

    핵심 개선:
    1. Risk hysteresis: r_on > r_off (방어 모드 튀김 방지)
    2. Goal switch 시 EMA 부분 리셋
    3. Dropout에서도 안정적 동작
    """

    def __init__(
        self,
        use_memory: bool = False,
        use_hierarchy: bool = False,
        d_safe: float = 0.3,
        ema_reset_factor: float = 0.3,
        # Risk hysteresis 파라미터
        risk_on_threshold: float = 0.4,   # 방어 모드 켜는 임계
        risk_off_threshold: float = 0.2,  # 방어 모드 끄는 임계
        risk_ema_alpha: float = 0.5,      # risk만 별도 스무딩 (빠르게)
    ):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy
        self.d_safe = d_safe
        self.ema_reset_factor = ema_reset_factor

        # Risk hysteresis
        self.risk_on_threshold = risk_on_threshold
        self.risk_off_threshold = risk_off_threshold
        self.risk_ema_alpha = risk_ema_alpha
        self.in_defense_mode = False

        # EMA states
        self.goal_ema = None
        self.risk_ema = None
        self.ema_alpha = 0.3

        # Goal switch detection
        self.prev_goal_dir = None
        self.switch_detected = False

    def reset(self):
        self.goal_ema = None
        self.risk_ema = None
        self.prev_goal_dir = None
        self.switch_detected = False
        self.in_defense_mode = False

    def on_goal_switch(self):
        """목표 전환 시 EMA 부분 리셋"""
        if self.goal_ema is not None:
            self.goal_ema = self.goal_ema * self.ema_reset_factor
        self.switch_detected = True

    def _detect_goal_switch(self, goal_dir: np.ndarray) -> bool:
        """목표 전환 감지"""
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
        """
        Risk hysteresis 적용

        - r_on (켜는 임계) > r_off (끄는 임계)
        - risk는 EMA 스무딩 적용 (센서 노이즈 방지)
        """
        # Risk EMA (빠른 스무딩)
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        smoothed_risk = self.risk_ema

        # Hysteresis 적용
        if self.in_defense_mode:
            # 방어 모드 중: 낮은 임계로 끄기
            if smoothed_risk < self.risk_off_threshold:
                self.in_defense_mode = False
        else:
            # 일반 모드: 높은 임계로 켜기
            if smoothed_risk > self.risk_on_threshold:
                self.in_defense_mode = True

        return smoothed_risk

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        """
        관측 구조:
        [0:2] pos_x, pos_y
        [2:4] vel_x, vel_y
        [4:7] goal_dx, goal_dy, dist_to_goal (dropout될 수 있음)
        [7:17] obstacle info
        [16] min_obstacle_dist
        """
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        goal_dir = np.array([goal_dx, goal_dy])

        # Goal switch 감지 및 처리
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

        # BASE: 노이즈 추가
        if not self.use_memory:
            noise = np.random.randn(2) * 0.15
            goal_dx += noise[0]
            goal_dy += noise[1]
        else:
            # Memory: EMA 스무딩
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

        # Hierarchy: t_goal + t_risk with hysteresis
        if self.use_hierarchy:
            t_goal = np.clip(dist_to_goal / 0.5, 0, 1)

            # Raw risk
            raw_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)

            # Hysteresis 적용된 risk
            smoothed_risk = self._update_risk_hysteresis(raw_risk)

            # 방어 모드에서는 더 보수적
            if self.in_defense_mode:
                t_risk = smoothed_risk * 1.2  # 방어 강화
                t_risk = min(t_risk, 1.0)
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

        # 속도 댐핑
        action -= np.array([vel_x, vel_y]) * damp

        return action


# ============================================================================
# Test Runners
# ============================================================================

def run_single_level_test(
    p_drop: float,
    n_seeds: int = 30,
    seed: int = 42,
) -> Dict:
    """단일 난이도 레벨 테스트"""
    print(f"\n  Testing p_drop = {p_drop}...")

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
        event_collisions_total = 0
        goal_switches_total = 0

        for s in range(n_seeds):
            env_config = PartialObsConfig(
                p_drop=p_drop,
                dropout_channel="goal",
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
            )
            env = PartialObsNavEnv(env_config, seed=seed + s)
            agent = PartialObsAgent(
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

            # Lag sensitivity
            lag_stats = env.get_lag_sensitivity_stats()
            event_collisions_total += lag_stats['event_collisions']
            goal_switches_total += lag_stats['total_goal_switches']

        event_collision_rate = event_collisions_total / max(1, goal_switches_total)

        results[config_name] = {
            'avg_reward': np.mean(total_rewards),
            'avg_goals': np.mean(goals_completed_list),
            'triple_rate': triple_successes / n_seeds,
            'collision_rate': collision_episodes / n_seeds,
            'event_collision_rate': event_collision_rate,
        }

        print(f"    {config_name}: Triple={triple_successes/n_seeds:.0%}, "
              f"Coll={collision_episodes/n_seeds:.0%}, "
              f"EventColl={event_collision_rate:.1%}")

    return results


def run_robustness_test(n_seeds: int = 30, seed: int = 42) -> Dict:
    """
    Robustness 테스트: 3 difficulty levels

    p_drop = 0.1, 0.3, 0.5
    """
    print("\n" + "=" * 70)
    print("  E6-4a: Partial Observability (Goal Dropout) - Robustness Test")
    print("=" * 70)

    levels = [0.1, 0.3, 0.5]
    level_results = {}

    start_time = time.time()

    for p_drop in levels:
        level_results[p_drop] = run_single_level_test(p_drop, n_seeds, seed)

    elapsed = time.time() - start_time

    # Gate 평가
    gate = E6_4Gate()
    robustness_result = gate.evaluate_robustness(level_results)

    # Lag sensitivity 분석 (평균)
    base_event_rates = [r['BASE']['event_collision_rate'] for r in level_results.values()]
    mem_event_rates = [r['+MEM']['event_collision_rate'] for r in level_results.values()]
    full_event_rates = [r['FULL']['event_collision_rate'] for r in level_results.values()]

    lag_result = gate.analyze_lag_sensitivity(
        np.mean(base_event_rates),
        np.mean(mem_event_rates),
        np.mean(full_event_rates),
    )

    # 결과 출력
    print(f"\n{'='*70}")
    print("  Robustness Gate Results")
    print(f"{'='*70}")

    for p_drop, results in level_results.items():
        threshold = gate.get_safety_threshold(p_drop)
        print(f"\n  p_drop = {p_drop} (safety threshold: {threshold:.0%})")
        for cfg, metrics in results.items():
            coll_status = "[OK]" if metrics['collision_rate'] <= threshold else "[HIGH]"
            print(f"    {cfg}: Triple={metrics['triple_rate']:.0%}, "
                  f"Coll={metrics['collision_rate']:.0%} {coll_status}, "
                  f"EventColl={metrics['event_collision_rate']:.1%}")

    print(f"\n  Robustness: [{'PASS' if robustness_result.passed else 'FAIL'}] {robustness_result.reason}")
    print(f"  FULL advantage: {robustness_result.full_advantage_levels}/{robustness_result.total_levels} levels")

    print(f"\n{'='*70}")
    print("  Lag Sensitivity Analysis")
    print(f"{'='*70}")
    print(f"\n  Event collision rate (avg across levels):")
    print(f"    BASE:  {lag_result.base_event_rate:.1%}")
    print(f"    +MEM:  {lag_result.mem_event_rate:.1%}")
    print(f"    FULL:  {lag_result.full_event_rate:.1%}")
    print(f"\n  +MEM lag penalty: {lag_result.mem_lag_penalty:+.1%}")
    if lag_result.mem_lag_penalty > 0:
        print(f"  (WARNING: +MEM increases event collisions by {lag_result.mem_lag_penalty:.1%})")
    else:
        print(f"  (+MEM reduces event collisions)")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'='*70}\n")

    return {
        'robustness_result': robustness_result,
        'lag_result': lag_result,
        'level_results': level_results,
        'elapsed_sec': elapsed,
    }


def run_full_e6_4a_test(n_seeds: int = 30) -> Dict:
    """E6-4a 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E6-4a: Partial Observability - Full Test")
    print("=" * 70)

    start_time = time.time()

    # Robustness 테스트 (3 levels)
    robustness = run_robustness_test(n_seeds)

    elapsed = time.time() - start_time

    # 최종 판정
    all_passed = robustness['robustness_result'].passed

    print("\n" + "=" * 70)
    print("  E6-4a Final Summary")
    print("=" * 70)
    print(f"\n  Robustness Gate: [{'PASS' if robustness['robustness_result'].passed else 'FAIL'}]")
    print(f"  FULL advantage in {robustness['robustness_result'].full_advantage_levels}/3 levels")

    lag = robustness['lag_result']
    print(f"\n  Lag penalty (+MEM vs BASE): {lag.mem_lag_penalty:+.1%}")
    print(f"  FULL event collision rate: {lag.full_event_rate:.1%}")

    print(f"\n  Overall: [{'PASS' if all_passed else 'FAIL'}]")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'passed': all_passed,
        'robustness': robustness,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=float, default=None, help="Test single level")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    if args.level is not None:
        result = run_single_level_test(args.level, args.seeds)
    else:
        result = run_full_e6_4a_test(args.seeds)
        exit(0 if result['passed'] else 1)
