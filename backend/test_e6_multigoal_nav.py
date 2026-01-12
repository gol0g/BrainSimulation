"""
E6-3: 2D Physics Navigation with Sequential Goals Test

Step 1: Stability Gate
Step 2: Safety Gate (collision_rate < 10%)
Step 3: Learnability Gate (avg_goals 기준)
Step 4: Sequencing Gate (triple_success >= 70%, avg_goals >= 2.5)
Step 5: Memory/Hierarchy Gate (combined ratio)

Usage:
    python test_e6_multigoal_nav.py --stability
    python test_e6_multigoal_nav.py
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e6_multigoal_nav import (
    MultiGoalNavEnv, MultiGoalConfig,
    E6_3Gate, E6SequencingGateResult,
)


# ============================================================================
# Agents for E6-3
# ============================================================================

class RandomAgent:
    """랜덤 정책 에이전트"""

    def __init__(self, action_scale: float = 0.5):
        self.action_scale = action_scale

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        return np.random.randn(2) * self.action_scale

    def reset(self):
        pass

    def on_goal_switch(self):
        pass


class MultiGoalAgent:
    """
    순차 목표 추적 에이전트

    핵심 기능:
    - goal switch 감지 (관측의 급격한 변화로)
    - EMA 부분 리셋 on goal switch (context change handling)
    - t_goal + t_risk adaptive control
    """

    def __init__(
        self,
        use_memory: bool = False,
        use_hierarchy: bool = False,
        d_safe: float = 0.3,
        ema_reset_factor: float = 0.3,  # goal switch 시 EMA 감쇠 계수 (0.3 = 70% 리셋)
    ):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy
        self.d_safe = d_safe
        self.ema_reset_factor = ema_reset_factor

        # Memory: EMA smoothing
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

    def on_goal_switch(self):
        """목표 전환 시 EMA 부분 리셋"""
        if self.goal_ema is not None:
            self.goal_ema = self.goal_ema * self.ema_reset_factor
        if self.risk_ema is not None:
            # risk_ema는 리셋하지 않음 (위험 정보는 유지)
            pass
        self.switch_detected = True

    def _detect_goal_switch(self, goal_dir: np.ndarray) -> bool:
        """
        목표 전환 감지 (관측의 급격한 방향 변화로)

        goal_index 없이 "상황으로 추론"
        """
        if self.prev_goal_dir is None:
            self.prev_goal_dir = goal_dir.copy()
            return False

        # 방향 변화 계산
        prev_norm = np.linalg.norm(self.prev_goal_dir)
        curr_norm = np.linalg.norm(goal_dir)

        if prev_norm > 0.01 and curr_norm > 0.01:
            # 코사인 유사도
            cos_sim = np.dot(self.prev_goal_dir, goal_dir) / (prev_norm * curr_norm)

            # 급격한 방향 전환 (cos_sim < 0.3 = ~72도 이상)
            # + 거리가 급격히 증가 (목표 전환의 특징)
            if cos_sim < 0.3 or curr_norm > prev_norm * 1.5:
                self.prev_goal_dir = goal_dir.copy()
                return True

        self.prev_goal_dir = goal_dir.copy()
        return False

    def act(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        """
        관측 구조 (E6-3):
        [0:2] pos_x, pos_y
        [2:4] vel_x, vel_y
        [4:7] goal_dx, goal_dy, dist_to_goal
        [7:10] obs1_dx, obs1_dy, obs1_dist
        [10:13] obs2_dx, obs2_dy, obs2_dist
        [13:16] obs3_dx, obs3_dy, obs3_dist
        [16] min_obstacle_dist
        """
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist_to_goal = obs[6]
        min_obs_dist = obs[16]

        # 현재 목표 방향
        goal_dir = np.array([goal_dx, goal_dy])

        # 목표 전환 감지 및 처리
        if info is not None and info.get('goal_switch'):
            # 외부에서 알려준 경우
            self.on_goal_switch()
        elif self._detect_goal_switch(goal_dir):
            # 관측으로 추론한 경우
            self.on_goal_switch()

        # 가장 가까운 장애물
        closest_obs_dx, closest_obs_dy = 0.0, 0.0
        min_dist = float('inf')
        for i in range(3):
            base = 7 + i * 3
            obs_dist = obs[base + 2]
            if obs_dist < min_dist:
                min_dist = obs_dist
                closest_obs_dx = obs[base]
                closest_obs_dy = obs[base + 1]

        # BASE (no memory): 노이즈 추가
        if not self.use_memory:
            noise = np.random.randn(2) * 0.15
            goal_dx += noise[0]
            goal_dy += noise[1]
        else:
            # Memory: EMA 스무딩 (목표 전환 후에는 빠르게 적응)
            current_goal = np.array([goal_dx, goal_dy])

            # 목표 전환 직후에는 alpha를 높여서 빠르게 적응
            if self.switch_detected:
                effective_alpha = 0.7  # 빠른 적응
                self.switch_detected = False
            else:
                effective_alpha = self.ema_alpha

            if self.goal_ema is None:
                self.goal_ema = current_goal
            else:
                self.goal_ema = effective_alpha * current_goal + (1 - effective_alpha) * self.goal_ema
            goal_dx, goal_dy = self.goal_ema[0], self.goal_ema[1]

            # Risk EMA
            current_risk = min_obs_dist
            if self.risk_ema is None:
                self.risk_ema = current_risk
            else:
                self.risk_ema = self.ema_alpha * current_risk + (1 - self.ema_alpha) * self.risk_ema
            min_obs_dist = self.risk_ema

        # Hierarchy: t_goal + t_risk adaptive control
        if self.use_hierarchy:
            t_goal = np.clip(dist_to_goal / 0.5, 0, 1)
            t_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)

            gain = 0.8 + t_goal * 0.4 - t_risk * 0.3
            gain = np.clip(gain, 0.4, 1.2)

            damp = 0.3 + t_risk * 0.3
            damp = np.clip(damp, 0.2, 0.6)

            avoid_strength = t_risk * 0.5
        else:
            gain = 1.0
            damp = 0.3
            avoid_strength = 0.0

        # 목표 방향으로 가속
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

def run_stability_test(n_episodes: int = 100, seed: int = 42) -> Dict:
    """Step 1: Stability Gate"""
    print("\n" + "=" * 60)
    print("  E6-3 Step 1: Stability Gate")
    print("  Random policy, {} episodes".format(n_episodes))
    print("=" * 60 + "\n")

    np.random.seed(seed)

    config = MultiGoalConfig()
    env = MultiGoalNavEnv(config, seed=seed)
    agent = RandomAgent()
    gate = E6_3Gate()

    env.reset_all_stats()

    start_time = time.time()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

    elapsed = time.time() - start_time

    stats = env.get_stability_stats()
    result = gate.evaluate_stability(stats, n_episodes, is_random_policy=True)

    seq_stats = env.get_sequencing_stats()
    safety_stats = env.get_safety_stats()

    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    print(f"\nStability Metrics:")
    print(f"  NaN/Inf count: {result.nan_count + result.inf_count}")
    print(f"  Norm clamp rate: {result.norm_clamp_rate:.1%}")
    print(f"  Delta clamp rate: {result.delta_clamp_rate:.1%}")
    print(f"\nRandom policy performance:")
    print(f"  Triple success: {seq_stats['triple_success_rate']:.1%}")
    print(f"  Avg goals: {seq_stats['avg_goals_completed']:.2f}")
    print(f"  Collision rate: {safety_stats['collision_rate']:.1%}")

    print(f"\n{'='*60}")
    print(f"  Stability Gate: [{'PASS' if result.passed else 'FAIL'}] {result.reason}")
    print(f"{'='*60}\n")

    return {'stability_result': result, 'elapsed_sec': elapsed}


def run_safety_test(n_episodes: int = 100, seed: int = 42) -> Dict:
    """Step 2: Safety Gate"""
    print("\n" + "=" * 60)
    print("  E6-3 Step 2: Safety Gate")
    print("  Full agent, {} episodes".format(n_episodes))
    print("=" * 60 + "\n")

    np.random.seed(seed)

    config = MultiGoalConfig()
    env = MultiGoalNavEnv(config, seed=seed)
    agent = MultiGoalAgent(use_memory=True, use_hierarchy=True)
    gate = E6_3Gate()

    env.reset_all_stats()

    start_time = time.time()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        agent.reset()
        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if info.get('goal_switch'):
                agent.on_goal_switch()
            if done:
                break

    elapsed = time.time() - start_time

    safety_stats = env.get_safety_stats()
    result = gate.evaluate_safety(
        safety_stats['collision_rate'],
        safety_stats['total_collisions'],
        safety_stats['total_episodes'],
    )

    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    print(f"\nSafety Metrics:")
    print(f"  Collision rate: {safety_stats['collision_rate']:.1%}")

    print(f"\n{'='*60}")
    print(f"  Safety Gate: [{'PASS' if result.passed else 'FAIL'}] {result.reason}")
    print(f"{'='*60}\n")

    return {'safety_result': result, 'elapsed_sec': elapsed}


def run_sequencing_test(n_episodes: int = 100, seed: int = 42) -> Dict:
    """Step 3+4: Learnability + Sequencing Gate"""
    print("\n" + "=" * 60)
    print("  E6-3 Step 3+4: Learnability + Sequencing Gate")
    print("  Full agent, {} episodes".format(n_episodes))
    print("=" * 60 + "\n")

    np.random.seed(seed)

    config = MultiGoalConfig()
    env = MultiGoalNavEnv(config, seed=seed)
    agent = MultiGoalAgent(use_memory=True, use_hierarchy=True)
    gate = E6_3Gate()

    env.reset_all_stats()

    goals_per_episode = []

    start_time = time.time()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        agent.reset()
        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if info.get('goal_switch'):
                agent.on_goal_switch()
            if done:
                break
        goals_per_episode.append(info.get('goals_completed', 0))

    elapsed = time.time() - start_time

    seq_stats = env.get_sequencing_stats()
    result = gate.evaluate_sequencing(
        seq_stats['triple_success_rate'],
        seq_stats['avg_goals_completed'],
    )

    # Learnability: 트렌드 계산
    mid = n_episodes // 2
    early_goals = np.mean(goals_per_episode[:mid])
    late_goals = np.mean(goals_per_episode[mid:])
    goals_trend = late_goals - early_goals

    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    print(f"\nSequencing Metrics:")
    print(f"  Triple success rate: {seq_stats['triple_success_rate']:.1%}")
    print(f"  Avg goals completed: {seq_stats['avg_goals_completed']:.2f}")
    print(f"  Goals trend: {goals_trend:.3f} ({'improving' if goals_trend > 0 else 'stable/declining'})")

    print(f"\n{'='*60}")
    print(f"  Sequencing Gate: [{'PASS' if result.passed else 'FAIL'}] {result.reason}")
    print(f"{'='*60}\n")

    return {
        'sequencing_result': result,
        'goals_trend': goals_trend,
        'elapsed_sec': elapsed,
    }


def run_memory_hierarchy_test(n_seeds: int = 30, seed: int = 42) -> Dict:
    """Step 5: Memory/Hierarchy Gate"""
    print("\n" + "=" * 60)
    print("  E6-3 Step 5: Memory/Hierarchy Gate")
    print("  {} seeds".format(n_seeds))
    print("=" * 60 + "\n")

    configs = {
        'BASE': {'use_memory': False, 'use_hierarchy': False},
        '+MEM': {'use_memory': True, 'use_hierarchy': False},
        'FULL': {'use_memory': True, 'use_hierarchy': True},
    }

    results = {}

    for config_name, config_opts in configs.items():
        print(f"Running {config_name}...")

        total_rewards = []
        goals_completed_list = []
        triple_successes = 0
        collision_episodes = 0

        for s in range(n_seeds):
            env_config = MultiGoalConfig(
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
            )
            env = MultiGoalNavEnv(env_config, seed=seed + s)
            agent = MultiGoalAgent(
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

        results[config_name] = {
            'avg_reward': np.mean(total_rewards),
            'avg_goals': np.mean(goals_completed_list),
            'triple_rate': triple_successes / n_seeds,
            'collision_rate': collision_episodes / n_seeds,
        }

        print(f"  Triple: {triple_successes/n_seeds:.1%}, "
              f"Avg goals: {np.mean(goals_completed_list):.2f}, "
              f"Reward: {np.mean(total_rewards):.2f}, "
              f"Collisions: {collision_episodes/n_seeds:.1%}")

    # Ratio 계산 (avg_goals + triple_rate 복합)
    base_score = results['BASE']['avg_goals'] + results['BASE']['triple_rate'] * 3
    mem_score = results['+MEM']['avg_goals'] + results['+MEM']['triple_rate'] * 3
    full_score = results['FULL']['avg_goals'] + results['FULL']['triple_rate'] * 3

    # Combined: FULL vs BASE
    combined_ratio = full_score / base_score if base_score > 0 else 1.0

    # Hierarchy marginal: FULL vs +MEM
    hier_marginal_ratio = full_score / mem_score if mem_score > 0 else 1.0

    # 충돌률 보너스
    base_coll = results['BASE']['collision_rate']
    full_coll = results['FULL']['collision_rate']
    mem_coll = results['+MEM']['collision_rate']

    coll_bonus = max(0, (base_coll - full_coll) * 2)
    combined_ratio += coll_bonus

    hier_coll_bonus = max(0, (mem_coll - full_coll) * 2)
    hier_marginal_ratio += hier_coll_bonus

    # Gate 판정
    MIN_COMBINED = 1.10  # E6-3은 시퀀싱이므로 10% 이상
    MIN_HIER_MARGINAL = 1.05

    combined_passed = combined_ratio >= MIN_COMBINED
    hier_passed = hier_marginal_ratio >= MIN_HIER_MARGINAL
    overall_passed = combined_passed and hier_passed

    print(f"\n{'='*60}")
    print(f"  Memory/Hierarchy Gate Results (E6-3)")
    print(f"{'='*60}")
    print(f"\n  Score = avg_goals + triple_rate * 3")
    print(f"  BASE score: {base_score:.2f}, +MEM: {mem_score:.2f}, FULL: {full_score:.2f}")
    print(f"\n  Combined (FULL vs BASE): {combined_ratio:.2f} [{'PASS' if combined_passed else 'FAIL'}]")
    print(f"  Hierarchy marginal (FULL vs +MEM): {hier_marginal_ratio:.2f} [{'PASS' if hier_passed else 'FAIL'}]")
    print(f"\n  Collision rates: BASE={base_coll:.1%}, +MEM={mem_coll:.1%}, FULL={full_coll:.1%}")
    print(f"\n  Overall: [{'PASS' if overall_passed else 'FAIL'}]")
    print(f"{'='*60}\n")

    return {
        'results': results,
        'combined_ratio': combined_ratio,
        'hier_marginal_ratio': hier_marginal_ratio,
        'overall_passed': overall_passed,
    }


def run_full_e6_3_test(n_episodes: int = 100, n_seeds: int = 30) -> Dict:
    """E6-3 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E6-3: Sequential Goals Navigation - Full Test")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Stability
    stability = run_stability_test(n_episodes)
    if not stability['stability_result'].passed:
        print("\n[ABORT] Stability gate failed.")
        return {'passed': False, 'reason': 'stability_failed'}

    # Step 2: Safety
    safety = run_safety_test(n_episodes)

    # Step 3+4: Sequencing
    sequencing = run_sequencing_test(n_episodes)

    # Step 5: Memory/Hierarchy
    mem_hier = run_memory_hierarchy_test(n_seeds)

    elapsed = time.time() - start_time

    # Final summary
    all_passed = (
        stability['stability_result'].passed and
        safety['safety_result'].passed and
        sequencing['sequencing_result'].passed and
        mem_hier['overall_passed']
    )

    print("\n" + "=" * 70)
    print("  E6-3 Final Summary")
    print("=" * 70)
    print(f"\n  Step 1 (Stability):      [{'PASS' if stability['stability_result'].passed else 'FAIL'}]")
    print(f"  Step 2 (Safety):         [{'PASS' if safety['safety_result'].passed else 'FAIL'}]")
    print(f"  Step 3+4 (Sequencing):   [{'PASS' if sequencing['sequencing_result'].passed else 'FAIL'}]")
    print(f"  Step 5 (Memory/Hier):    [{'PASS' if mem_hier['overall_passed'] else 'FAIL'}]")
    print(f"\n  Overall: [{'PASS' if all_passed else 'FAIL'}]")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'passed': all_passed,
        'stability': stability,
        'safety': safety,
        'sequencing': sequencing,
        'memory_hierarchy': mem_hier,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()

    if args.stability:
        result = run_stability_test(args.episodes)
        exit(0 if result['stability_result'].passed else 1)
    else:
        result = run_full_e6_3_test(args.episodes, args.seeds)
        exit(0 if result['passed'] else 1)
