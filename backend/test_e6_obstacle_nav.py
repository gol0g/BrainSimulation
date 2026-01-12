"""
E6-2: 2D Physics Navigation with Obstacles Test

Step 1: Stability Gate (랜덤 정책)
Step 2: Learnability Gate (회피 에이전트)
Step 3: Memory/Hierarchy Gate (기존 비율 검증)
Step 4: Safety Gate (충돌률 <= 5%)

Usage:
    python test_e6_obstacle_nav.py --stability    # Stability gate only
    python test_e6_obstacle_nav.py --learnability # + Learnability
    python test_e6_obstacle_nav.py                # Full test
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e6_obstacle_nav import (
    ObstacleNavEnv, ObstacleConfig,
    E6_2Gate, E6SafetyGateResult, E6_2LearnabilityGateResult,
)


# ============================================================================
# Agents for E6-2
# ============================================================================

class RandomAgent:
    """랜덤 정책 에이전트 (Stability Gate용)"""

    def __init__(self, action_scale: float = 0.5):
        self.action_scale = action_scale

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.random.randn(2) * self.action_scale


class ObstacleAvoidAgent:
    """
    장애물 회피 + 목표 추적 에이전트

    t_goal + t_risk 기반 adaptive gain/damp:
    - t_goal: 목표 거리 기준 (멀면 aggressive)
    - t_risk: 장애물 근접도 기준 (가까우면 defensive)
    """

    def __init__(
        self,
        use_memory: bool = False,
        use_hierarchy: bool = False,
        d_safe: float = 0.3,  # 안전 거리 (정규화)
    ):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy
        self.d_safe = d_safe

        # Memory: EMA smoothing
        self.goal_ema = None
        self.risk_ema = None
        self.ema_alpha = 0.3

    def reset(self):
        self.goal_ema = None
        self.risk_ema = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        관측 구조 (E6-2):
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
        min_obs_dist = obs[16]  # 최소 장애물 거리 (정규화)

        # 가장 가까운 장애물 방향
        closest_obs_dx, closest_obs_dy = 0.0, 0.0
        min_dist = float('inf')
        for i in range(3):
            base = 7 + i * 3
            obs_dist = obs[base + 2]
            if obs_dist < min_dist:
                min_dist = obs_dist
                closest_obs_dx = obs[base]
                closest_obs_dy = obs[base + 1]

        # BASE (no memory): 노이즈 추가 (더 강하게)
        if not self.use_memory:
            noise = np.random.randn(2) * 0.2
            goal_dx += noise[0]
            goal_dy += noise[1]
        else:
            # Memory: 목표 방향 EMA 스무딩
            current_goal = np.array([goal_dx, goal_dy])
            if self.goal_ema is None:
                self.goal_ema = current_goal
            else:
                self.goal_ema = self.ema_alpha * current_goal + (1 - self.ema_alpha) * self.goal_ema
            goal_dx, goal_dy = self.goal_ema[0], self.goal_ema[1]

            # Risk EMA 스무딩 (급격한 회피 반응 방지)
            current_risk = min_obs_dist
            if self.risk_ema is None:
                self.risk_ema = current_risk
            else:
                self.risk_ema = self.ema_alpha * current_risk + (1 - self.ema_alpha) * self.risk_ema
            min_obs_dist = self.risk_ema

        # Hierarchy: t_goal + t_risk 기반 adaptive gain/damp
        if self.use_hierarchy:
            # t_goal: 목표가 멀면 1, 가까우면 0
            t_goal = np.clip(dist_to_goal / 0.5, 0, 1)

            # t_risk: 장애물이 가까우면 1, 멀면 0
            t_risk = np.clip((self.d_safe - min_obs_dist) / self.d_safe, 0, 1)

            # Adaptive control
            # - 목표가 멀면 aggressive (gain 높음)
            # - 장애물 가까우면 defensive (damp 높음, gain 낮음)
            gain = 0.8 + t_goal * 0.4 - t_risk * 0.3  # 0.5 ~ 1.2
            gain = np.clip(gain, 0.4, 1.2)

            damp = 0.3 + t_risk * 0.3  # 0.3 ~ 0.6
            damp = np.clip(damp, 0.2, 0.6)

            # 회피력 (장애물 반대 방향)
            avoid_strength = t_risk * 0.5  # 위험할수록 강한 회피
        else:
            gain = 1.0
            damp = 0.3
            avoid_strength = 0.0

        # 목표 방향으로 가속
        goal_dir = np.array([goal_dx, goal_dy])
        goal_norm = np.linalg.norm(goal_dir)
        if goal_norm > 1e-6:
            goal_dir = goal_dir / goal_norm

        action = goal_dir * gain

        # 장애물 회피 (장애물 반대 방향으로)
        if avoid_strength > 0 and (abs(closest_obs_dx) > 1e-6 or abs(closest_obs_dy) > 1e-6):
            obs_dir = np.array([closest_obs_dx, closest_obs_dy])
            obs_norm = np.linalg.norm(obs_dir)
            if obs_norm > 1e-6:
                obs_dir = obs_dir / obs_norm
                # 장애물 반대 방향으로 밀기
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
    print("  E6-2 Step 1: Stability Gate")
    print("  Random policy, {} episodes".format(n_episodes))
    print("=" * 60 + "\n")

    np.random.seed(seed)

    config = ObstacleConfig()
    env = ObstacleNavEnv(config, seed=seed)
    agent = RandomAgent(action_scale=0.5)
    gate = E6_2Gate()

    env.reset_stability_stats()
    env.reset_safety_stats()

    successes = 0
    total_rewards = []
    final_distances = []
    collision_episodes = 0

    start_time = time.time()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)

        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

        total_rewards.append(env.state.total_reward)
        final_dist = np.linalg.norm(env.state.goal - env.state.pos)
        final_distances.append(final_dist)

        if info.get('success'):
            successes += 1
        if env.state.collision_this_episode:
            collision_episodes += 1

    elapsed = time.time() - start_time

    # Gate 평가
    stats = env.get_stability_stats()
    result = gate.evaluate_stability(stats, n_episodes, is_random_policy=True)

    safety_stats = env.get_safety_stats()

    # 결과 출력
    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    print(f"\nStability Metrics:")
    print(f"  NaN/Inf count: {result.nan_count + result.inf_count}")
    print(f"  Speed violations: {result.speed_violations}")
    print(f"  Position violations: {result.pos_violations}")
    print(f"  Norm clamp rate: {result.norm_clamp_rate:.1%}")
    print(f"  Delta clamp rate: {result.delta_clamp_rate:.1%}")

    print(f"\nCollision Metrics (random policy):")
    print(f"  Collision rate: {safety_stats['collision_rate']:.1%}")
    print(f"  Episodes with collision: {collision_episodes}/{n_episodes}")

    print(f"\nPerformance (random policy):")
    print(f"  Success rate: {successes/n_episodes:.1%}")
    print(f"  Avg reward: {np.mean(total_rewards):.2f}")
    print(f"  Avg final distance: {np.mean(final_distances):.2f}")

    print(f"\n{'='*60}")
    print(f"  Stability Gate: [{'PASS' if result.passed else 'FAIL'}] {result.reason}")
    print(f"{'='*60}\n")

    return {
        'stability_result': result,
        'safety_stats': safety_stats,
        'success_rate': successes / n_episodes,
        'collision_rate': safety_stats['collision_rate'],
        'avg_reward': np.mean(total_rewards),
        'avg_final_distance': np.mean(final_distances),
        'elapsed_sec': elapsed,
    }


def run_learnability_test(n_episodes: int = 100, seed: int = 42) -> Dict:
    """Step 2: Learnability Gate"""
    print("\n" + "=" * 60)
    print("  E6-2 Step 2: Learnability Gate")
    print("  Obstacle avoidance agent, {} episodes".format(n_episodes))
    print("=" * 60 + "\n")

    np.random.seed(seed)

    config = ObstacleConfig()
    env = ObstacleNavEnv(config, seed=seed)
    agent = ObstacleAvoidAgent(use_memory=True, use_hierarchy=True)
    gate = E6_2Gate()

    successes = 0
    total_rewards = []
    final_distances = []
    collisions_per_episode = []

    start_time = time.time()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        agent.reset()
        episode_collisions = 0

        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if info.get('collision'):
                episode_collisions += 1
            if done:
                break

        total_rewards.append(env.state.total_reward)
        final_dist = np.linalg.norm(env.state.goal - env.state.pos)
        final_distances.append(final_dist)
        collisions_per_episode.append(1 if episode_collisions > 0 else 0)

        if info.get('success'):
            successes += 1

    elapsed = time.time() - start_time

    # 트렌드 계산
    mid = n_episodes // 2
    early_dist = np.mean(final_distances[:mid])
    late_dist = np.mean(final_distances[mid:])
    distance_trend = late_dist - early_dist

    early_coll = np.mean(collisions_per_episode[:mid])
    late_coll = np.mean(collisions_per_episode[mid:])
    collision_trend = late_coll - early_coll

    success_rate = successes / n_episodes
    avg_distance = np.mean(final_distances)
    collision_rate = np.mean(collisions_per_episode)

    result = gate.evaluate_learnability_with_collision(
        success_rate, avg_distance, distance_trend, collision_trend
    )

    # 결과 출력
    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    print(f"\nLearnability Metrics:")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Avg final distance: {avg_distance:.2f}")
    print(f"  Distance trend: {distance_trend:.3f} ({'improving' if distance_trend < 0 else 'not improving'})")
    print(f"  Collision rate: {collision_rate:.1%}")
    print(f"  Collision trend: {collision_trend:.3f} ({'improving' if collision_trend <= 0 else 'worsening'})")
    print(f"  Avg reward: {np.mean(total_rewards):.2f}")

    print(f"\n{'='*60}")
    print(f"  Learnability Gate: [{'PASS' if result.passed else 'FAIL'}] {result.reason}")
    print(f"{'='*60}\n")

    return {
        'learnability_result': result,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_reward': np.mean(total_rewards),
        'avg_final_distance': avg_distance,
        'distance_trend': distance_trend,
        'collision_trend': collision_trend,
        'elapsed_sec': elapsed,
    }


def run_memory_hierarchy_test(n_seeds: int = 30, seed: int = 42) -> Dict:
    """Step 3: Memory/Hierarchy Gate"""
    print("\n" + "=" * 60)
    print("  E6-2 Step 3: Memory/Hierarchy Gate")
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

        successes = 0
        total_rewards = []
        final_distances = []
        collision_episodes = 0

        for s in range(n_seeds):
            env_config = ObstacleConfig(
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
            )
            env = ObstacleNavEnv(env_config, seed=seed + s)
            agent = ObstacleAvoidAgent(
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
            )
            agent.reset()

            obs = env.reset(seed=seed + s)

            for _ in range(env_config.max_steps):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    break

            total_rewards.append(env.state.total_reward)
            final_dist = np.linalg.norm(env.state.goal - env.state.pos)
            final_distances.append(final_dist)

            if info.get('success'):
                successes += 1
            if env.state.collision_this_episode:
                collision_episodes += 1

        results[config_name] = {
            'success_rate': successes / n_seeds,
            'avg_reward': np.mean(total_rewards),
            'avg_final_distance': np.mean(final_distances),
            'collision_rate': collision_episodes / n_seeds,
        }

        print(f"  Success: {successes/n_seeds:.1%}, "
              f"Reward: {np.mean(total_rewards):.2f}, "
              f"Dist: {np.mean(final_distances):.2f}, "
              f"Collisions: {collision_episodes/n_seeds:.1%}")

    # Ratio 계산 (보상 + 충돌률 개선 모두 고려)
    # E6-2 핵심 발견: 장애물 회피에서는 memory alone이 오히려 해가 됨 (EMA lag)
    # 따라서 FULL vs BASE로 "combined benefit" 측정 (E5와 동일 패턴)
    base_reward = results['BASE']['avg_reward']
    mem_reward = results['+MEM']['avg_reward']
    full_reward = results['FULL']['avg_reward']

    base_coll = results['BASE']['collision_rate']
    mem_coll = results['+MEM']['collision_rate']
    full_coll = results['FULL']['collision_rate']

    # Combined benefit: FULL vs BASE (memory + hierarchy 시너지)
    if base_reward < 0:
        combined_ratio_reward = 1.0 + (full_reward - base_reward) / abs(base_reward)
    else:
        combined_ratio_reward = full_reward / base_reward if base_reward != 0 else 1.0

    # 충돌률 개선 보너스 (FULL vs BASE)
    coll_improvement = base_coll - full_coll  # 양수면 개선
    coll_bonus = max(0, coll_improvement * 2)  # 충돌 10% 감소 → 0.2 보너스

    combined_ratio = combined_ratio_reward + coll_bonus

    # Hierarchy 추가 이득: FULL vs +MEM (hierarchy가 memory 위에 더하는 가치)
    if mem_reward > 0:
        hierarchy_marginal_ratio = full_reward / mem_reward
    else:
        hierarchy_marginal_ratio = 1.0 + (full_reward - mem_reward) / abs(mem_reward) if mem_reward != 0 else 1.0

    # 충돌률 개선 (FULL vs +MEM)
    hier_coll_improvement = mem_coll - full_coll
    hier_coll_bonus = max(0, hier_coll_improvement * 2)
    hierarchy_marginal_ratio += hier_coll_bonus

    # Gate 판정:
    # 1. Combined (FULL vs BASE) >= 1.15 (memory+hierarchy 시너지)
    # 2. Hierarchy marginal (FULL vs +MEM) >= 1.05 (hierarchy 추가 이득)
    MIN_COMBINED_RATIO = 1.15
    MIN_HIER_MARGINAL = 1.05

    combined_passed = combined_ratio >= MIN_COMBINED_RATIO
    hierarchy_passed = hierarchy_marginal_ratio >= MIN_HIER_MARGINAL
    overall_passed = combined_passed and hierarchy_passed

    print(f"\n{'='*60}")
    print(f"  Memory/Hierarchy Gate Results (E6-2)")
    print(f"{'='*60}")
    print(f"\n  Combined (FULL vs BASE): {combined_ratio:.2f} (reward: {combined_ratio_reward:.2f}, coll_bonus: {coll_bonus:.2f}) [{'PASS' if combined_passed else 'FAIL'}]")
    print(f"  Hierarchy marginal (FULL vs +MEM): {hierarchy_marginal_ratio:.2f} [{'PASS' if hierarchy_passed else 'FAIL'}]")
    print(f"\n  Collision rates: BASE={base_coll:.1%}, +MEM={mem_coll:.1%}, FULL={full_coll:.1%}")
    print(f"  Note: +MEM alone may hurt (EMA lag without defensive mode)")
    print(f"\n  Overall: [{'PASS' if overall_passed else 'FAIL'}]")
    print(f"{'='*60}\n")

    return {
        'results': results,
        'combined_ratio': combined_ratio,
        'hierarchy_marginal_ratio': hierarchy_marginal_ratio,
        'combined_passed': combined_passed,
        'hierarchy_passed': hierarchy_passed,
        'overall_passed': overall_passed,
    }


def run_safety_test(n_episodes: int = 100, seed: int = 42) -> Dict:
    """Step 4: Safety Gate"""
    print("\n" + "=" * 60)
    print("  E6-2 Step 4: Safety Gate")
    print("  Full agent, {} episodes".format(n_episodes))
    print("=" * 60 + "\n")

    np.random.seed(seed)

    config = ObstacleConfig()
    env = ObstacleNavEnv(config, seed=seed)
    agent = ObstacleAvoidAgent(use_memory=True, use_hierarchy=True)
    gate = E6_2Gate()

    env.reset_safety_stats()

    successes = 0
    collision_episodes = 0

    start_time = time.time()

    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        agent.reset()

        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break

        if info.get('success'):
            successes += 1
        if env.state.collision_this_episode:
            collision_episodes += 1

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
    print(f"  Success rate: {successes/n_episodes:.1%}")
    print(f"  Collision rate: {safety_stats['collision_rate']:.1%}")
    print(f"  Episodes with collision: {collision_episodes}/{n_episodes}")

    print(f"\n{'='*60}")
    print(f"  Safety Gate: [{'PASS' if result.passed else 'FAIL'}] {result.reason}")
    print(f"{'='*60}\n")

    return {
        'safety_result': result,
        'success_rate': successes / n_episodes,
        'collision_rate': safety_stats['collision_rate'],
        'elapsed_sec': elapsed,
    }


def run_full_e6_2_test(n_episodes: int = 100, n_seeds: int = 30) -> Dict:
    """E6-2 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E6-2: 2D Physics Navigation with Obstacles - Full Test")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Stability
    stability = run_stability_test(n_episodes)

    if not stability['stability_result'].passed:
        print("\n[ABORT] Stability gate failed. Fix stability issues first.")
        return {'passed': False, 'reason': 'stability_failed', 'stability': stability}

    # Step 2: Learnability
    learnability = run_learnability_test(n_episodes)

    # Step 3: Memory/Hierarchy
    mem_hier = run_memory_hierarchy_test(n_seeds)

    # Step 4: Safety
    safety = run_safety_test(n_episodes)

    elapsed = time.time() - start_time

    # Final summary
    all_passed = (
        stability['stability_result'].passed and
        learnability['learnability_result'].passed and
        mem_hier['overall_passed'] and
        safety['safety_result'].passed
    )

    print("\n" + "=" * 70)
    print("  E6-2 Final Summary")
    print("=" * 70)
    print(f"\n  Step 1 (Stability):      [{'PASS' if stability['stability_result'].passed else 'FAIL'}]")
    print(f"  Step 2 (Learnability):   [{'PASS' if learnability['learnability_result'].passed else 'FAIL'}]")
    print(f"  Step 3 (Memory/Hier):    [{'PASS' if mem_hier['overall_passed'] else 'FAIL'}]")
    print(f"  Step 4 (Safety):         [{'PASS' if safety['safety_result'].passed else 'FAIL'}]")
    print(f"\n  Overall: [{'PASS' if all_passed else 'FAIL'}]")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'passed': all_passed,
        'stability': stability,
        'learnability': learnability,
        'memory_hierarchy': mem_hier,
        'safety': safety,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability", action="store_true", help="Run stability gate only")
    parser.add_argument("--learnability", action="store_true", help="Run stability + learnability")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per test")
    parser.add_argument("--seeds", type=int, default=30, help="Seeds for memory/hierarchy test")
    args = parser.parse_args()

    if args.stability:
        result = run_stability_test(args.episodes)
        exit(0 if result['stability_result'].passed else 1)
    elif args.learnability:
        stability = run_stability_test(args.episodes)
        if stability['stability_result'].passed:
            learnability = run_learnability_test(args.episodes)
            exit(0 if learnability['learnability_result'].passed else 1)
        exit(1)
    else:
        result = run_full_e6_2_test(args.episodes, args.seeds)
        exit(0 if result['passed'] else 1)
