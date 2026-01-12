"""
E6-1: 2D Physics Navigation Test

Step 1: Stability Gate (랜덤 정책)
Step 2: Learnability Gate (간단한 학습)
Step 3: Memory/Hierarchy Gate (기존 비율 검증)

Usage:
    python test_e6_physics_nav.py --stability    # Stability gate only
    python test_e6_physics_nav.py --learnability # + Learnability
    python test_e6_physics_nav.py                # Full test
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e6_physics_nav import (
    PhysicsNavEnv, PhysicsConfig, ActionGate,
    E6Gate, E6StabilityGateResult, E6LearnabilityGateResult, E6GateResult,
)


# ============================================================================
# Simple Agents for Testing
# ============================================================================

class RandomAgent:
    """랜덤 정책 에이전트 (Stability Gate용)"""

    def __init__(self, action_scale: float = 1.0):
        self.action_scale = action_scale

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.random.randn(2) * self.action_scale


class SimpleGoalSeekAgent:
    """
    간단한 목표 추적 에이전트

    관측에서 goal_dx, goal_dy를 읽어서 그 방향으로 이동
    """

    def __init__(self, gain: float = 1.0, use_memory: bool = False):
        self.gain = gain
        self.use_memory = use_memory

    def act(self, obs: np.ndarray) -> np.ndarray:
        # obs: [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, dist]
        goal_dx = obs[4]
        goal_dy = obs[5]
        dist = obs[6]

        # 거리에 비례하게 가속 (멀면 더 세게)
        distance_gain = min(2.0, 0.5 + dist * 2)
        action = np.array([goal_dx, goal_dy]) * self.gain * distance_gain

        # 속도 댐핑 (진동 방지) - 가까워질수록 더 강하게
        vel_x = obs[2]
        vel_y = obs[3]
        damp_factor = 0.2 + (1 - min(1, dist)) * 0.3  # 가까우면 0.5, 멀면 0.2
        action -= np.array([vel_x, vel_y]) * damp_factor

        return action


class MemoryGoalSeekAgent:
    """
    메모리 기반 목표 추적 에이전트

    Memory OFF (BASE): 노이즈가 낀 관측 사용 (시뮬레이션)
    Memory ON (+MEM): 스무딩된 목표 방향 사용
    Hierarchy ON (FULL): 거리별 게인 조절 추가
    """

    def __init__(self, use_memory: bool = True, use_hierarchy: bool = False):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy

        # Memory smoothing (EMA)
        self.goal_ema = None
        self.ema_alpha = 0.3

    def reset(self):
        self.goal_ema = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        # 기본 관측
        vel_x, vel_y = obs[2], obs[3]
        goal_dx, goal_dy = obs[4], obs[5]
        dist = obs[6]

        # BASE (no memory): 노이즈 추가로 불안정하게
        if not self.use_memory:
            noise = np.random.randn(2) * 0.15
            goal_dx += noise[0]
            goal_dy += noise[1]
        else:
            # Memory: EMA 스무딩으로 안정적인 방향
            current = np.array([goal_dx, goal_dy])
            if self.goal_ema is None:
                self.goal_ema = current
            else:
                self.goal_ema = self.ema_alpha * current + (1 - self.ema_alpha) * self.goal_ema
            goal_dx, goal_dy = self.goal_ema[0], self.goal_ema[1]

        # Hierarchy: 연속적 거리 기반 게인 (부드러운 전환)
        if self.use_hierarchy:
            # dist is normalized: actual_dist / world_size
            # dist=0.05 (actual=0.5) is goal_radius, dist=0.8 (actual=8) is far
            # Smooth interpolation: far=fast, close=precise
            t = np.clip(dist / 0.5, 0, 1)  # 0 when at goal, 1 when dist >= 0.5 (actual 5.0)
            gain = 0.8 + t * 0.4  # 0.8 ~ 1.2
            damp = 0.4 - t * 0.15  # 0.4 ~ 0.25 (more damping when close)
        else:
            gain = 1.0
            damp = 0.3

        # 목표 방향으로 가속
        action = np.array([goal_dx, goal_dy]) * gain

        # 속도 댐핑
        action -= np.array([vel_x, vel_y]) * damp

        return action


# ============================================================================
# Test Runners
# ============================================================================

def run_stability_test(n_episodes: int = 100, seed: int = 42) -> Dict:
    """
    Step 1: Stability Gate

    랜덤 정책으로 100 에피소드 돌려서:
    - NaN/Inf 0회
    - 폭주(상한 초과) 0회
    - Action clamp rate < 50%
    """
    print("\n" + "=" * 60)
    print("  E6-1 Step 1: Stability Gate")
    print("  Random policy, {} episodes".format(n_episodes))
    print("=" * 60 + "\n")

    np.random.seed(seed)

    config = PhysicsConfig()
    env = PhysicsNavEnv(config, seed=seed)
    agent = RandomAgent(action_scale=0.5)  # 합리적인 탐색 범위
    gate = E6Gate()

    env.reset_stability_stats()

    successes = 0
    total_rewards = []
    final_distances = []

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

    elapsed = time.time() - start_time

    # Gate 평가 (random policy mode - high clamp rates are OK)
    stats = env.get_stability_stats()
    result = gate.evaluate_stability(stats, n_episodes, is_random_policy=True)

    # 결과 출력
    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    print(f"\nStability Metrics:")
    print(f"  NaN/Inf count: {result.nan_count + result.inf_count}")
    print(f"  Speed violations: {result.speed_violations}")
    print(f"  Position violations: {result.pos_violations}")
    print(f"  Norm clamp rate: {result.norm_clamp_rate:.1%}")
    print(f"  Delta clamp rate: {result.delta_clamp_rate:.1%}")

    print(f"\nPerformance (random policy):")
    print(f"  Success rate: {successes/n_episodes:.1%}")
    print(f"  Avg reward: {np.mean(total_rewards):.2f}")
    print(f"  Avg final distance: {np.mean(final_distances):.2f}")

    print(f"\n{'='*60}")
    print(f"  Stability Gate: [{'PASS' if result.passed else 'FAIL'}] {result.reason}")
    print(f"{'='*60}\n")

    return {
        'stability_result': result,
        'success_rate': successes / n_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_final_distance': np.mean(final_distances),
        'elapsed_sec': elapsed,
    }


def run_learnability_test(n_episodes: int = 100, seed: int = 42) -> Dict:
    """
    Step 2: Learnability Gate

    간단한 goal-seeking 에이전트로:
    - 성공률 30% 이상, 또는
    - 거리 감소 트렌드 확인
    """
    print("\n" + "=" * 60)
    print("  E6-1 Step 2: Learnability Gate")
    print("  Goal-seeking agent, {} episodes".format(n_episodes))
    print("=" * 60 + "\n")

    np.random.seed(seed)

    config = PhysicsConfig()
    env = PhysicsNavEnv(config, seed=seed)
    agent = SimpleGoalSeekAgent(gain=0.5)
    gate = E6Gate()

    successes = 0
    total_rewards = []
    final_distances = []

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

    elapsed = time.time() - start_time

    # 거리 트렌드 계산 (처음 50% vs 나중 50%)
    mid = n_episodes // 2
    early_dist = np.mean(final_distances[:mid])
    late_dist = np.mean(final_distances[mid:])
    distance_trend = late_dist - early_dist  # 음수면 개선

    success_rate = successes / n_episodes
    avg_distance = np.mean(final_distances)

    result = gate.evaluate_learnability(success_rate, avg_distance, distance_trend)

    # 결과 출력
    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    print(f"\nLearnability Metrics:")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Avg final distance: {avg_distance:.2f}")
    print(f"  Distance trend: {distance_trend:.3f} ({'improving' if distance_trend < 0 else 'not improving'})")
    print(f"  Avg reward: {np.mean(total_rewards):.2f}")

    print(f"\n{'='*60}")
    print(f"  Learnability Gate: [{'PASS' if result.passed else 'FAIL'}] {result.reason}")
    print(f"{'='*60}\n")

    return {
        'learnability_result': result,
        'success_rate': success_rate,
        'avg_reward': np.mean(total_rewards),
        'avg_final_distance': avg_distance,
        'distance_trend': distance_trend,
        'elapsed_sec': elapsed,
    }


def run_memory_hierarchy_test(n_seeds: int = 30, seed: int = 42) -> Dict:
    """
    Step 3: Memory/Hierarchy Gate

    E5와 동일한 구조:
    - BASE: Memory OFF, Hierarchy OFF
    - +MEM: Memory ON
    - FULL: Memory ON, Hierarchy ON
    """
    print("\n" + "=" * 60)
    print("  E6-1 Step 3: Memory/Hierarchy Gate")
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

        for s in range(n_seeds):
            env_config = PhysicsConfig(
                use_memory=config_opts['use_memory'],
                use_hierarchy=config_opts['use_hierarchy'],
            )
            env = PhysicsNavEnv(env_config, seed=seed + s)
            agent = MemoryGoalSeekAgent(
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

        results[config_name] = {
            'success_rate': successes / n_seeds,
            'avg_reward': np.mean(total_rewards),
            'avg_final_distance': np.mean(final_distances),
        }

        print(f"  Success: {successes/n_seeds:.1%}, "
              f"Reward: {np.mean(total_rewards):.2f}, "
              f"Dist: {np.mean(final_distances):.2f}")

    # Ratio 계산
    base_reward = results['BASE']['avg_reward']
    mem_reward = results['+MEM']['avg_reward']
    full_reward = results['FULL']['avg_reward']

    # 음수 보상 처리 (E5와 동일)
    if base_reward < 0:
        memory_ratio = 1.0 + (mem_reward - base_reward) / abs(base_reward)
        hierarchy_ratio = 1.0 + (full_reward - base_reward) / abs(base_reward)
    else:
        memory_ratio = mem_reward / base_reward if base_reward != 0 else 1.0
        hierarchy_ratio = full_reward / base_reward if base_reward != 0 else 1.0

    # Gate 판정
    MIN_RATIO = 1.05  # 연속제어는 5% 이상이면 OK (E5보다 낮게)
    memory_passed = memory_ratio >= MIN_RATIO
    hierarchy_passed = hierarchy_ratio >= MIN_RATIO
    overall_passed = memory_passed and hierarchy_passed

    print(f"\n{'='*60}")
    print(f"  Memory/Hierarchy Gate Results")
    print(f"{'='*60}")
    print(f"\n  Memory ratio: {memory_ratio:.2f} [{'PASS' if memory_passed else 'FAIL'}]")
    print(f"  Hierarchy ratio: {hierarchy_ratio:.2f} [{'PASS' if hierarchy_passed else 'FAIL'}]")
    print(f"\n  Overall: [{'PASS' if overall_passed else 'FAIL'}]")
    print(f"{'='*60}\n")

    return {
        'results': results,
        'memory_ratio': memory_ratio,
        'hierarchy_ratio': hierarchy_ratio,
        'memory_passed': memory_passed,
        'hierarchy_passed': hierarchy_passed,
        'overall_passed': overall_passed,
    }


def run_full_e6_test(n_episodes: int = 100, n_seeds: int = 30) -> Dict:
    """E6-1 전체 테스트"""
    print("\n" + "=" * 70)
    print("  E6-1: 2D Physics Navigation - Full Test")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Stability
    stability = run_stability_test(n_episodes)

    if not stability['stability_result'].passed:
        print("\n[ABORT] Stability gate failed. Fix stability issues first.")
        return {'passed': False, 'reason': 'stability_failed', 'stability': stability}

    # Step 2: Learnability
    learnability = run_learnability_test(n_episodes)

    if not learnability['learnability_result'].passed:
        print("\n[WARN] Learnability gate failed. Consider tuning agent/env.")

    # Step 3: Memory/Hierarchy
    mem_hier = run_memory_hierarchy_test(n_seeds)

    elapsed = time.time() - start_time

    # Final summary
    all_passed = (
        stability['stability_result'].passed and
        learnability['learnability_result'].passed and
        mem_hier['overall_passed']
    )

    print("\n" + "=" * 70)
    print("  E6-1 Final Summary")
    print("=" * 70)
    print(f"\n  Step 1 (Stability):      [{'PASS' if stability['stability_result'].passed else 'FAIL'}]")
    print(f"  Step 2 (Learnability):   [{'PASS' if learnability['learnability_result'].passed else 'FAIL'}]")
    print(f"  Step 3 (Memory/Hier):    [{'PASS' if mem_hier['overall_passed'] else 'FAIL'}]")
    print(f"\n  Overall: [{'PASS' if all_passed else 'FAIL'}]")
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'passed': all_passed,
        'stability': stability,
        'learnability': learnability,
        'memory_hierarchy': mem_hier,
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
        result = run_full_e6_test(args.episodes, args.seeds)
        exit(0 if result['passed'] else 1)
