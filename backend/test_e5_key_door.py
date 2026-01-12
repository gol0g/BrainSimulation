"""
E5 Task A: Key → Door Validation

목표:
- Memory ratio > 1.0 증명
- '열쇠를 본 경험'이 없으면 문을 열 수 없는 구조

실험 매트릭스:
- BASE: Memory OFF, Hierarchy OFF
- +MEM: Memory ON
- +HIE: Hierarchy ON
- FULL: 둘 다 ON

Usage:
    python test_e5_key_door.py                # Smoke (10 seeds)
    python test_e5_key_door.py --seeds 100    # Release (100 seeds)
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e5_key_door import (
    KeyDoorEnv, KeyDoorConfig, Action, CellType,
    E5Gate, E5GateResult, E5RunStats
)
from genesis.pc_z_dynamics import PCZDynamics


class SimpleMemoryAgent:
    """
    간단한 메모리 기반 에이전트

    Memory OFF: 현재 시야만 사용
    Memory ON: 마지막으로 본 Key/Door 위치 기억
    """

    def __init__(self, use_memory: bool = True, use_hierarchy: bool = False):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy

        # 내부 상태
        self.key_memory: Optional[Tuple[int, int]] = None
        self.door_memory: Optional[Tuple[int, int]] = None
        self.has_key = False
        self.door_passed = False  # 문을 통과했는지 추적
        self.current_goal = "find_key"  # find_key → go_to_door → go_to_goal

        # 탐색용 상태
        self.visited_recently: List[Tuple[int, int]] = []
        self.explore_direction = 0
        self.explore_counter = 0

    def reset(self):
        """에이전트 상태 리셋"""
        self.key_memory = None
        self.door_memory = None
        self.has_key = False
        self.door_passed = False
        self.current_goal = "find_key"
        self.visited_recently = []
        self.explore_direction = 0
        self.explore_counter = 0

    def act(self, obs: np.ndarray, agent_pos: Tuple[int, int]) -> int:
        """
        행동 선택

        obs 구조 (8D 또는 12D):
        [key_visible, key_dx, key_dy, door_visible, door_dx, door_dy, has_key, goal_visible]
        + [key_mem_dx, key_mem_dy, door_mem_dx, door_mem_dy] (memory 모드)
        """
        key_visible = obs[0] > 0.5
        key_dx, key_dy = obs[1], obs[2]
        door_visible = obs[3] > 0.5
        door_dx, door_dy = obs[4], obs[5]
        has_key = obs[6] > 0.5
        goal_visible = obs[7] > 0.5

        self.has_key = has_key

        # 메모리 업데이트
        if key_visible and self.use_memory:
            self.key_memory = (key_dx, key_dy)
        if door_visible and self.use_memory:
            self.door_memory = (door_dx, door_dy)

        # 메모리 관측 (12D 모드)
        if len(obs) >= 12 and self.use_memory:
            key_mem_dx, key_mem_dy = obs[8], obs[9]
            door_mem_dx, door_mem_dy = obs[10], obs[11]
        else:
            key_mem_dx, key_mem_dy = 0, 0
            door_mem_dx, door_mem_dy = 0, 0

        # 계층적 목표 설정 (Hierarchy)
        if self.use_hierarchy:
            if not has_key:
                self.current_goal = "find_key"
            elif not door_visible and self.door_memory:
                self.current_goal = "go_to_door"
            elif goal_visible:
                self.current_goal = "go_to_goal"
            else:
                self.current_goal = "explore"

        # 행동 결정
        # 1. 키 위에 있으면 → 줍기 (최우선)
        if key_visible and abs(key_dx) < 0.2 and abs(key_dy) < 0.2 and not has_key:
            return Action.PICKUP.value

        # 2. 키가 보이면 → 키로 이동
        if key_visible and not has_key:
            return self._move_towards(key_dx, key_dy)

        # 3. 키 있고 문 보이면 (문을 아직 통과하지 않은 경우만)
        if has_key and door_visible and not self.door_passed:
            # 문 위에 있으면 → 통과 (아래로)
            if abs(door_dx) < 0.2 and abs(door_dy) < 0.2:
                self.door_passed = True  # 문을 통과함!
                return Action.DOWN.value  # 문을 통과해서 아래 방으로
            else:
                return self._move_towards(door_dx, door_dy)

        # 4. 키 있고 문 기억 있으면 → 기억된 문으로 이동 (Memory 효과!)
        # 문을 이미 통과했으면 스킵
        if has_key and not self.door_passed and self.use_memory and (abs(door_mem_dx) > 0.01 or abs(door_mem_dy) > 0.01):
            return self._move_towards(door_mem_dx, door_mem_dy)

        # 5. 키 없고 키 기억 있으면 → 기억된 키로 이동 (Memory 효과!)
        if not has_key and self.use_memory and (abs(key_mem_dx) > 0.01 or abs(key_mem_dy) > 0.01):
            return self._move_towards(key_mem_dx, key_mem_dy)

        # 6. 문을 통과한 후 → Goal로 이동 (아래 방에 있음)
        if self.door_passed:
            # Goal이 보이면 아래로 이동
            if goal_visible:
                return Action.DOWN.value
            # Goal이 안 보여도 아래 방향으로 탐색
            return Action.DOWN.value

        # 7. Goal 보이면 → Goal 방향으로 이동 (문 통과 전)
        if goal_visible:
            return Action.DOWN.value  # goal은 아래 방에 있음

        # 8. 그 외 → 탐색
        return self._random_explore()

    def _move_towards(self, dx: float, dy: float) -> int:
        """방향으로 이동"""
        if abs(dx) > abs(dy):
            if dx > 0:
                return Action.DOWN.value
            else:
                return Action.UP.value
        else:
            if dy > 0:
                return Action.RIGHT.value
            else:
                return Action.LEFT.value

    def _random_explore(self) -> int:
        """체계적 탐색 - 벽에 부딪히면 방향 변경"""
        self.explore_counter += 1

        # 일정 시간 같은 방향, 또는 3스텝마다 방향 변경
        if self.explore_counter >= 3:
            self.explore_counter = 0
            # 랜덤하게 방향 변경
            self.explore_direction = np.random.randint(0, 4)

        return self.explore_direction


def run_single_episode(
    env: KeyDoorEnv,
    agent: SimpleMemoryAgent,
    dynamics: PCZDynamics,
    seed: int,
    config_name: str,
) -> E5RunStats:
    """단일 에피소드 실행"""
    np.random.seed(seed)
    obs = env.reset(seed=seed)
    agent.reset()

    # PC-Z 추적
    phase_violations = 0
    early_recovery_count = 0
    last_residual = 0.5
    last_openness = 0.0
    in_high_residual = False
    entry_weight = 0.0

    steps = 0
    success = False

    while True:
        action = agent.act(obs, env.state.agent_pos)
        obs, reward, done, info = env.step(action)
        steps += 1

        # PC 신호 계산 (환경 상태 기반)
        # 복잡한 상황일수록 residual 높음
        complexity = 0.2
        if not env.state.has_key:
            complexity += 0.1
        if not env.state.door_open:
            complexity += 0.1

        residual = np.clip(complexity + np.random.randn() * 0.05, 0.1, 0.8)
        error_norm = residual * 0.5

        epsilon = np.ones(8) * residual
        signals = dynamics.compute_pc_signals(
            epsilon=epsilon,
            error_norm=error_norm,
            iterations=15,
            max_iterations=30,
            converged=True,
            prior_force_norm=0.2,
            data_force_norm=error_norm,
            action_margin=0.5,
        )

        # Z modulation
        if residual > 0.5:
            z = 2
        elif residual > 0.3:
            z = 1
        else:
            z = 0

        dynamics.get_modulation_for_pc(
            z=z,
            z_confidence=0.7,
            regime_change_score=residual
        )

        current_residual = signals.residual_error
        current_weight = dynamics.dynamic_past_regime.w_applied
        current_openness = dynamics.dynamic_past_regime.compute_internal_stability()

        # Wrong-confidence 체크
        if current_residual > 0.5 and not in_high_residual:
            in_high_residual = True
            entry_weight = current_weight

        if current_residual <= 0.5 and in_high_residual:
            in_high_residual = False
            if current_weight - entry_weight > 0.02:
                phase_violations += 1

        if in_high_residual:
            if current_openness - last_openness > 0.2 and current_openness > 0.4:
                early_recovery_count += 1

        last_residual = current_residual
        last_openness = current_openness

        if done:
            success = env.state.goal_reached
            break

    # 최적 경로 대비 효율
    optimal = env.get_optimal_path_length()
    path_efficiency = optimal / steps if steps > 0 else 0.0

    return E5RunStats(
        config_name=config_name,
        seed=seed,
        task="key_door",
        success=success,
        steps_to_goal=steps if success else env.config.max_steps,
        total_reward=env.state.total_reward,
        optimal_path=optimal,
        path_efficiency=path_efficiency,
        key_remembered=env.state.key_seen,
        door_remembered=env.state.door_seen,
        phase_violations=phase_violations,
        early_recovery_count=early_recovery_count,
    )


def run_e5_task_a_validation(
    n_seeds: int = 10,
    max_steps: int = 500,
) -> Dict:
    """E5 Task A 검증 실행"""
    print(f"\n{'='*60}")
    print(f"  E5 Task A: Key → Door Validation")
    print(f"  {n_seeds} seeds, {max_steps} max steps")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Seeds
    seeds = list(range(n_seeds))

    # 실험 매트릭스
    configs = {
        "BASE": {"use_memory": False, "use_hierarchy": False},
        "+MEM": {"use_memory": True, "use_hierarchy": False},
        "+HIE": {"use_memory": False, "use_hierarchy": True},
        "FULL": {"use_memory": True, "use_hierarchy": True},
    }

    all_stats: Dict[str, List[E5RunStats]] = {name: [] for name in configs}

    for config_name, config_opts in configs.items():
        print(f"\nRunning {config_name}...")

        for seed in seeds:
            # 환경 설정
            env_config = KeyDoorConfig(
                max_steps=max_steps,
                use_memory=config_opts["use_memory"],
                use_hierarchy=config_opts["use_hierarchy"],
            )
            env = KeyDoorEnv(env_config, seed=seed)

            # 에이전트
            agent = SimpleMemoryAgent(
                use_memory=config_opts["use_memory"],
                use_hierarchy=config_opts["use_hierarchy"],
            )

            # PC-Z dynamics
            dynamics = PCZDynamics()

            # 에피소드 실행
            stats = run_single_episode(env, agent, dynamics, seed, config_name)
            all_stats[config_name].append(stats)

        # 요약 출력
        success_rate = np.mean([1 if s.success else 0 for s in all_stats[config_name]])
        avg_reward = np.mean([s.total_reward for s in all_stats[config_name]])
        avg_steps = np.mean([s.steps_to_goal for s in all_stats[config_name]])
        print(f"  Success: {success_rate:.1%}, Reward: {avg_reward:.2f}, Steps: {avg_steps:.0f}")

    # 게이트 평가
    gate = E5Gate()
    gate_result = gate.evaluate(
        all_stats["BASE"],
        all_stats["+MEM"],
        all_stats["FULL"],
    )

    elapsed = time.time() - start_time

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"  E5 Gate Results")
    print(f"{'='*60}\n")

    print(f"E5a Long-horizon Success:")
    print(f"  Baseline: {gate_result.baseline_success_rate:.1%}")
    print(f"  FULL:     {gate_result.success_rate:.1%}")
    print(f"  [{'PASS' if gate_result.e5a_passed else 'FAIL'}]")

    print(f"\nE5b Memory Benefit:")
    print(f"  Memory ratio: {gate_result.memory_benefit_ratio:.2f} (min: 1.10)")
    print(f"  Hierarchy ratio: {gate_result.hierarchy_benefit_ratio:.2f}")
    print(f"  [{'PASS' if gate_result.e5b_passed else 'FAIL'}]")

    print(f"\nE5c Wrong-confidence:")
    print(f"  Phase violations: {gate_result.phase_violation_total}")
    print(f"  Early recovery: {gate_result.early_recovery_total}")
    print(f"  [{'PASS' if gate_result.e5c_passed else 'FAIL'}]")

    print(f"\nE5d Sample Efficiency:")
    print(f"  BASE avg steps: {gate_result.avg_steps_base:.0f}")
    print(f"  FULL avg steps: {gate_result.avg_steps_full:.0f}")
    print(f"  Improvement: {gate_result.efficiency_improvement:.1%}")
    print(f"  [{'PASS' if gate_result.e5d_passed else 'FAIL'}]")

    print(f"\n{'='*60}")
    print(f"  Overall: [{'PASS' if gate_result.passed else 'FAIL'}] {gate_result.reason}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # 상세 비교표
    print("Configuration Comparison:")
    print(f"{'Config':<10} {'Success':<10} {'Reward':<12} {'Steps':<10} {'Efficiency':<12}")
    print("-" * 54)
    for config_name in configs:
        stats_list = all_stats[config_name]
        success = np.mean([1 if s.success else 0 for s in stats_list])
        reward = np.mean([s.total_reward for s in stats_list])
        steps = np.mean([s.steps_to_goal for s in stats_list])
        eff = np.mean([s.path_efficiency for s in stats_list])
        print(f"{config_name:<10} {success:<10.1%} {reward:<12.2f} {steps:<10.0f} {eff:<12.2%}")

    return {
        'n_seeds': n_seeds,
        'stats': {k: [s.__dict__ for s in v] for k, v in all_stats.items()},
        'gate_result': gate_result,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (smoke=10, release=100)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    args = parser.parse_args()

    results = run_e5_task_a_validation(
        n_seeds=args.seeds,
        max_steps=args.steps,
    )
    exit(0 if results['gate_result'].passed else 1)
