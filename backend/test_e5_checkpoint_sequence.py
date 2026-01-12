"""
E5 Task B: Checkpoint Sequence Validation

목표:
- Hierarchy ratio > 1.0 증명
- 순서를 기억하고, 현재 목표를 추적해야 함

실험 매트릭스:
- BASE: Memory OFF, Hierarchy OFF
- +MEM: Memory ON
- +HIE: Hierarchy ON
- FULL: 둘 다 ON

Usage:
    python test_e5_checkpoint_sequence.py                # Smoke (10 seeds)
    python test_e5_checkpoint_sequence.py --seeds 100    # Release (100 seeds)
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.e5_checkpoint_sequence import (
    CheckpointEnv, CheckpointConfig, ActionB,
    E5BGate, E5BGateResult, E5BRunStats
)
from genesis.pc_z_dynamics import PCZDynamics


class SimpleHierarchyAgent:
    """
    간단한 계층적 에이전트

    Memory OFF: 순서 기억 못함
    Memory ON: 순서 기억
    Hierarchy OFF: 현재 서브골 추적 안 함
    Hierarchy ON: 현재 서브골 추적
    """

    def __init__(self, use_memory: bool = True, use_hierarchy: bool = False):
        self.use_memory = use_memory
        self.use_hierarchy = use_hierarchy

        # 내부 상태
        self.remembered_sequence: List[int] = []
        self.current_subgoal_idx: int = 0
        self.completed_checkpoints: List[bool] = [False, False, False]

        # 탐색 상태
        self.explore_direction = 0
        self.explore_counter = 0

    def reset(self):
        """에이전트 상태 리셋"""
        self.remembered_sequence = []
        self.current_subgoal_idx = 0
        self.completed_checkpoints = [False, False, False]
        self.explore_direction = 0
        self.explore_counter = 0

    def act(self, obs: np.ndarray, step: int) -> int:
        """
        행동 선택

        핵심 로직:
        - BASE: 아무 체크포인트나 가서 활성화 (순서 무시) → 많은 wrong 페널티
        - +MEM: 순서 기억, 현재 단계 추적 → 정확한 순서로 활성화
        - +HIE: 기본 순서 A→B→C + 단계 추적 → 운에 따라
        - FULL: 순서 기억 + 단계 추적 → 정확한 순서로 활성화, 페널티 없음
        """
        # 기본 관측 파싱
        cp_a_dx, cp_a_dy = obs[1], obs[2]
        cp_b_dx, cp_b_dy = obs[4], obs[5]
        cp_c_dx, cp_c_dy = obs[7], obs[8]
        goal_visible = obs[9] > 0.5
        goal_dx, goal_dy = obs[10], obs[11]
        n_completed = int(obs[12] * 3 + 0.5)  # 완료된 체크포인트 수
        all_completed = obs[13] > 0.5

        # 힌트 파싱 (처음 5스텝)
        if len(obs) >= 17:
            hint_seq = [int(obs[14] * 2 + 0.5), int(obs[15] * 2 + 0.5), int(obs[16] * 2 + 0.5)]
        else:
            hint_seq = [0, 1, 2]

        # 메모리 업데이트 (처음에 순서 기억)
        if step < 5 and self.use_memory and not self.remembered_sequence:
            self.remembered_sequence = list(hint_seq)

        # 체크포인트 위치 정보 (dx, dy만)
        cp_dxdy = [(cp_a_dx, cp_a_dy), (cp_b_dx, cp_b_dy), (cp_c_dx, cp_c_dy)]

        # 1. 모든 체크포인트 완료 → Goal로 이동
        if all_completed:
            if goal_visible and (abs(goal_dx) > 0.01 or abs(goal_dy) > 0.01):
                return self._move_towards(goal_dx, goal_dy)
            return ActionB.DOWN.value

        # 2. 현재 목표 결정
        if self.use_memory and self.remembered_sequence:
            # Memory 있음: 기억한 순서 사용
            if self.use_hierarchy:
                # FULL: 순서 + n_completed로 현재 목표 결정
                current_target = self.remembered_sequence[n_completed] if n_completed < 3 else -1
            else:
                # +MEM only: 순서는 알지만 n_completed 없으면 첫 번째부터
                current_target = self.remembered_sequence[0]
        elif self.use_hierarchy:
            # +HIE only: 기본 순서 A→B→C + n_completed
            default_seq = [0, 1, 2]
            current_target = default_seq[n_completed] if n_completed < 3 else -1
        else:
            # BASE: 아무거나 (첫 번째 보이는 것)
            current_target = -1

        # 3. 목표로 이동 또는 활성화
        if current_target >= 0:
            dx, dy = cp_dxdy[current_target]
            if abs(dx) < 0.05 and abs(dy) < 0.05:
                return ActionB.ACTIVATE.value
            return self._move_towards(dx, dy)

        # 4. BASE 모드: 아무 체크포인트나 찾기
        for i, (dx, dy) in enumerate(cp_dxdy):
            if abs(dx) < 0.05 and abs(dy) < 0.05:
                return ActionB.ACTIVATE.value
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                return self._move_towards(dx, dy)

        return self._explore()

    def _move_towards(self, dx: float, dy: float) -> int:
        """방향으로 이동"""
        if abs(dx) > abs(dy):
            if dx > 0:
                return ActionB.DOWN.value
            else:
                return ActionB.UP.value
        else:
            if dy > 0:
                return ActionB.RIGHT.value
            else:
                return ActionB.LEFT.value

    def _explore(self) -> int:
        """체계적 탐색 - 더 넓은 영역 커버"""
        self.explore_counter += 1

        # 더 오래 한 방향으로 이동 (10스텝)
        if self.explore_counter >= 10:
            self.explore_counter = 0
            # 순환: DOWN → RIGHT → DOWN → LEFT → ...
            directions = [ActionB.DOWN.value, ActionB.RIGHT.value,
                         ActionB.DOWN.value, ActionB.LEFT.value]
            self.explore_direction = directions[self.explore_direction % 4]
            self.explore_direction = (self.explore_direction + 1) % 4

        # 주로 아래로 이동하면서 좌우로 sweep
        if self.explore_counter < 5:
            return ActionB.DOWN.value  # 아래로
        else:
            return ActionB.RIGHT.value if (self.explore_counter // 5) % 2 == 0 else ActionB.LEFT.value


def run_single_episode(
    env: CheckpointEnv,
    agent: SimpleHierarchyAgent,
    dynamics: PCZDynamics,
    seed: int,
    config_name: str,
) -> E5BRunStats:
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

    correct_activations = 0
    wrong_activations = 0
    steps = 0
    success = False

    while True:
        action = agent.act(obs, steps)
        obs, reward, done, info = env.step(action)
        steps += 1

        if info.get('correct_checkpoint'):
            correct_activations += 1
        if info.get('wrong_checkpoint'):
            wrong_activations += 1

        # PC 신호 계산
        complexity = 0.2 + 0.1 * (3 - sum(env.state.completed))
        residual = np.clip(complexity + np.random.randn() * 0.05, 0.1, 0.8)
        error_norm = residual * 0.5

        epsilon = np.ones(12) * residual
        signals = dynamics.compute_pc_signals(
            epsilon=epsilon[:8],
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

    return E5BRunStats(
        config_name=config_name,
        seed=seed,
        success=success,
        steps_to_goal=steps if success else env.config.max_steps,
        total_reward=env.state.total_reward,
        correct_activations=correct_activations,
        wrong_activations=wrong_activations,
        optimal_path=optimal,
        path_efficiency=path_efficiency,
        phase_violations=phase_violations,
        early_recovery_count=early_recovery_count,
    )


def run_e5_task_b_validation(
    n_seeds: int = 10,
    max_steps: int = 200,
) -> Dict:
    """E5 Task B 검증 실행"""
    print(f"\n{'='*60}")
    print(f"  E5 Task B: Checkpoint Sequence Validation")
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

    all_stats: Dict[str, List[E5BRunStats]] = {name: [] for name in configs}

    for config_name, config_opts in configs.items():
        print(f"\nRunning {config_name}...")

        for seed in seeds:
            # 환경 설정
            env_config = CheckpointConfig(
                max_steps=max_steps,
                use_memory=config_opts["use_memory"],
                use_hierarchy=config_opts["use_hierarchy"],
            )
            env = CheckpointEnv(env_config, seed=seed)

            # 에이전트
            agent = SimpleHierarchyAgent(
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
        correct = np.mean([s.correct_activations for s in all_stats[config_name]])
        wrong = np.mean([s.wrong_activations for s in all_stats[config_name]])
        print(f"  Success: {success_rate:.1%}, Reward: {avg_reward:.2f}, "
              f"Steps: {avg_steps:.0f}, Correct: {correct:.1f}, Wrong: {wrong:.1f}")

    # 게이트 평가
    gate = E5BGate()
    gate_result = gate.evaluate(
        all_stats["BASE"],
        all_stats["+HIE"],
        all_stats["FULL"],
    )

    elapsed = time.time() - start_time

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"  E5B Gate Results")
    print(f"{'='*60}\n")

    print(f"E5a Long-horizon Success:")
    print(f"  Baseline: {gate_result.baseline_success_rate:.1%}")
    print(f"  FULL:     {gate_result.success_rate:.1%}")
    print(f"  [{'PASS' if gate_result.e5a_passed else 'FAIL'}]")

    print(f"\nE5b Hierarchy Benefit:")
    print(f"  Hierarchy ratio: {gate_result.hierarchy_benefit_ratio:.2f} (min: 1.10)")
    print(f"  Memory ratio: {gate_result.memory_benefit_ratio:.2f}")
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
    print(f"{'Config':<10} {'Success':<10} {'Reward':<12} {'Steps':<10} {'Correct':<10} {'Wrong':<10}")
    print("-" * 62)
    for config_name in configs:
        stats_list = all_stats[config_name]
        success = np.mean([1 if s.success else 0 for s in stats_list])
        reward = np.mean([s.total_reward for s in stats_list])
        steps = np.mean([s.steps_to_goal for s in stats_list])
        correct = np.mean([s.correct_activations for s in stats_list])
        wrong = np.mean([s.wrong_activations for s in stats_list])
        print(f"{config_name:<10} {success:<10.1%} {reward:<12.2f} {steps:<10.0f} {correct:<10.1f} {wrong:<10.1f}")

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
    parser.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    args = parser.parse_args()

    results = run_e5_task_b_validation(
        n_seeds=args.seeds,
        max_steps=args.steps,
    )
    exit(0 if results['gate_result'].passed else 1)
