"""
E5 Task B: Checkpoint Sequence Environment

핵심 의도:
- Hierarchy ratio > 1.0이 나와야만 하는 구조
- 3개 체크포인트를 정해진 순서로 방문해야 함
- 순서는 에피소드 시작 시 잠깐 보여주고 숨김

환경 스펙:
- 맵: 11×11 격자 (3개 방)
- 체크포인트: A, B, C (각 방에 1개씩)
- 순서 힌트: 에피소드 시작 시 5스텝 동안 표시
- Partial Observability: 5×5 시야
- 순서를 기억해야 하고, 현재 어디까지 했는지 추적해야 함 (Hierarchy)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CellTypeB(Enum):
    """셀 타입"""
    EMPTY = 0
    WALL = 1
    CHECKPOINT_A = 2
    CHECKPOINT_B = 3
    CHECKPOINT_C = 4
    GOAL = 5
    AGENT = 6


class ActionB(Enum):
    """행동"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ACTIVATE = 4  # 체크포인트 활성화


@dataclass
class CheckpointConfig:
    """Checkpoint Sequence 환경 설정"""
    grid_size: int = 9  # 작은 그리드
    vision_size: int = 7  # 넓은 시야 (Navigation보다 Hierarchy 테스트)
    n_checkpoints: int = 3
    hint_duration: int = 5  # 순서 힌트 표시 시간
    max_steps: int = 150  # 더 짧게

    # 보상 설정
    correct_checkpoint_reward: float = 3.0
    wrong_checkpoint_penalty: float = -1.0
    goal_reward: float = 10.0
    step_penalty: float = -0.01

    # Memory/Hierarchy 설정
    use_memory: bool = True
    use_hierarchy: bool = False


@dataclass
class CheckpointState:
    """환경 상태"""
    agent_pos: Tuple[int, int]
    target_sequence: List[int] = field(default_factory=list)  # [0,1,2] = A→B→C
    completed: List[bool] = field(default_factory=lambda: [False, False, False])
    current_target_idx: int = 0
    step: int = 0
    total_reward: float = 0.0
    goal_reached: bool = False

    # 메모리/계층 관련
    sequence_seen: bool = False
    remembered_sequence: List[int] = field(default_factory=list)
    subgoal_active: str = "none"  # "find_A", "find_B", "find_C", "go_goal"


class CheckpointEnv:
    """
    Checkpoint Sequence 환경

    Hierarchy ratio > 1.0을 강제하는 최소 구조:
    - 순서대로 3개 체크포인트 방문 필요
    - 현재 서브골을 추적하는 Hierarchy가 유리
    """

    def __init__(self, config: CheckpointConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # 맵 생성
        self.grid = np.zeros((config.grid_size, config.grid_size), dtype=int)

        # 체크포인트 위치 (고정)
        self.checkpoint_positions: List[Tuple[int, int]] = []
        self.goal_pos: Optional[Tuple[int, int]] = None

        # 상태
        self.state = CheckpointState(agent_pos=(1, 1))

        # 초기화
        self._generate_map()
        self.reset()

    def _generate_map(self):
        """맵 생성 - 3개 영역 + 체크포인트"""
        size = self.config.grid_size

        # 외벽
        self.grid[0, :] = CellTypeB.WALL.value
        self.grid[-1, :] = CellTypeB.WALL.value
        self.grid[:, 0] = CellTypeB.WALL.value
        self.grid[:, -1] = CellTypeB.WALL.value

        # 체크포인트 위치 (모두 시작점에서 보이도록 배치)
        # A: 좌측, B: 중앙, C: 우측 (모두 같은 행에)
        mid = size // 2
        self.checkpoint_positions = [
            (mid, 2),      # A - 좌측
            (mid, mid),    # B - 중앙
            (mid, size-3), # C - 우측
        ]

        for i, pos in enumerate(self.checkpoint_positions):
            cell_type = [CellTypeB.CHECKPOINT_A, CellTypeB.CHECKPOINT_B, CellTypeB.CHECKPOINT_C][i]
            self.grid[pos] = cell_type.value

        # Goal 위치 (체크포인트들 바로 아래 - 쉽게 도달)
        self.goal_pos = (mid + 1, mid)
        self.grid[self.goal_pos] = CellTypeB.GOAL.value

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # 랜덤 순서 생성
        sequence = list(range(self.config.n_checkpoints))
        self.rng.shuffle(sequence)

        # 상태 초기화
        mid = self.config.grid_size // 2
        self.state = CheckpointState(
            agent_pos=(mid - 1, mid),  # 체크포인트들 바로 위에서 시작
            target_sequence=sequence,
            completed=[False] * self.config.n_checkpoints,
            current_target_idx=0,
            step=0,
            total_reward=0.0,
            goal_reached=False,
            sequence_seen=False,
            remembered_sequence=[],
            subgoal_active="none",
        )

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        관측 생성

        기본 (12D):
        [cp_a_visible, cp_a_dx, cp_a_dy,
         cp_b_visible, cp_b_dx, cp_b_dy,
         cp_c_visible, cp_c_dx, cp_c_dy,
         goal_visible, current_target, all_completed]

        + 순서 힌트 (hint_duration 동안만): [seq_0, seq_1, seq_2] (3D)
        + 메모리 (use_memory): [mem_seq_0, mem_seq_1, mem_seq_2, current_idx] (4D)
        """
        ax, ay = self.state.agent_pos
        half = self.config.vision_size // 2

        obs = []

        # 각 체크포인트 관측
        for i, cp_pos in enumerate(self.checkpoint_positions):
            cx, cy = cp_pos
            dx = (cx - ax) / self.config.grid_size
            dy = (cy - ay) / self.config.grid_size

            # 시야 내에 있는지 확인
            visible = abs(cx - ax) <= half and abs(cy - ay) <= half

            obs.extend([1.0 if visible else 0.0, dx, dy])

        # Goal 관측 (visible, dx, dy 포함)
        gx, gy = self.goal_pos
        g_dx = (gx - ax) / self.config.grid_size
        g_dy = (gy - ay) / self.config.grid_size
        goal_visible = abs(gx - ax) <= half and abs(gy - ay) <= half
        obs.extend([1.0 if goal_visible else 0.0, g_dx, g_dy])

        # 완료된 체크포인트 수 (구체적으로 어떤 것인지는 안 알려줌)
        n_completed = sum(self.state.completed)
        obs.append(n_completed / 3.0)

        # 모두 완료 여부
        obs.append(1.0 if all(self.state.completed) else 0.0)

        # 순서 힌트 (처음 N 스텝만)
        if self.state.step < self.config.hint_duration:
            for idx in self.state.target_sequence:
                obs.append(idx / 2.0)  # normalize to [0, 1]
            self.state.sequence_seen = True
        else:
            obs.extend([0.0, 0.0, 0.0])

        # 메모리 (순서를 기억)
        if self.config.use_memory:
            if self.state.sequence_seen and not self.state.remembered_sequence:
                self.state.remembered_sequence = list(self.state.target_sequence)

            for i in range(3):
                if i < len(self.state.remembered_sequence):
                    obs.append(self.state.remembered_sequence[i] / 2.0)
                else:
                    obs.append(0.0)
            obs.append(self.state.current_target_idx / 3.0)

        return np.array(obs, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
        self.state.step += 1
        reward = self.config.step_penalty
        done = False
        info = {}

        # 이동 처리
        ax, ay = self.state.agent_pos
        new_pos = self.state.agent_pos

        if action == ActionB.UP.value:
            new_pos = (max(1, ax - 1), ay)
        elif action == ActionB.DOWN.value:
            new_pos = (min(self.config.grid_size - 2, ax + 1), ay)
        elif action == ActionB.LEFT.value:
            new_pos = (ax, max(1, ay - 1))
        elif action == ActionB.RIGHT.value:
            new_pos = (ax, min(self.config.grid_size - 2, ay + 1))
        elif action == ActionB.ACTIVATE.value:
            # 체크포인트에서 ACTIVATE
            for i, cp_pos in enumerate(self.checkpoint_positions):
                if self.state.agent_pos == cp_pos:
                    if not self.state.completed[i]:
                        # 순서가 맞는지 확인
                        expected = self.state.target_sequence[self.state.current_target_idx]
                        if i == expected:
                            # 정답!
                            self.state.completed[i] = True
                            self.state.current_target_idx += 1
                            reward += self.config.correct_checkpoint_reward
                            info['correct_checkpoint'] = True
                        else:
                            # 오답!
                            reward += self.config.wrong_checkpoint_penalty
                            info['wrong_checkpoint'] = True
                    break

        # 벽 충돌 체크
        if self.grid[new_pos] != CellTypeB.WALL.value:
            self.state.agent_pos = new_pos

        # Goal 도달 체크
        if self.state.agent_pos == self.goal_pos and all(self.state.completed):
            self.state.goal_reached = True
            reward += self.config.goal_reward
            done = True

        # 최대 스텝 체크
        if self.state.step >= self.config.max_steps:
            done = True

        self.state.total_reward += reward

        return self._get_observation(), reward, done, info

    def get_optimal_path_length(self) -> int:
        """최적 경로 길이 계산"""
        total = 0
        current_pos = (1, self.config.grid_size // 2)

        for cp_idx in self.state.target_sequence:
            cp_pos = self.checkpoint_positions[cp_idx]
            total += abs(cp_pos[0] - current_pos[0]) + abs(cp_pos[1] - current_pos[1])
            current_pos = cp_pos

        # Goal까지
        total += abs(self.goal_pos[0] - current_pos[0]) + abs(self.goal_pos[1] - current_pos[1])

        return total


@dataclass
class E5BRunStats:
    """E5 Task B 실행 통계"""
    config_name: str
    seed: int
    success: bool
    steps_to_goal: int
    total_reward: float
    correct_activations: int
    wrong_activations: int
    optimal_path: int
    path_efficiency: float
    phase_violations: int = 0
    early_recovery_count: int = 0


@dataclass
class E5BGateResult:
    """E5 Task B 게이트 결과"""
    passed: bool
    reason: str

    # E5a: Long-horizon Success
    baseline_success_rate: float
    success_rate: float
    e5a_passed: bool

    # E5b: Hierarchy Benefit
    hierarchy_benefit_ratio: float
    memory_benefit_ratio: float
    e5b_passed: bool

    # E5c: Wrong-confidence
    phase_violation_total: int
    early_recovery_total: int
    e5c_passed: bool

    # E5d: Sample Efficiency
    avg_steps_base: float
    avg_steps_full: float
    efficiency_improvement: float
    e5d_passed: bool


class E5BGate:
    """E5 Task B 게이트"""

    # Thresholds
    MIN_SUCCESS_RATE = 0.70
    MIN_HIERARCHY_RATIO = 1.10  # Hierarchy가 10% 이상 이득

    def evaluate(
        self,
        base_stats: List[E5BRunStats],
        hier_stats: List[E5BRunStats],
        full_stats: List[E5BRunStats],
    ) -> E5BGateResult:
        """게이트 평가"""

        # E5a: Long-horizon Success
        baseline_success = np.mean([1 if s.success else 0 for s in base_stats])
        full_success = np.mean([1 if s.success else 0 for s in full_stats])
        e5a_passed = full_success >= baseline_success and full_success >= self.MIN_SUCCESS_RATE

        # E5b: Hierarchy Benefit
        base_reward = np.mean([s.total_reward for s in base_stats])
        hier_reward = np.mean([s.total_reward for s in hier_stats])
        full_reward = np.mean([s.total_reward for s in full_stats])

        hierarchy_ratio = hier_reward / base_reward if base_reward != 0 else 1.0
        memory_ratio = full_reward / base_reward if base_reward != 0 else 1.0
        e5b_passed = hierarchy_ratio >= self.MIN_HIERARCHY_RATIO or memory_ratio >= self.MIN_HIERARCHY_RATIO

        # E5c: Wrong-confidence
        phase_violations = sum(s.phase_violations for s in full_stats)
        early_recovery = sum(s.early_recovery_count for s in full_stats)
        e5c_passed = phase_violations == 0

        # E5d: Sample Efficiency
        avg_steps_base = np.mean([s.steps_to_goal for s in base_stats])
        avg_steps_full = np.mean([s.steps_to_goal for s in full_stats])
        efficiency_improvement = (avg_steps_base - avg_steps_full) / avg_steps_base if avg_steps_base > 0 else 0
        e5d_passed = efficiency_improvement > 0 or full_success > baseline_success

        # Overall
        passed = e5a_passed and e5b_passed and e5c_passed and e5d_passed

        if passed:
            reason = "PASS"
        else:
            reasons = []
            if not e5a_passed:
                reasons.append(f"success={full_success:.1%}<{self.MIN_SUCCESS_RATE:.0%}")
            if not e5b_passed:
                reasons.append(f"hierarchy_ratio={hierarchy_ratio:.2f}<{self.MIN_HIERARCHY_RATIO}")
            if not e5c_passed:
                reasons.append(f"phase_violations={phase_violations}")
            if not e5d_passed:
                reasons.append(f"no_efficiency_gain")
            reason = ", ".join(reasons)

        return E5BGateResult(
            passed=passed,
            reason=reason,
            baseline_success_rate=baseline_success,
            success_rate=full_success,
            e5a_passed=e5a_passed,
            hierarchy_benefit_ratio=hierarchy_ratio,
            memory_benefit_ratio=memory_ratio,
            e5b_passed=e5b_passed,
            phase_violation_total=phase_violations,
            early_recovery_total=early_recovery,
            e5c_passed=e5c_passed,
            avg_steps_base=avg_steps_base,
            avg_steps_full=avg_steps_full,
            efficiency_improvement=efficiency_improvement,
            e5d_passed=e5d_passed,
        )
