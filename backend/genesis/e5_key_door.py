"""
E5 Task A: Key → Door Environment

핵심 의도:
- '열쇠를 본 경험'이 없으면 문을 열 수 없음
- Memory ratio > 1.0이 나와야만 하는 구조

환경 스펙:
- 맵: 11×11 격자 + 벽/방 2~3개
- Key: 한 곳에 존재 (라운드마다 위치 변동)
- Door: 다른 방의 입구. Key가 없으면 통과 불가
- Goal: Door 뒤에만 존재 (희소 보상)
- Partial Observability: 5×5 시야
- Distractor: 가짜 키("fake key") 여러 개
- Delayed reward: Goal reward는 door 통과 이후 N step 뒤 지급
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


class CellType(Enum):
    """셀 타입"""
    EMPTY = 0
    WALL = 1
    KEY = 2
    FAKE_KEY = 3
    DOOR = 4
    GOAL = 5
    AGENT = 6


class Action(Enum):
    """행동"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PICKUP = 4  # 아이템 줍기


@dataclass
class KeyDoorConfig:
    """Key-Door 환경 설정"""
    grid_size: int = 13  # 더 큰 그리드 (Memory가 필수가 되려면)
    vision_size: int = 5  # 5×5 시야 (제한된 시야)
    n_fake_keys: int = 2
    reward_delay: int = 3  # Goal 도달 후 보상 지연
    max_steps: int = 300  # 더 긴 시간 허용

    # 보상 설정
    goal_reward: float = 10.0
    step_penalty: float = -0.01
    fake_key_penalty: float = -0.2
    wall_penalty: float = -0.05

    # Memory/Hierarchy 설정
    use_memory: bool = True
    use_hierarchy: bool = False

    # 난이도
    key_door_same_room: bool = False  # True면 쉬움 (단기로 풀림)


@dataclass
class KeyDoorState:
    """환경 상태"""
    agent_pos: Tuple[int, int]
    has_key: bool = False
    door_open: bool = False
    goal_reached: bool = False
    step: int = 0
    total_reward: float = 0.0
    pending_rewards: List[Tuple[int, float]] = field(default_factory=list)

    # 메모리 관련
    key_seen: bool = False  # 키를 본 적 있는지
    key_last_seen_pos: Optional[Tuple[int, int]] = None
    door_seen: bool = False
    door_last_seen_pos: Optional[Tuple[int, int]] = None


class KeyDoorEnv:
    """
    Key → Door 환경

    Memory ratio > 1.0을 강제하는 최소 구조:
    - Key와 Door가 다른 방에 있어서 동시에 볼 수 없음
    - Key 위치를 기억해야 Door를 열 수 있음
    """

    def __init__(self, config: KeyDoorConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # 맵 생성
        self.grid = np.zeros((config.grid_size, config.grid_size), dtype=int)
        self.rooms: List[Tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)

        # 오브젝트 위치
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.fake_key_positions: List[Tuple[int, int]] = []

        # 상태
        self.state = KeyDoorState(agent_pos=(0, 0))

        # 초기화
        self._generate_map()
        self.reset()

    def _generate_map(self):
        """맵 생성 - 2개 방 + 벽"""
        size = self.config.grid_size

        # 외벽
        self.grid[0, :] = CellType.WALL.value
        self.grid[-1, :] = CellType.WALL.value
        self.grid[:, 0] = CellType.WALL.value
        self.grid[:, -1] = CellType.WALL.value

        # 중앙 벽으로 2개 방 분리
        mid = size // 2
        self.grid[mid, 1:-1] = CellType.WALL.value

        # Door 위치 (벽 중앙에)
        door_x = mid
        door_y = size // 2  # 중앙
        self.grid[door_x, door_y] = CellType.DOOR.value
        self.door_pos = (door_x, door_y)

        # 방 정의
        self.rooms = [
            (1, 1, mid - 1, size - 2),  # 상단 방
            (mid + 1, 1, size - 2, size - 2),  # 하단 방
        ]

        # Key는 상단 방 구석에 (문과 떨어진 곳)
        room1 = self.rooms[0]
        # 문이 중앙이니 키는 양 끝 중 하나에
        key_y = 1 if self.rng.random() < 0.5 else size - 2
        self.key_pos = (room1[0] + 1, key_y)  # 상단 방 안쪽
        self.grid[self.key_pos] = CellType.KEY.value

        # Goal은 하단 방 중앙에
        room2 = self.rooms[1]
        self.goal_pos = ((room2[0] + room2[2]) // 2, size // 2)
        self.grid[self.goal_pos] = CellType.GOAL.value

        # Fake keys - 상단 방에만 (혼란 유발)
        self.fake_key_positions = []
        for _ in range(self.config.n_fake_keys):
            for attempt in range(10):
                pos = (
                    self.rng.randint(room1[0], room1[2] + 1),
                    self.rng.randint(room1[1], room1[3] + 1),
                )
                # 다른 오브젝트와 겹치지 않게
                if (pos != self.key_pos and pos != self.door_pos and
                    pos != self.goal_pos and pos not in self.fake_key_positions):
                    self.fake_key_positions.append(pos)
                    self.grid[pos] = CellType.FAKE_KEY.value
                    break

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # 에이전트 시작 위치 (상단 방 랜덤)
        room1 = self.rooms[0]
        start_pos = (
            self.rng.randint(room1[0], room1[2] + 1),
            self.rng.randint(room1[1], room1[3] + 1),
        )
        # Key와 겹치지 않게
        while start_pos == self.key_pos:
            start_pos = (
                self.rng.randint(room1[0], room1[2] + 1),
                self.rng.randint(room1[1], room1[3] + 1),
            )

        self.state = KeyDoorState(agent_pos=start_pos)

        # Return memory-augmented observation when use_memory=True (same as step())
        if self.config.use_memory:
            return self._get_memory_augmented_observation()
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        5×5 시야 관측

        각 셀: [wall, key, fake_key, door, goal, agent, has_key]
        → 25 * 6 + 1 = 151D ... 너무 큼

        간소화: 8D 벡터로 압축
        [key_visible, key_dx, key_dy, door_visible, door_dx, door_dy, has_key, goal_visible]
        """
        ax, ay = self.state.agent_pos
        half = self.config.vision_size // 2

        # 시야 내 오브젝트 탐지
        key_visible = False
        key_dx, key_dy = 0.0, 0.0
        door_visible = False
        door_dx, door_dy = 0.0, 0.0
        goal_visible = False

        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < self.config.grid_size and 0 <= ny < self.config.grid_size:
                    cell = self.grid[nx, ny]

                    if cell == CellType.KEY.value and not self.state.has_key:
                        key_visible = True
                        key_dx = dx / half  # 정규화
                        key_dy = dy / half
                        # 메모리 업데이트
                        self.state.key_seen = True
                        self.state.key_last_seen_pos = (nx, ny)

                    if cell == CellType.DOOR.value:
                        door_visible = True
                        door_dx = dx / half
                        door_dy = dy / half
                        self.state.door_seen = True
                        self.state.door_last_seen_pos = (nx, ny)

                    if cell == CellType.GOAL.value and self.state.door_open:
                        goal_visible = True

        obs = np.array([
            1.0 if key_visible else 0.0,
            key_dx,
            key_dy,
            1.0 if door_visible else 0.0,
            door_dx,
            door_dy,
            1.0 if self.state.has_key else 0.0,
            1.0 if goal_visible else 0.0,
        ], dtype=np.float32)

        return obs

    def _get_memory_augmented_observation(self) -> np.ndarray:
        """
        메모리 보강 관측 (Memory ON일 때)

        기본 8D + 메모리 4D = 12D
        [기본 8D..., key_memory_dx, key_memory_dy, door_memory_dx, door_memory_dy]
        """
        base_obs = self._get_observation()

        if not self.config.use_memory:
            return base_obs

        ax, ay = self.state.agent_pos

        # Key 메모리
        if self.state.key_last_seen_pos and not self.state.has_key:
            kx, ky = self.state.key_last_seen_pos
            key_mem_dx = (kx - ax) / self.config.grid_size
            key_mem_dy = (ky - ay) / self.config.grid_size
        else:
            key_mem_dx, key_mem_dy = 0.0, 0.0

        # Door 메모리
        if self.state.door_last_seen_pos:
            dx, dy = self.state.door_last_seen_pos
            door_mem_dx = (dx - ax) / self.config.grid_size
            door_mem_dy = (dy - ay) / self.config.grid_size
        else:
            door_mem_dx, door_mem_dy = 0.0, 0.0

        memory_obs = np.array([
            key_mem_dx, key_mem_dy,
            door_mem_dx, door_mem_dy,
        ], dtype=np.float32)

        return np.concatenate([base_obs, memory_obs])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경 스텝

        Returns:
            obs: 관측
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        self.state.step += 1
        reward = self.config.step_penalty
        done = False
        info = {}

        ax, ay = self.state.agent_pos

        # 이동 처리
        if action == Action.UP.value:
            new_pos = (ax - 1, ay)
        elif action == Action.DOWN.value:
            new_pos = (ax + 1, ay)
        elif action == Action.LEFT.value:
            new_pos = (ax, ay - 1)
        elif action == Action.RIGHT.value:
            new_pos = (ax, ay + 1)
        elif action == Action.PICKUP.value:
            new_pos = (ax, ay)
            # 픽업 처리
            if (ax, ay) == self.key_pos and not self.state.has_key:
                self.state.has_key = True
                info['picked_up'] = 'key'
            elif (ax, ay) in self.fake_key_positions:
                reward += self.config.fake_key_penalty
                info['picked_up'] = 'fake_key'
        else:
            new_pos = (ax, ay)

        # 이동 유효성 검사
        nx, ny = new_pos
        if 0 <= nx < self.config.grid_size and 0 <= ny < self.config.grid_size:
            cell = self.grid[nx, ny]

            if cell == CellType.WALL.value:
                reward += self.config.wall_penalty
                # 이동 안 함
            elif cell == CellType.DOOR.value:
                if self.state.has_key:
                    # Door 열림!
                    self.state.door_open = True
                    self.state.agent_pos = new_pos
                    info['door_opened'] = True
                else:
                    # 키 없으면 통과 불가
                    reward += self.config.wall_penalty
            elif cell == CellType.GOAL.value:
                if self.state.door_open:
                    # Goal 도달!
                    self.state.goal_reached = True
                    # 보상 지연
                    deliver_at = self.state.step + self.config.reward_delay
                    self.state.pending_rewards.append((deliver_at, self.config.goal_reward))
                    info['goal_reached'] = True
                    done = True
                self.state.agent_pos = new_pos
            else:
                self.state.agent_pos = new_pos

        # 지연 보상 처리
        delivered = []
        for deliver_at, r in self.state.pending_rewards:
            if self.state.step >= deliver_at:
                reward += r
                delivered.append((deliver_at, r))
        for item in delivered:
            self.state.pending_rewards.remove(item)

        self.state.total_reward += reward

        # 타임아웃
        if self.state.step >= self.config.max_steps:
            done = True
            info['timeout'] = True

        # 관측
        if self.config.use_memory:
            obs = self._get_memory_augmented_observation()
        else:
            obs = self._get_observation()

        return obs, reward, done, info

    def get_optimal_path_length(self) -> int:
        """최적 경로 길이 계산 (BFS)"""
        from collections import deque

        start = self.state.agent_pos
        key_pos = self.key_pos
        door_pos = self.door_pos
        goal_pos = self.goal_pos

        def bfs(start, end, can_pass_door=False):
            queue = deque([(start, 0)])
            visited = {start}

            while queue:
                pos, dist = queue.popleft()
                if pos == end:
                    return dist

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if (nx, ny) in visited:
                        continue
                    if not (0 <= nx < self.config.grid_size and 0 <= ny < self.config.grid_size):
                        continue

                    cell = self.grid[nx, ny]
                    if cell == CellType.WALL.value:
                        continue
                    # Door 처리: 목적지면 허용, 통과는 can_pass_door에 따라
                    if cell == CellType.DOOR.value:
                        if (nx, ny) == end:
                            # 목적지가 door면 도달 가능
                            pass
                        elif not can_pass_door:
                            continue

                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))

            return float('inf')

        # start → key → door → goal
        d1 = bfs(start, key_pos, can_pass_door=False)
        d2 = bfs(key_pos, door_pos, can_pass_door=False)  # door가 목적지이므로 도달 가능
        d3 = bfs(door_pos, goal_pos, can_pass_door=True)

        return d1 + d2 + d3


@dataclass
class E5RunStats:
    """E5 단일 실행 통계"""
    config_name: str  # BASE, +MEM, +HIE, FULL
    seed: int
    task: str  # "key_door" or "checkpoint"

    # 성과
    success: bool  # goal 도달 여부
    steps_to_goal: int  # goal까지 걸린 스텝 (실패 시 max_steps)
    total_reward: float

    # 효율
    optimal_path: int
    path_efficiency: float  # optimal / actual

    # Memory 활용
    key_remembered: bool  # 키 위치 기억했는지
    door_remembered: bool

    # Wrong-confidence (PC-Z 지표)
    phase_violations: int
    early_recovery_count: int


@dataclass
class E5GateResult:
    """E5 게이트 결과"""
    # E5a: Long-horizon success rate
    e5a_passed: bool
    success_rate: float
    baseline_success_rate: float

    # E5b: Memory benefit ratio
    e5b_passed: bool
    memory_benefit_ratio: float  # target: > 1.0, 권장 ≥ 1.10
    hierarchy_benefit_ratio: float  # FULL vs +MEM

    # E5c: Wrong-confidence
    e5c_passed: bool
    phase_violation_total: int
    early_recovery_total: int

    # E5d: Sample efficiency
    e5d_passed: bool
    avg_steps_base: float
    avg_steps_full: float
    efficiency_improvement: float  # (base - full) / base

    # 전체 판정
    passed: bool
    reason: str


class E5Gate:
    """
    E5 게이트

    E5a: Long-horizon success rate ≥ baseline
    E5b: Memory benefit ratio > 1.0 (권장 ≥ 1.10)
    E5c: Wrong-confidence = 0
    E5d: Sample efficiency improvement
    """

    def __init__(
        self,
        memory_benefit_min: float = 1.10,  # 10% 이상 개선
        hierarchy_benefit_min: float = 1.05,  # 5% 추가 개선
        wrong_confidence_max: int = 0,  # 0건
    ):
        self.memory_benefit_min = memory_benefit_min
        self.hierarchy_benefit_min = hierarchy_benefit_min
        self.wrong_confidence_max = wrong_confidence_max

    def evaluate(
        self,
        base_stats: List[E5RunStats],
        mem_stats: List[E5RunStats],
        full_stats: List[E5RunStats],
    ) -> E5GateResult:
        """E5 게이트 평가"""

        # E5a: Success rate
        base_success = np.mean([1 if s.success else 0 for s in base_stats])
        full_success = np.mean([1 if s.success else 0 for s in full_stats])
        e5a_passed = full_success >= base_success

        # E5b: Memory benefit ratio
        base_reward = np.mean([s.total_reward for s in base_stats])
        mem_reward = np.mean([s.total_reward for s in mem_stats])
        full_reward = np.mean([s.total_reward for s in full_stats])

        # Avoid division by zero
        if abs(base_reward) < 0.01:
            base_reward = 0.01 if base_reward >= 0 else -0.01

        memory_benefit_ratio = mem_reward / base_reward if base_reward > 0 else (
            mem_reward - base_reward + 1  # 둘 다 음수면 개선량으로
        )

        # Hierarchy benefit: FULL vs +MEM
        if abs(mem_reward) < 0.01:
            mem_reward_safe = 0.01 if mem_reward >= 0 else -0.01
        else:
            mem_reward_safe = mem_reward
        hierarchy_benefit_ratio = full_reward / mem_reward_safe if mem_reward_safe > 0 else 1.0

        e5b_passed = memory_benefit_ratio >= self.memory_benefit_min

        # E5c: Wrong-confidence
        total_phase = sum(s.phase_violations for s in full_stats)
        total_early = sum(s.early_recovery_count for s in full_stats)
        e5c_passed = (total_phase + total_early) <= self.wrong_confidence_max

        # E5d: Sample efficiency
        # 성공한 경우의 평균 스텝
        base_steps = [s.steps_to_goal for s in base_stats if s.success]
        full_steps = [s.steps_to_goal for s in full_stats if s.success]

        avg_steps_base = np.mean(base_steps) if base_steps else float('inf')
        avg_steps_full = np.mean(full_steps) if full_steps else float('inf')

        if avg_steps_base > 0 and avg_steps_base != float('inf'):
            efficiency_improvement = (avg_steps_base - avg_steps_full) / avg_steps_base
        else:
            efficiency_improvement = 0.0

        e5d_passed = efficiency_improvement >= 0  # 최소한 악화되지 않음

        # 전체 판정
        passed = e5a_passed and e5b_passed and e5c_passed and e5d_passed

        reasons = []
        if not e5a_passed:
            reasons.append(f"success_rate={full_success:.1%}<baseline({base_success:.1%})")
        if not e5b_passed:
            reasons.append(f"memory_ratio={memory_benefit_ratio:.2f}<{self.memory_benefit_min:.2f}")
        if not e5c_passed:
            reasons.append(f"wrong_confidence={total_phase + total_early}>{self.wrong_confidence_max}")
        if not e5d_passed:
            reasons.append(f"efficiency_worse={efficiency_improvement:.1%}")

        reason = "PASS" if passed else "; ".join(reasons)

        return E5GateResult(
            e5a_passed=e5a_passed,
            success_rate=full_success,
            baseline_success_rate=base_success,
            e5b_passed=e5b_passed,
            memory_benefit_ratio=memory_benefit_ratio,
            hierarchy_benefit_ratio=hierarchy_benefit_ratio,
            e5c_passed=e5c_passed,
            phase_violation_total=total_phase,
            early_recovery_total=total_early,
            e5d_passed=e5d_passed,
            avg_steps_base=avg_steps_base,
            avg_steps_full=avg_steps_full,
            efficiency_improvement=efficiency_improvement,
            passed=passed,
            reason=reason,
        )
