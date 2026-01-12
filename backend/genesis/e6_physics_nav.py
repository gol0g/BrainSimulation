"""
E6-1: 2D Physics Navigation Environment

목표: 연속 action + 2D 물리 + 목표 도달
- 가장 얇은 세팅으로 시작
- 학습/메모리/계층 이득이 연속제어에서도 유지되는지 확인

관측: [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, dist_to_goal]
행동: [ax, ay] (연속, -1 ~ 1)
물리: vel += action * dt, pos += vel * dt, friction
보상: -dist + goal_bonus - control_cost
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
from enum import Enum


@dataclass
class PhysicsConfig:
    """2D Physics Navigation 환경 설정"""
    # 공간
    world_size: float = 10.0  # -10 ~ 10
    goal_radius: float = 0.5  # 목표 도달 반경

    # 물리
    dt: float = 0.1  # 시간 스텝
    friction: float = 0.1  # 마찰 계수 (속도 감쇠)
    max_speed: float = 2.0  # 최대 속도

    # Action Gate (MVP)
    a_max: float = 1.0  # ||a|| 상한
    delta_max: float = 0.2  # ||Δa|| 상한 (폭주 방지 핵심)

    # 보상
    goal_reward: float = 10.0
    step_penalty: float = -0.01
    control_cost_weight: float = 0.01  # ||a||^2 페널티

    # 에피소드
    max_steps: int = 500  # Increased for physics to reach goal

    # Memory/Hierarchy (E5와 동일한 구조)
    use_memory: bool = False
    use_hierarchy: bool = False


@dataclass
class PhysicsState:
    """환경 상태"""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2))

    step: int = 0
    total_reward: float = 0.0
    goal_reached: bool = False

    # Action gate tracking
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Memory tracking
    visited_positions: List[np.ndarray] = field(default_factory=list)
    goal_memory: Optional[np.ndarray] = None


class ActionGate:
    """
    연속 Action Safety Gate (MVP)

    1. 크기 클램프: ||a|| <= a_max
    2. 변화량 클램프: ||Δa|| <= delta_max
    """

    def __init__(self, a_max: float = 1.0, delta_max: float = 0.2):
        self.a_max = a_max
        self.delta_max = delta_max

        # Statistics
        self.norm_clamp_count = 0
        self.delta_clamp_count = 0
        self.total_count = 0

    def clip_norm(self, v: np.ndarray, max_norm: float) -> np.ndarray:
        """벡터 노름 클램프"""
        norm = np.linalg.norm(v)
        if norm > max_norm:
            return v * (max_norm / norm)
        return v

    def apply(self, action: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        """
        Action gate 적용

        1. ||a|| <= a_max
        2. ||Δa|| <= delta_max
        """
        self.total_count += 1

        # 1. 크기 클램프
        a_norm = np.linalg.norm(action)
        if a_norm > self.a_max:
            action = self.clip_norm(action, self.a_max)
            self.norm_clamp_count += 1

        # 2. 변화량 클램프
        delta = action - prev_action
        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.delta_max:
            delta = self.clip_norm(delta, self.delta_max)
            action = prev_action + delta
            self.delta_clamp_count += 1

        return action

    def get_stats(self) -> Dict:
        """Gate 통계"""
        return {
            'total': self.total_count,
            'norm_clamps': self.norm_clamp_count,
            'delta_clamps': self.delta_clamp_count,
            'norm_clamp_rate': self.norm_clamp_count / max(1, self.total_count),
            'delta_clamp_rate': self.delta_clamp_count / max(1, self.total_count),
        }

    def reset_stats(self):
        self.norm_clamp_count = 0
        self.delta_clamp_count = 0
        self.total_count = 0


class PhysicsNavEnv:
    """
    2D Physics Navigation Environment

    E6-1 핵심: 연속 action + 물리 시뮬레이션
    """

    def __init__(self, config: PhysicsConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Action gate
        self.action_gate = ActionGate(
            a_max=config.a_max,
            delta_max=config.delta_max
        )

        # State
        self.state = PhysicsState()

        # Stability tracking
        self.nan_count = 0
        self.inf_count = 0
        self.speed_violation_count = 0
        self.pos_violation_count = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # 랜덤 시작 위치 (중앙 근처)
        start_pos = self.rng.uniform(-3, 3, size=2)

        # 랜덤 목표 위치 (가장자리 근처)
        angle = self.rng.uniform(0, 2 * np.pi)
        radius = self.rng.uniform(5, 8)
        goal_pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        self.state = PhysicsState(
            pos=start_pos.astype(np.float32),
            vel=np.zeros(2, dtype=np.float32),
            goal=goal_pos.astype(np.float32),
            step=0,
            total_reward=0.0,
            goal_reached=False,
            prev_action=np.zeros(2, dtype=np.float32),
            visited_positions=[start_pos.copy()],
            goal_memory=None,
        )

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        관측 생성

        기본 7D: [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, dist_to_goal]
        + Memory 4D: [goal_mem_dx, goal_mem_dy, avg_visited_dx, avg_visited_dy]
        """
        pos = self.state.pos
        vel = self.state.vel
        goal = self.state.goal

        # 상대 위치
        goal_delta = goal - pos
        dist = np.linalg.norm(goal_delta)

        # 정규화 (world_size 기준)
        ws = self.config.world_size

        obs = np.array([
            pos[0] / ws,
            pos[1] / ws,
            vel[0] / self.config.max_speed,
            vel[1] / self.config.max_speed,
            goal_delta[0] / ws,
            goal_delta[1] / ws,
            dist / ws,
        ], dtype=np.float32)

        # Memory augmentation
        if self.config.use_memory:
            # Goal memory (마지막으로 본 목표 방향)
            if self.state.goal_memory is not None:
                goal_mem_delta = self.state.goal_memory - pos
            else:
                goal_mem_delta = goal_delta

            # 방문 위치 평균 (탐색 정보)
            if len(self.state.visited_positions) > 1:
                visited_center = np.mean(self.state.visited_positions, axis=0)
                visited_delta = visited_center - pos
            else:
                visited_delta = np.zeros(2)

            memory_obs = np.array([
                goal_mem_delta[0] / ws,
                goal_mem_delta[1] / ws,
                visited_delta[0] / ws,
                visited_delta[1] / ws,
            ], dtype=np.float32)

            obs = np.concatenate([obs, memory_obs])

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경 스텝

        물리:
        1. Action gate 적용
        2. vel += action * dt
        3. vel *= (1 - friction)
        4. vel = clip(vel, max_speed)
        5. pos += vel * dt
        """
        self.state.step += 1

        # NaN/Inf 체크
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            self.nan_count += 1
            action = np.zeros(2)

        # Action gate 적용
        action = self.action_gate.apply(action, self.state.prev_action)
        self.state.prev_action = action.copy()

        # 물리 시뮬레이션
        dt = self.config.dt

        # 가속
        self.state.vel = self.state.vel + action * dt

        # 마찰
        self.state.vel = self.state.vel * (1 - self.config.friction)

        # 속도 제한
        speed = np.linalg.norm(self.state.vel)
        if speed > self.config.max_speed:
            self.state.vel = self.state.vel * (self.config.max_speed / speed)
            self.speed_violation_count += 1

        # 위치 업데이트
        self.state.pos = self.state.pos + self.state.vel * dt

        # 경계 체크 (soft boundary - 반발)
        ws = self.config.world_size
        if np.any(np.abs(self.state.pos) > ws):
            self.pos_violation_count += 1
            # 경계 밖으로 나가면 반대로 튕김
            for i in range(2):
                if self.state.pos[i] > ws:
                    self.state.pos[i] = ws
                    self.state.vel[i] = -abs(self.state.vel[i]) * 0.5
                elif self.state.pos[i] < -ws:
                    self.state.pos[i] = -ws
                    self.state.vel[i] = abs(self.state.vel[i]) * 0.5

        # NaN/Inf 체크 (물리 시뮬레이션 후)
        if np.any(np.isnan(self.state.pos)) or np.any(np.isinf(self.state.pos)):
            self.nan_count += 1
            self.state.pos = np.zeros(2)
            self.state.vel = np.zeros(2)

        # Memory 업데이트
        if self.config.use_memory:
            self.state.goal_memory = self.state.goal.copy()
            # 일정 간격으로만 방문 위치 저장 (메모리 효율)
            if self.state.step % 10 == 0:
                self.state.visited_positions.append(self.state.pos.copy())

        # 보상 계산
        dist_to_goal = np.linalg.norm(self.state.goal - self.state.pos)

        reward = self.config.step_penalty
        reward -= dist_to_goal * 0.01  # 거리 페널티
        reward -= self.config.control_cost_weight * np.sum(action ** 2)  # 제어 비용

        done = False
        info = {}

        # 목표 도달 체크
        if dist_to_goal < self.config.goal_radius:
            self.state.goal_reached = True
            reward += self.config.goal_reward
            done = True
            info['success'] = True

        # 타임아웃
        if self.state.step >= self.config.max_steps:
            done = True
            info['timeout'] = True

        self.state.total_reward += reward

        return self._get_observation(), reward, done, info

    def get_stability_stats(self) -> Dict:
        """Stability Gate 통계"""
        return {
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'speed_violations': self.speed_violation_count,
            'pos_violations': self.pos_violation_count,
            'action_gate': self.action_gate.get_stats(),
        }

    def reset_stability_stats(self):
        self.nan_count = 0
        self.inf_count = 0
        self.speed_violation_count = 0
        self.pos_violation_count = 0
        self.action_gate.reset_stats()


# ============================================================================
# E6-1 Gates
# ============================================================================

@dataclass
class E6StabilityGateResult:
    """Stability Gate 결과"""
    passed: bool
    reason: str

    nan_count: int
    inf_count: int
    speed_violations: int
    pos_violations: int

    norm_clamp_rate: float
    delta_clamp_rate: float


@dataclass
class E6LearnabilityGateResult:
    """Learnability Gate 결과"""
    passed: bool
    reason: str

    success_rate: float
    avg_distance: float
    distance_trend: float  # 음수면 개선


@dataclass
class E6GateResult:
    """E6-1 전체 Gate 결과"""
    passed: bool
    reason: str

    stability: E6StabilityGateResult
    learnability: Optional[E6LearnabilityGateResult]


class E6Gate:
    """E6-1 Gate 평가"""

    # Stability thresholds
    MAX_NAN_INF = 0
    # Clamp rate thresholds:
    # - For random policy: high clamp rate is expected and OK
    # - For learned policy: should be < 50%
    MAX_CLAMP_RATE_RANDOM = 0.99  # Random policy can have high clamp rate
    MAX_CLAMP_RATE_LEARNED = 0.5  # Learned policy should be smooth

    # Learnability thresholds
    MIN_SUCCESS_RATE = 0.3  # 30% 이상 성공

    def evaluate_stability(
        self,
        stats: Dict,
        n_episodes: int,
        is_random_policy: bool = True,
    ) -> E6StabilityGateResult:
        """Stability Gate 평가"""

        nan_inf = stats['nan_count'] + stats.get('inf_count', 0)

        action_stats = stats.get('action_gate', {})
        norm_clamp_rate = action_stats.get('norm_clamp_rate', 0)
        delta_clamp_rate = action_stats.get('delta_clamp_rate', 0)

        # Select threshold based on policy type
        max_clamp = self.MAX_CLAMP_RATE_RANDOM if is_random_policy else self.MAX_CLAMP_RATE_LEARNED

        passed = (
            nan_inf <= self.MAX_NAN_INF and
            norm_clamp_rate <= max_clamp and
            delta_clamp_rate <= max_clamp
        )

        if passed:
            reason = "PASS"
        else:
            reasons = []
            if nan_inf > self.MAX_NAN_INF:
                reasons.append(f"nan_inf={nan_inf}>{self.MAX_NAN_INF}")
            if norm_clamp_rate > max_clamp:
                reasons.append(f"norm_clamp={norm_clamp_rate:.1%}>{max_clamp:.0%}")
            if delta_clamp_rate > max_clamp:
                reasons.append(f"delta_clamp={delta_clamp_rate:.1%}>{max_clamp:.0%}")
            reason = ", ".join(reasons)

        return E6StabilityGateResult(
            passed=passed,
            reason=reason,
            nan_count=stats['nan_count'],
            inf_count=stats.get('inf_count', 0),
            speed_violations=stats.get('speed_violations', 0),
            pos_violations=stats.get('pos_violations', 0),
            norm_clamp_rate=norm_clamp_rate,
            delta_clamp_rate=delta_clamp_rate,
        )

    def evaluate_learnability(
        self,
        success_rate: float,
        avg_distance: float,
        distance_trend: float,
    ) -> E6LearnabilityGateResult:
        """Learnability Gate 평가"""

        passed = success_rate >= self.MIN_SUCCESS_RATE or distance_trend < 0

        if passed:
            reason = "PASS"
        else:
            reason = f"success={success_rate:.1%}<{self.MIN_SUCCESS_RATE:.0%}, trend={distance_trend:.3f}"

        return E6LearnabilityGateResult(
            passed=passed,
            reason=reason,
            success_rate=success_rate,
            avg_distance=avg_distance,
            distance_trend=distance_trend,
        )
