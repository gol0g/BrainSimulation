"""
E7-A1: Map Randomization (Static Generalization)

E6 구조(방어계층 + 위험추정)가 분포 변화에서도 유지되는지 검증

매 에피소드마다:
- 장애물 개수: 1~5 랜덤
- 장애물 위치/반경 랜덤 (시작/목표 주변 안전 반경 확보)
- 목표 위치 랜덤
- (선택) 물리 파라미터 ±10-20% 랜덤

게이트:
1. O.O.D. collision: 평균 collision < 15%
2. Tail risk: collision p95 < 25%
3. Success: triple_success >= 90%
4. Robustness delta: E6 대비 성능 하락폭 제한
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List

from .e6_multigoal_nav import (
    MultiGoalConfig,
    MultiGoalState,
    MultiGoalNavEnv,
    E6_3Gate,
    E6SequencingGateResult,
)
from .e6_obstacle_nav import Obstacle, E6SafetyGateResult


@dataclass
class GeneralizationConfig(MultiGoalConfig):
    """E7-A1 환경 설정"""
    # 장애물 랜덤화
    min_obstacles: int = 1
    max_obstacles: int = 5
    min_obstacle_radius: float = 0.8
    max_obstacle_radius: float = 1.5

    # 물리 파라미터 랜덤화
    randomize_physics: bool = True
    friction_range: Tuple[float, float] = (0.08, 0.12)  # ±20%
    max_speed_range: Tuple[float, float] = (1.6, 2.4)   # ±20%

    # 목표 랜덤화
    min_goal_distance: float = 3.0  # 목표 간 최소 거리
    goal_radius_range: Tuple[float, float] = (0.4, 0.6)  # 목표 반경 변동

    # 안전 반경 (시작점/목표 근처 장애물 금지)
    safe_radius: float = 2.0


@dataclass
class GeneralizationState(MultiGoalState):
    """E7-A1 환경 상태"""
    # 현재 에피소드의 랜덤 설정
    current_n_obstacles: int = 0
    current_friction: float = 0.1
    current_max_speed: float = 2.0
    current_goal_radius: float = 0.5


class GeneralizationNavEnv(MultiGoalNavEnv):
    """
    Map Randomization Navigation Environment

    E7-A1 핵심: 매 에피소드마다 맵 구조 랜덤화
    """

    def __init__(self, config: GeneralizationConfig, seed: Optional[int] = None):
        # 부모 초기화 전에 config 타입 설정
        super().__init__(config, seed)
        self.config = config

        # 에피소드별 통계
        self.episode_collisions = []  # 각 에피소드 충돌 여부
        self.episode_goals = []       # 각 에피소드 완료 목표 수

    def _randomize_episode(self):
        """에피소드별 랜덤화 적용"""
        # 장애물 개수 랜덤
        n_obs = self.rng.randint(
            self.config.min_obstacles,
            self.config.max_obstacles + 1
        )

        # 물리 파라미터 랜덤
        if self.config.randomize_physics:
            friction = self.rng.uniform(*self.config.friction_range)
            max_speed = self.rng.uniform(*self.config.max_speed_range)
        else:
            friction = self.config.friction
            max_speed = self.config.max_speed

        # 목표 반경 랜덤
        goal_radius = self.rng.uniform(*self.config.goal_radius_range)

        return n_obs, friction, max_speed, goal_radius

    def _generate_goals(self, start_pos: np.ndarray) -> List[np.ndarray]:
        """순차 목표 3개 생성 (랜덤화된 위치)"""
        goals = []
        min_dist_between = self.config.min_goal_distance

        for i in range(self.config.n_goals):
            max_attempts = 100
            for attempt in range(max_attempts):
                # 랜덤 위치 (가장자리 영역)
                angle = self.rng.uniform(0, 2 * np.pi)
                radius = self.rng.uniform(4, 8)  # 약간 더 넓은 범위
                goal_pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])

                # 경계 체크
                ws = self.config.world_size
                if np.any(np.abs(goal_pos) > ws * 0.9):
                    continue

                # 시작점과의 거리 체크
                if np.linalg.norm(goal_pos - start_pos) < self.config.safe_radius:
                    continue

                # 기존 목표와의 거리 체크
                valid = True
                for existing_goal in goals:
                    if np.linalg.norm(goal_pos - existing_goal) < min_dist_between:
                        valid = False
                        break

                if valid:
                    goals.append(goal_pos.astype(np.float32))
                    break

        return goals

    def _generate_obstacles(
        self,
        start_pos: np.ndarray,
        goals: List[np.ndarray],
        n_obstacles: int,
    ) -> List[Obstacle]:
        """장애물 생성 (랜덤 개수, 랜덤 반경)"""
        obstacles = []
        max_attempts = 100

        for _ in range(n_obstacles):
            for attempt in range(max_attempts):
                # 랜덤 반경
                radius = self.rng.uniform(
                    self.config.min_obstacle_radius,
                    self.config.max_obstacle_radius
                )

                # 랜덤 위치
                pos = self.rng.uniform(
                    -self.config.world_size * 0.7,
                    self.config.world_size * 0.7,
                    size=2
                )

                # 시작점 안전 반경 체크
                if np.linalg.norm(pos - start_pos) < radius + self.config.safe_radius:
                    continue

                # 모든 목표와의 안전 반경 체크
                valid = True
                for goal in goals:
                    if np.linalg.norm(pos - goal) < radius + self.config.safe_radius:
                        valid = False
                        break

                # 기존 장애물과의 거리 체크
                for obs in obstacles:
                    if np.linalg.norm(pos - obs.pos) < radius + obs.radius + 0.5:
                        valid = False
                        break

                if valid:
                    obstacles.append(Obstacle(
                        pos=pos.astype(np.float32),
                        radius=radius
                    ))
                    break

        return obstacles

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋 (에피소드별 랜덤화 포함)"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # 에피소드별 랜덤화
        n_obs, friction, max_speed, goal_radius = self._randomize_episode()

        # 시작 위치 (중앙 근처, 약간 랜덤)
        start_pos = self.rng.uniform(-2, 2, size=2)

        # 순차 목표 생성
        goals = self._generate_goals(start_pos)

        # 장애물 생성 (랜덤 개수)
        obstacles = self._generate_obstacles(start_pos, goals, n_obs)

        # 상태 초기화
        self.state = GeneralizationState(
            pos=start_pos.astype(np.float32),
            vel=np.zeros(2, dtype=np.float32),
            goal=goals[0] if goals else np.zeros(2),
            step=0,
            total_reward=0.0,
            goal_reached=False,
            prev_action=np.zeros(2, dtype=np.float32),
            visited_positions=[start_pos.copy()],
            goal_memory=None,
            obstacles=obstacles,
            collision_count=0,
            collision_this_episode=False,
            goals=goals,
            current_goal_idx=0,
            goals_completed=0,
            goal_switch_step=0,
            post_switch_collisions=0,
            # E7 추가 상태
            current_n_obstacles=n_obs,
            current_friction=friction,
            current_max_speed=max_speed,
            current_goal_radius=goal_radius,
        )

        # 물리 파라미터 업데이트
        self.config.friction = friction
        self.config.max_speed = max_speed
        self.config.goal_radius = goal_radius

        self.total_episodes += 1

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝 (랜덤화된 물리 적용)"""
        # 부모 step 호출
        obs, reward, done, info = super().step(action)

        # 에피소드 종료 시 통계 기록
        if done:
            self.episode_collisions.append(1 if self.state.collision_this_episode else 0)
            self.episode_goals.append(self.state.goals_completed)

        # 추가 정보
        info['n_obstacles'] = self.state.current_n_obstacles
        info['friction'] = self.state.current_friction
        info['max_speed'] = self.state.current_max_speed

        return obs, reward, done, info

    def get_generalization_stats(self) -> Dict:
        """일반화 통계"""
        if len(self.episode_collisions) == 0:
            return {
                'mean_collision': 0.0,
                'collision_variance': 0.0,
                'triple_success_rate': 0.0,
                'avg_goals': 0.0,
                'n_episodes': 0,
            }

        collisions = np.array(self.episode_collisions)
        goals = np.array(self.episode_goals)

        # 충돌률 통계
        mean_collision = np.mean(collisions)
        collision_variance = np.var(collisions)  # 분산 (안정성 지표)

        # 성공률 통계
        triple_success = np.sum(goals == 3) / len(goals)
        avg_goals = np.mean(goals)

        return {
            'mean_collision': mean_collision,
            'collision_variance': collision_variance,
            'triple_success_rate': triple_success,
            'avg_goals': avg_goals,
            'n_episodes': len(collisions),
        }

    def reset_all_stats(self):
        super().reset_all_stats()
        self.episode_collisions = []
        self.episode_goals = []


# ============================================================================
# E7-A1 Gates
# ============================================================================

@dataclass
class E7GeneralizationGateResult:
    """Generalization Gate 결과"""
    passed: bool
    reason: str

    # 세부 지표
    mean_collision: float
    triple_success_rate: float
    avg_goals: float

    # 게이트별 통과 여부
    ood_collision_passed: bool
    success_passed: bool


@dataclass
class E7RobustnessGateResult:
    """Robustness Delta Gate 결과"""
    passed: bool
    reason: str

    # E6 baseline vs E7 비교
    e6_collision: float
    e7_collision: float
    collision_delta: float

    e6_triple_success: float
    e7_triple_success: float
    success_delta: float


class E7_A1Gate(E6_3Gate):
    """E7-A1 Gate 평가"""

    # O.O.D. Collision 임계값
    MAX_MEAN_COLLISION = 0.15  # 평균 15%

    # Success 임계값
    MIN_TRIPLE_SUCCESS = 0.90  # 90% 완주

    # Robustness Delta 임계값
    MAX_COLLISION_DELTA = 0.05   # E6 대비 +5%p 이내
    MAX_SUCCESS_DELTA = -0.10    # E6 대비 -10%p 이내

    def evaluate_generalization(
        self,
        mean_collision: float,
        triple_success_rate: float,
        avg_goals: float,
    ) -> E7GeneralizationGateResult:
        """
        Generalization Gate 평가

        PASS 조건:
        1. mean_collision < 15%
        2. triple_success >= 90%
        """
        ood_passed = mean_collision <= self.MAX_MEAN_COLLISION
        success_passed = triple_success_rate >= self.MIN_TRIPLE_SUCCESS

        all_passed = ood_passed and success_passed

        if all_passed:
            reason = "PASS"
        else:
            reasons = []
            if not ood_passed:
                reasons.append(f"mean_coll={mean_collision:.1%}>{self.MAX_MEAN_COLLISION:.0%}")
            if not success_passed:
                reasons.append(f"triple={triple_success_rate:.1%}<{self.MIN_TRIPLE_SUCCESS:.0%}")
            reason = ", ".join(reasons)

        return E7GeneralizationGateResult(
            passed=all_passed,
            reason=reason,
            mean_collision=mean_collision,
            triple_success_rate=triple_success_rate,
            avg_goals=avg_goals,
            ood_collision_passed=ood_passed,
            success_passed=success_passed,
        )

    def evaluate_robustness_delta(
        self,
        e6_collision: float,
        e7_collision: float,
        e6_triple_success: float,
        e7_triple_success: float,
    ) -> E7RobustnessGateResult:
        """
        Robustness Delta Gate 평가

        E6 baseline 대비 성능 하락폭 제한
        """
        collision_delta = e7_collision - e6_collision
        success_delta = e7_triple_success - e6_triple_success

        collision_ok = collision_delta <= self.MAX_COLLISION_DELTA
        success_ok = success_delta >= self.MAX_SUCCESS_DELTA

        passed = collision_ok and success_ok

        if passed:
            reason = f"collision_delta={collision_delta:+.1%}, success_delta={success_delta:+.1%}"
        else:
            reasons = []
            if not collision_ok:
                reasons.append(f"collision_delta={collision_delta:+.1%}>{self.MAX_COLLISION_DELTA:+.0%}")
            if not success_ok:
                reasons.append(f"success_delta={success_delta:+.1%}<{self.MAX_SUCCESS_DELTA:+.0%}")
            reason = ", ".join(reasons)

        return E7RobustnessGateResult(
            passed=passed,
            reason=reason,
            e6_collision=e6_collision,
            e7_collision=e7_collision,
            collision_delta=collision_delta,
            e6_triple_success=e6_triple_success,
            e7_triple_success=e7_triple_success,
            success_delta=success_delta,
        )
