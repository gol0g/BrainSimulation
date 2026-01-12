"""
E6-2: 2D Physics Navigation with Obstacles

E6-1 확장: 정적 원형 장애물 추가
- 장애물 1~3개 (고정 위치)
- 충돌 판정 (반사) + 큰 페널티
- 관측에 장애물 상대벡터 추가

관측: [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, dist_to_goal,
       obs1_dx, obs1_dy, obs1_dist, obs2_dx, obs2_dy, obs2_dist, obs3_dx, obs3_dy, obs3_dist,
       min_obstacle_dist]
행동: [ax, ay] (연속, -1 ~ 1)

게이트:
1. Stability: E6-1과 동일
2. Learnability: 성공률 + 충돌률 감소 트렌드
3. Memory/Hierarchy: ratios 유지
4. Safety: collision_rate <= 5%
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List

from .e6_physics_nav import (
    PhysicsConfig as E6Config,
    PhysicsState as E6State,
    ActionGate,
    E6Gate,
    E6StabilityGateResult,
    E6LearnabilityGateResult,
)


@dataclass
class Obstacle:
    """원형 장애물"""
    pos: np.ndarray  # 중심 위치
    radius: float  # 반경


@dataclass
class ObstacleConfig(E6Config):
    """E6-2 환경 설정 (E6-1 확장)"""
    # 장애물 설정
    n_obstacles: int = 3  # 장애물 개수 (1~3) - 3개로 증가
    obstacle_radius: float = 1.2  # 장애물 반경 - 더 크게
    obstacle_margin: float = 1.2  # 시작/목표에서 최소 거리 - 더 타이트하게

    # 충돌 페널티
    collision_penalty: float = -3.0  # 충돌 시 페널티 - 더 크게

    # Safety gate
    max_collision_rate: float = 0.10  # 10% 이하 (난이도 증가로 완화)

    # Action gate (회피 시 급조향 방지 위해 더 타이트하게)
    delta_max: float = 0.15  # E6-1: 0.2 → E6-2: 0.15


@dataclass
class ObstacleState(E6State):
    """E6-2 환경 상태 (E6-1 확장)"""
    obstacles: List[Obstacle] = field(default_factory=list)

    # 충돌 추적
    collision_count: int = 0
    collision_this_episode: bool = False


class ObstacleNavEnv:
    """
    2D Physics Navigation with Obstacles

    E6-2 핵심: 회피 + 도달
    """

    def __init__(self, config: ObstacleConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Action gate (tighter delta for obstacle avoidance)
        self.action_gate = ActionGate(
            a_max=config.a_max,
            delta_max=config.delta_max
        )

        # State
        self.state = ObstacleState()

        # Stability tracking
        self.nan_count = 0
        self.inf_count = 0
        self.speed_violation_count = 0
        self.pos_violation_count = 0

        # Collision tracking (across episodes)
        self.total_collisions = 0
        self.total_episodes = 0

    def _generate_obstacles(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[Obstacle]:
        """장애물 생성 (시작/목표와 겹치지 않게)"""
        obstacles = []
        max_attempts = 100

        for _ in range(self.config.n_obstacles):
            for attempt in range(max_attempts):
                # 랜덤 위치 생성
                pos = self.rng.uniform(
                    -self.config.world_size * 0.7,
                    self.config.world_size * 0.7,
                    size=2
                )

                # 시작/목표와의 거리 체크
                dist_to_start = np.linalg.norm(pos - start_pos)
                dist_to_goal = np.linalg.norm(pos - goal_pos)

                min_dist = self.config.obstacle_radius + self.config.obstacle_margin

                if dist_to_start > min_dist and dist_to_goal > min_dist:
                    # 기존 장애물과의 거리 체크
                    valid = True
                    for obs in obstacles:
                        if np.linalg.norm(pos - obs.pos) < 2 * self.config.obstacle_radius + 0.5:
                            valid = False
                            break

                    if valid:
                        obstacles.append(Obstacle(
                            pos=pos.astype(np.float32),
                            radius=self.config.obstacle_radius
                        ))
                        break

        return obstacles

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

        # 장애물 생성
        obstacles = self._generate_obstacles(start_pos, goal_pos)

        self.state = ObstacleState(
            pos=start_pos.astype(np.float32),
            vel=np.zeros(2, dtype=np.float32),
            goal=goal_pos.astype(np.float32),
            step=0,
            total_reward=0.0,
            goal_reached=False,
            prev_action=np.zeros(2, dtype=np.float32),
            visited_positions=[start_pos.copy()],
            goal_memory=None,
            obstacles=obstacles,
            collision_count=0,
            collision_this_episode=False,
        )

        self.total_episodes += 1

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        관측 생성

        기본 7D: [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, dist_to_goal]
        + 장애물 10D: [obs1_dx, obs1_dy, obs1_dist, obs2_dx, obs2_dy, obs2_dist,
                       obs3_dx, obs3_dy, obs3_dist, min_obstacle_dist]
        """
        pos = self.state.pos
        vel = self.state.vel
        goal = self.state.goal

        # 상대 위치
        goal_delta = goal - pos
        dist = np.linalg.norm(goal_delta)

        # 정규화
        ws = self.config.world_size

        obs = [
            pos[0] / ws,
            pos[1] / ws,
            vel[0] / self.config.max_speed,
            vel[1] / self.config.max_speed,
            goal_delta[0] / ws,
            goal_delta[1] / ws,
            dist / ws,
        ]

        # 장애물 관측 (최대 3개)
        min_obstacle_dist = float('inf')

        for i in range(3):
            if i < len(self.state.obstacles):
                obstacle = self.state.obstacles[i]
                delta = obstacle.pos - pos
                dist_to_obs = np.linalg.norm(delta) - obstacle.radius  # 표면까지 거리
                min_obstacle_dist = min(min_obstacle_dist, dist_to_obs)

                obs.extend([
                    delta[0] / ws,
                    delta[1] / ws,
                    max(0, dist_to_obs) / ws,  # 음수 방지 (내부에 있을 때)
                ])
            else:
                # 장애물 없으면 멀리 있는 것처럼
                obs.extend([0.0, 0.0, 1.0])

        # 최소 장애물 거리 (정규화)
        if min_obstacle_dist == float('inf'):
            min_obstacle_dist = ws
        obs.append(min(1.0, max(0, min_obstacle_dist) / ws))

        return np.array(obs, dtype=np.float32)

    def _check_collision(self) -> bool:
        """장애물 충돌 체크"""
        for obstacle in self.state.obstacles:
            dist = np.linalg.norm(self.state.pos - obstacle.pos)
            if dist < obstacle.radius:
                return True
        return False

    def _resolve_collision(self, obstacle: Obstacle):
        """충돌 해결 (반사)"""
        # 장애물 중심에서 에이전트로의 방향
        delta = self.state.pos - obstacle.pos
        dist = np.linalg.norm(delta)

        if dist < 1e-6:
            # 정확히 중심에 있으면 랜덤 방향으로 밀어냄
            delta = self.rng.randn(2)
            dist = np.linalg.norm(delta)

        normal = delta / dist

        # 에이전트를 장애물 표면으로 밀어냄
        self.state.pos = obstacle.pos + normal * (obstacle.radius + 0.05)

        # 속도 반사 (탄성 계수 0.5)
        vel_normal = np.dot(self.state.vel, normal) * normal
        vel_tangent = self.state.vel - vel_normal
        self.state.vel = vel_tangent - vel_normal * 0.5

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
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

        # 장애물 충돌 체크 및 해결
        collision_this_step = False
        for obstacle in self.state.obstacles:
            dist = np.linalg.norm(self.state.pos - obstacle.pos)
            if dist < obstacle.radius:
                self._resolve_collision(obstacle)
                collision_this_step = True
                self.state.collision_count += 1
                if not self.state.collision_this_episode:
                    self.state.collision_this_episode = True
                    self.total_collisions += 1

        # 경계 체크 (soft boundary - 반발)
        ws = self.config.world_size
        if np.any(np.abs(self.state.pos) > ws):
            self.pos_violation_count += 1
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
            if self.state.step % 10 == 0:
                self.state.visited_positions.append(self.state.pos.copy())

        # 보상 계산
        dist_to_goal = np.linalg.norm(self.state.goal - self.state.pos)

        reward = self.config.step_penalty
        reward -= dist_to_goal * 0.01  # 거리 페널티
        reward -= self.config.control_cost_weight * np.sum(action ** 2)  # 제어 비용

        # 충돌 페널티
        if collision_this_step:
            reward += self.config.collision_penalty

        done = False
        info = {'collision': collision_this_step}

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

    def get_safety_stats(self) -> Dict:
        """Safety Gate 통계"""
        collision_rate = self.total_collisions / max(1, self.total_episodes)
        return {
            'total_collisions': self.total_collisions,
            'total_episodes': self.total_episodes,
            'collision_rate': collision_rate,
        }

    def reset_stability_stats(self):
        self.nan_count = 0
        self.inf_count = 0
        self.speed_violation_count = 0
        self.pos_violation_count = 0
        self.action_gate.reset_stats()

    def reset_safety_stats(self):
        self.total_collisions = 0
        self.total_episodes = 0


# ============================================================================
# E6-2 Gates
# ============================================================================

@dataclass
class E6SafetyGateResult:
    """Safety Gate 결과"""
    passed: bool
    reason: str

    collision_rate: float
    total_collisions: int
    total_episodes: int


@dataclass
class E6_2LearnabilityGateResult:
    """E6-2 Learnability Gate 결과 (충돌 트렌드 포함)"""
    passed: bool
    reason: str

    success_rate: float
    avg_distance: float
    distance_trend: float

    collision_trend: float  # 음수면 개선


class E6_2Gate(E6Gate):
    """E6-2 Gate 평가 (E6Gate 확장)"""

    # Safety thresholds
    MAX_COLLISION_RATE = 0.05  # 5% 이하

    def evaluate_safety(
        self,
        collision_rate: float,
        total_collisions: int,
        total_episodes: int,
    ) -> E6SafetyGateResult:
        """Safety Gate 평가"""

        passed = collision_rate <= self.MAX_COLLISION_RATE

        if passed:
            reason = "PASS"
        else:
            reason = f"collision_rate={collision_rate:.1%}>{self.MAX_COLLISION_RATE:.0%}"

        return E6SafetyGateResult(
            passed=passed,
            reason=reason,
            collision_rate=collision_rate,
            total_collisions=total_collisions,
            total_episodes=total_episodes,
        )

    def evaluate_learnability_with_collision(
        self,
        success_rate: float,
        avg_distance: float,
        distance_trend: float,
        collision_trend: float,
    ) -> E6_2LearnabilityGateResult:
        """E6-2 Learnability Gate (충돌 트렌드 포함)"""

        # 성공률 또는 거리/충돌 개선
        passed = (
            success_rate >= self.MIN_SUCCESS_RATE or
            (distance_trend < 0 and collision_trend <= 0)
        )

        if passed:
            reason = "PASS"
        else:
            reason = f"success={success_rate:.1%}, dist_trend={distance_trend:.3f}, coll_trend={collision_trend:.3f}"

        return E6_2LearnabilityGateResult(
            passed=passed,
            reason=reason,
            success_rate=success_rate,
            avg_distance=avg_distance,
            distance_trend=distance_trend,
            collision_trend=collision_trend,
        )
