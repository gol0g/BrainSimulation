"""
E6-3: 2D Physics Navigation with Sequential Goals

E6-2 확장: 순차 목표 3개 (A→B→C)
- 각 목표 도달 시 다음 목표 활성화 (리셋 없음)
- 장애물 1-2개 (E6-2보다 낮은 난이도)
- goal_index 미제공 (상황으로 추론하게)

관측: [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, dist_to_goal,
       obs1_dx, obs1_dy, obs1_dist, obs2_dx, obs2_dy, obs2_dist, obs3_dx, obs3_dy, obs3_dist,
       min_obstacle_dist]
       (goal은 현재 활성 목표 기준)

게이트 (5개):
1. Stability: 그대로
2. Safety: collision_rate < 10%
3. Learnability: 평균 목표 달성 수 기준
4. Memory/Hierarchy: combined ratio + sequencing 보조지표
5. Sequencing: triple_success_rate >= 70%, avg_goals >= 2.5
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List

from .e6_obstacle_nav import (
    ObstacleConfig,
    ObstacleState,
    Obstacle,
    ActionGate,
    E6_2Gate,
    E6SafetyGateResult,
)


@dataclass
class MultiGoalConfig(ObstacleConfig):
    """E6-3 환경 설정"""
    # 목표 설정
    n_goals: int = 3  # 순차 목표 개수

    # 장애물 (E6-2보다 낮은 난이도)
    n_obstacles: int = 2
    obstacle_radius: float = 1.0

    # 보상
    goal_reward: float = 5.0  # 각 목표 도달 보상
    final_goal_bonus: float = 10.0  # 마지막 목표 추가 보너스

    # Sequencing gate thresholds
    min_triple_success_rate: float = 0.70  # 70% 완주
    min_avg_goals: float = 2.5  # 평균 2.5개 이상


@dataclass
class MultiGoalState(ObstacleState):
    """E6-3 환경 상태"""
    # 다중 목표
    goals: List[np.ndarray] = field(default_factory=list)
    current_goal_idx: int = 0
    goals_completed: int = 0

    # 목표 전환 추적
    goal_switch_step: int = 0  # 마지막 목표 전환 시점
    post_switch_collisions: int = 0  # 전환 직후 충돌 수


class MultiGoalNavEnv:
    """
    2D Physics Navigation with Sequential Goals

    E6-3 핵심: 순차 목표 도달 (A→B→C)
    """

    def __init__(self, config: MultiGoalConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Action gate
        self.action_gate = ActionGate(
            a_max=config.a_max,
            delta_max=config.delta_max
        )

        # State
        self.state = MultiGoalState()

        # Stability tracking
        self.nan_count = 0
        self.inf_count = 0
        self.speed_violation_count = 0
        self.pos_violation_count = 0

        # Safety tracking
        self.total_collisions = 0
        self.total_episodes = 0

        # Sequencing tracking
        self.triple_successes = 0
        self.total_goals_completed = 0

    def _generate_goals(self, start_pos: np.ndarray) -> List[np.ndarray]:
        """순차 목표 3개 생성 (서로 떨어져 있게)"""
        goals = []
        min_dist_between = 4.0  # 목표 간 최소 거리

        for i in range(self.config.n_goals):
            max_attempts = 100
            for attempt in range(max_attempts):
                # 랜덤 위치 (가장자리 영역)
                angle = self.rng.uniform(0, 2 * np.pi)
                radius = self.rng.uniform(4, 7)
                goal_pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])

                # 시작점과의 거리 체크
                if np.linalg.norm(goal_pos - start_pos) < 2.0:
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

    def _generate_obstacles(self, start_pos: np.ndarray, goals: List[np.ndarray]) -> List[Obstacle]:
        """장애물 생성 (목표들과 겹치지 않게)"""
        obstacles = []
        max_attempts = 100

        for _ in range(self.config.n_obstacles):
            for attempt in range(max_attempts):
                pos = self.rng.uniform(
                    -self.config.world_size * 0.6,
                    self.config.world_size * 0.6,
                    size=2
                )

                # 시작점 체크
                if np.linalg.norm(pos - start_pos) < self.config.obstacle_radius + 1.5:
                    continue

                # 모든 목표와의 거리 체크
                valid = True
                for goal in goals:
                    if np.linalg.norm(pos - goal) < self.config.obstacle_radius + 1.5:
                        valid = False
                        break

                # 기존 장애물과의 거리 체크
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

        # 시작 위치 (중앙 근처)
        start_pos = self.rng.uniform(-2, 2, size=2)

        # 순차 목표 생성
        goals = self._generate_goals(start_pos)

        # 장애물 생성
        obstacles = self._generate_obstacles(start_pos, goals)

        self.state = MultiGoalState(
            pos=start_pos.astype(np.float32),
            vel=np.zeros(2, dtype=np.float32),
            goal=goals[0] if goals else np.zeros(2),  # 첫 번째 목표
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
        )

        self.total_episodes += 1

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """관측 생성 (현재 활성 목표 기준)"""
        pos = self.state.pos
        vel = self.state.vel

        # 현재 활성 목표
        if self.state.current_goal_idx < len(self.state.goals):
            current_goal = self.state.goals[self.state.current_goal_idx]
        else:
            current_goal = self.state.goals[-1] if self.state.goals else pos

        # 상대 위치
        goal_delta = current_goal - pos
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

        # 장애물 관측
        min_obstacle_dist = float('inf')

        for i in range(3):
            if i < len(self.state.obstacles):
                obstacle = self.state.obstacles[i]
                delta = obstacle.pos - pos
                dist_to_obs = np.linalg.norm(delta) - obstacle.radius
                min_obstacle_dist = min(min_obstacle_dist, dist_to_obs)

                obs.extend([
                    delta[0] / ws,
                    delta[1] / ws,
                    max(0, dist_to_obs) / ws,
                ])
            else:
                obs.extend([0.0, 0.0, 1.0])

        if min_obstacle_dist == float('inf'):
            min_obstacle_dist = ws
        obs.append(min(1.0, max(0, min_obstacle_dist) / ws))

        # goal_index는 의도적으로 미제공 (상황 추론)

        return np.array(obs, dtype=np.float32)

    def _resolve_collision(self, obstacle: Obstacle):
        """충돌 해결"""
        delta = self.state.pos - obstacle.pos
        dist = np.linalg.norm(delta)

        if dist < 1e-6:
            delta = self.rng.randn(2)
            dist = np.linalg.norm(delta)

        normal = delta / dist
        self.state.pos = obstacle.pos + normal * (obstacle.radius + 0.05)

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

        # Action gate
        action = self.action_gate.apply(action, self.state.prev_action)
        self.state.prev_action = action.copy()

        # 물리 시뮬레이션
        dt = self.config.dt
        self.state.vel = self.state.vel + action * dt
        self.state.vel = self.state.vel * (1 - self.config.friction)

        speed = np.linalg.norm(self.state.vel)
        if speed > self.config.max_speed:
            self.state.vel = self.state.vel * (self.config.max_speed / speed)
            self.speed_violation_count += 1

        self.state.pos = self.state.pos + self.state.vel * dt

        # 충돌 체크
        collision_this_step = False
        for obstacle in self.state.obstacles:
            dist = np.linalg.norm(self.state.pos - obstacle.pos)
            if dist < obstacle.radius:
                self._resolve_collision(obstacle)
                collision_this_step = True
                self.state.collision_count += 1

                # 목표 전환 직후 충돌 추적 (10 step 이내)
                if self.state.step - self.state.goal_switch_step <= 10:
                    self.state.post_switch_collisions += 1

                if not self.state.collision_this_episode:
                    self.state.collision_this_episode = True
                    self.total_collisions += 1

        # 경계 체크
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

        # NaN 체크
        if np.any(np.isnan(self.state.pos)) or np.any(np.isinf(self.state.pos)):
            self.nan_count += 1
            self.state.pos = np.zeros(2)
            self.state.vel = np.zeros(2)

        # Memory 업데이트
        if self.config.use_memory:
            self.state.goal_memory = self.state.goals[self.state.current_goal_idx].copy() \
                if self.state.current_goal_idx < len(self.state.goals) else None
            if self.state.step % 10 == 0:
                self.state.visited_positions.append(self.state.pos.copy())

        # 보상 계산
        current_goal = self.state.goals[self.state.current_goal_idx] \
            if self.state.current_goal_idx < len(self.state.goals) else self.state.pos
        dist_to_goal = np.linalg.norm(current_goal - self.state.pos)

        reward = self.config.step_penalty
        reward -= dist_to_goal * 0.01
        reward -= self.config.control_cost_weight * np.sum(action ** 2)

        if collision_this_step:
            reward += self.config.collision_penalty

        done = False
        info = {
            'collision': collision_this_step,
            'goal_switch': False,
            'current_goal_idx': self.state.current_goal_idx,
        }

        # 목표 도달 체크
        if dist_to_goal < self.config.goal_radius:
            self.state.goals_completed += 1
            reward += self.config.goal_reward

            if self.state.current_goal_idx < len(self.state.goals) - 1:
                # 다음 목표로 전환
                self.state.current_goal_idx += 1
                self.state.goal = self.state.goals[self.state.current_goal_idx]
                self.state.goal_switch_step = self.state.step
                info['goal_switch'] = True
            else:
                # 마지막 목표 완료
                reward += self.config.final_goal_bonus
                self.state.goal_reached = True
                self.triple_successes += 1
                done = True
                info['triple_success'] = True

        # 타임아웃
        if self.state.step >= self.config.max_steps:
            done = True
            info['timeout'] = True

        self.state.total_reward += reward

        # Sequencing tracking
        if done:
            self.total_goals_completed += self.state.goals_completed

        info['goals_completed'] = self.state.goals_completed

        return self._get_observation(), reward, done, info

    def get_stability_stats(self) -> Dict:
        """Stability 통계"""
        return {
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'speed_violations': self.speed_violation_count,
            'pos_violations': self.pos_violation_count,
            'action_gate': self.action_gate.get_stats(),
        }

    def get_safety_stats(self) -> Dict:
        """Safety 통계"""
        collision_rate = self.total_collisions / max(1, self.total_episodes)
        return {
            'total_collisions': self.total_collisions,
            'total_episodes': self.total_episodes,
            'collision_rate': collision_rate,
        }

    def get_sequencing_stats(self) -> Dict:
        """Sequencing 통계"""
        triple_rate = self.triple_successes / max(1, self.total_episodes)
        avg_goals = self.total_goals_completed / max(1, self.total_episodes)
        return {
            'triple_successes': self.triple_successes,
            'total_episodes': self.total_episodes,
            'triple_success_rate': triple_rate,
            'avg_goals_completed': avg_goals,
        }

    def reset_all_stats(self):
        self.nan_count = 0
        self.inf_count = 0
        self.speed_violation_count = 0
        self.pos_violation_count = 0
        self.action_gate.reset_stats()
        self.total_collisions = 0
        self.total_episodes = 0
        self.triple_successes = 0
        self.total_goals_completed = 0


# ============================================================================
# E6-3 Gates
# ============================================================================

@dataclass
class E6SequencingGateResult:
    """Sequencing Gate 결과"""
    passed: bool
    reason: str

    triple_success_rate: float
    avg_goals_completed: float


class E6_3Gate(E6_2Gate):
    """E6-3 Gate 평가"""

    # Safety threshold (E6-3는 경로가 길어서 완화)
    MAX_COLLISION_RATE = 0.15  # 15% 이하 (E6-2: 10%)

    # Sequencing thresholds
    MIN_TRIPLE_SUCCESS_RATE = 0.70  # 70% 완주
    MIN_AVG_GOALS = 2.5  # 평균 2.5개 이상

    def evaluate_safety(
        self,
        collision_rate: float,
        total_collisions: int,
        total_episodes: int,
    ) -> E6SafetyGateResult:
        """Safety Gate 평가 (E6-3용 완화된 임계값)"""

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

    def evaluate_sequencing(
        self,
        triple_success_rate: float,
        avg_goals_completed: float,
    ) -> E6SequencingGateResult:
        """Sequencing Gate 평가"""

        passed = (
            triple_success_rate >= self.MIN_TRIPLE_SUCCESS_RATE and
            avg_goals_completed >= self.MIN_AVG_GOALS
        )

        if passed:
            reason = "PASS"
        else:
            reasons = []
            if triple_success_rate < self.MIN_TRIPLE_SUCCESS_RATE:
                reasons.append(f"triple={triple_success_rate:.1%}<{self.MIN_TRIPLE_SUCCESS_RATE:.0%}")
            if avg_goals_completed < self.MIN_AVG_GOALS:
                reasons.append(f"avg_goals={avg_goals_completed:.2f}<{self.MIN_AVG_GOALS}")
            reason = ", ".join(reasons)

        return E6SequencingGateResult(
            passed=passed,
            reason=reason,
            triple_success_rate=triple_success_rate,
            avg_goals_completed=avg_goals_completed,
        )
