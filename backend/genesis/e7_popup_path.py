"""
E7-B1b: Path-biased Pop-up Obstacle

B1 확장: Pop-up이 에이전트 진행 방향 앞에 생성
→ +MEM lag 병리를 극대화하여 "행동 EMA는 구조적으로 위험" 완전 증명

Pop-up 생성 규칙:
- 기준 벡터 v: 속도 > threshold면 vel 방향, 아니면 goal 방향
- pos_spawn = pos_agent + v * d_forward + n_perp * d_lateral
- 즉시 충돌 금지, 회피 공간 보장

추가 지표:
- Near-miss rate: 충돌 안 했지만 min_dist < ε (회피가 늦었다)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List

from .e7_popup import (
    PopupConfig,
    PopupState,
    PopupNavEnv,
    E7_B1Gate,
    E7EventGateResult,
)
from .e6_obstacle_nav import Obstacle


@dataclass
class PathPopupConfig(PopupConfig):
    """E7-B1b 환경 설정"""
    # 경로 편향 설정
    path_bias_enabled: bool = True
    min_speed_for_vel: float = 0.3      # 속도 기준 (이하면 goal 방향 사용)
    forward_distance_min: float = 1.5   # 전방 거리 최소
    forward_distance_max: float = 2.5   # 전방 거리 최대
    lateral_offset_max: float = 0.3     # 측면 오프셋 최대

    # Near-miss 설정
    near_miss_threshold: float = 1.5    # 충돌 반경의 1.5배


@dataclass
class PathPopupState(PopupState):
    """E7-B1b 환경 상태"""
    # Near-miss 추적
    near_miss_count: int = 0
    min_popup_distance: float = float('inf')  # Pop-up 장애물과의 최소 거리


class PathPopupNavEnv(PopupNavEnv):
    """
    Path-biased Pop-up Navigation Environment

    E7-B1b 핵심: 진행 방향 앞에 장애물 생성 → +MEM lag 병리 극대화
    """

    def __init__(self, config: PathPopupConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.config = config

        # Near-miss 통계
        self.near_miss_episodes = []

    def _get_path_direction(self) -> np.ndarray:
        """에이전트 진행 방향 계산"""
        speed = np.linalg.norm(self.state.vel)

        if speed > self.config.min_speed_for_vel:
            # 속도가 충분하면 속도 방향
            direction = self.state.vel / speed
        else:
            # 속도가 작으면 목표 방향
            current_goal = self.state.goals[self.state.current_goal_idx] \
                if self.state.current_goal_idx < len(self.state.goals) else self.state.goal
            goal_rel = current_goal - self.state.pos
            goal_dist = np.linalg.norm(goal_rel)

            if goal_dist > 1e-6:
                direction = goal_rel / goal_dist
            else:
                # 랜덤 방향 (fallback)
                angle = self.rng.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])

        return direction

    def _spawn_popup_obstacle(self) -> Optional[Obstacle]:
        """경로 앞에 장애물 생성"""
        if not self.config.path_bias_enabled:
            return super()._spawn_popup_obstacle()

        max_attempts = 50

        # 진행 방향
        forward = self._get_path_direction()

        # 수직 방향 (좌/우)
        perpendicular = np.array([-forward[1], forward[0]])

        for _ in range(max_attempts):
            # 전방 거리
            d_forward = self.rng.uniform(
                self.config.forward_distance_min,
                self.config.forward_distance_max
            )

            # 측면 오프셋 (좌/우 랜덤)
            d_lateral = self.rng.uniform(
                -self.config.lateral_offset_max,
                self.config.lateral_offset_max
            )

            # 생성 위치
            pos = (self.state.pos +
                   forward * d_forward +
                   perpendicular * d_lateral)

            # 경계 체크
            ws = self.config.world_size
            if np.any(np.abs(pos) > ws * 0.9):
                continue

            # 즉시 충돌 방지
            dist_to_agent = np.linalg.norm(pos - self.state.pos)
            if dist_to_agent < self.config.popup_radius + 0.5:
                continue

            # 기존 장애물과 겹침 체크
            valid = True
            for obs in self.state.obstacles:
                if np.linalg.norm(pos - obs.pos) < self.config.popup_radius + obs.radius + 0.3:
                    valid = False
                    break

            # 목표와 겹침 체크
            for goal in self.state.goals:
                if np.linalg.norm(pos - goal) < self.config.popup_radius + self.config.goal_radius + 0.5:
                    valid = False
                    break

            if valid:
                return Obstacle(
                    pos=pos.astype(np.float32),
                    radius=self.config.popup_radius
                )

        # 경로 편향 실패 시 기본 방식으로 fallback
        return super()._spawn_popup_obstacle()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        obs = super().reset(seed)

        # PathPopupState로 확장
        self.state = PathPopupState(
            # 기존 상태 복사
            pos=self.state.pos,
            vel=self.state.vel,
            goal=self.state.goal,
            step=self.state.step,
            total_reward=self.state.total_reward,
            goal_reached=self.state.goal_reached,
            prev_action=self.state.prev_action,
            visited_positions=self.state.visited_positions,
            goal_memory=self.state.goal_memory,
            obstacles=self.state.obstacles,
            collision_count=self.state.collision_count,
            collision_this_episode=self.state.collision_this_episode,
            goals=self.state.goals,
            current_goal_idx=self.state.current_goal_idx,
            goals_completed=self.state.goals_completed,
            goal_switch_step=self.state.goal_switch_step,
            post_switch_collisions=self.state.post_switch_collisions,
            current_n_obstacles=self.state.current_n_obstacles,
            current_friction=self.state.current_friction,
            current_max_speed=self.state.current_max_speed,
            current_goal_radius=self.state.current_goal_radius,
            # Pop-up 상태 (super().reset()에서 설정됨)
            popup_triggered=False,
            popup_step=self.state.popup_step,
            popup_obstacle_idx=-1,
            event_window_collision=False,
            event_window_steps=0,
            risk_detected_step=-1,
            defense_started_step=-1,
            reaction_time=-1,
            pre_event_defense_steps=0,
            pre_event_total_steps=0,
            # B1b 추가
            near_miss_count=0,
            min_popup_distance=float('inf'),
        )

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
        obs, reward, done, info = super().step(action)

        # Near-miss 추적 (pop-up 장애물과의 거리)
        if self.state.popup_triggered and self.state.popup_obstacle_idx >= 0:
            popup_obs = self.state.obstacles[self.state.popup_obstacle_idx]
            dist_to_popup = np.linalg.norm(self.state.pos - popup_obs.pos) - popup_obs.radius

            # 최소 거리 업데이트
            self.state.min_popup_distance = min(
                self.state.min_popup_distance,
                dist_to_popup
            )

            # Near-miss 체크 (충돌은 안 했지만 가까웠다)
            near_miss_threshold = self.config.popup_radius * (self.config.near_miss_threshold - 1)
            if dist_to_popup < near_miss_threshold and dist_to_popup > 0:
                if not hasattr(self.state, '_near_miss_recorded'):
                    self.state.near_miss_count += 1
                    self.state._near_miss_recorded = True

        info['min_popup_distance'] = self.state.min_popup_distance

        # 에피소드 종료 시 통계 기록
        if done:
            # Near-miss 발생 여부
            had_near_miss = self.state.near_miss_count > 0 or \
                (self.state.min_popup_distance < self.config.popup_radius * 0.5 and
                 not self.state.event_window_collision)
            self.near_miss_episodes.append(1 if had_near_miss else 0)

        return obs, reward, done, info

    def get_path_popup_stats(self) -> Dict:
        """Path pop-up 통계"""
        base_stats = self.get_popup_stats()

        # Near-miss rate
        if len(self.near_miss_episodes) > 0:
            near_miss_rate = np.mean(self.near_miss_episodes)
        else:
            near_miss_rate = 0.0

        base_stats['near_miss_rate'] = near_miss_rate
        return base_stats

    def reset_all_stats(self):
        super().reset_all_stats()
        self.near_miss_episodes = []


# ============================================================================
# E7-B1b Gates
# ============================================================================

@dataclass
class E7PathEventGateResult:
    """Path Event Gate 결과"""
    passed: bool
    reason: str

    # 세부 지표
    event_collision_rate: float
    near_miss_rate: float
    mean_reaction_time: float
    p95_reaction_time: float

    # 게이트별 통과 여부
    event_collision_passed: bool
    near_miss_passed: bool


class E7_B1bGate(E7_B1Gate):
    """E7-B1b Gate 평가"""

    # Near-miss 임계값
    MAX_NEAR_MISS_RATE = 0.10  # 10%

    # Event collision은 더 엄격하게
    MAX_EVENT_COLLISION = 0.02  # 2%

    def evaluate_path_event_response(
        self,
        event_collision_rate: float,
        near_miss_rate: float,
        mean_reaction_time: float,
        p95_reaction_time: float,
    ) -> E7PathEventGateResult:
        """
        Path Event Response Gate 평가

        PASS 조건:
        1. event_collision_rate < 2%
        2. near_miss_rate < 10%
        """
        event_coll_passed = event_collision_rate <= self.MAX_EVENT_COLLISION
        near_miss_passed = near_miss_rate <= self.MAX_NEAR_MISS_RATE

        all_passed = event_coll_passed and near_miss_passed

        if all_passed:
            reason = "PASS"
        else:
            reasons = []
            if not event_coll_passed:
                reasons.append(f"event_coll={event_collision_rate:.1%}>{self.MAX_EVENT_COLLISION:.0%}")
            if not near_miss_passed:
                reasons.append(f"near_miss={near_miss_rate:.1%}>{self.MAX_NEAR_MISS_RATE:.0%}")
            reason = ", ".join(reasons)

        return E7PathEventGateResult(
            passed=all_passed,
            reason=reason,
            event_collision_rate=event_collision_rate,
            near_miss_rate=near_miss_rate,
            mean_reaction_time=mean_reaction_time,
            p95_reaction_time=p95_reaction_time,
            event_collision_passed=event_coll_passed,
            near_miss_passed=near_miss_passed,
        )
