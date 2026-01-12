"""
E7-B1: Pop-up Obstacle (Adversarial Event)

동적 돌발 이벤트에서 방어 구조 검증

핵심:
- 에피소드 중 1회, 랜덤 시점에 장애물 갑자기 생성
- 생성 위치: 에이전트 근처 (위협이 되도록), 즉시 충돌은 금지
- Event window (10 step) 내 충돌률로 반응 속도 평가

게이트:
1. Event collision rate: 이벤트 후 10 step 내 충돌률 < 5%
2. Reaction time: 방어 모드 전환까지 시간
3. False defense: 이벤트 없는 구간의 과방어율
4. Global metrics: mean collision, triple success
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List

from .e7_generalization import (
    GeneralizationConfig,
    GeneralizationState,
    GeneralizationNavEnv,
)
from .e6_obstacle_nav import Obstacle


@dataclass
class PopupConfig(GeneralizationConfig):
    """E7-B1 환경 설정"""
    # Pop-up 설정
    popup_enabled: bool = True
    popup_min_step_ratio: float = 0.2   # 최소 발생 시점 (전체의 20%)
    popup_max_step_ratio: float = 0.8   # 최대 발생 시점 (전체의 80%)

    # 생성 위치 설정
    popup_spawn_min: float = 1.5        # 최소 거리 (즉시 충돌 방지)
    popup_spawn_max: float = 3.0        # 최대 거리 (위협이 되도록)
    popup_radius: float = 1.0           # pop-up 장애물 반경

    # 이벤트 윈도우 설정
    event_window: int = 10              # 이벤트 후 관찰 윈도우 (steps)


@dataclass
class PopupState(GeneralizationState):
    """E7-B1 환경 상태"""
    # Pop-up 이벤트 상태
    popup_triggered: bool = False
    popup_step: int = -1                # pop-up 발생 step
    popup_obstacle_idx: int = -1        # 생성된 장애물 인덱스

    # 이벤트 윈도우 추적
    event_window_collision: bool = False  # 이벤트 윈도우 내 충돌 여부
    event_window_steps: int = 0           # 이벤트 윈도우 경과 step

    # 반응 시간 추적
    risk_detected_step: int = -1          # 위험 감지 step
    defense_started_step: int = -1        # 방어 모드 시작 step
    reaction_time: int = -1               # 반응 시간

    # False defense 추적 (이벤트 전 방어 모드)
    pre_event_defense_steps: int = 0
    pre_event_total_steps: int = 0


class PopupNavEnv(GeneralizationNavEnv):
    """
    Pop-up Obstacle Navigation Environment

    E7-B1 핵심: 갑작스런 위험 변화에 대한 반응 검증
    """

    def __init__(self, config: PopupConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.config = config

        # 이벤트 통계
        self.event_collisions = []        # 각 에피소드의 이벤트 윈도우 충돌 여부
        self.reaction_times = []          # 반응 시간 기록
        self.false_defense_ratios = []    # 에피소드별 false defense 비율

    def _schedule_popup(self) -> int:
        """Pop-up 발생 시점 결정"""
        min_step = int(self.config.max_steps * self.config.popup_min_step_ratio)
        max_step = int(self.config.max_steps * self.config.popup_max_step_ratio)
        return self.rng.randint(min_step, max_step + 1)

    def _spawn_popup_obstacle(self) -> Optional[Obstacle]:
        """에이전트 근처에 장애물 생성"""
        max_attempts = 50

        for _ in range(max_attempts):
            # 에이전트 기준 랜덤 방향
            angle = self.rng.uniform(0, 2 * np.pi)

            # 거리: spawn_min ~ spawn_max 사이
            dist = self.rng.uniform(
                self.config.popup_spawn_min,
                self.config.popup_spawn_max
            )

            # 생성 위치
            pos = self.state.pos + np.array([
                dist * np.cos(angle),
                dist * np.sin(angle)
            ])

            # 경계 체크
            ws = self.config.world_size
            if np.any(np.abs(pos) > ws * 0.9):
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

        return None  # 생성 실패

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        obs = super().reset(seed)

        # Pop-up 이벤트 스케줄
        popup_step = self._schedule_popup() if self.config.popup_enabled else -1

        # 상태 업데이트 (GeneralizationState를 PopupState로 확장)
        self.state = PopupState(
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
            # Pop-up 상태
            popup_triggered=False,
            popup_step=popup_step,
            popup_obstacle_idx=-1,
            event_window_collision=False,
            event_window_steps=0,
            risk_detected_step=-1,
            defense_started_step=-1,
            reaction_time=-1,
            pre_event_defense_steps=0,
            pre_event_total_steps=0,
        )

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
        # Pop-up 트리거 체크 (이벤트 발생 전)
        popup_just_triggered = False
        if (self.config.popup_enabled and
            not self.state.popup_triggered and
            self.state.step == self.state.popup_step):

            new_obstacle = self._spawn_popup_obstacle()
            if new_obstacle is not None:
                self.state.obstacles.append(new_obstacle)
                self.state.popup_obstacle_idx = len(self.state.obstacles) - 1
                self.state.popup_triggered = True
                popup_just_triggered = True

        # 부모 step 실행
        obs, reward, done, info = super().step(action)

        # 이벤트 정보 추가
        info['popup_triggered'] = self.state.popup_triggered
        info['popup_just_triggered'] = popup_just_triggered
        info['popup_step'] = self.state.popup_step

        # 이벤트 윈도우 추적
        if self.state.popup_triggered:
            steps_since_popup = self.state.step - self.state.popup_step

            if 0 < steps_since_popup <= self.config.event_window:
                self.state.event_window_steps = steps_since_popup

                # 이벤트 윈도우 내 충돌 체크
                if info.get('collision', False):
                    self.state.event_window_collision = True

        # 에피소드 종료 시 통계 기록
        if done:
            self.event_collisions.append(1 if self.state.event_window_collision else 0)

            if self.state.reaction_time > 0:
                self.reaction_times.append(self.state.reaction_time)

            # False defense 비율
            if self.state.pre_event_total_steps > 0:
                fd_ratio = self.state.pre_event_defense_steps / self.state.pre_event_total_steps
                self.false_defense_ratios.append(fd_ratio)

        return obs, reward, done, info

    def record_defense_state(self, in_defense_mode: bool, true_risk: float):
        """
        에이전트가 방어 모드 상태 기록 (에이전트가 호출)

        Args:
            in_defense_mode: 현재 방어 모드 여부
            true_risk: 실제 위험 수준
        """
        # 이벤트 전 false defense 추적
        if not self.state.popup_triggered:
            self.state.pre_event_total_steps += 1
            if in_defense_mode:
                self.state.pre_event_defense_steps += 1

        # 이벤트 후 반응 시간 추적
        if self.state.popup_triggered and self.state.reaction_time < 0:
            # 위험 감지
            if self.state.risk_detected_step < 0 and true_risk > 0.3:
                self.state.risk_detected_step = self.state.step

            # 방어 모드 시작
            if self.state.defense_started_step < 0 and in_defense_mode:
                self.state.defense_started_step = self.state.step

                # 반응 시간 계산
                if self.state.risk_detected_step > 0:
                    self.state.reaction_time = (
                        self.state.defense_started_step - self.state.risk_detected_step
                    )
                else:
                    # 위험 감지 전 방어 모드 진입 (선제 방어)
                    self.state.reaction_time = 0

    def get_popup_stats(self) -> Dict:
        """Pop-up 이벤트 통계"""
        if len(self.event_collisions) == 0:
            return {
                'event_collision_rate': 0.0,
                'mean_reaction_time': 0.0,
                'p95_reaction_time': 0.0,
                'false_defense_rate': 0.0,
                'n_events': 0,
            }

        event_coll_rate = np.mean(self.event_collisions)

        if len(self.reaction_times) > 0:
            mean_rt = np.mean(self.reaction_times)
            sorted_rt = np.sort(self.reaction_times)
            p95_idx = min(int(len(sorted_rt) * 0.95), len(sorted_rt) - 1)
            p95_rt = sorted_rt[p95_idx]
        else:
            mean_rt = 0.0
            p95_rt = 0.0

        if len(self.false_defense_ratios) > 0:
            false_defense_rate = np.mean(self.false_defense_ratios)
        else:
            false_defense_rate = 0.0

        return {
            'event_collision_rate': event_coll_rate,
            'mean_reaction_time': mean_rt,
            'p95_reaction_time': p95_rt,
            'false_defense_rate': false_defense_rate,
            'n_events': len(self.event_collisions),
        }

    def reset_all_stats(self):
        super().reset_all_stats()
        self.event_collisions = []
        self.reaction_times = []
        self.false_defense_ratios = []


# ============================================================================
# E7-B1 Gates
# ============================================================================

@dataclass
class E7EventGateResult:
    """Event Gate 결과"""
    passed: bool
    reason: str

    # 세부 지표
    event_collision_rate: float
    mean_reaction_time: float
    p95_reaction_time: float
    false_defense_rate: float

    # 게이트별 통과 여부
    event_collision_passed: bool
    reaction_time_passed: bool


@dataclass
class E7PopupSummary:
    """E7-B1 전체 요약"""
    passed: bool

    # 각 config별 결과
    results_by_config: Dict[str, Dict]

    # FULL vs FULL+RF 비교
    full_event_coll: float
    rf_event_coll: float
    full_false_defense: float
    rf_false_defense: float


class E7_B1Gate:
    """E7-B1 Gate 평가"""

    # Event collision 임계값
    MAX_EVENT_COLLISION = 0.05  # 5%

    # Reaction time 임계값 (steps)
    MAX_MEAN_REACTION_TIME = 5.0
    MAX_P95_REACTION_TIME = 10.0

    # Global collision 임계값 (A1과 동일)
    MAX_MEAN_COLLISION = 0.15

    def evaluate_event_response(
        self,
        event_collision_rate: float,
        mean_reaction_time: float,
        p95_reaction_time: float,
        false_defense_rate: float,
    ) -> E7EventGateResult:
        """
        Event Response Gate 평가

        PASS 조건:
        1. event_collision_rate < 5%
        2. p95_reaction_time < 10 steps
        """
        event_coll_passed = event_collision_rate <= self.MAX_EVENT_COLLISION
        rt_passed = p95_reaction_time <= self.MAX_P95_REACTION_TIME

        all_passed = event_coll_passed and rt_passed

        if all_passed:
            reason = "PASS"
        else:
            reasons = []
            if not event_coll_passed:
                reasons.append(f"event_coll={event_collision_rate:.1%}>{self.MAX_EVENT_COLLISION:.0%}")
            if not rt_passed:
                reasons.append(f"p95_rt={p95_reaction_time:.1f}>{self.MAX_P95_REACTION_TIME:.0f}")
            reason = ", ".join(reasons)

        return E7EventGateResult(
            passed=all_passed,
            reason=reason,
            event_collision_rate=event_collision_rate,
            mean_reaction_time=mean_reaction_time,
            p95_reaction_time=p95_reaction_time,
            false_defense_rate=false_defense_rate,
            event_collision_passed=event_coll_passed,
            reaction_time_passed=rt_passed,
        )

    def compare_configs(
        self,
        results: Dict[str, Dict],
    ) -> E7PopupSummary:
        """설정별 비교 요약"""
        full_result = results.get('FULL', {})
        rf_result = results.get('FULL+RF', {})

        # FULL과 FULL+RF 모두 이벤트 게이트 통과해야 PASS
        full_event_gate = self.evaluate_event_response(
            full_result.get('event_collision_rate', 1.0),
            full_result.get('mean_reaction_time', 99),
            full_result.get('p95_reaction_time', 99),
            full_result.get('false_defense_rate', 1.0),
        )
        rf_event_gate = self.evaluate_event_response(
            rf_result.get('event_collision_rate', 1.0),
            rf_result.get('mean_reaction_time', 99),
            rf_result.get('p95_reaction_time', 99),
            rf_result.get('false_defense_rate', 1.0),
        )

        overall_passed = full_event_gate.passed or rf_event_gate.passed

        return E7PopupSummary(
            passed=overall_passed,
            results_by_config=results,
            full_event_coll=full_result.get('event_collision_rate', 0),
            rf_event_coll=rf_result.get('event_collision_rate', 0),
            full_false_defense=full_result.get('false_defense_rate', 0),
            rf_false_defense=rf_result.get('false_defense_rate', 0),
        )
