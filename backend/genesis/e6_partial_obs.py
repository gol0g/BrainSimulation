"""
E6-4a: Partial Observability (Goal Vector Dropout)

E6-3 확장: 목표 벡터를 확률적으로 가림
- p_drop 확률로 goal_dx, goal_dy, dist_to_goal을 가림
- ZOH (Zero-Order Hold): dropout 시 마지막 관측 유지
- 메모리가 진짜 필요한 상황 강제

관측: [pos_x, pos_y, vel_x, vel_y, goal_dx, goal_dy, dist_to_goal, ...]
      (goal 관련 3개가 확률적으로 가려짐)

게이트:
1. Stability / Safety (난이도에 따라 임계 완화)
2. Robustness (성능 완만 하락)
3. Lag sensitivity (event_collision_rate)
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
from .e6_obstacle_nav import Obstacle, ActionGate, E6SafetyGateResult


@dataclass
class PartialObsConfig(MultiGoalConfig):
    """E6-4a 환경 설정"""
    # Dropout 설정
    p_drop: float = 0.1  # 목표 벡터 dropout 확률
    dropout_channel: str = "goal"  # "goal", "obstacle", "both"

    # Safety threshold (난이도에 따라 조정)
    # p_drop 0.1: 15%, 0.3: 20%, 0.5: 25%
    max_collision_rate: float = 0.15


@dataclass
class PartialObsState(MultiGoalState):
    """E6-4a 환경 상태"""
    # ZOH (Zero-Order Hold) 버퍼
    last_goal_obs: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_obstacle_obs: np.ndarray = field(default_factory=lambda: np.zeros(10))

    # Dropout 추적
    goal_dropout_count: int = 0
    obstacle_dropout_count: int = 0
    total_obs_count: int = 0

    # Lag sensitivity 추적
    post_switch_steps: int = 0  # goal switch 이후 경과 step
    in_post_switch_window: bool = False  # goal switch 후 10 step 이내


class PartialObsNavEnv(MultiGoalNavEnv):
    """
    Partial Observability Navigation Environment

    E6-4a 핵심: 목표 벡터 dropout으로 메모리 필요성 강제
    """

    def __init__(self, config: PartialObsConfig, seed: Optional[int] = None):
        # 부모 클래스 초기화
        super().__init__(config, seed)
        self.config = config  # 타입 힌트를 위해 재할당

        # Event collision 추적
        self.event_collisions = 0  # goal switch 후 10 step 내 충돌
        self.total_goal_switches = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        obs = super().reset(seed)

        # ZOH 버퍼 초기화
        self.state.last_goal_obs = obs[4:7].copy()  # goal_dx, goal_dy, dist
        self.state.last_obstacle_obs = obs[7:17].copy()  # obstacle info

        # Dropout 추적 초기화
        self.state.goal_dropout_count = 0
        self.state.obstacle_dropout_count = 0
        self.state.total_obs_count = 0
        self.state.post_switch_steps = 0
        self.state.in_post_switch_window = False

        return obs

    def _apply_dropout(self, obs: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """
        관측에 dropout 적용

        Returns:
            obs: dropout이 적용된 관측
            goal_dropped: 목표가 가려졌는지
            obstacle_dropped: 장애물이 가려졌는지
        """
        obs = obs.copy()
        goal_dropped = False
        obstacle_dropped = False

        self.state.total_obs_count += 1

        # Goal dropout
        if self.config.dropout_channel in ["goal", "both"]:
            if self.rng.random() < self.config.p_drop:
                # ZOH: 마지막 관측 유지
                obs[4:7] = self.state.last_goal_obs
                goal_dropped = True
                self.state.goal_dropout_count += 1
            else:
                # 현재 값 저장
                self.state.last_goal_obs = obs[4:7].copy()

        # Obstacle dropout
        if self.config.dropout_channel in ["obstacle", "both"]:
            if self.rng.random() < self.config.p_drop:
                obs[7:17] = self.state.last_obstacle_obs
                obstacle_dropped = True
                self.state.obstacle_dropout_count += 1
            else:
                self.state.last_obstacle_obs = obs[7:17].copy()

        return obs, goal_dropped, obstacle_dropped

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝 (dropout 적용)"""
        # 부모 스텝 실행
        obs, reward, done, info = super().step(action)

        # Post-switch window 추적
        if info.get('goal_switch'):
            self.state.in_post_switch_window = True
            self.state.post_switch_steps = 0
            self.total_goal_switches += 1

        if self.state.in_post_switch_window:
            self.state.post_switch_steps += 1
            if self.state.post_switch_steps > 10:
                self.state.in_post_switch_window = False

        # Event collision 추적 (goal switch 후 10 step 내 충돌)
        if info.get('collision') and self.state.in_post_switch_window:
            self.event_collisions += 1

        # Dropout 적용
        obs, goal_dropped, obstacle_dropped = self._apply_dropout(obs)
        info['goal_dropped'] = goal_dropped
        info['obstacle_dropped'] = obstacle_dropped

        return obs, reward, done, info

    def get_dropout_stats(self) -> Dict:
        """Dropout 통계"""
        total = max(1, self.state.total_obs_count)
        return {
            'goal_dropout_rate': self.state.goal_dropout_count / total,
            'obstacle_dropout_rate': self.state.obstacle_dropout_count / total,
            'total_obs_count': self.state.total_obs_count,
        }

    def get_lag_sensitivity_stats(self) -> Dict:
        """Lag sensitivity 통계"""
        event_collision_rate = self.event_collisions / max(1, self.total_goal_switches)
        return {
            'event_collisions': self.event_collisions,
            'total_goal_switches': self.total_goal_switches,
            'event_collision_rate': event_collision_rate,
        }

    def reset_all_stats(self):
        super().reset_all_stats()
        self.event_collisions = 0
        self.total_goal_switches = 0


# ============================================================================
# E6-4a Gates
# ============================================================================

@dataclass
class E6RobustnessGateResult:
    """Robustness Gate 결과"""
    passed: bool
    reason: str

    # 각 난이도 레벨별 결과
    level_results: Dict  # {p_drop: {config_name: metrics}}

    # FULL이 BASE보다 우위인 레벨 수
    full_advantage_levels: int
    total_levels: int


@dataclass
class E6LagSensitivityResult:
    """Lag Sensitivity 분석 결과"""
    # 각 config별 event_collision_rate
    base_event_rate: float
    mem_event_rate: float
    full_event_rate: float

    # +MEM의 lag 부작용 정량화
    mem_lag_penalty: float  # mem_event_rate - base_event_rate


class E6_4Gate(E6_3Gate):
    """E6-4a Gate 평가"""

    # Safety thresholds per difficulty
    SAFETY_THRESHOLDS = {
        0.1: 0.15,  # 15%
        0.3: 0.20,  # 20%
        0.5: 0.25,  # 25%
    }

    def get_safety_threshold(self, p_drop: float) -> float:
        """난이도에 따른 safety threshold"""
        if p_drop <= 0.1:
            return 0.15
        elif p_drop <= 0.3:
            return 0.20
        else:
            return 0.25

    def evaluate_robustness(
        self,
        level_results: Dict,
    ) -> E6RobustnessGateResult:
        """
        Robustness Gate 평가

        PASS 조건:
        1. FULL이 최소 2개 레벨에서 BASE 대비 우위 유지
        2. +MEM의 충돌률이 BASE를 넘지 않아야 함 (안전 우선)
        """
        full_advantage_count = 0
        mem_safe = True
        total_levels = len(level_results)

        for p_drop, results in level_results.items():
            base_coll = results.get('BASE', {}).get('collision_rate', 1.0)
            mem_coll = results.get('+MEM', {}).get('collision_rate', 1.0)
            full_coll = results.get('FULL', {}).get('collision_rate', 1.0)

            # FULL이 BASE보다 나은지
            if full_coll < base_coll:
                full_advantage_count += 1

            # +MEM이 BASE보다 안 좋으면 경고 (FAIL은 아님)
            if mem_coll > base_coll * 1.1:  # 10% 이상 나쁘면
                mem_safe = False

        passed = full_advantage_count >= 2

        if passed:
            reason = f"FULL advantage in {full_advantage_count}/{total_levels} levels"
            if not mem_safe:
                reason += " (WARNING: +MEM collision rate exceeds BASE)"
        else:
            reason = f"FULL advantage only in {full_advantage_count}/{total_levels} levels (<2)"

        return E6RobustnessGateResult(
            passed=passed,
            reason=reason,
            level_results=level_results,
            full_advantage_levels=full_advantage_count,
            total_levels=total_levels,
        )

    def analyze_lag_sensitivity(
        self,
        base_event_rate: float,
        mem_event_rate: float,
        full_event_rate: float,
    ) -> E6LagSensitivityResult:
        """Lag sensitivity 분석"""
        mem_lag_penalty = mem_event_rate - base_event_rate

        return E6LagSensitivityResult(
            base_event_rate=base_event_rate,
            mem_event_rate=mem_event_rate,
            full_event_rate=full_event_rate,
            mem_lag_penalty=mem_lag_penalty,
        )
