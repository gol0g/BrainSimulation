"""
v5.4: Interaction Gating - Act/Learn Coupling Split

핵심 변경 (v5.3-2 → v5.4):
- external_coupling을 act_coupling + learn_coupling으로 분리
- "행동은 줄여도 학습은 유지/증폭" 가능
- locked-in 문제 해결: 쉬면서도 보는 뇌

Gating 원리:
- act_coupling: 행동 실행 확률 (몸을 세상에 얼마나 걸지)
- learn_coupling: 관측 업데이트, 학습률 (세상을 보긴 보는데 얼마나 반영할지)
- internal_coupling: 기억 재생, 프로토타입 생성 결합 강도

z별 처방:
- z=0 (stable): act=normal, learn=normal
- z=1 (uncertain): act=↓가능, learn=↑ (quiet learning)
- z=2 (regret): act=normal, learn=↑
- z=3 (fatigue): act=↓, learn=normal, consolidation=↑

성공 조건:
- G2a (적응 속도) 유지 또는 개선
- G2c (efficiency) 유지 또는 개선
- Locked-in 없이 적응
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class GatingState:
    """Interaction gating 상태"""
    # z 히스토리
    z_history: deque = field(default_factory=lambda: deque(maxlen=50))
    fatigue_streak: int = 0
    uncertainty_streak: int = 0  # v5.4: z=1 연속 추적

    # Efficiency 추적
    efficiency_history: deque = field(default_factory=lambda: deque(maxlen=30))
    efficiency_at_fatigue_start: float = 0.5

    # v5.4: Act/Learn 분리된 gating 상태
    current_act_coupling: float = 1.0
    current_learn_coupling: float = 1.0
    current_internal_coupling: float = 1.0

    # 회복 추적
    recovery_phase: bool = False
    steps_in_recovery: int = 0


@dataclass
class GatingModifiers:
    """v5.4 Interaction gating 결과 - Act/Learn 분리"""
    # v5.4 핵심: Act와 Learn 분리
    act_coupling: float = 1.0     # 행동 실행 결합 (0.2 ~ 1.0)
    learn_coupling: float = 1.0   # 관측/학습 결합 (0.5 ~ 1.5)

    # 내부 결합 (1.0 ~ 3.0)
    internal_coupling: float = 1.0  # 통합 트리거 민감도, 재생 강도

    # 파생 값
    action_execution_prob: float = 1.0      # act_coupling 기반
    observation_update_weight: float = 1.0  # learn_coupling 기반
    learning_rate_boost: float = 1.0        # learn_coupling 기반
    consolidation_boost: float = 1.0        # internal_coupling 기반
    replay_intensity: float = 1.0           # internal_coupling 기반

    # v5.3-2 호환성을 위한 레거시
    @property
    def external_coupling(self) -> float:
        """v5.3-2 호환: act와 learn의 평균"""
        return (self.act_coupling + self.learn_coupling) / 2

    def to_dict(self) -> Dict:
        return {
            'act_coupling': self.act_coupling,
            'learn_coupling': self.learn_coupling,
            'internal_coupling': self.internal_coupling,
            'action_execution_prob': self.action_execution_prob,
            'observation_update_weight': self.observation_update_weight,
            'learning_rate_boost': self.learning_rate_boost,
            'consolidation_boost': self.consolidation_boost,
            'replay_intensity': self.replay_intensity,
        }


@dataclass
class InteractionGatingConfig:
    """Interaction gating 설정"""
    # Fatigue 감지
    fatigue_streak_threshold: int = 8  # z=3가 이 스텝 이상 지속되면 gating 시작
    uncertainty_streak_threshold: int = 5  # v5.4: z=1 quiet learning 트리거

    # v5.4: Act coupling 범위
    min_act_coupling: float = 0.2   # 최소 행동 결합 (완전 차단 X)
    act_decay_rate: float = 0.10    # fatigue streak당 감소율

    # v5.4: Learn coupling 범위
    min_learn_coupling: float = 0.5   # 최소 학습 결합 (data starvation 방지)
    max_learn_coupling: float = 1.5   # 최대 학습 결합
    learn_boost_rate: float = 0.08    # uncertainty streak당 증가율

    # Internal coupling 범위
    max_internal_coupling: float = 3.0  # 최대 내부 결합
    internal_boost_rate: float = 0.15   # fatigue streak당 증가율

    # 회복 조건
    efficiency_recovery_threshold: float = 0.3
    recovery_steps: int = 15


class InteractionGating:
    """
    v5.4 Interaction Gating - Act/Learn Coupling Split

    핵심 변경:
    - external_coupling → act_coupling + learn_coupling 분리
    - "행동은 줄여도(Act↓) 관측/학습은 유지·증폭(Learn↑)"
    - locked-in 없이 적응 가능

    z별 처방:
    - z=0 (stable): act=1.0, learn=1.0 (기본)
    - z=1 (uncertain): act=유지/↓, learn=↑ (quiet learning)
    - z=2 (regret): act=1.0, learn=↑ (반성적 학습)
    - z=3 (fatigue): act=↓, learn=유지, internal=↑ (쉬면서 통합)

    Usage:
        gating = InteractionGating()

        # 매 스텝
        modifiers = gating.update(z=current_z, efficiency=current_efficiency)

        # 행동 실행 시
        if np.random.random() < modifiers.action_execution_prob:
            execute_action()

        # 관측 업데이트 시
        belief_update *= modifiers.observation_update_weight

        # 학습 시
        learning_rate *= modifiers.learning_rate_boost

        # 통합 트리거 시
        if should_consolidate() or modifiers.consolidation_boost > 1.5:
            consolidate(batch_size_multiplier=modifiers.replay_intensity)
    """

    def __init__(self, config: Optional[InteractionGatingConfig] = None):
        self.config = config or InteractionGatingConfig()
        self.state = GatingState()

        # 통계
        self._step_count = 0
        self._fatigue_episodes = 0
        self._uncertainty_episodes = 0  # v5.4
        self._recovery_count = 0
        self._avg_recovery_steps: List[int] = []

    def update(
        self,
        z: int,
        efficiency: float,
        uncertainty: float = 0.3,
        Q_z: list = None
    ) -> GatingModifiers:
        """
        z와 efficiency 기반으로 gating modifier 계산

        Args:
            z: 현재 self-state (0=stable, 1=uncertain, 2=regret, 3=fatigue)
            efficiency: 현재 에너지 효율 (0-1+)
            uncertainty: 현재 불확실성 (보조 신호)
            Q_z: Q(z) 분포 [q0, q1, q2, q3]

        Returns:
            GatingModifiers: act/learn 분리된 결합 강도
        """
        self._step_count += 1
        cfg = self.config
        state = self.state

        # === 1. z history 업데이트 ===
        state.z_history.append(z)
        state.efficiency_history.append(efficiency)

        # === 2. Fatigue streak 계산 ===
        is_fatigue = z == 3 or (Q_z is not None and len(Q_z) > 3 and Q_z[3] > 0.4)
        if is_fatigue:
            state.fatigue_streak += 1
            if state.fatigue_streak == 1:
                state.efficiency_at_fatigue_start = np.mean(
                    list(state.efficiency_history)[-5:]
                ) if len(state.efficiency_history) >= 5 else efficiency
                self._fatigue_episodes += 1
        else:
            if state.fatigue_streak >= cfg.fatigue_streak_threshold:
                state.recovery_phase = True
                state.steps_in_recovery = 0
            state.fatigue_streak = 0

        # === 3. v5.4: Uncertainty streak 계산 ===
        is_uncertain = z == 1 or (Q_z is not None and len(Q_z) > 1 and Q_z[1] > 0.4)
        if is_uncertain:
            state.uncertainty_streak += 1
            if state.uncertainty_streak == 1:
                self._uncertainty_episodes += 1
        else:
            state.uncertainty_streak = 0

        # === 4. Recovery phase 처리 ===
        if state.recovery_phase:
            state.steps_in_recovery += 1

            if len(state.efficiency_history) >= 5:
                recent_efficiency = np.mean(list(state.efficiency_history)[-5:])
                efficiency_gain = recent_efficiency - state.efficiency_at_fatigue_start

                if efficiency_gain > cfg.efficiency_recovery_threshold:
                    self._avg_recovery_steps.append(state.steps_in_recovery)
                    self._recovery_count += 1
                    state.recovery_phase = False

            if state.steps_in_recovery >= cfg.recovery_steps:
                state.recovery_phase = False

        # === 5. v5.4: Act/Learn 분리 gating 계산 ===
        modifiers = self._compute_gating_v54(z, uncertainty, Q_z)

        # 상태 저장
        state.current_act_coupling = modifiers.act_coupling
        state.current_learn_coupling = modifiers.learn_coupling
        state.current_internal_coupling = modifiers.internal_coupling

        return modifiers

    def _compute_gating_v54(
        self,
        z: int,
        uncertainty: float,
        Q_z: Optional[list]
    ) -> GatingModifiers:
        """
        v5.4: Act/Learn 분리 gating 계산

        핵심 원칙:
        - act↓일 때도 learn은 유지/증가 가능 (data starvation 방지)
        - z=1: act 유지/↓, learn ↑ (quiet learning)
        - z=3: act ↓, learn 유지, internal ↑ (쉬면서 통합)
        """
        cfg = self.config
        state = self.state

        # Base values
        act = 1.0
        learn = 1.0
        internal = 1.0

        # === z=3 (fatigue): Act↓, Learn=유지, Internal↑ ===
        if state.fatigue_streak >= cfg.fatigue_streak_threshold:
            excess_streak = state.fatigue_streak - cfg.fatigue_streak_threshold

            # Act: 점진적 감소 (행동 줄임)
            decay = cfg.act_decay_rate * excess_streak
            act = max(cfg.min_act_coupling, 1.0 - decay)

            # Learn: 유지 (쉬면서도 관측은 계속)
            # data starvation 방지를 위해 learn은 줄이지 않음
            learn = 1.0

            # Internal: 점진적 증가 (통합 강화)
            boost = cfg.internal_boost_rate * excess_streak
            internal = min(cfg.max_internal_coupling, 1.0 + boost)

        # === z=1 (uncertain): Act=유지(높게), Learn↑ (quiet learning) ===
        # v5.4-fix: z=1은 "불확실 → 탐색↑"이 자연스러움
        # act를 너무 낮추면 데이터 수집이 막힘 → 하한을 높게 유지
        elif state.uncertainty_streak >= cfg.uncertainty_streak_threshold:
            excess_streak = state.uncertainty_streak - cfg.uncertainty_streak_threshold

            # Act: 높게 유지 (탐색/데이터 수집 필요)
            # 0.6 → 0.80: "아끼는 모드"가 너무 자주 켜지는 것 방지
            act = max(0.80, 1.0 - 0.03 * excess_streak)  # 감소율도 줄임

            # Learn: 증가 (새로운 환경 빨리 배움)
            boost = cfg.learn_boost_rate * excess_streak
            learn = min(cfg.max_learn_coupling, 1.0 + boost)

            # Internal: 약간 감소 (과거 기억보다 새 학습에 집중)
            internal = max(0.8, 1.0 - 0.03 * excess_streak)

        # === Recovery phase: 점진적 복귀 ===
        elif state.recovery_phase:
            progress = state.steps_in_recovery / cfg.recovery_steps
            act = 0.6 + 0.4 * progress     # 0.6 → 1.0
            learn = 1.2 - 0.2 * progress   # 1.2 → 1.0
            internal = 1.3 - 0.3 * progress  # 1.3 → 1.0

        # === z=2 (regret): Act=유지, Learn↑ ===
        if z == 2:
            learn = min(cfg.max_learn_coupling, learn * 1.2)

        # === 파생 값 계산 ===
        # Action execution: act coupling 기반
        action_prob = 0.2 + 0.8 * act  # 최소 20%는 실행

        # Observation update: learn coupling 직접 적용
        obs_weight = learn

        # Learning rate: learn coupling 기반
        lr_boost = learn

        # Consolidation: internal coupling 기반
        consolidation_boost = internal
        replay_intensity = internal

        return GatingModifiers(
            act_coupling=round(act, 3),
            learn_coupling=round(learn, 3),
            internal_coupling=round(internal, 3),
            action_execution_prob=round(action_prob, 3),
            observation_update_weight=round(obs_weight, 3),
            learning_rate_boost=round(lr_boost, 3),
            consolidation_boost=round(consolidation_boost, 3),
            replay_intensity=round(replay_intensity, 3),
        )

    def get_status(self) -> Dict:
        """현재 gating 상태"""
        state = self.state

        return {
            'enabled': True,
            'version': 'v5.4',
            'current_z': int(state.z_history[-1]) if state.z_history else 0,
            'fatigue_streak': state.fatigue_streak,
            'uncertainty_streak': state.uncertainty_streak,
            'in_fatigue_gating': state.fatigue_streak >= self.config.fatigue_streak_threshold,
            'in_quiet_learning': state.uncertainty_streak >= self.config.uncertainty_streak_threshold,
            'recovery_phase': state.recovery_phase,
            'steps_in_recovery': state.steps_in_recovery,
            'current_gating': {
                'act_coupling': state.current_act_coupling,
                'learn_coupling': state.current_learn_coupling,
                'internal_coupling': state.current_internal_coupling,
            },
            'stats': {
                'total_steps': self._step_count,
                'fatigue_episodes': self._fatigue_episodes,
                'uncertainty_episodes': self._uncertainty_episodes,
                'recovery_count': self._recovery_count,
                'avg_recovery_steps': (
                    round(np.mean(self._avg_recovery_steps), 1)
                    if self._avg_recovery_steps else 0.0
                ),
            },
        }

    def get_efficiency_recovery_metrics(self) -> Dict:
        """효율 회복 지표 (G2c 검증용)"""
        if self._fatigue_episodes == 0:
            return {
                'fatigue_to_recovery_ratio': 0.0,
                'avg_recovery_steps': 0.0,
                'efficiency_improvement': 0.0,
            }

        return {
            'fatigue_to_recovery_ratio': self._recovery_count / self._fatigue_episodes,
            'avg_recovery_steps': (
                np.mean(self._avg_recovery_steps) if self._avg_recovery_steps else 0.0
            ),
            'efficiency_improvement': 0.0,
        }

    def reset(self):
        """리셋"""
        self.state = GatingState()
        self._step_count = 0
        self._fatigue_episodes = 0
        self._uncertainty_episodes = 0
        self._recovery_count = 0
        self._avg_recovery_steps = []
