"""
Precision Learning - FEP의 핵심 메커니즘

Precision = 1/variance = "이 정보를 얼마나 신뢰할 것인가"

핵심 원리:
- 예측 오차가 작으면 → precision ↑ → 이 정보를 더 신뢰
- 예측 오차가 크면 → precision ↓ → 이 정보를 덜 신뢰
- 환경이 변동성이 크면 → precision ↓ → 과거 경험에 덜 의존

뇌에서의 역할:
- Sensory precision: 감각 정보 vs 사전 믿음
- Transition precision: 전이 모델 신뢰도
- Goal precision: 목표의 강도 (동기/감정과 연결)

이 구현에서:
- 각 관측 차원별 precision 학습
- 예측 오차 기반 자동 조정
- Risk/Ambiguity 계산에 precision 가중치 적용
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PrecisionState:
    """현재 precision 상태"""
    sensory_precision: np.ndarray  # 각 관측 차원별 (6,)
    transition_precision: np.ndarray  # 각 행동별 (5,)
    goal_precision: float  # 목표 추구 강도 (동기)

    # 메타 정보
    volatility: float  # 환경 변동성 (0-1)
    confidence: float  # 전체 신뢰도 (0-1)


class PrecisionLearner:
    """
    Precision Learning System

    FEP에서 Precision은 핵심:
    - 높은 precision = 이 정보에 주의 집중
    - 낮은 precision = 이 정보 무시

    학습 원리:
    - 예측 오차 작음 → precision 증가
    - 예측 오차 큼 → precision 감소
    - 변동성 높음 → 전체적으로 precision 감소
    """

    def __init__(self, n_obs: int = 6, n_actions: int = 5):
        self.n_obs = n_obs
        self.n_actions = n_actions

        # === Precision 파라미터 ===
        # 각 관측 차원별 sensory precision
        self.sensory_precision = np.ones(n_obs)

        # 각 행동별 transition precision
        self.transition_precision = np.ones(n_actions)

        # 목표 precision (동기 강도)
        self.goal_precision = 1.0

        # === 학습 파라미터 ===
        self.learning_rate = 0.1
        self.precision_decay = 0.01  # 시간에 따른 자연 감소
        self.min_precision = 0.1
        self.max_precision = 5.0  # 상한 낮춤 (발산 방지)

        # === EMA (Exponential Moving Average) for error smoothing ===
        self.ema_alpha = 0.2  # EMA 계수 (작을수록 더 smooth)
        self.ema_error = np.ones(n_obs) * 0.1  # 초기 EMA 오차

        # === 예측 오차 추적 ===
        self.prediction_errors: List[np.ndarray] = []
        self.max_history = 50

        # === 변동성 추적 ===
        self.volatility = 0.0
        self._prev_obs: Optional[np.ndarray] = None
        self._obs_changes: List[float] = []

    def update(self,
               predicted_obs: np.ndarray,
               actual_obs: np.ndarray,
               action: int) -> PrecisionState:
        """
        관측 후 precision 업데이트

        핵심 로직:
        1. 예측 오차 계산
        2. sensory precision 조정 (차원별)
        3. transition precision 조정 (행동별)
        4. 변동성 추적 및 goal precision 조정

        Args:
            predicted_obs: 예측했던 관측
            actual_obs: 실제 관측
            action: 수행한 행동
        """
        if len(predicted_obs) != self.n_obs or len(actual_obs) != self.n_obs:
            return self.get_state()

        # === 1. 예측 오차 계산 ===
        prediction_error = np.abs(actual_obs - predicted_obs)
        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > self.max_history:
            self.prediction_errors.pop(0)

        # === 2. Sensory Precision 업데이트 (EMA 사용) ===
        # EMA로 순간 오차 스파이크 완화
        for i in range(self.n_obs):
            # EMA 업데이트: ema = alpha * new + (1-alpha) * old
            self.ema_error[i] = (
                self.ema_alpha * prediction_error[i] +
                (1 - self.ema_alpha) * self.ema_error[i]
            )

            # Precision = 1 / (smoothed_error + eps)
            # EMA 오차 사용으로 안정적인 precision 계산
            expected_precision = 1.0 / (self.ema_error[i] + 0.1)
            delta = self.learning_rate * (expected_precision - self.sensory_precision[i])

            self.sensory_precision[i] += delta

        # 범위 제한 (발산 방지)
        self.sensory_precision = np.clip(
            self.sensory_precision,
            self.min_precision,
            self.max_precision
        )

        # === 3. Transition Precision 업데이트 ===
        # 이 행동의 예측이 정확했으면 precision ↑
        mean_error = np.mean(prediction_error[:2])  # proximity 차원만
        expected_trans_precision = 1.0 / (mean_error + 0.1)
        delta_trans = self.learning_rate * (
            expected_trans_precision - self.transition_precision[action]
        )
        self.transition_precision[action] += delta_trans
        self.transition_precision[action] = np.clip(
            self.transition_precision[action],
            self.min_precision,
            self.max_precision
        )

        # === 4. 변동성 추적 ===
        if self._prev_obs is not None:
            obs_change = np.mean(np.abs(actual_obs - self._prev_obs))
            self._obs_changes.append(obs_change)
            if len(self._obs_changes) > self.max_history:
                self._obs_changes.pop(0)

            # 변동성 = 최근 변화량의 분산
            if len(self._obs_changes) > 5:
                self.volatility = np.std(self._obs_changes)

        self._prev_obs = actual_obs.copy()

        # === 5. Goal Precision (= Preference Sharpness) 조정 ===
        # 이것은 "목표 추구 강도"가 아니라 "선호 분포 P(o)의 온도/샤프니스"
        # - 높으면 → P(o)가 뾰족 → 선호에 확신 → exploitation 모드
        # - 낮으면 → P(o)가 평평 → 선호에 불확신 → exploration 모드
        #
        # 변동성 기반 조정:
        # - 환경이 불안정하면 → 선호에 덜 확신 → 더 탐색적
        # - 환경이 안정적이면 → 선호에 더 확신 → 더 활용적
        target_goal_precision = 1.0 / (self.volatility + 0.5)
        self.goal_precision += 0.05 * (target_goal_precision - self.goal_precision)
        self.goal_precision = np.clip(self.goal_precision, 0.3, 2.0)  # 상한 낮춤

        return self.get_state()

    def get_state(self) -> PrecisionState:
        """현재 precision 상태 반환"""
        # 전체 신뢰도 = sensory precision의 평균
        confidence = np.mean(self.sensory_precision) / self.max_precision

        return PrecisionState(
            sensory_precision=self.sensory_precision.copy(),
            transition_precision=self.transition_precision.copy(),
            goal_precision=self.goal_precision,
            volatility=self.volatility,
            confidence=confidence
        )

    def get_risk_weights(self) -> np.ndarray:
        """
        Risk 계산에 사용할 가중치 반환

        높은 precision = 이 차원의 선호 위반에 더 민감
        """
        # Normalize to sum to n_obs
        weights = self.sensory_precision / np.mean(self.sensory_precision)
        return weights

    def get_ambiguity_weight(self, action: int) -> float:
        """
        Ambiguity 계산에 사용할 가중치 반환

        높은 transition precision = 이 행동의 불확실성에 더 민감
        """
        return self.transition_precision[action] / np.mean(self.transition_precision)

    def get_goal_weight(self) -> float:
        """
        Goal precision (= Preference Sharpness) 반환

        이것은 "목표 추구 강도"가 아니라 P(o) 선호 분포의 온도/샤프니스.
        - 높으면 → P(o)가 뾰족 → exploitation 모드
        - 낮으면 → P(o)가 평평 → exploration 모드
        """
        return self.goal_precision

    def reset(self):
        """상태 초기화"""
        self.sensory_precision = np.ones(self.n_obs)
        self.transition_precision = np.ones(self.n_actions)
        self.goal_precision = 1.0
        self.volatility = 0.0
        self.ema_error = np.ones(self.n_obs) * 0.1  # EMA 오차 초기화
        self.prediction_errors = []
        self._prev_obs = None
        self._obs_changes = []

    def get_attention_map(self) -> Dict[str, float]:
        """
        각 관측 차원에 대한 "주의" 수준 반환

        관찰자 해석: "에이전트가 어디에 주의를 기울이고 있는가"
        실제: sensory precision이 높은 차원
        """
        dim_names = [
            'food_proximity', 'danger_proximity',
            'food_dx', 'food_dy',
            'danger_dx', 'danger_dy'
        ]

        normalized = self.sensory_precision / np.sum(self.sensory_precision)

        return {name: float(normalized[i]) for i, name in enumerate(dim_names)}

    def interpret_state(self) -> Dict:
        """
        현재 precision 상태의 해석 (관찰자용)

        이것은 기계적 상태를 인간이 이해할 수 있는 용어로 매핑
        """
        state = self.get_state()

        # 주의 분배
        attention = self.get_attention_map()
        max_attention = max(attention, key=attention.get)

        # 전체 상태 해석
        if state.volatility > 0.3:
            volatility_interpretation = "환경 불안정 - 유연한 대응 모드"
        elif state.volatility < 0.1:
            volatility_interpretation = "환경 안정 - 학습된 패턴 신뢰"
        else:
            volatility_interpretation = "보통 변동성"

        if state.goal_precision > 1.5:
            goal_interpretation = "목표 강하게 추구 (높은 동기)"
        elif state.goal_precision < 0.5:
            goal_interpretation = "탐색적/개방적 상태"
        else:
            goal_interpretation = "균형적 목표 추구"

        return {
            'primary_attention': max_attention,
            'attention_map': attention,
            'volatility': volatility_interpretation,
            'goal_state': goal_interpretation,
            'overall_confidence': state.confidence
        }
