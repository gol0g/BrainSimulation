"""
Uncertainty/Confidence Tracking - v4.3

"대시보드"가 아니라 "자기조절 신호"로 만들기

핵심 목표:
에이전트가 불확실함을 '느끼고', 그 결과로:
- 더 관찰(THINK)하거나
- 더 탐색하거나
- 더 보수적으로 행동하거나
- 기억을 더 강하게 저장/통합하도록
내부 메커니즘이 자동으로 바뀌게 만드는 것.

4가지 불확실성 소스:
1) Belief Uncertainty: H(Q(c)) - "내가 지금 어떤 상황인지 모르겠다"
2) Action Uncertainty: H(π(a)) - "뭘 해야 할지 모르겠다"
3) Model Uncertainty: transition_std - "이 행동의 결과를 잘 모르겠다"
4) Surprise: prediction_error - "세상이 내 예측과 다르다"

행동 연결 (대시보드가 아닌 실제 조절):
A) THINK 선택 확률/비용을 불확실성에 연동
B) Precision을 불확실성에 의해 조절 (메타-정밀도)
C) 기억 저장 강도 게이트 (v4.0 Memory 준비)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class UncertaintyComponents:
    """4가지 불확실성 구성요소"""
    belief: float          # H(Q(c)) - context belief 엔트로피
    action: float          # H(π(a)) - action 분포 엔트로피
    model: float           # transition_std - 전이 불확실성
    surprise: float        # prediction_error - 예측 오차

    def to_dict(self) -> Dict:
        return {
            'belief': self.belief,
            'action': self.action,
            'model': self.model,
            'surprise': self.surprise,
        }


@dataclass
class UncertaintyState:
    """종합 불확실성 상태"""
    global_uncertainty: float  # 0~1
    global_confidence: float   # 0~1 = 1 - uncertainty
    components: UncertaintyComponents

    # 최근 변화 추적
    delta_uncertainty: float   # 불확실성 변화율
    top_factor: str            # 현재 불확실성의 주 원인

    def to_dict(self) -> Dict:
        return {
            'global_uncertainty': self.global_uncertainty,
            'global_confidence': self.global_confidence,
            'components': self.components.to_dict(),
            'delta_uncertainty': self.delta_uncertainty,
            'top_factor': self.top_factor,
        }


@dataclass
class UncertaintyModulation:
    """불확실성 기반 행동 조절 신호"""
    # A) THINK 조절
    think_bias: float          # THINK G에 더할 bias (음수=더 유리)

    # B) Precision 조절 (메타-정밀도)
    sensory_precision_mult: float   # 관측 precision 배율
    goal_precision_mult: float      # 목표 precision 배율

    # C) 탐색/회피 조절
    exploration_bonus: float        # ambiguity 감소 보너스
    risk_sensitivity: float         # risk 민감도 배율

    # D) 기억 저장 게이트 (v4.0 준비)
    memory_gate: float              # 0~1: 이 경험을 기억할 가치

    def to_dict(self) -> Dict:
        return {
            'think_bias': self.think_bias,
            'sensory_precision_mult': self.sensory_precision_mult,
            'goal_precision_mult': self.goal_precision_mult,
            'exploration_bonus': self.exploration_bonus,
            'risk_sensitivity': self.risk_sensitivity,
            'memory_gate': self.memory_gate,
        }


class UncertaintyTracker:
    """
    불확실성 추적 및 행동 조절 신호 생성.

    핵심 원리:
    - 4가지 소스에서 불확실성을 계산
    - 가중 평균으로 global uncertainty 계산
    - 불확실성에 따라 행동 조절 신호 생성

    이것은 "감정"이 아님. FEP 에이전트가 자기 상태를 모니터링하고
    그에 따라 메타-파라미터를 조절하는 것.
    """

    def __init__(self,
                 # 구성요소 가중치
                 belief_weight: float = 0.25,
                 action_weight: float = 0.30,
                 model_weight: float = 0.20,
                 surprise_weight: float = 0.25,

                 # EMA 스무딩
                 ema_alpha: float = 0.1,

                 # 정규화 파라미터
                 max_entropy: float = 1.6,      # log(5) ≈ 1.61
                 max_transition_std: float = 0.5,
                 max_surprise: float = 2.0,

                 # 조절 민감도
                 modulation_sensitivity: float = 1.0):

        # 가중치 (합 = 1.0)
        total_weight = belief_weight + action_weight + model_weight + surprise_weight
        self.belief_weight = belief_weight / total_weight
        self.action_weight = action_weight / total_weight
        self.model_weight = model_weight / total_weight
        self.surprise_weight = surprise_weight / total_weight

        # 스무딩
        self.ema_alpha = ema_alpha

        # 정규화
        self.max_entropy = max_entropy
        self.max_transition_std = max_transition_std
        self.max_surprise = max_surprise

        # 민감도
        self.modulation_sensitivity = modulation_sensitivity

        # 상태
        self._belief_ema = 0.5
        self._action_ema = 0.5
        self._model_ema = 0.3
        self._surprise_ema = 0.3

        self._global_uncertainty_ema = 0.5
        self._prev_uncertainty = 0.5

        # 히스토리 (분위수 계산용)
        self._surprise_history = []
        self._history_size = 100

        # 통계
        self._update_count = 0
        self._top_factor_history = []

    def update(self,
               decision_entropy: float = 0.0,
               context_entropy: Optional[float] = None,
               transition_std: float = 0.2,
               prediction_error: float = 0.0) -> UncertaintyState:
        """
        불확실성 업데이트.

        Args:
            decision_entropy: H(π(a)) - action 선택 엔트로피
            context_entropy: H(Q(c)) - context belief 엔트로피 (None이면 사용 안함)
            transition_std: 평균 transition std
            prediction_error: -log P(o|s) 또는 |o - o_pred|

        Returns:
            UncertaintyState: 현재 불확실성 상태
        """
        self._update_count += 1

        # === 1. 각 구성요소 정규화 (0~1) ===

        # Belief uncertainty: context entropy
        if context_entropy is not None:
            belief_norm = min(1.0, context_entropy / self.max_entropy)
        else:
            belief_norm = 0.5  # 기본값 (hierarchy 없을 때)

        # Action uncertainty: decision entropy
        action_norm = min(1.0, decision_entropy / self.max_entropy)

        # Model uncertainty: transition std
        model_norm = min(1.0, transition_std / self.max_transition_std)

        # Surprise: prediction error
        surprise_norm = min(1.0, prediction_error / self.max_surprise)

        # === 2. EMA 스무딩 ===
        self._belief_ema = (1 - self.ema_alpha) * self._belief_ema + self.ema_alpha * belief_norm
        self._action_ema = (1 - self.ema_alpha) * self._action_ema + self.ema_alpha * action_norm
        self._model_ema = (1 - self.ema_alpha) * self._model_ema + self.ema_alpha * model_norm
        self._surprise_ema = (1 - self.ema_alpha) * self._surprise_ema + self.ema_alpha * surprise_norm

        # === 3. 가중 평균으로 global uncertainty ===
        raw_uncertainty = (
            self.belief_weight * self._belief_ema +
            self.action_weight * self._action_ema +
            self.model_weight * self._model_ema +
            self.surprise_weight * self._surprise_ema
        )

        # Global EMA
        self._prev_uncertainty = self._global_uncertainty_ema
        self._global_uncertainty_ema = (
            (1 - self.ema_alpha) * self._global_uncertainty_ema +
            self.ema_alpha * raw_uncertainty
        )

        # === 4. 변화율 및 top factor ===
        delta_uncertainty = self._global_uncertainty_ema - self._prev_uncertainty

        # 어떤 구성요소가 가장 높은가?
        factors = {
            'belief': self._belief_ema,
            'action': self._action_ema,
            'model': self._model_ema,
            'surprise': self._surprise_ema,
        }
        top_factor = max(factors, key=factors.get)

        # History 업데이트
        self._surprise_history.append(surprise_norm)
        if len(self._surprise_history) > self._history_size:
            self._surprise_history.pop(0)

        self._top_factor_history.append(top_factor)
        if len(self._top_factor_history) > 20:
            self._top_factor_history.pop(0)

        # === 5. 상태 반환 ===
        components = UncertaintyComponents(
            belief=self._belief_ema,
            action=self._action_ema,
            model=self._model_ema,
            surprise=self._surprise_ema,
        )

        return UncertaintyState(
            global_uncertainty=self._global_uncertainty_ema,
            global_confidence=1.0 - self._global_uncertainty_ema,
            components=components,
            delta_uncertainty=delta_uncertainty,
            top_factor=top_factor,
        )

    def get_modulation(self) -> UncertaintyModulation:
        """
        현재 불확실성에 기반한 행동 조절 신호 계산.

        핵심: 불확실성이 "자동으로" 행동을 바꾸게 만듦.
        """
        u = self._global_uncertainty_ema  # 0~1
        s = self.modulation_sensitivity

        # === A) THINK bias ===
        # 불확실성 높음 → THINK가 유리해짐 (bias 음수)
        # 불확실성 낮음 → THINK 안 해도 됨 (bias 양수)
        # 선형 매핑: u=0 → bias=+0.2, u=1 → bias=-0.3
        think_bias = s * (0.2 - 0.5 * u)

        # === B) Precision 조절 (메타-정밀도) ===
        # 불확실성 높음 → precision 낮춤 (관측을 덜 믿음, 더 넓게 봄)
        # 불확실성 낮음 → precision 높임 (집중/확신)
        # 범위: 0.7 ~ 1.3
        sensory_precision_mult = 1.0 - 0.3 * s * u  # u=0: 1.0, u=1: 0.7
        goal_precision_mult = 1.0 + 0.3 * s * (1 - u)  # u=0: 1.3, u=1: 1.0

        # === C) 탐색/회피 조절 ===
        # 불확실성 높음 → 탐색 보너스, risk 민감도 감소
        # "모르니까 조심스럽게 탐색"
        # 단, 안전 하한선 유지 (v4.3.1 안전점검)
        exploration_bonus = s * 0.2 * u  # u=1일 때 최대 0.2
        # risk_sensitivity: 최소 0.6 (위험 근처에서 자살 방지)
        risk_sensitivity_raw = 1.0 - 0.3 * s * u  # u=1일 때 0.7
        risk_sensitivity = max(0.6, min(1.2, risk_sensitivity_raw))

        # === D) 기억 저장 게이트 ===
        # 불확실성 또는 surprise가 높으면 → 기억할 가치 있음
        # 안정적이면 → 굳이 저장 안 함
        #
        # 기준: 상대적 surprise (최근 히스토리 대비)
        if len(self._surprise_history) >= 10:
            recent_mean = np.mean(self._surprise_history[-10:])
            surprise_relative = self._surprise_ema - recent_mean
            memory_gate = np.clip(0.3 + surprise_relative + 0.5 * u, 0.0, 1.0)
        else:
            # 초기: 불확실성 높으면 저장
            memory_gate = u

        return UncertaintyModulation(
            think_bias=think_bias,
            sensory_precision_mult=sensory_precision_mult,
            goal_precision_mult=goal_precision_mult,
            exploration_bonus=exploration_bonus,
            risk_sensitivity=risk_sensitivity,
            memory_gate=memory_gate,
        )

    def get_state(self) -> UncertaintyState:
        """현재 상태 반환 (update 없이)"""
        # 가장 높은 factor 결정
        factors = {
            'belief': self._belief_ema,
            'action': self._action_ema,
            'model': self._model_ema,
            'surprise': self._surprise_ema,
        }
        top_factor = max(factors, key=factors.get)

        components = UncertaintyComponents(
            belief=self._belief_ema,
            action=self._action_ema,
            model=self._model_ema,
            surprise=self._surprise_ema,
        )

        return UncertaintyState(
            global_uncertainty=self._global_uncertainty_ema,
            global_confidence=1.0 - self._global_uncertainty_ema,
            components=components,
            delta_uncertainty=self._global_uncertainty_ema - self._prev_uncertainty,
            top_factor=top_factor,
        )

    def get_status(self) -> Dict:
        """API용 상태 반환"""
        state = self.get_state()
        modulation = self.get_modulation()

        # 최근 top factor 분포
        if self._top_factor_history:
            factor_counts = {}
            for f in self._top_factor_history:
                factor_counts[f] = factor_counts.get(f, 0) + 1
            total = len(self._top_factor_history)
            factor_dist = {k: v/total for k, v in factor_counts.items()}
        else:
            factor_dist = {}

        return {
            'state': state.to_dict(),
            'modulation': modulation.to_dict(),
            'update_count': self._update_count,
            'top_factor_distribution': factor_dist,
            'surprise_history_size': len(self._surprise_history),
            'weights': {
                'belief': self.belief_weight,
                'action': self.action_weight,
                'model': self.model_weight,
                'surprise': self.surprise_weight,
            },
        }

    def reset(self):
        """상태 리셋"""
        self._belief_ema = 0.5
        self._action_ema = 0.5
        self._model_ema = 0.3
        self._surprise_ema = 0.3
        self._global_uncertainty_ema = 0.5
        self._prev_uncertainty = 0.5
        self._surprise_history = []
        self._update_count = 0
        self._top_factor_history = []

    def set_weights(self,
                    belief_weight: float = None,
                    action_weight: float = None,
                    model_weight: float = None,
                    surprise_weight: float = None):
        """가중치 조정"""
        if belief_weight is not None:
            self.belief_weight = belief_weight
        if action_weight is not None:
            self.action_weight = action_weight
        if model_weight is not None:
            self.model_weight = model_weight
        if surprise_weight is not None:
            self.surprise_weight = surprise_weight

        # 정규화
        total = (self.belief_weight + self.action_weight +
                 self.model_weight + self.surprise_weight)
        if total > 0:
            self.belief_weight /= total
            self.action_weight /= total
            self.model_weight /= total
            self.surprise_weight /= total

    def set_sensitivity(self, sensitivity: float):
        """조절 민감도 설정 (0.0~2.0 권장)"""
        self.modulation_sensitivity = max(0.0, min(2.0, sensitivity))


# === Convenience functions ===

def compute_context_entropy(Q_context: np.ndarray) -> float:
    """Context belief 엔트로피 계산"""
    Q = np.clip(Q_context, 1e-10, 1.0)
    Q = Q / Q.sum()  # 정규화
    return -np.sum(Q * np.log(Q))


def compute_action_entropy(G_values: np.ndarray, temperature: float = 0.3) -> float:
    """Action 분포 엔트로피 계산 (softmax 기반)"""
    log_probs = -G_values / temperature
    log_probs = log_probs - np.max(log_probs)
    probs = np.exp(log_probs)
    probs = probs / (probs.sum() + 1e-10)
    return -np.sum(probs * np.log(probs + 1e-10))
