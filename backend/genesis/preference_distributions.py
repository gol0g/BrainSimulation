"""
Preference Distributions - P(o) as Proper Probability Distributions

핵심 개념:
    기존: preferred_obs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] (목표 벡터)
    변경: P(o) = Product of distributions for each observation dimension

    이제 Risk = KL[Q(o|a) || P(o)]가 "진짜" KL divergence가 됨.

각 관측 차원별 분포:
    - food_proximity [0]: Beta(5, 1) - 1에 가까울수록 선호 (음식 위에 있고 싶음)
    - danger_proximity [1]: Beta(1, 5) - 0에 가까울수록 선호 (위험에서 멀리)
    - food_dx [2]: Categorical(uniform) - 방향 선호 없음
    - food_dy [3]: Categorical(uniform) - 방향 선호 없음
    - danger_dx [4]: Categorical(uniform) - 방향 선호 없음
    - danger_dy [5]: Categorical(uniform) - 방향 선호 없음

Beta 분포 특성:
    - Beta(α, β): α > β → 1 쪽으로 치우침
    - Beta(α, β): α < β → 0 쪽으로 치우침
    - Beta(1, 1) = Uniform(0, 1)

왜 Beta인가?
    - proximity는 [0, 1] bounded continuous
    - Beta는 [0, 1]에서 정의된 자연스러운 분포
    - KL divergence 계산 가능
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BetaParams:
    """Beta distribution parameters."""
    alpha: float
    beta: float

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def pdf(self, x: float) -> float:
        """Probability density at x."""
        x = np.clip(x, 1e-10, 1 - 1e-10)
        return stats.beta.pdf(x, self.alpha, self.beta)

    def log_pdf(self, x: float) -> float:
        """Log probability density at x."""
        x = np.clip(x, 1e-10, 1 - 1e-10)
        return stats.beta.logpdf(x, self.alpha, self.beta)

    def entropy(self) -> float:
        """Differential entropy of the distribution."""
        return stats.beta.entropy(self.alpha, self.beta)


@dataclass
class CategoricalParams:
    """Categorical distribution parameters (for discrete values)."""
    probs: np.ndarray  # Probability for each category

    def pmf(self, x: int) -> float:
        """Probability mass at x."""
        if 0 <= x < len(self.probs):
            return self.probs[x]
        return 0.0

    def log_pmf(self, x: int) -> float:
        """Log probability mass at x."""
        p = self.pmf(x)
        return np.log(p + 1e-10)

    def entropy(self) -> float:
        """Entropy of the distribution."""
        p = np.clip(self.probs, 1e-10, 1.0)
        return -np.sum(p * np.log(p))


class PreferenceDistributions:
    """
    P(o) - Prior preference over observations as probability distributions.

    각 관측 차원이 독립이라고 가정:
    P(o) = P(o_0) × P(o_1) × ... × P(o_5)

    Risk = KL[Q(o|a) || P(o)] 계산 가능.
    """

    def __init__(self):
        # === PROXIMITY PREFERENCES (Beta distributions) ===

        # food_proximity [0]: Want to be ON food (value near 1)
        # Beta(5, 1): mode at 1, mean = 5/6 ≈ 0.83
        self.food_prox_pref = BetaParams(alpha=5.0, beta=1.0)

        # danger_proximity [1]: Want to be FAR from danger (value near 0)
        # Beta(1, 5): mode at 0, mean = 1/6 ≈ 0.17
        self.danger_prox_pref = BetaParams(alpha=1.0, beta=5.0)

        # === DIRECTION PREFERENCES (Categorical, uniform) ===
        # Values: -1, 0, +1 mapped to indices 0, 1, 2
        uniform_3 = np.array([1/3, 1/3, 1/3])

        self.food_dx_pref = CategoricalParams(probs=uniform_3.copy())
        self.food_dy_pref = CategoricalParams(probs=uniform_3.copy())
        self.danger_dx_pref = CategoricalParams(probs=uniform_3.copy())
        self.danger_dy_pref = CategoricalParams(probs=uniform_3.copy())

        # === MOVEMENT PREFERENCE (absorbed STAY penalty) ===
        # "정지 상태 관측"에 대한 비선호 → P(o)에 흡수
        # 만약 이전 관측과 같은 관측이면 → 약간의 비선호
        # 이건 나중에 구현 (우선은 proximity/direction만)
        self.static_obs_penalty = 0.0  # TODO: implement later

    def log_prob_food_prox(self, value: float) -> float:
        """log P(food_proximity = value)"""
        return self.food_prox_pref.log_pdf(value)

    def log_prob_danger_prox(self, value: float) -> float:
        """log P(danger_proximity = value)"""
        return self.danger_prox_pref.log_pdf(value)

    def log_prob_direction(self, value: float, dim: int) -> float:
        """
        log P(direction = value) for dimension dim (2-5).
        value: -1, 0, or +1
        """
        # Map value to index: -1 → 0, 0 → 1, +1 → 2
        idx = int(value + 1)
        idx = np.clip(idx, 0, 2)

        if dim == 2:
            return self.food_dx_pref.log_pmf(idx)
        elif dim == 3:
            return self.food_dy_pref.log_pmf(idx)
        elif dim == 4:
            return self.danger_dx_pref.log_pmf(idx)
        elif dim == 5:
            return self.danger_dy_pref.log_pmf(idx)
        return 0.0

    def log_prob_observation(self, obs: np.ndarray) -> float:
        """
        log P(o) for full observation vector.

        Assumes independence:
        log P(o) = sum_i log P(o_i)
        """
        if len(obs) < 6:
            return 0.0

        log_p = 0.0
        log_p += self.log_prob_food_prox(obs[0])
        log_p += self.log_prob_danger_prox(obs[1])
        log_p += self.log_prob_direction(obs[2], 2)
        log_p += self.log_prob_direction(obs[3], 3)
        log_p += self.log_prob_direction(obs[4], 4)
        log_p += self.log_prob_direction(obs[5], 5)

        return log_p

    def kl_divergence_beta(self,
                           q_alpha: float, q_beta: float,
                           p_alpha: float, p_beta: float) -> float:
        """
        KL[Q || P] where Q = Beta(q_alpha, q_beta), P = Beta(p_alpha, p_beta).

        KL divergence between two Beta distributions.
        """
        from scipy.special import betaln, digamma

        # KL[Beta(a1,b1) || Beta(a2,b2)] =
        #   ln(B(a2,b2)/B(a1,b1)) + (a1-a2)ψ(a1) + (b1-b2)ψ(b1) + (a2-a1+b2-b1)ψ(a1+b1)

        a1, b1 = q_alpha, q_beta
        a2, b2 = p_alpha, p_beta

        term1 = betaln(a2, b2) - betaln(a1, b1)
        term2 = (a1 - a2) * digamma(a1)
        term3 = (b1 - b2) * digamma(b1)
        term4 = (a2 - a1 + b2 - b1) * digamma(a1 + b1)

        kl = term1 + term2 + term3 + term4
        return max(0.0, kl)

    def kl_divergence_categorical(self, q_probs: np.ndarray, p_probs: np.ndarray) -> float:
        """
        KL[Q || P] where Q and P are categorical distributions.
        """
        q = np.clip(q_probs, 1e-10, 1.0)
        p = np.clip(p_probs, 1e-10, 1.0)
        q = q / q.sum()
        p = p / p.sum()

        kl = np.sum(q * (np.log(q) - np.log(p)))
        return max(0.0, kl)

    def compute_risk(self,
                     q_obs: np.ndarray,
                     q_uncertainty: Optional[np.ndarray] = None) -> float:
        """
        Risk = -log P(predicted_obs) (simplified from KL divergence)

        더 간단하고 직관적인 접근:
        - 예측된 관측값이 선호에 맞으면 → 낮은 risk
        - 예측된 관측값이 선호에서 벗어나면 → 높은 risk

        -log P(o)를 사용하면:
        - food_prox = 1.0 (음식 위) → P(1.0) 높음 → -log 낮음 → risk 낮음 ✓
        - food_prox = 0.5 (멀리) → P(0.5) 낮음 → -log 높음 → risk 높음 ✓

        Returns:
            Risk value (negative log probability under preference)
        """
        if len(q_obs) < 6:
            return 0.0

        total_risk = 0.0

        # === Food proximity risk ===
        # P(o) = Beta(5, 1) prefers high values (near 1.0)
        # Risk = -log P(food_prox_pred)
        food_prox_pred = np.clip(q_obs[0], 0.01, 0.99)
        log_p_food = self.food_prox_pref.log_pdf(food_prox_pred)

        # Normalize to reasonable scale (Beta(5,1) has log_pdf range roughly -5 to +1.5)
        # Shift so that preferred value (food_prox=1.0) has risk ≈ 0
        max_log_p_food = self.food_prox_pref.log_pdf(0.99)  # Near-max probability
        risk_food = -(log_p_food - max_log_p_food)  # 0 at preferred, positive elsewhere
        risk_food = max(0.0, risk_food)  # Ensure non-negative

        total_risk += risk_food

        # === Danger proximity risk ===
        # P(o) = Beta(1, 5) prefers low values (near 0.0)
        danger_prox_pred = np.clip(q_obs[1], 0.01, 0.99)
        log_p_danger = self.danger_prox_pref.log_pdf(danger_prox_pred)

        max_log_p_danger = self.danger_prox_pref.log_pdf(0.01)  # Near-max probability
        risk_danger = -(log_p_danger - max_log_p_danger)
        risk_danger = max(0.0, risk_danger)

        total_risk += risk_danger

        # === Direction risks (categorical) ===
        # Uniform preference = no contribution to risk
        # Direction is informational, not preferential

        return total_risk

    def compute_ambiguity(self, transition_uncertainty: float = 0.1) -> float:
        """
        Ambiguity = E_{Q(s'|a)}[H[P(o|s')]]

        FEP 정의 그대로:
        - "행동 a를 하면 도달하는 상태 s'에서, 관측이 얼마나 모호한가?"
        - 관측 모델 P(o|s)가 deterministic이면 H[P(o|s')] ≈ 0
        - 따라서 ambiguity는 순수하게 **전이 모델의 불확실성**에서 옴

        Args:
            transition_uncertainty: 전이 모델의 표준편차 (delta_std)
                - 이 값은 경험(학습)을 통해 자연스럽게 감소함
                - 휴리스틱이 아님 - 모델 파라미터 자체가 경험을 반영

        Returns:
            Ambiguity value

        핵심 원리:
        - "경험 → confidence → ambiguity 감소" (❌ 휴리스틱)
        - "경험 → 모델 학습 → delta_std 감소 → ambiguity 감소" (✅ FEP)
        - transition_std는 update_transition_model()에서 학습됨
        """
        # Ambiguity = 전이 불확실성의 함수
        # transition_uncertainty가 높으면 → 어디로 갈지 모름 → 높은 ambiguity
        # transition_uncertainty가 낮으면 → 예측 가능 → 낮은 ambiguity
        #
        # 스케일링: transition_std는 보통 0.01~0.5 범위
        # ambiguity가 risk와 비슷한 스케일이 되도록 조정
        ambiguity = transition_uncertainty * 1.5

        return max(0.01, ambiguity)

    def get_preference_summary(self) -> Dict:
        """Get human-readable summary of preferences."""
        return {
            'food_proximity': {
                'distribution': 'Beta',
                'params': f'α={self.food_prox_pref.alpha}, β={self.food_prox_pref.beta}',
                'mean': self.food_prox_pref.mean(),
                'interpretation': 'Prefers food_proximity near 1 (on food)'
            },
            'danger_proximity': {
                'distribution': 'Beta',
                'params': f'α={self.danger_prox_pref.alpha}, β={self.danger_prox_pref.beta}',
                'mean': self.danger_prox_pref.mean(),
                'interpretation': 'Prefers danger_proximity near 0 (far from danger)'
            },
            'directions': {
                'distribution': 'Categorical (uniform)',
                'interpretation': 'No directional preference'
            }
        }
