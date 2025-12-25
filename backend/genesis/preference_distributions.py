"""
Preference Distributions - P(o) as Proper Probability Distributions

핵심 개념:
    기존: preferred_obs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] (목표 벡터)
    변경: P(o) = Product of distributions for each observation dimension

    이제 Risk = KL[Q(o|a) || P(o)]가 "진짜" KL divergence가 됨.

각 관측 차원별 분포 (v2.5: 8차원):
    === EXTEROCEPTION (외부 세계) ===
    - food_proximity [0]: Beta(5, 1) - 1에 가까울수록 선호 (음식 위에 있고 싶음)
    - danger_proximity [1]: Beta(1, 5) - 0에 가까울수록 선호 (위험에서 멀리)
    - food_dx [2]: Categorical(uniform) - 방향 선호 없음
    - food_dy [3]: Categorical(uniform) - 방향 선호 없음
    - danger_dx [4]: Categorical(uniform) - 방향 선호 없음
    - danger_dy [5]: Categorical(uniform) - 방향 선호 없음

    === INTEROCEPTION (내부 상태) - v2.5 ===
    - energy [6]: Beta(3, 2) - ~0.6 선호 (항상성: 적당한 에너지)
    - pain [7]: Beta(1, 5) - 0 선호 (통증 없음)

    Phase 0: 외부 선호 유지, 내부 선호는 약하게 시작
    Phase 1: lambda로 외부/내부 혼합
    Phase 2: 내부 선호만 (외부는 학습으로 발견)

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
    P(o) = P(o_0) × P(o_1) × ... × P(o_7)

    Risk = KL[Q(o|a) || P(o)] 계산 가능.

    v2.5: 8차원 (6 exteroception + 2 interoception)
    """

    def __init__(self, internal_pref_weight: float = 1.0):
        """
        Args:
            internal_pref_weight: Phase 0-2 전환용 lambda (0.0 = 외부만, 1.0 = 내부만)
                                 v2.5: 기본값 1.0 (내부 선호만) - 테스트에서 더 좋은 성능
        """
        # === EXTEROCEPTION PREFERENCES (외부 세계) ===

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

        # === INTEROCEPTION PREFERENCES (내부 상태) - v2.5 ===

        # energy [6]: Homeostatic setpoint around 0.6
        # Beta(3, 2): mode at 0.67, mean = 0.6
        # 이것이 "진짜" 선호 - 에너지가 적당해야 함
        self.energy_pref = BetaParams(alpha=3.0, beta=2.0)

        # pain [7]: Want NO pain (value near 0)
        # Beta(1, 5): mode at 0, mean = 0.17
        # 이것이 "진짜" 선호 - 통증이 없어야 함
        self.pain_pref = BetaParams(alpha=1.0, beta=5.0)

        # === PHASE CONTROL (v2.5) ===
        # lambda = 0.0: 외부 선호만 (Phase 0)
        # lambda = 0.5: 혼합 (Phase 1)
        # lambda = 1.0: 내부 선호만 (Phase 2)
        self.internal_pref_weight = internal_pref_weight

        # === LEGACY ===
        self.static_obs_penalty = 0.0

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
                     q_uncertainty: Optional[np.ndarray] = None,
                     precision_weights: Optional[np.ndarray] = None) -> float:
        """
        Risk = -log P(predicted_obs) with PRECISION weighting

        v2.5: 8차원 지원 (exteroception + interoception)
        - internal_pref_weight로 외부/내부 선호 혼합 조절

        더 간단하고 직관적인 접근:
        - 예측된 관측값이 선호에 맞으면 → 낮은 risk
        - 예측된 관측값이 선호에서 벗어나면 → 높은 risk

        Precision 가중치:
        - precision_weights[i]가 높으면 → 이 차원의 선호 위반에 더 민감
        - precision은 예측 오차가 작을수록 높아짐 (신뢰할 수 있는 정보)

        Returns:
            Risk value (negative log probability under preference)
        """
        if len(q_obs) < 6:
            return 0.0

        # Default precision weights (no weighting) - now 8 dimensions
        if precision_weights is None:
            precision_weights = np.ones(8)
        elif len(precision_weights) < 8:
            # Extend to 8 dimensions if needed
            precision_weights = np.concatenate([precision_weights, np.ones(8 - len(precision_weights))])

        external_risk = 0.0
        internal_risk = 0.0

        # === EXTEROCEPTION RISK (외부 세계) ===
        # 외부 선호 가중치: (1 - internal_pref_weight)
        ext_weight = 1.0 - self.internal_pref_weight

        # Food proximity risk
        # P(o) = Beta(5, 1) prefers high values (near 1.0)
        food_prox_pred = np.clip(q_obs[0], 0.01, 0.99)
        log_p_food = self.food_prox_pref.log_pdf(food_prox_pred)
        max_log_p_food = self.food_prox_pref.log_pdf(0.99)
        risk_food = max(0.0, -(log_p_food - max_log_p_food))
        external_risk += risk_food * precision_weights[0]

        # Danger proximity risk
        # P(o) = Beta(1, 5) prefers low values (near 0.0)
        danger_prox_pred = np.clip(q_obs[1], 0.01, 0.99)
        log_p_danger = self.danger_prox_pref.log_pdf(danger_prox_pred)
        max_log_p_danger = self.danger_prox_pref.log_pdf(0.01)
        risk_danger = max(0.0, -(log_p_danger - max_log_p_danger))
        external_risk += risk_danger * precision_weights[1]

        # Direction risks (categorical) = 0 (uniform preference)

        # === INTEROCEPTION RISK (내부 상태) - v2.5 ===
        # 내부 선호 가중치: internal_pref_weight
        int_weight = self.internal_pref_weight

        if len(q_obs) >= 8:
            # Energy risk
            # P(o) = Beta(3, 2) prefers ~0.6
            energy_pred = np.clip(q_obs[6], 0.01, 0.99)
            log_p_energy = self.energy_pref.log_pdf(energy_pred)
            max_log_p_energy = self.energy_pref.log_pdf(0.6)  # Near mode
            risk_energy = max(0.0, -(log_p_energy - max_log_p_energy))
            internal_risk += risk_energy * precision_weights[6]

            # Pain risk
            # P(o) = Beta(1, 5) prefers 0
            pain_pred = np.clip(q_obs[7], 0.01, 0.99)
            log_p_pain = self.pain_pref.log_pdf(pain_pred)
            max_log_p_pain = self.pain_pref.log_pdf(0.01)
            risk_pain = max(0.0, -(log_p_pain - max_log_p_pain))
            internal_risk += risk_pain * precision_weights[7]

        # === TOTAL RISK ===
        # Phase 0: ext_weight=1.0, int_weight=0.0 → 외부만
        # Phase 1: 혼합
        # Phase 2: ext_weight=0.0, int_weight=1.0 → 내부만
        total_risk = ext_weight * external_risk + int_weight * internal_risk

        return total_risk

    def set_internal_weight(self, weight: float):
        """Phase 전환을 위한 internal_pref_weight 설정"""
        self.internal_pref_weight = max(0.0, min(1.0, weight))

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


class StatePreferenceDistribution:
    """
    P(s) - Prior preference over hidden states.

    Complexity = KL[Q(s) || P(s)]
    = "믿음이 선호 상태 분포에서 얼마나 벗어났나"

    핵심 개념:
    - P(s)는 "에이전트가 어떤 상태에 있고 싶은가"
    - 상태 s는 관측 o와 연결됨 (상태가 좋으면 좋은 관측)
    - P(s)가 높은 상태: 안전하고, 음식에 가깝고, 예측 가능한 상태

    Complexity의 역할:
    - "믿음을 너무 급격하게 바꾸지 마라" (인지적 관성)
    - "선호 상태에서 벗어난 믿음을 가지면 비용이 든다"

    Expected Complexity for action selection:
    - E[KL[Q(s'|a) || P(s')]]
    - "이 행동을 하면 믿음이 선호에서 얼마나 벗어날 것인가"
    """

    def __init__(self, n_states: int = 16):
        """
        Initialize state preference distribution.

        Args:
            n_states: Number of hidden states
        """
        self.n_states = n_states

        # P(s) - 상태 선호 분포
        # 초기에는 균등하지만, 관측 선호에서 유도됨
        self.P_s = np.ones(n_states) / n_states

        # 상태-관측 매핑을 통해 P(s) 구성
        # 상태 i → 예상 관측 o_i → P(o_i)에서 P(s) 유도
        self._build_from_observation_preferences()

    def _build_from_observation_preferences(self):
        """
        관측 선호 P(o)에서 상태 선호 P(s) 유도.

        아이디어:
        - 상태 s가 "좋은" 관측 o를 생성할 확률이 높으면 → P(s) 높음
        - P(s) ∝ E_{P(o|s)}[P(o)]

        단순화:
        - n_states를 격자로 해석
        - 각 상태를 (food_prox_level, danger_prox_level)로 매핑
        """
        # 4x4 격자로 해석 (16 states)
        # state i → (food_level, danger_level)
        # food_level: 0-3 (0=far, 3=on food)
        # danger_level: 0-3 (0=safe, 3=dangerous)

        n_grid = int(np.sqrt(self.n_states))
        if n_grid * n_grid != self.n_states:
            n_grid = 4  # Default to 4x4

        obs_prefs = PreferenceDistributions()

        for s in range(self.n_states):
            food_level = s % n_grid
            danger_level = s // n_grid

            # 관측으로 매핑 (0-3 → 0.0-1.0)
            food_prox = food_level / (n_grid - 1)
            danger_prox = danger_level / (n_grid - 1)

            # P(o) 하에서의 확률
            log_p_food = obs_prefs.food_prox_pref.log_pdf(np.clip(food_prox, 0.01, 0.99))
            log_p_danger = obs_prefs.danger_prox_pref.log_pdf(np.clip(danger_prox, 0.01, 0.99))

            # 결합 확률 (독립 가정)
            log_p_s = log_p_food + log_p_danger
            self.P_s[s] = np.exp(log_p_s)

        # 정규화
        self.P_s = self.P_s / self.P_s.sum()

    def compute_complexity(self, Q_s: np.ndarray) -> float:
        """
        Complexity = KL[Q(s) || P(s)]

        현재 믿음이 선호 상태 분포에서 얼마나 벗어났나.

        Args:
            Q_s: Current belief over states

        Returns:
            Complexity value (KL divergence)
        """
        Q_s = np.clip(Q_s, 1e-10, 1.0)
        Q_s = Q_s / Q_s.sum()

        P_s = np.clip(self.P_s, 1e-10, 1.0)

        # KL[Q || P] = sum Q(s) * log(Q(s) / P(s))
        kl = np.sum(Q_s * (np.log(Q_s) - np.log(P_s)))

        return max(0.0, kl)

    def compute_expected_complexity(self,
                                     Q_s_current: np.ndarray,
                                     transition_matrix: np.ndarray) -> float:
        """
        Expected Complexity = KL[Q(s'|a) || P(s')]

        행동 후 예측 믿음이 선호에서 얼마나 벗어날 것인가.

        Args:
            Q_s_current: Current belief Q(s)
            transition_matrix: P(s'|s, a) for specific action, shape (n_states, n_states)

        Returns:
            Expected complexity after action
        """
        # Q(s'|a) = sum_s P(s'|s,a) Q(s)
        Q_s_next = transition_matrix.T @ Q_s_current
        Q_s_next = np.clip(Q_s_next, 1e-10, 1.0)
        Q_s_next = Q_s_next / Q_s_next.sum()

        # Complexity of predicted belief
        return self.compute_complexity(Q_s_next)

    def compute_expected_complexity_from_obs(self,
                                              current_obs: np.ndarray,
                                              predicted_obs: np.ndarray) -> float:
        """
        관측 공간에서 직접 Expected Complexity 계산.

        상태 공간을 거치지 않고 관측 예측에서 직접 계산.
        현재 시스템이 관측 기반이므로 더 실용적.

        Args:
            current_obs: Current observation
            predicted_obs: Predicted observation after action

        Returns:
            Expected complexity (proxy)

        핵심 아이디어:
        - 예측 관측이 "선호 상태"에서 멀어지면 → 높은 complexity
        - "선호 상태" = food_prox 높고, danger_prox 낮은 상태
        """
        if len(predicted_obs) < 2:
            return 0.0

        obs_prefs = PreferenceDistributions()

        # 예측 관측이 선호에서 얼마나 벗어나는가?
        food_prox_pred = np.clip(predicted_obs[0], 0.01, 0.99)
        danger_prox_pred = np.clip(predicted_obs[1], 0.01, 0.99)

        # P(s) = P(food_prox) * P(danger_prox)
        log_p_food = obs_prefs.food_prox_pref.log_pdf(food_prox_pred)
        log_p_danger = obs_prefs.danger_prox_pref.log_pdf(danger_prox_pred)

        # 선호 상태의 최대 log prob (food=1, danger=0)
        max_log_p = (obs_prefs.food_prox_pref.log_pdf(0.99) +
                     obs_prefs.danger_prox_pref.log_pdf(0.01))

        # Complexity = -(log_p - max_log_p) = 선호에서 벗어난 정도
        log_p_pred = log_p_food + log_p_danger
        complexity = -(log_p_pred - max_log_p)

        # 스케일 조정: Risk와 비슷한 스케일로
        complexity = complexity * 0.3

        return max(0.0, complexity)
