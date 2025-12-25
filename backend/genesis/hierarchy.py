"""
Hierarchical Models - Slow Layer for Context Tracking

v3.2: Context별 전이 모델 학습 (World Models)

핵심 원칙:
1. 내부 표현: c ∈ {0..K-1} (라벨 없음)
2. 사후 해석: UI/디버깅용 라벨만 (모델에 피드백 안 함)
3. FEP 일관성: Context도 Free Energy 최소화로 추론
4. Precision 조절만: 행동 직접 지시 안 함

v3.2 핵심:
- Context별 전이 모델: 각 context가 "행동 → 관측 변화" 예측 학습
- 전이 예측 정확도: context likelihood에 40% 반영
- 세상 모델 분화: 지배적 context가 더 정확한 세상 예측 획득

v3.1 개선 (유지):
- expected[k] 학습: 각 context가 자기 특징을 스스로 배움 (EMA)
- modulation 범위 축소: 0.85~1.15 (숨은 손 방지)
- 컨텍스트 전환 비용: transition_self로 관성 유지

Slow Layer가 하는 일:
- 긴 시간 스케일에서 "상황/컨텍스트" 추론
- Fast Layer의 precision/선호 샤프니스 조절
- **v3.2**: 각 context별로 다른 "세상 작동 방식" 학습

Slow Layer가 안 하는 일:
- 행동 직접 지정
- 감정 라벨 사용
- 외부에서 의미 주입
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class SlowLayerState:
    """Slow Layer의 상태"""

    # 핵심: Pure indices (K=4)
    Q_context: np.ndarray  # shape (K,) - context belief

    # 관측 통계 (EMA로 smoothing)
    obs_stats: Dict[str, float] = field(default_factory=dict)

    # 메타 정보
    step_count: int = 0
    last_update_step: int = 0
    dwell_time: int = 0
    dominant_context: int = 0

    # Slow layer F
    slow_F: float = 0.0


class SlowLayerInference:
    """
    Slow Layer 추론 엔진

    핵심 메커니즘:
    1. 관측 통계를 EMA로 smoothing (매 step)
    2. Context likelihood 계산 (설명력 기반)
    3. Slow Bayesian update (매 update_interval step)
    4. 강한 자기 유지 (transition_self ≈ 0.95)
    """

    def __init__(self,
                 K: int = 4,
                 update_interval: int = 10,
                 transition_self: float = 0.95,
                 ema_alpha: float = 0.1,
                 expectation_lr: float = 0.05):
        """
        Args:
            K: Context 수 (추천: 4)
            update_interval: Slow layer 업데이트 주기 (step)
            transition_self: P(c_t = c_{t-1}) - 자기 유지 확률
            ema_alpha: 관측 통계 EMA alpha
            expectation_lr: expected[k] 학습률 (v3.1)
        """
        self.K = K
        self.update_interval = update_interval
        self.ema_alpha = ema_alpha
        self.expectation_lr = expectation_lr

        # Context belief 초기화 (uniform)
        self.Q_context = np.ones(K) / K

        # 전이 모델: 강한 자기 유지
        # P(c_t | c_{t-1}) = transition_self if same, else (1-transition_self)/(K-1)
        off_diag = (1.0 - transition_self) / (K - 1)
        self.transition = np.ones((K, K)) * off_diag
        np.fill_diagonal(self.transition, transition_self)

        # Context별 "예상" 파라미터 (설명력 계산용)
        # [expected_pred_error, expected_ambiguity, expected_complexity,
        #  expected_energy, expected_danger_prox]
        # v3.1: 초기값은 골고루 분산, 이후 EMA로 학습됨
        self.context_expectations = np.array([
            [0.3, 0.3, 0.3, 0.4, 0.2],  # Context 0: 초기 추정
            [0.4, 0.3, 0.3, 0.5, 0.5],  # Context 1: 초기 추정
            [0.3, 0.4, 0.3, 0.6, 0.2],  # Context 2: 초기 추정
            [0.2, 0.2, 0.3, 0.7, 0.1],  # Context 3: 초기 추정
        ])

        # Context별 precision modulation 파라미터
        # v3.1: 범위 축소 (0.85~1.15) - 숨은 손 방지
        # [goal_mult, sensory[0..7], internal_weight, rollout_budget]
        self.context_params = np.array([
            [1.0, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 0.9, 0.3],  # Context 0
            [1.1, 1.0, 1.15, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4],  # Context 1
            [0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.5],  # Context 2
            [0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.25],  # Context 3
        ])

        # === v3.2: Context별 전이 모델 ===
        # 각 context k가 "세상이 어떻게 작동하는가"를 학습
        # context_transition_delta[k][action] = expected observation delta (8-dim)
        # 초기값: 모든 context가 동일한 prior (물리적 기본값)
        n_actions = 5
        n_obs = 8
        self.context_transition_delta = np.zeros((K, n_actions, n_obs))
        # 초기 prior: 이동 행동의 기본 효과 (약한 prior)
        # action 0=stay, 1=up, 2=down, 3=left, 4=right
        # 방향 행동이 위치 관측에 영향을 줄 것이라는 약한 기대
        for k in range(K):
            # 모든 context가 동일하게 시작
            self.context_transition_delta[k] = np.zeros((n_actions, n_obs))

        # 전이 예측 정확도 추적 (context별)
        self.context_transition_error = np.ones(K) * 0.5  # 초기 불확실

        # 최근 전이 정보 저장 (학습용)
        self.last_obs = None
        self.last_action = None
        self.transition_lr = 0.02  # 전이 모델 학습률

        # 관측 통계 (EMA)
        self.obs_stats = {
            'avg_pred_error': 0.3,
            'avg_ambiguity': 0.2,
            'avg_complexity': 0.3,
            'avg_energy': 0.7,
            'avg_danger_prox': 0.2,
            'avg_food_prox': 0.3,
            'avg_pain': 0.0,
        }

        # 상태 추적
        self.step_count = 0
        self.last_update_step = 0
        self.dwell_time = 0
        self.dominant_context = 0

    def _update_obs_stats(self,
                          pred_error: float,
                          ambiguity: float,
                          complexity: float,
                          observation: np.ndarray):
        """관측 통계 EMA 업데이트 (매 step)"""
        alpha = self.ema_alpha

        self.obs_stats['avg_pred_error'] = (
            (1 - alpha) * self.obs_stats['avg_pred_error'] + alpha * pred_error
        )
        self.obs_stats['avg_ambiguity'] = (
            (1 - alpha) * self.obs_stats['avg_ambiguity'] + alpha * ambiguity
        )
        self.obs_stats['avg_complexity'] = (
            (1 - alpha) * self.obs_stats['avg_complexity'] + alpha * complexity
        )

        if len(observation) >= 8:
            self.obs_stats['avg_food_prox'] = (
                (1 - alpha) * self.obs_stats['avg_food_prox'] + alpha * observation[0]
            )
            self.obs_stats['avg_danger_prox'] = (
                (1 - alpha) * self.obs_stats['avg_danger_prox'] + alpha * observation[1]
            )
            self.obs_stats['avg_energy'] = (
                (1 - alpha) * self.obs_stats['avg_energy'] + alpha * observation[6]
            )
            self.obs_stats['avg_pain'] = (
                (1 - alpha) * self.obs_stats['avg_pain'] + alpha * observation[7]
            )

    def _update_context_expectations(self):
        """
        v3.1: expected[k] 학습 (EMA)

        각 context k가 자기 특징을 스스로 배움:
        - 현재 context belief Q(c)로 가중 평균하여 업데이트
        - 지배적 context일수록 더 많이 학습

        핵심: context가 "숨은 원인 발견"이 되게 함
        (단순히 클러스터 라벨 맞추기가 아니라)
        """
        # 현재 관측 통계
        current = np.array([
            self.obs_stats['avg_pred_error'],
            self.obs_stats['avg_ambiguity'],
            self.obs_stats['avg_complexity'],
            self.obs_stats['avg_energy'],
            self.obs_stats['avg_danger_prox'],
        ])

        # 각 context k에 대해 Q(k)로 가중된 학습
        for k in range(self.K):
            # Q(k)가 높을수록 더 많이 학습
            effective_lr = self.expectation_lr * self.Q_context[k]

            # EMA 업데이트: expected[k] ← (1-lr) * expected[k] + lr * current
            self.context_expectations[k] = (
                (1 - effective_lr) * self.context_expectations[k] +
                effective_lr * current
            )

    def _update_context_transitions(self, obs_before: np.ndarray, action: int,
                                     obs_after: np.ndarray):
        """
        v3.2: Context별 전이 모델 학습

        각 context k가 "이 행동을 하면 관측이 어떻게 변하는가"를 배움
        - Q(k)로 가중된 학습: 지배적 context가 더 많이 배움
        - 결과: 각 context가 다른 "세상 모델"을 가지게 됨

        예:
        - Context A: "위험 지역에서는 움직이면 danger가 증가"
        - Context B: "안전 지역에서는 움직여도 danger가 안정적"
        """
        # 실제 전이 (delta)
        actual_delta = obs_after - obs_before
        # delta clipping for numerical stability
        actual_delta = np.clip(actual_delta, -1.0, 1.0)

        # 각 context k에 대해 전이 예측 오차 계산 및 학습
        for k in range(self.K):
            # 이 context의 예측
            predicted_delta = self.context_transition_delta[k, action]

            # 예측 오차
            error = actual_delta - predicted_delta
            error_magnitude = np.mean(np.abs(error))

            # 전이 예측 정확도 EMA 업데이트
            self.context_transition_error[k] = (
                0.9 * self.context_transition_error[k] + 0.1 * error_magnitude
            )

            # Q(k)로 가중된 학습
            effective_lr = self.transition_lr * self.Q_context[k]

            # 전이 모델 업데이트
            self.context_transition_delta[k, action] = (
                (1 - effective_lr) * self.context_transition_delta[k, action] +
                effective_lr * actual_delta
            )

    def _compute_context_likelihood(self) -> np.ndarray:
        """
        각 context k가 최근 관측/오차를 얼마나 잘 설명하는가?

        P(recent_observations | context=k)

        v3.2: 두 가지 설명력을 결합
        1. 관측 통계 설명력 (v3.1): 평균 관측 패턴이 예상과 맞는가
        2. 전이 예측 정확도 (v3.2): 이 context의 세상 모델이 전이를 잘 예측하는가
        """
        log_lik = np.zeros(self.K)

        # 현재 관측 통계
        current = np.array([
            self.obs_stats['avg_pred_error'],
            self.obs_stats['avg_ambiguity'],
            self.obs_stats['avg_complexity'],
            self.obs_stats['avg_energy'],
            self.obs_stats['avg_danger_prox'],
        ])

        for k in range(self.K):
            # 1. 관측 통계 설명력 (v3.1)
            expected = self.context_expectations[k]
            diff = current - expected
            obs_log_lik = -0.5 * np.sum(diff ** 2 / 0.1)  # precision = 10

            # 2. 전이 예측 정확도 (v3.2)
            # 전이 오차가 낮을수록 이 context가 세상을 잘 설명
            trans_error = self.context_transition_error[k]
            trans_log_lik = -5.0 * trans_error  # 오차에 비례해 likelihood 감소

            # 결합 (가중 평균)
            # 관측 통계 60%, 전이 예측 40%
            log_lik[k] = 0.6 * obs_log_lik + 0.4 * trans_log_lik

        return log_lik

    def _compute_slow_F(self) -> float:
        """
        Slow Layer의 Free Energy

        F_slow = -H[Q(context)] + E_Q[-log P(obs_stats | context)]

        Slow layer도 F를 최소화함 (FEP 일관성)
        """
        # Entropy term
        Q = np.clip(self.Q_context, 1e-10, 1.0)
        H = -np.sum(Q * np.log(Q))

        # Expected negative log likelihood
        log_lik = self._compute_context_likelihood()
        E_nll = -np.sum(Q * log_lik)

        F_slow = -H + E_nll
        return float(F_slow)

    def update(self,
               pred_error: float,
               ambiguity: float,
               complexity: float,
               observation: np.ndarray,
               action: Optional[int] = None) -> SlowLayerState:
        """
        Slow Layer 업데이트

        Args:
            pred_error: Fast layer의 prediction error
            ambiguity: Fast layer의 ambiguity
            complexity: Fast layer의 complexity
            observation: 현재 관측 (8차원)
            action: 직전에 수행한 행동 (v3.2: 전이 학습용)

        Returns:
            SlowLayerState: 업데이트된 상태
        """
        self.step_count += 1

        # v3.2: Context별 전이 모델 학습
        if self.last_obs is not None and action is not None:
            self._update_context_transitions(self.last_obs, action, observation)

        # 다음 스텝을 위해 저장
        self.last_obs = observation.copy()
        self.last_action = action

        # 1. 관측 통계 EMA 업데이트 (매 step)
        self._update_obs_stats(pred_error, ambiguity, complexity, observation)

        # 2. Context belief 업데이트 (매 update_interval step)
        if self.step_count % self.update_interval == 0:
            self.last_update_step = self.step_count

            # Compute likelihood
            log_lik = self._compute_context_likelihood()

            # Slow Bayes: Q(c_t) ∝ P(obs|c) * Σ_c' P(c_t|c') Q(c')
            prior = self.transition.T @ self.Q_context
            log_posterior = log_lik + np.log(prior + 1e-10)

            # Softmax normalization
            log_posterior -= np.max(log_posterior)  # numerical stability
            posterior = np.exp(log_posterior)
            self.Q_context = posterior / posterior.sum()

            # v3.1: expected[k] 학습 (EMA)
            # 각 context k가 자기 특징을 스스로 배움
            self._update_context_expectations()

        # 3. Dwell time 추적
        new_dominant = int(np.argmax(self.Q_context))
        if new_dominant == self.dominant_context:
            self.dwell_time += 1
        else:
            self.dominant_context = new_dominant
            self.dwell_time = 0

        # 4. Slow F 계산
        slow_F = self._compute_slow_F()

        return SlowLayerState(
            Q_context=self.Q_context.copy(),
            obs_stats=self.obs_stats.copy(),
            step_count=self.step_count,
            last_update_step=self.last_update_step,
            dwell_time=self.dwell_time,
            dominant_context=self.dominant_context,
            slow_F=slow_F,
        )


class PrecisionModulator:
    """
    Slow Layer → Fast Layer Precision 조절

    Context가 하층에 주는 영향 (행동 직접 지시 안 함):
    1. goal_precision 조절 (exploitation vs exploration)
    2. sensory_precision 조절 (주의 할당)
    3. internal_pref_weight 조절 (내부/외부 선호 비중)
    4. rollout_budget 조절 (thinking 확률)

    v3.1: 범위 축소 (숨은 손 방지)
    - goal_precision_mult: 0.85~1.15 (기존 0.3~2.0)
    - sensory_mod: 0.9~1.1 (기존 0.5~2.0)
    """

    def get_modulation(self,
                       Q_context: np.ndarray,
                       context_params: np.ndarray) -> Dict[str, any]:
        """
        현재 context belief에 따른 precision 조절값

        Args:
            Q_context: Context belief (K,)
            context_params: Context별 파라미터 (K, n_params)

        Returns:
            Dict with modulation values
        """
        # Weighted average by context belief
        effective_params = Q_context @ context_params

        # v3.1: 범위 축소 - 숨은 손 방지
        return {
            'goal_precision_mult': float(np.clip(effective_params[0], 0.85, 1.15)),
            'sensory_mod': np.clip(effective_params[1:9], 0.9, 1.1),
            'internal_pref_weight': float(np.clip(effective_params[9], 0.8, 1.0)),
            'rollout_budget': float(np.clip(effective_params[10], 0.0, 1.0)),
        }


class HierarchicalController:
    """
    계층적 제어기 - Slow Layer와 Fast Layer 통합

    사용법:
        controller = HierarchicalController(K=4)
        state = controller.update(pred_error, ambiguity, complexity, observation)
        modulation = controller.get_modulation()
    """

    def __init__(self,
                 K: int = 4,
                 update_interval: int = 10,
                 transition_self: float = 0.95,
                 ema_alpha: float = 0.1):
        """
        Args:
            K: Context 수
            update_interval: Slow layer 업데이트 주기
            transition_self: Context 자기 유지 확률
            ema_alpha: 관측 통계 EMA alpha
        """
        self.inference = SlowLayerInference(
            K=K,
            update_interval=update_interval,
            transition_self=transition_self,
            ema_alpha=ema_alpha,
        )
        self.modulator = PrecisionModulator()
        self.state: Optional[SlowLayerState] = None

    def update(self,
               pred_error: float,
               ambiguity: float,
               complexity: float,
               observation: np.ndarray,
               action: Optional[int] = None) -> SlowLayerState:
        """Slow Layer 업데이트 (v3.2: action 추가로 전이 학습)"""
        self.state = self.inference.update(
            pred_error, ambiguity, complexity, observation, action
        )
        return self.state

    def get_modulation(self) -> Dict[str, any]:
        """현재 context에 따른 precision modulation 값"""
        if self.state is None:
            # 기본값 반환
            return {
                'goal_precision_mult': 1.0,
                'sensory_mod': np.ones(8),
                'internal_pref_weight': 1.0,
                'rollout_budget': 0.3,
            }

        return self.modulator.get_modulation(
            self.state.Q_context,
            self.inference.context_params,
        )

    def get_state(self) -> Optional[SlowLayerState]:
        """현재 Slow Layer 상태"""
        return self.state

    def get_slow_F(self) -> float:
        """Slow Layer Free Energy"""
        if self.state is None:
            return 0.0
        return self.state.slow_F

    def get_context_belief(self) -> np.ndarray:
        """Context belief 분포"""
        if self.state is None:
            return np.ones(self.inference.K) / self.inference.K
        return self.state.Q_context.copy()

    def get_dominant_context(self) -> int:
        """지배적 context (argmax)"""
        if self.state is None:
            return 0
        return self.state.dominant_context

    def get_context_weighted_transition(self, action: int) -> np.ndarray:
        """
        v3.3: Context-weighted 전이 예측

        delta_ctx = Σ_k Q(k) * context_transition_delta[k][action]

        Args:
            action: 행동 인덱스 (0-4)

        Returns:
            8차원 관측 delta 예측 (context belief로 가중 평균)
        """
        Q = self.get_context_belief()
        # (K, 8) @ (K,) transposed = weighted sum
        delta_ctx = np.zeros(8)
        for k in range(self.inference.K):
            delta_ctx += Q[k] * self.inference.context_transition_delta[k, action]
        return delta_ctx


def infer_post_hoc_label(state: SlowLayerState) -> str:
    """
    사후 라벨링 - 관찰자 해석용

    중요: 이 라벨은 모델에 피드백하지 않음!
    순수하게 디버깅/시각화용

    Args:
        state: SlowLayerState

    Returns:
        str: "Context X (LABEL-like)" 형식
    """
    dominant = state.dominant_context
    stats = state.obs_stats

    # 최근 관측 통계 기반 해석
    if stats['avg_energy'] < 0.4:
        label = "FORAGING-like"
    elif stats['avg_danger_prox'] > 0.4 or stats['avg_pain'] > 0.2:
        label = "AVOIDANCE-like"
    elif stats['avg_ambiguity'] > 0.35:
        label = "EXPLORATION-like"
    else:
        label = "RESTING-like"

    return f"Context {dominant} ({label})"
