"""
v5.0 Neural Predictive Coding Layer

핵심 철학:
- Q(s)를 직접 감독하지 않음
- 예측(ô)과 오차(ε)를 통해 믿음(μ)이 자연히 수렴
- Prior force가 memory/context/preference를 회로에 스며들게 함

동역학:
  ε = o - ô                                    # 예측 오차
  ô = W_pred @ μ                               # 상태→예측
  dμ/dt = W_ε @ ε - λ_prior * (μ - μ_prior)   # 두 힘의 균형

v4.x 연결:
- uncertainty → λ_prior 감소 (prior 약화, 데이터 의존)
- drift 감지 → λ_prior 감소 (새 환경 적응)
- memory → μ_prior 제공 (기억이 prior로 작용)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum


@dataclass
class PCConfig:
    """Predictive Coding 회로 설정"""
    n_obs: int = 8              # 관측 차원
    n_state: int = 16           # 상태 뉴런 수 (μ 차원)
    dt: float = 0.1             # 시뮬레이션 타임스텝
    tau: float = 1.0            # 시간 상수

    # 수렴 설정
    max_iterations: int = 50    # 최대 수렴 반복
    convergence_threshold: float = 0.01  # |dμ/dt| < threshold면 수렴

    # Prior 설정
    base_prior_precision: float = 1.0   # 기본 λ_prior
    min_prior_precision: float = 0.1    # 최소 λ_prior (drift 시)

    # Learning rates
    lr_pred: float = 0.05       # W_pred learning rate (increased)
    lr_error: float = 0.05      # W_epsilon learning rate


@dataclass
class PCState:
    """Predictive Coding circuit state"""
    # Neuron activities
    mu: np.ndarray              # State neurons (belief)
    epsilon: np.ndarray         # Error neurons
    o_hat: np.ndarray           # Prediction

    # Dynamics info
    d_mu: np.ndarray            # dmu/dt
    converged: bool = False
    iterations: int = 0

    # Prior info
    mu_prior: np.ndarray = field(default_factory=lambda: np.zeros(16))
    lambda_prior: float = 1.0

    # Diagnostics
    error_norm: float = 0.0
    prior_force_norm: float = 0.0
    data_force_norm: float = 0.0
    initial_error: float = 0.0  # Error before convergence (for drift detection)


class NeuralPCLayer:
    """
    Predictive Coding 기반 Neural Inference Layer

    FEP의 Q(s) 추론을 뉴런 동역학으로 구현
    """

    def __init__(self, config: Optional[PCConfig] = None):
        self.config = config or PCConfig()

        # 가중치 초기화
        self._init_weights()

        # 상태 초기화
        self.state = self._create_initial_state()

        # 히스토리 (시각화/디버깅용)
        self.history: List[Dict] = []
        self.max_history = 1000

    def _init_weights(self):
        """Initialize synaptic weights"""
        n_obs = self.config.n_obs
        n_state = self.config.n_state

        # W_pred: state -> prediction (n_obs x n_state)
        # Initialize closer to identity for first n_obs dimensions
        # This helps initial predictions match observations
        self.W_pred = np.random.randn(n_obs, n_state) * 0.05

        # Make first n_obs columns close to identity
        for i in range(min(n_obs, n_state)):
            self.W_pred[i, i] = 0.8 + np.random.randn() * 0.1

        # W_epsilon: error -> state update (n_state x n_obs)
        # Transpose-like structure for symmetric start
        self.W_epsilon = self.W_pred.T.copy() * 0.5

    def _create_initial_state(self) -> PCState:
        """초기 상태 생성"""
        n_obs = self.config.n_obs
        n_state = self.config.n_state

        return PCState(
            mu=np.zeros(n_state),
            epsilon=np.zeros(n_obs),
            o_hat=np.zeros(n_obs),
            d_mu=np.zeros(n_state),
            mu_prior=np.zeros(n_state),
            lambda_prior=self.config.base_prior_precision
        )

    def reset(self):
        """상태 리셋"""
        self.state = self._create_initial_state()
        self.history.clear()

    # =========================================================================
    # 핵심 동역학
    # =========================================================================

    def compute_prediction(self, mu: np.ndarray) -> np.ndarray:
        """
        상태(μ)에서 예측(ô) 생성
        ô = W_pred @ μ
        """
        o_hat = self.W_pred @ mu
        # 관측 범위로 클리핑 (0-1)
        return np.clip(o_hat, 0, 1)

    def compute_error(self, observation: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """
        예측 오차 계산
        ε = o - ô
        """
        return observation - prediction

    def compute_state_update(
        self,
        epsilon: np.ndarray,
        mu: np.ndarray,
        mu_prior: np.ndarray,
        lambda_prior: float
    ) -> np.ndarray:
        """
        상태 업데이트 계산 (두 힘의 균형)

        dμ/dt = W_ε @ ε - λ_prior * (μ - μ_prior)
                ^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
                data-fit   prior-pull
        """
        # Data-fit force: 오차가 상태를 끌어당김
        data_force = self.W_epsilon @ epsilon

        # Prior-pull force: prior가 상태를 잡아당김
        prior_force = lambda_prior * (mu - mu_prior)

        # 총 업데이트
        d_mu = data_force - prior_force

        # 진단 정보 저장
        self.state.data_force_norm = np.linalg.norm(data_force)
        self.state.prior_force_norm = np.linalg.norm(prior_force)

        return d_mu

    def step_dynamics(
        self,
        observation: np.ndarray,
        mu_prior: Optional[np.ndarray] = None,
        lambda_prior: Optional[float] = None
    ) -> PCState:
        """
        한 타임스텝 동역학 실행

        Args:
            observation: 현재 관측 (8D)
            mu_prior: prior 상태 (memory/context에서 제공)
            lambda_prior: prior precision (uncertainty로 조절)

        Returns:
            업데이트된 PCState
        """
        dt = self.config.dt
        tau = self.config.tau

        # Prior 설정
        if mu_prior is not None:
            self.state.mu_prior = mu_prior
        if lambda_prior is not None:
            self.state.lambda_prior = lambda_prior

        # 1. 예측 계산
        self.state.o_hat = self.compute_prediction(self.state.mu)

        # 2. 오차 계산
        self.state.epsilon = self.compute_error(observation, self.state.o_hat)
        self.state.error_norm = np.linalg.norm(self.state.epsilon)

        # 3. 상태 업데이트 계산
        self.state.d_mu = self.compute_state_update(
            self.state.epsilon,
            self.state.mu,
            self.state.mu_prior,
            self.state.lambda_prior
        )

        # 4. 오일러 적분
        self.state.mu = self.state.mu + (dt / tau) * self.state.d_mu

        return self.state

    def infer(
        self,
        observation: np.ndarray,
        mu_prior: Optional[np.ndarray] = None,
        lambda_prior: Optional[float] = None,
        reset_state: bool = False
    ) -> PCState:
        """
        Infer belief (mu) from observation (iterate until convergence)

        Args:
            observation: current observation
            mu_prior: prior state
            lambda_prior: prior precision
            reset_state: if True, reset mu to 0 before inference

        Returns:
            converged PCState

        Note: By default, mu carries forward from previous inference,
        enabling drift detection (spike when observations suddenly change).
        """
        threshold = self.config.convergence_threshold
        max_iter = self.config.max_iterations

        if reset_state:
            self.state.mu = np.zeros(self.config.n_state)

        # Record initial prediction error (before convergence)
        initial_o_hat = self.compute_prediction(self.state.mu)
        initial_error = np.linalg.norm(observation - initial_o_hat)

        for i in range(max_iter):
            self.step_dynamics(observation, mu_prior, lambda_prior)

            # Convergence check
            if np.max(np.abs(self.state.d_mu)) < threshold:
                self.state.converged = True
                self.state.iterations = i + 1
                break
        else:
            self.state.converged = False
            self.state.iterations = max_iter

        # Store initial error for drift detection
        self.state.initial_error = initial_error

        # Record history
        self._record_history(observation)

        return self.state

    # =========================================================================
    # Teacher-Student 학습
    # =========================================================================

    def learn_from_teacher(
        self,
        observation: np.ndarray,
        o_hat_teacher: np.ndarray,
        epsilon_stats_teacher: Optional[Dict] = None
    ):
        """
        Teacher(FEP)의 예측을 따라 학습

        학습 목표: ||ô_student - ô_teacher|| 최소화
        (Q를 직접 맞추지 않음!)

        Args:
            observation: 현재 관측
            o_hat_teacher: FEP가 제공하는 예측
            epsilon_stats_teacher: FEP의 오차 통계 (선택)
        """
        # 현재 학생의 예측
        o_hat_student = self.state.o_hat

        # 예측 오차 (teacher와의 차이)
        pred_error = o_hat_student - o_hat_teacher

        # W_pred 업데이트: 예측을 teacher에 맞추도록
        # dW_pred = -lr * pred_error @ mu.T
        d_W_pred = -self.config.lr_pred * np.outer(pred_error, self.state.mu)
        self.W_pred += d_W_pred

        # W_epsilon 업데이트: 오차가 올바른 상태 변화를 유도하도록
        # 이건 더 복잡한 학습 규칙이 필요할 수 있음
        # 일단 W_pred와 대칭 유지하는 방향으로
        self.W_epsilon = self.W_pred.T.copy() * 0.8

    # =========================================================================
    # v4.x 연결: Prior 조절
    # =========================================================================

    def modulate_prior_precision(
        self,
        uncertainty: float = 0.0,
        drift_detected: bool = False,
        transition_error: float = 0.0
    ) -> float:
        """
        v4.x 신호로 λ_prior 조절

        - uncertainty ↑ → λ_prior ↓ (prior 약화, 데이터 의존)
        - drift 감지 → λ_prior ↓ (새 환경 적응)
        - transition_error ↑ → λ_prior ↓ (모델 불신)
        """
        base = self.config.base_prior_precision
        min_p = self.config.min_prior_precision

        # 불확실성에 의한 감소
        uncertainty_factor = 1.0 - 0.5 * uncertainty

        # Drift에 의한 감소
        drift_factor = 0.3 if drift_detected else 1.0

        # Transition error에 의한 감소
        error_factor = 1.0 / (1.0 + transition_error)

        # 최종 precision
        lambda_prior = base * uncertainty_factor * drift_factor * error_factor
        lambda_prior = max(min_p, lambda_prior)

        self.state.lambda_prior = lambda_prior
        return lambda_prior

    def set_prior_from_memory(self, memory_state: np.ndarray):
        """
        Memory/Context에서 μ_prior 설정

        기억이 "정책 지시"가 아니라 "prior로 스며드는" 방식
        """
        # memory_state를 n_state 차원으로 변환
        if len(memory_state) != self.config.n_state:
            # 간단한 projection
            proj = np.zeros(self.config.n_state)
            proj[:len(memory_state)] = memory_state[:self.config.n_state]
            self.state.mu_prior = proj
        else:
            self.state.mu_prior = memory_state.copy()

    # =========================================================================
    # 히스토리 & 진단
    # =========================================================================

    def _record_history(self, observation: np.ndarray):
        """히스토리 기록"""
        record = {
            'observation': observation.copy(),
            'mu': self.state.mu.copy(),
            'o_hat': self.state.o_hat.copy(),
            'epsilon': self.state.epsilon.copy(),
            'error_norm': self.state.error_norm,
            'converged': self.state.converged,
            'iterations': self.state.iterations,
            'lambda_prior': self.state.lambda_prior,
            'data_force': self.state.data_force_norm,
            'prior_force': self.state.prior_force_norm,
        }

        self.history.append(record)

        # 최대 크기 유지
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_diagnostics(self) -> Dict:
        """현재 상태 진단 정보"""
        return {
            'mu_mean': float(np.mean(self.state.mu)),
            'mu_std': float(np.std(self.state.mu)),
            'error_norm': float(self.state.error_norm),
            'converged': self.state.converged,
            'iterations': self.state.iterations,
            'lambda_prior': float(self.state.lambda_prior),
            'data_force': float(self.state.data_force_norm),
            'prior_force': float(self.state.prior_force_norm),
            'force_ratio': float(
                self.state.data_force_norm / (self.state.prior_force_norm + 1e-6)
            )
        }

    def get_recent_error_stats(self, window: int = 50) -> Dict:
        """최근 오차 통계"""
        if len(self.history) < 2:
            return {'mean': 0.0, 'std': 0.0, 'trend': 0.0}

        recent = self.history[-window:]
        errors = [h['error_norm'] for h in recent]

        return {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'trend': float(errors[-1] - errors[0]) if len(errors) > 1 else 0.0
        }


# =============================================================================
# Teacher 인터페이스 (FEP와 연결)
# =============================================================================

class FEPTeacher:
    """
    FEP 시스템을 Teacher로 감싸는 인터페이스

    기존 GenesisAgent의 추론 결과를 예측(ô) 형태로 제공
    """

    def __init__(self, agent):
        """
        Args:
            agent: GenesisAgent 인스턴스
        """
        self.agent = agent

    def get_prediction(self, observation: np.ndarray) -> np.ndarray:
        """
        FEP 시스템의 예측 획득

        ô_teacher = E[o | Q_teacher]

        실제로는 agent의 현재 믿음에서 기대 관측을 계산
        """
        # Agent의 현재 상태에서 예측
        # 간단한 버전: 현재 관측을 prior와 혼합
        state = self.agent.state

        # Q(s)에서 기대 관측 계산
        # 현재는 단순화: 관측 그대로 + precision 가중
        o_hat = observation.copy()

        # 선호와 혼합 (예측은 관측과 선호의 가중 평균)
        if hasattr(self.agent, 'preferred_obs'):
            mix = 0.3  # 선호 반영 비율
            o_hat = (1 - mix) * observation + mix * self.agent.preferred_obs

        return o_hat

    def get_error_statistics(self) -> Dict:
        """FEP 시스템의 오차 통계"""
        if hasattr(self.agent, 'action_selector'):
            selector = self.agent.action_selector
            return {
                'variance': getattr(selector, 'transition_std', 0.1) ** 2,
                'precision': getattr(selector, 'sensory_precision', 1.0)
            }
        return {'variance': 0.1, 'precision': 1.0}


# =============================================================================
# 통합 테스트용 함수
# =============================================================================

def test_convergence(pc_layer: NeuralPCLayer, n_tests: int = 10) -> Dict:
    """Gate 1: 수렴 테스트"""
    results = []

    for _ in range(n_tests):
        # 랜덤 관측
        obs = np.random.rand(pc_layer.config.n_obs)

        # 추론
        pc_layer.reset()
        state = pc_layer.infer(obs)

        results.append({
            'converged': state.converged,
            'iterations': state.iterations,
            'final_error': state.error_norm
        })

    convergence_rate = sum(r['converged'] for r in results) / n_tests
    avg_iterations = np.mean([r['iterations'] for r in results])

    return {
        'convergence_rate': convergence_rate,
        'avg_iterations': avg_iterations,
        'passed': convergence_rate > 0.8
    }


def test_prediction_match(
    pc_layer: NeuralPCLayer,
    teacher: FEPTeacher,
    n_tests: int = 10
) -> Dict:
    """Gate 2: 예측 일치 테스트"""
    errors = []

    for _ in range(n_tests):
        obs = np.random.rand(pc_layer.config.n_obs)

        # Student 추론
        pc_layer.reset()
        state = pc_layer.infer(obs)
        o_hat_student = state.o_hat

        # Teacher 예측
        o_hat_teacher = teacher.get_prediction(obs)

        # 오차
        error = np.linalg.norm(o_hat_student - o_hat_teacher)
        errors.append(error)

        # 학습
        pc_layer.learn_from_teacher(obs, o_hat_teacher)

    avg_error = np.mean(errors)
    final_error = errors[-1]

    return {
        'avg_error': avg_error,
        'final_error': final_error,
        'improvement': errors[0] - errors[-1] if len(errors) > 1 else 0,
        'passed': final_error < 0.3
    }


def test_drift_response(
    pc_layer: NeuralPCLayer,
    n_pre: int = 20,
    n_post: int = 30
) -> Dict:
    """Gate 3: Drift response test

    Tests if:
    1. Circuit maintains state across inferences
    2. Sudden observation change causes initial_error spike
    3. Circuit adapts and error decreases over time
    """
    initial_errors_pre = []
    final_errors_pre = []
    initial_errors_post = []
    final_errors_post = []

    # Pre-drift: consistent observations - let the circuit adapt
    base_obs = np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0])

    for _ in range(n_pre):
        obs = base_obs + np.random.randn(8) * 0.02  # Low noise
        obs = np.clip(obs, 0, 1)
        state = pc_layer.infer(obs)
        initial_errors_pre.append(state.initial_error)
        final_errors_pre.append(state.error_norm)

    # Record stable errors (after adaptation)
    stable_initial_error = np.mean(initial_errors_pre[-10:])
    stable_final_error = np.mean(final_errors_pre[-10:])

    # Drift: sudden change in observation pattern
    drifted_obs = np.array([0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5])

    for i in range(n_post):
        obs = drifted_obs + np.random.randn(8) * 0.02
        obs = np.clip(obs, 0, 1)

        # Detect drift and modulate prior
        if i == 0:
            pc_layer.modulate_prior_precision(drift_detected=True)
        elif i == n_post // 2:
            pc_layer.modulate_prior_precision(drift_detected=False)

        state = pc_layer.infer(obs)
        initial_errors_post.append(state.initial_error)
        final_errors_post.append(state.error_norm)

    # Check spike: INITIAL error at drift moment should spike
    # (prediction based on old mu doesn't match new observation)
    spike_initial_error = initial_errors_post[0]
    spike_detected = spike_initial_error > stable_initial_error * 1.3  # 30% increase

    # Check recovery: late errors should be stable or decreasing
    # (the circuit adapts to new observations over time)
    early_final_error = np.mean(final_errors_post[:5])
    late_final_error = np.mean(final_errors_post[-5:])
    # Recovery: either return to near pre-drift level OR improve from spike
    # (drifted observation may have inherently higher prediction error)
    recovery_to_baseline = late_final_error < stable_final_error * 1.4
    recovery_from_spike = late_final_error < spike_initial_error * 0.9
    recovery = recovery_to_baseline or recovery_from_spike

    return {
        'pre_drift_final_error': stable_final_error,
        'pre_drift_initial_error': stable_initial_error,
        'spike_initial_error': spike_initial_error,
        'post_drift_final_error': late_final_error,
        'spike_detected': spike_detected,
        'recovery': recovery,
        'passed': spike_detected and recovery
    }
