"""
Action Selection - Expected Free Energy with Risk/Ambiguity Decomposition

G(a) = Risk + Ambiguity

Where:
    Risk = KL[Q(o|a) || P(o)]
         = "How far are expected observations from preferences?"
         = "선호 위반 가능성"

    Ambiguity = E_Q(s'|a)[H[P(o|s')]]
              = "How uncertain are observations given the predicted state?"
              = "상태에서 관측이 얼마나 애매한가?"

This decomposition is crucial because:
- Risk explains AVOIDANCE (what observers call "fear")
- Ambiguity reduction explains EXPLORATION (what observers call "curiosity")

No emotion labels needed. The decomposition itself shows the cause.

VERSION 2.0: True FEP Implementation
- P(o) is now proper probability distributions (Beta for proximity, Categorical for direction)
- Risk = KL divergence (not squared error)
- Ambiguity = Entropy (not constant 0.1)
- STAY penalty removed (absorbed into P(o) as observation dynamics preference)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .free_energy import FreeEnergyEngine
from .generative_model import GenerativeModel
from .preference_distributions import PreferenceDistributions, StatePreferenceDistribution, PreferenceLearner
from .precision import PrecisionLearner, PrecisionState
from .temporal import TemporalPlanner, RolloutResult
from .hierarchy import HierarchicalController
from .uncertainty import UncertaintyTracker, UncertaintyState, UncertaintyModulation, compute_context_entropy
from .memory import LTMStore, Episode, RecallResult, compute_outcome_score, RegimeLTMStore
from .consolidation import MemoryConsolidator, ConsolidationResult
from .regret import CounterfactualEngine, CounterfactualResult, RegretState
from .regime import RegimeTracker, RegimeConfig, RegimeState


@dataclass
class GDecomposition:
    """Complete decomposition of Expected Free Energy."""
    G: float              # Total expected free energy
    risk: float           # KL[Q(o|a) || P(o)] - preference violation
    ambiguity: float      # E[H[P(o|s')]] - observation uncertainty
    complexity: float     # KL[Q(s'|a) || P(s')] - belief divergence from preferred states
    action_cost: float    # Small cost for movement (deprecated, kept for compatibility)


@dataclass
class ActionResult:
    """Result of action selection."""
    action: int
    G_all: Dict[int, GDecomposition]  # Full decomposition for each action
    probabilities: np.ndarray

    # For logging/debugging
    why_risk: str         # Which action had high risk
    why_ambiguity: str    # Which action reduces ambiguity most


class ActionSelector:
    """
    Selects actions by minimizing Expected Free Energy.

    G(a) = Risk(a) + Ambiguity(a) + ActionCost(a)

    Key insight:
    - Minimizing Risk → avoid states that violate preferences (looks like "fear")
    - Minimizing Ambiguity → seek informative observations (looks like "curiosity")
    - Action cost → prevents "do nothing" from being optimal

    The agent doesn't "feel" fear or curiosity.
    It just minimizes G.
    But the decomposition shows WHY each action was chosen.
    """

    def __init__(self,
                 model: GenerativeModel,
                 free_energy: FreeEnergyEngine,
                 n_actions: int):
        self.model = model
        self.fe = free_energy
        self.n_actions = n_actions

        # === TRUE FEP: P(o) as probability distributions ===
        self.preferences = PreferenceDistributions()

        # === TRUE FEP v2.2: P(s) for Complexity ===
        self.state_preferences = StatePreferenceDistribution(n_states=model.n_states)

        # === TRUE FEP v2.3: Precision Learning ===
        self.precision_learner = PrecisionLearner(n_obs=6, n_actions=n_actions)
        self._predicted_obs: Optional[np.ndarray] = None  # 예측 저장 (precision 업데이트용)

        # === TRUE FEP v3.5: Online Preference Learning ===
        # 내부 선호(energy, pain)의 Beta 파라미터를 경험에서 학습
        self.preference_learner: Optional[PreferenceLearner] = None
        self.preference_learning_enabled = False

        # === TRUE FEP v2.4: Temporal Depth ===
        self.temporal_planner: Optional[TemporalPlanner] = None
        self.use_rollout = False  # 기본: 1-step만
        self.rollout_horizon = 3
        self.rollout_discount = 0.9
        self.rollout_n_samples = 3       # Monte Carlo 샘플 수
        self.rollout_complexity_decay = 0.7  # Complexity 감쇠율
        self._last_rollout_results: Optional[Dict[int, RolloutResult]] = None

        # === TRUE FEP v2.4.3: Adaptive Rollout (Enhanced) ===
        # 결정 불확실성 기반 rollout 트리거
        #
        # v2.4.2: G_spread = G_second_best - G_best (스케일 민감)
        # v2.4.3: softmax entropy + 분위수 기반 + ambiguity 보완
        #
        # 핵심 개선:
        # 1. decision_entropy = H(softmax(-G)) - 스케일 불변
        # 2. threshold = 최근 N스텝 중 상위 q% (상대 기준)
        # 3. ambiguity 높으면 spread 커도 rollout (불확실성 보완)
        # 4. rollout이 선택을 바꿨는지 로그

        self.adaptive_rollout = False
        self.rollout_quantile = 0.3  # 상위 30%에서 rollout (0.0=항상, 1.0=안함)
        self._rollout_triggered = False

        # 결정 불확실성 측정 (v2.4.3)
        self._last_G_spread = 0.0
        self._last_decision_entropy = 0.0  # H(softmax(-G))
        self._last_p_best = 0.0  # softmax(-G)의 최대값
        self._last_max_ambiguity = 0.0  # 최대 ambiguity

        # 분위수 계산용 히스토리
        self._entropy_history = []  # 최근 N스텝의 entropy
        self._entropy_history_size = 100  # 히스토리 크기

        # rollout 선택 변경 로그
        self._last_1step_action = 0
        self._last_rollout_action = 0
        self._last_action_changed = False
        self._last_change_reason = ""

        # === PARAMETERS ===
        self.temperature = 0.3  # Lower = more deterministic
        self.complexity_weight = 0.5  # Weight for complexity in G(a)

        # REMOVED: action_cost_weight - absorbed into P(o)
        # REMOVED: novelty_bonus - was causing oscillation

        # Track last action and observations for transition learning
        self._last_action = 0
        self._last_obs = None
        self._action_history = []
        self._action_history_max = 1000  # v4.6.3: 메모리 제한 추가

        # === TRANSITION MODEL (learned, not hardcoded) ===
        # P(o'|o, a): 각 행동이 관측을 어떻게 바꾸는지 학습
        # Shape: (n_actions, 8) - v2.5: 8차원
        # [0-5]: exteroception (food/danger proximity, directions)
        # [6-7]: interoception (energy, pain) - 환경에서 결정되므로 예측 불확실
        self.transition_model = {
            'delta_mean': np.zeros((n_actions, 8)),  # 예측된 평균 변화
            'delta_std': np.ones((n_actions, 8)) * 0.2,  # 변화의 불확실성
            'count': np.ones((n_actions,))  # 행동별 경험 횟수
        }
        # Internal states (energy, pain)는 높은 불확실성으로 시작
        # 에이전트가 "음식 먹으면 energy↑" 등을 학습해야 함
        self.transition_model['delta_std'][:, 6:8] = 0.5

        # Learning rate for transition model
        self.transition_lr = 0.1

        # v4.2: Volatility tracking for transition model
        self._last_transition_error = None

        # === TRUE FEP v3.0: Hierarchical Models ===
        # Slow layer가 fast layer의 precision을 조절
        self.hierarchy_controller: Optional[HierarchicalController] = None

        # === TRUE FEP v3.3.1: Context-weighted Transition (Stabilized) ===
        # Slow layer의 학습된 전이 모델을 행동 선택에 사용
        #
        # v3.3 문제점:
        # - delta_ctx 스케일이 physics보다 커서 alpha 조금만 올려도 지배
        # - DOWN 같은 특정 행동이 internal state 예측에서 과하게 유리
        # - Q(c)가 one-hot으로 굳으면 더 빠르게 붕괴
        #
        # v3.3.1 개선:
        # 1. delta_ctx clamp: [-0.05, +0.05]로 제한
        # 2. alpha 분리: internal(energy/pain)은 더 보수적
        # 3. 신뢰도 기반 alpha_eff: context 신뢰도 높을 때만 alpha 상승

        self.context_transition_alpha_external = 0.2  # 외부 (food/danger prox)
        self.context_transition_alpha_internal = 0.1  # 내부 (energy/pain) - 더 보수적
        self.delta_ctx_clamp = 0.05  # delta_ctx 최대 변화량
        self.use_confidence_alpha = True  # 신뢰도 기반 alpha 활성화

        # 디버그 로깅 (마지막 스텝 정보)
        self._last_delta_debug = None

        # === TRUE FEP v3.4: THINK Action (Metacognition) ===
        # THINK = action 5: "생각할지 말지"를 G로 선택
        # G(THINK) = Expected G after deliberation
        #
        # 핵심 원리:
        # - THINK 선택 시 rollout 실행 → 더 좋은 행동 발견
        # - G(THINK)가 G(best_physical)보다 낮으면 THINK 선택
        # - 비용: 시간이 흐름 (energy 감소, 위험 이동)
        # - 이것은 "생각 벌점"이 아니라 환경 다이나믹스

        self.THINK_ACTION = 5  # stay=0, up=1, down=2, left=3, right=4, THINK=5
        self.N_PHYSICAL_ACTIONS = 5  # THINK 제외한 물리 행동 수
        self.think_enabled = False  # 기본: 비활성화
        self.think_energy_cost = 0.003  # THINK 시 energy 감소 (환경과 동일)

        # === v3.4.1: THINK 최적화 ===
        # 1) 2단계 게이트: entropy 높을 때만 THINK 평가
        self.think_entropy_threshold = 1.0  # log(5)≈1.61의 ~62%
        self.think_G_spread_threshold = 0.1  # G 차이가 이것보다 작으면 애매한 상황
        self._think_gate_passed = False  # 디버그용: 게이트 통과 여부

        # 2) THINK rollout 예산 하드캡
        self.think_rollout_horizon = 2  # THINK용 rollout은 짧게
        self.think_rollout_samples = 1  # 샘플 1개만

        # 3) THINK 쿨다운
        self.think_cooldown = 5  # THINK 후 N step 동안 THINK 평가 스킵
        self._think_cooldown_counter = 0  # 현재 남은 쿨다운

        # THINK 선택 로그
        self._last_think_selected = False
        self._last_think_reason = ""
        self._last_expected_improvement = 0.0
        self._think_count = 0
        self._physical_action_after_think = None

        # === TRUE FEP v4.3: Uncertainty/Confidence Tracking ===
        # 불확실성 기반 자기조절 시스템
        # - THINK 선택 확률/비용 연동
        # - Precision 메타-조절
        # - 탐색/회피 균형
        # - 기억 저장 게이트 (v4.0 준비)
        self.uncertainty_tracker: Optional[UncertaintyTracker] = None
        self.uncertainty_enabled = False
        self._last_uncertainty_state: Optional[UncertaintyState] = None
        self._last_uncertainty_modulation: Optional[UncertaintyModulation] = None

        # === TRUE FEP v4.0: Long-Term Memory ===
        # 기억 = 미래 F/G를 줄이는 압축 모델
        # - 저장: memory_gate > threshold일 때 확률적 저장
        # - 회상: 유사 상황 검색 → G bias (행동 직접 지시 X)
        # - 압축: 유사 에피소드 병합
        self.ltm_store: Optional[LTMStore] = None
        self.memory_enabled = False
        # v4.6: 분해 실험용 세분화 제어
        self.memory_store_enabled = True   # 새 에피소드 저장 허용
        self.memory_recall_enabled = True  # 회상 bias 적용 허용
        # v4.6: Drift-aware recall suppression
        self.drift_aware_suppression = False  # drift 감지 시 recall 억제
        self._recall_suppression_factor = 1.0  # 1.0=정상, 0.0=완전억제
        self._prediction_error_baseline = 0.3  # EMA baseline
        self._prediction_error_ema = 0.3
        self._suppression_recovery_rate = 0.05  # 회복 속도
        self._last_recall_result: Optional[RecallResult] = None
        self._last_store_result: Optional[Dict] = None
        self._pending_episode_data: Optional[Dict] = None  # 저장 대기 중 에피소드 정보

        # === TRUE FEP v4.7: Regime-tagged Memory ===
        # 레짐별 메모리 분리 - pre-drift 기억이 post-drift에서 독이 되는 문제 해결
        # - regime_tracker: Q(r) belief 기반 레짐 감지
        # - regime_ltm: 레짐별 분리된 메모리 뱅크
        # - 현재 레짐 뱅크에서만 회상 (MVP)
        self.regime_tracker: Optional[RegimeTracker] = None
        self.regime_ltm: Optional[RegimeLTMStore] = None
        self.regime_memory_enabled = False
        self._last_regime_update: Optional[Dict] = None

        # === TRUE FEP v4.1: Memory Consolidation (Sleep) ===
        # 수면/통합: LTM을 "조언자"에서 "prior"로 변환
        # - 트리거: low_surprise + high_redundancy + stable_context
        # - 효과: transition_std 감소, uncertainty 감소, G 감소
        self.consolidator: Optional[MemoryConsolidator] = None
        self.consolidation_enabled = False
        self._last_consolidation_result: Optional[ConsolidationResult] = None
        self._consolidation_auto = True  # 자동 통합 트리거 여부

        # === TRUE FEP v4.4: Counterfactual + Regret ===
        # 후회 = 선택한 행동이 대안보다 얼마나 더 큰 G를 초래했는지의 '사후 EFE 차이'
        # 연결 방식 (FEP스럽게):
        # - 정책 직접 변경 X
        # - memory_gate, lr_boost, THINK 비용/편익 보정 O
        self.counterfactual_engine: Optional[CounterfactualEngine] = None
        self.regret_enabled = False
        # v4.6: 분해 실험용 - regret 계산은 하되 modulation 끄기
        self.regret_modulation_enabled = True  # memory_gate_boost, lr_boost, think_benefit 적용
        self._last_cf_result: Optional[CounterfactualResult] = None
        self._last_G_pred: Dict[int, float] = {}  # 선택 시점 G 저장용

    def compute_G(self, Q_s: Optional[np.ndarray] = None, current_obs: Optional[np.ndarray] = None) -> Dict[int, GDecomposition]:
        """
        Compute Expected Free Energy G(a) with TRUE FEP decomposition.

        G(a) = Risk(a) + Ambiguity(a)

        Where:
            Risk = KL[Q(o|a) || P(o)]  - NOT squared error!
            Ambiguity = E[H[P(o|s')]] - NOT constant 0.1!

        Observation structure (6 dims):
        - [0]: food_proximity (0-1, 1=on food)
        - [1]: danger_proximity (0-1, 1=on danger)
        - [2]: food_dx (-1=left, 0=same, +1=right)
        - [3]: food_dy (-1=up, 0=same, +1=down)
        - [4]: danger_dx
        - [5]: danger_dy

        Actions: 0=stay, 1=up, 2=down, 3=left, 4=right

        P(o) as distributions:
        - food_proximity: Beta(5, 1) - prefer near 1
        - danger_proximity: Beta(1, 5) - prefer near 0
        - directions: Uniform - no preference

        Transition prediction uses:
        - Learned delta (from experience)
        - PHYSICS PRIOR: Direction info tells us which way moving changes proximity
          This is NOT value - it's geometry. "Moving right increases proximity to things on right"
        """
        if Q_s is None:
            Q_s = self.model.Q_s

        results = {}

        # v3.4: THINK는 여기서 계산하지 않음 (compute_G_think에서 별도 계산)
        n_physical = self.N_PHYSICAL_ACTIONS if hasattr(self, 'N_PHYSICAL_ACTIONS') else self.n_actions
        for a in range(n_physical):
            # === PREDICT OBSERVATIONS USING PHYSICS + LEARNING ===
            if current_obs is not None and len(current_obs) >= 6:
                food_prox = current_obs[0]
                danger_prox = current_obs[1]
                food_dx = current_obs[2]   # -1=left, 0=same, +1=right
                food_dy = current_obs[3]   # -1=up, 0=same, +1=down
                danger_dx = current_obs[4]
                danger_dy = current_obs[5]

                # Get learned adjustment (starts at 0, learns from experience)
                learned_delta = self.transition_model['delta_mean'][a]
                delta_std = self.transition_model['delta_std'][a]

                # === PHYSICS PRIOR ===
                # Moving toward something increases proximity.
                # This is geometry, not value judgment.
                # delta_prox_base = how much proximity changes when moving 1 step
                delta_prox_base = 0.1

                # Predict food proximity change based on action and direction
                delta_food_prox = 0.0
                if a == 1:  # UP (dy decreases)
                    if food_dy < 0:  # Food is up
                        delta_food_prox = delta_prox_base
                    elif food_dy > 0:  # Food is down
                        delta_food_prox = -delta_prox_base
                elif a == 2:  # DOWN (dy increases)
                    if food_dy > 0:  # Food is down
                        delta_food_prox = delta_prox_base
                    elif food_dy < 0:  # Food is up
                        delta_food_prox = -delta_prox_base
                elif a == 3:  # LEFT (dx decreases)
                    if food_dx < 0:  # Food is left
                        delta_food_prox = delta_prox_base
                    elif food_dx > 0:  # Food is right
                        delta_food_prox = -delta_prox_base
                elif a == 4:  # RIGHT (dx increases)
                    if food_dx > 0:  # Food is right
                        delta_food_prox = delta_prox_base
                    elif food_dx < 0:  # Food is left
                        delta_food_prox = -delta_prox_base
                # a == 0 (STAY) - no change

                # Predict danger proximity change (scaled by current proximity)
                # Key insight: If danger is far (prox≈0), moving toward it barely matters
                danger_delta_scale = danger_prox * 0.4  # 0 when far, 0.4 when close

                delta_danger_prox = 0.0
                if a == 1:  # UP
                    if danger_dy < 0:
                        delta_danger_prox = danger_delta_scale
                    elif danger_dy > 0:
                        delta_danger_prox = -danger_delta_scale
                elif a == 2:  # DOWN
                    if danger_dy > 0:
                        delta_danger_prox = danger_delta_scale
                    elif danger_dy < 0:
                        delta_danger_prox = -danger_delta_scale
                elif a == 3:  # LEFT
                    if danger_dx < 0:
                        delta_danger_prox = danger_delta_scale
                    elif danger_dx > 0:
                        delta_danger_prox = -danger_delta_scale
                elif a == 4:  # RIGHT
                    if danger_dx > 0:
                        delta_danger_prox = danger_delta_scale
                    elif danger_dx < 0:
                        delta_danger_prox = -danger_delta_scale

                # === v3.3.1: Context-weighted Transition Blending (Stabilized) ===
                # Physics prior만 사용하다가, Slow layer가 학습한 전이 모델을 혼합
                #
                # v3.3.1 개선:
                # 1. delta_ctx를 [-clamp, +clamp]로 제한 (스케일 폭주 방지)
                # 2. alpha_external / alpha_internal 분리 (내부 상태는 더 보수적)
                # 3. 신뢰도 기반 alpha_eff (context 불확실하면 physics 의존)

                # 1. Physics delta (기존)
                delta_physics = np.zeros(8)
                delta_physics[0] = delta_food_prox
                delta_physics[1] = delta_danger_prox

                # 2. Context-weighted delta with CLAMP (v3.3.1)
                delta_ctx = np.zeros(8)
                alpha_ext = 0.0
                alpha_int = 0.0

                if self.hierarchy_controller is not None:
                    ctx_delta = self.hierarchy_controller.get_context_weighted_transition(a)
                    if ctx_delta is not None:
                        # v3.3.1: Clamp delta_ctx to prevent scale explosion
                        delta_ctx = np.clip(ctx_delta, -self.delta_ctx_clamp, self.delta_ctx_clamp)

                        # 기본 alpha 값
                        alpha_ext = self.context_transition_alpha_external
                        alpha_int = self.context_transition_alpha_internal

                        # v3.3.1: 신뢰도 기반 alpha 조절
                        if self.use_confidence_alpha:
                            # Context belief entropy가 낮을수록 (확실할수록) alpha 상승
                            Q_ctx = self.hierarchy_controller.get_context_belief()
                            entropy = -np.sum(Q_ctx * np.log(Q_ctx + 1e-10))
                            max_entropy = np.log(len(Q_ctx))  # log(K)

                            # entropy_ratio: 0=확실, 1=불확실
                            entropy_ratio = entropy / (max_entropy + 1e-10)

                            # 확실할 때만 alpha 유지, 불확실하면 감소
                            confidence_mult = 1.0 - entropy_ratio  # 0~1
                            alpha_ext *= confidence_mult
                            alpha_int *= confidence_mult

                # 3. Blend separately for external and internal
                # External: food_prox, danger_prox (indices 0, 1)
                delta_blended_ext = (1 - alpha_ext) * delta_physics[:2] + alpha_ext * delta_ctx[:2]
                # v4.6.3: 최종 안전장치 (블렌딩 후 재클리핑)
                delta_blended_ext = np.clip(delta_blended_ext, -0.15, 0.15)

                total_delta_food = delta_blended_ext[0]
                total_delta_danger = delta_blended_ext[1]

                # Predicted observation
                Q_o = current_obs.copy()
                Q_o[0] = np.clip(food_prox + total_delta_food, 0.0, 1.0)
                Q_o[1] = np.clip(danger_prox + total_delta_danger, 0.0, 1.0)

                # v3.3.1: 내부 상태는 더 보수적인 alpha_internal 사용
                if alpha_int > 0:
                    delta_blended_int = alpha_int * delta_ctx[6:8]  # physics는 0이라 가정
                    # v4.6.3: 내부 상태도 클리핑
                    delta_blended_int = np.clip(delta_blended_int, -0.1, 0.1)
                    Q_o[6] = np.clip(current_obs[6] + delta_blended_int[0], 0.0, 1.0)
                    Q_o[7] = np.clip(current_obs[7] + delta_blended_int[1], 0.0, 1.0)

                # 디버그 로깅 (첫 번째 행동만)
                if a == 0:
                    self._last_delta_debug = {
                        'delta_physics': delta_physics[:2].tolist(),
                        'delta_ctx': delta_ctx[:2].tolist(),
                        'delta_ctx_internal': delta_ctx[6:8].tolist(),
                        'alpha_ext': alpha_ext,
                        'alpha_int': alpha_int,
                        'Q_o_pred': [Q_o[0], Q_o[1], Q_o[6], Q_o[7]]
                    }

                # Uncertainty combines physics uncertainty with learned uncertainty
                q_uncertainty = delta_std.copy()
                q_uncertainty[0] = max(0.05, delta_std[0])  # Minimum uncertainty
                q_uncertainty[1] = max(0.05, delta_std[1])

            else:
                Q_o = np.zeros(8)  # v2.5: 8차원
                q_uncertainty = np.ones(8) * 0.5

            # === RISK: KL[Q(o|a) || P(o)] with PRECISION weighting ===
            # Precision = "이 차원의 선호 위반에 얼마나 민감한가"
            risk_weights = self.precision_learner.get_risk_weights()

            # v3.0: Hierarchical modulation
            effective_goal_precision = self.precision_learner.goal_precision
            if self.hierarchy_controller is not None:
                mod = self.hierarchy_controller.get_modulation()
                # Sensory precision modulation (only first 6 dims for external prefs)
                sensory_mod = mod['sensory_mod']
                risk_weights = risk_weights * sensory_mod[:len(risk_weights)]
                # Goal precision modulation
                effective_goal_precision *= mod['goal_precision_mult']
                # Internal preference weight modulation
                self.preferences.internal_pref_weight = mod['internal_pref_weight']

            # v4.3: Uncertainty-based precision modulation
            # 불확실성 높음 → precision 낮춤 → 관측을 덜 믿음
            # 불확실성 낮음 → precision 높임 → 집중/확신
            risk_sensitivity = 1.0
            if self.uncertainty_enabled and self._last_uncertainty_modulation is not None:
                unc_mod = self._last_uncertainty_modulation
                risk_weights = risk_weights * unc_mod.sensory_precision_mult
                effective_goal_precision *= unc_mod.goal_precision_mult
                risk_sensitivity = unc_mod.risk_sensitivity

                # v4.3.1 안전점검: 위험 근접 시 risk_sensitivity 하한 상향
                # danger_proximity(Q_o[1])가 높으면 → 최소 민감도 올림
                danger_proximity = Q_o[1] if len(Q_o) > 1 else 0.0
                if danger_proximity > 0.3:
                    # 위험 가까울수록 민감도 최소값 상향 (0.6 → 0.9)
                    min_sensitivity = 0.6 + 0.3 * min(1.0, (danger_proximity - 0.3) / 0.7)
                    risk_sensitivity = max(min_sensitivity, risk_sensitivity)

            risk = self.preferences.compute_risk(Q_o, q_uncertainty, risk_weights) * risk_sensitivity

            # === AMBIGUITY: E_{Q(s'|a)}[H[P(o|s')]] with PRECISION weighting ===
            # FEP 정의: 전이 모델의 불확실성만 사용
            # Precision이 높으면 이 행동의 불확실성에 더 민감
            transition_std = self.transition_model['delta_std'][a]
            avg_uncertainty = np.mean(transition_std[:2])  # proximity 차원
            ambiguity_weight = self.precision_learner.get_ambiguity_weight(a)
            ambiguity = self.preferences.compute_ambiguity(avg_uncertainty) * ambiguity_weight

            # v4.3: Exploration bonus from uncertainty
            # 불확실성 높음 → ambiguity가 낮아짐 → 탐색 행동 장려
            if self.uncertainty_enabled and self._last_uncertainty_modulation is not None:
                ambiguity = max(0.0, ambiguity - self._last_uncertainty_modulation.exploration_bonus)

            # === COMPLEXITY: KL[Q(s'|a) || P(s')] ===
            # "이 행동을 하면 믿음이 선호 상태에서 얼마나 벗어날 것인가"
            # 관측 공간에서 직접 계산 (상태 공간 거치지 않음)
            complexity = self.state_preferences.compute_expected_complexity_from_obs(
                current_obs, Q_o
            )

            # === TOTAL G ===
            # G(a) = Risk + Ambiguity + Complexity
            # Risk: 선호 관측에서 벗어남 (precision 가중)
            # Ambiguity: 전이 불확실성 (precision 가중)
            # Complexity: 선호 상태에서 벗어남
            G = risk + ambiguity + self.complexity_weight * complexity

            # === v4.5.1: STAY ambiguity floor ===
            # STAY가 자주 선택되어 ambiguity가 극도로 낮아지는 것을 방지
            # "움직이지 않으면 안전하다"는 학습이 과적합되는 것을 막음
            # 최소 ambiguity를 movement와 비슷한 수준으로 유지
            if a == 0:
                min_stay_ambiguity = 0.15  # movement 평균의 약 50%
                if ambiguity < min_stay_ambiguity:
                    G = G + (min_stay_ambiguity - ambiguity)

            # === v4.0: Memory Bias ===
            # 기억이 G를 조정 (행동 직접 지시 X)
            # memory_bias < 0 → 과거 좋았던 행동 → G ↓ → 더 선택됨
            # memory_bias > 0 → 과거 나빴던 행동 → G ↑ → 덜 선택됨
            # v4.6: memory_recall_enabled로 회상 bias 적용 분리 제어
            # v4.6: drift_aware_suppression으로 급성기 recall 억제
            if self.memory_enabled and self.memory_recall_enabled and self._last_recall_result is not None:
                memory_bias = self._last_recall_result.memory_bias[a]
                # Apply drift-aware suppression
                if self.drift_aware_suppression:
                    memory_bias = memory_bias * self._recall_suppression_factor
                G = G + memory_bias

            # 최소 G 행동의 예측 관측 저장 (precision 업데이트용)
            min_G_so_far = min((r.G for r in results.values()), default=float('inf'))
            if a == 0 or G < min_G_so_far:
                self._predicted_obs = Q_o.copy()

            results[a] = GDecomposition(
                G=G,
                risk=risk,
                ambiguity=ambiguity,
                complexity=complexity,
                action_cost=0.0
            )

        return results

    def compute_G_think(self,
                        G_physical: Dict[int, GDecomposition],
                        current_obs: np.ndarray,
                        decision_entropy: float) -> GDecomposition:
        """
        v3.4: Compute G(THINK) - Expected Free Energy for deliberation action.

        핵심 원리:
        G(THINK) = E[G_after_deliberation] + deliberation_cost

        여기서:
        - E[G_after_deliberation]: rollout 후 선택할 행동의 예상 G
        - deliberation_cost: 시간이 흐르면서 발생하는 자연스러운 비용
          (energy 감소, 위험 이동 가능성)

        THINK가 유리한 경우:
        - decision_entropy가 높음 (현재 선택이 불확실)
        - G_best와 G_second_best 차이가 작음 (경쟁 중)
        - rollout이 결정을 바꿀 가능성이 높음

        THINK가 불리한 경우:
        - 명확한 최선 행동이 있음 (entropy 낮음)
        - 긴급 상황 (위험 가까움, energy 낮음)
        """
        # 물리 행동들의 G 값
        G_values = np.array([G_physical[a].G for a in range(5)])  # 0-4만

        G_best = np.min(G_values)
        best_action = int(np.argmin(G_values))
        G_sorted = np.sort(G_values)
        G_second_best = G_sorted[1] if len(G_sorted) > 1 else G_sorted[0]

        # === Expected improvement from deliberation ===
        # rollout이 결정을 바꿀 확률은 decision_entropy에 비례
        # entropy 높으면 → 현재 1-step 선택이 불확실 → rollout 효과 큼

        max_entropy = np.log(5)  # log(5) ≈ 1.61 for 5 physical actions
        entropy_ratio = decision_entropy / (max_entropy + 1e-10)

        # 예상되는 G 개선량
        # - entropy_ratio가 1에 가까우면 (불확실) → 개선 가능성 높음
        # - G_spread가 작으면 → 경쟁 중 → 신중한 선택이 중요
        G_spread = G_second_best - G_best
        potential_improvement = entropy_ratio * G_spread * 0.5

        # === Deliberation cost (환경 다이나믹스) ===
        # THINK하는 동안 시간이 흐름:
        # - energy 감소 (-0.003)
        # - 위험이 움직일 수 있음 (확률적)

        # energy 감소로 인한 risk 증가
        if current_obs is not None and len(current_obs) >= 8:
            current_energy = current_obs[6]
            energy_after_think = max(0, current_energy - self.think_energy_cost)

            # energy가 낮을수록 THINK 비용이 높아짐 (급한 상황)
            # energy_risk_increase = P(o)에서 energy 감소의 KL 증가
            energy_penalty = self.think_energy_cost * (1.0 / (current_energy + 0.1))
        else:
            energy_penalty = 0.01

        # 위험 proximity가 높을수록 THINK 비용 증가 (급한 상황)
        if current_obs is not None:
            danger_prox = current_obs[1]
            danger_urgency = danger_prox * 0.1  # 위험 가까우면 생각할 시간 없음
        else:
            danger_urgency = 0.0

        deliberation_cost = energy_penalty + danger_urgency

        # === G(THINK) = Expected G after deliberation ===
        # = G_best (현재 최선) - potential_improvement (개선 가능성) + cost (시간 비용)
        #
        # 만약 potential_improvement > cost이면 THINK가 유리

        # === v4.3: Uncertainty-based THINK bias ===
        # 불확실성이 높으면 THINK가 더 유리해짐
        think_bias = 0.0
        if self.uncertainty_enabled and self._last_uncertainty_modulation is not None:
            think_bias = self._last_uncertainty_modulation.think_bias
            # think_bias < 0 → THINK 유리, think_bias > 0 → THINK 불리

        # === v4.4: Regret-based THINK benefit boost ===
        # 누적 regret가 높으면 메타인지 가치 ↑ → potential_improvement 증가
        # v4.6: regret_modulation_enabled로 분리 제어
        think_benefit_boost = 0.0
        if self.regret_enabled and self.regret_modulation_enabled:
            regret_mod = self.get_regret_modulation()
            think_benefit_boost = regret_mod.get('think_benefit_boost', 0.0)

        # Apply regret boost to potential improvement
        effective_improvement = potential_improvement + think_benefit_boost

        G_think = G_best - effective_improvement + deliberation_cost + think_bias

        # Risk/Ambiguity/Complexity 분해 (THINK 전용)
        # - Risk: deliberation_cost (시간 비용)
        # - Ambiguity: 현재 결정 불확실성 (역수로, 불확실할수록 THINK가 줄여줌)
        # - Complexity: 0 (THINK는 믿음을 직접 바꾸지 않음)

        risk_component = deliberation_cost
        ambiguity_component = -effective_improvement  # 음수: THINK가 불확실성을 줄임

        # 저장 (디버깅용)
        self._last_expected_improvement = potential_improvement

        return GDecomposition(
            G=G_think,
            risk=risk_component,
            ambiguity=ambiguity_component,
            complexity=0.0,
            action_cost=0.0
        )

    def update_transition_model(self, action: int, obs_before: np.ndarray, obs_after: np.ndarray):
        """
        Learn transition model from experience.

        P(o'|o, a) - How does action a change observation?

        v4.2: Variance-based std update with volatility detection
        - var <- (1-β) * var + β * err^2
        - std = sqrt(var)
        - Volatility detection: 예측오차 급증 시 lr 일시 증가

        This allows std to INCREASE when prediction errors spike (drift detection).

        Args:
            action: Action taken
            obs_before: Observation before action
            obs_after: Observation after action
        """
        if len(obs_before) < 6 or len(obs_after) < 6:
            return

        # Actual change
        actual_delta = obs_after - obs_before

        # Predicted change
        predicted_delta = self.transition_model['delta_mean'][action]

        # Prediction error
        error = actual_delta - predicted_delta
        err_sq = error ** 2  # Squared error for variance

        # Update count
        self.transition_model['count'][action] += 1
        count = self.transition_model['count'][action]

        # === v4.2: Volatility-aware learning rate ===
        # Base learning rate decays with experience
        base_lr = self.transition_lr / np.sqrt(count)

        # Current std (for volatility detection)
        # IMPORTANT: Make a copy to preserve original value for debugging
        current_std = self.transition_model['delta_std'][action].copy()

        # Volatility detection: if |error| >> current_std, increase lr temporarily
        # This allows rapid adaptation when environment changes (drift)
        err_magnitude = np.abs(error)
        volatility_ratio = np.mean(err_magnitude) / (np.mean(current_std) + 1e-6)

        # If error is larger than expected, boost learning rate
        # v4.2.3: Lowered threshold to 0.4 for realistic drift detection
        # Observed: volatility_ratio ~0.5-0.6 during drift
        # For std to increase naturally: need |error| > current_std (ratio > 1.0)
        # So we need to boost learning when ratio is high but < 1.0
        if volatility_ratio > 0.4:
            lr_boost = min(5.0, 1.0 + volatility_ratio)  # 1.4x ~ 5x boost
            lr = base_lr * lr_boost
            # Also discount count (partial forgetting for change-point)
            self.transition_model['count'][action] = max(1, count * 0.8)
        else:
            lr = base_lr

        # === v4.4: Regret-based lr boost ===
        # regret + surprise = 모델 재학습 필요 → lr 증가
        # v4.6: regret_modulation_enabled로 분리 제어
        if self.regret_enabled and self.regret_modulation_enabled:
            regret_mod = self.get_regret_modulation()
            regret_lr_boost = regret_mod.get('lr_boost_factor', 1.0)
            lr = lr * regret_lr_boost

        # Ensure minimum learning rate for std updates
        # v4.2.3: Increased min from 0.05 to 0.1 for faster std adaptation
        lr_std = max(lr, 0.1)  # At least 10% update rate for std

        # Update mean (move toward actual)
        self.transition_model['delta_mean'][action] += lr * error

        # === v4.2.3: Variance-based std update with volatility boost ===
        # var <- (1-β) * var + β * err^2
        # std = sqrt(var)
        # This naturally allows std to INCREASE when errors are large
        current_var = current_std ** 2

        # v4.2.4: When volatility detected, SCALE UP errors to force std increase
        # Key insight: for std to increase, need err_sq > current_var
        # With volatility_ratio = |error|/std, if ratio < 1, err_sq < var (std decreases)
        # So we boost err_sq to ensure it exceeds current variance
        if volatility_ratio > 0.3:
            # Boost factor: ensure err_sq_boosted > current_var when volatility detected
            # err_sq_boosted = err_sq * boost, we want err_sq_boosted > std^2
            # err = volatility_ratio * std, so err^2 = volatility_ratio^2 * std^2
            # err_sq_boosted = volatility_ratio^2 * std^2 * boost > std^2
            # boost > 1/volatility_ratio^2
            # For safety, use boost = max(2.0, 1/(volatility_ratio^2))
            boost_factor = max(2.0, 1.5 / (volatility_ratio ** 2 + 0.01))
            err_sq_boosted = err_sq * boost_factor
        else:
            err_sq_boosted = err_sq

        new_var = (1 - lr_std) * current_var + lr_std * err_sq_boosted
        new_std = np.sqrt(new_var)

        self.transition_model['delta_std'][action] = new_std

        # Clip std to reasonable range (wide range to allow increase)
        self.transition_model['delta_std'][action] = np.clip(
            self.transition_model['delta_std'][action], 0.01, 2.0  # Upper limit raised
        )

        # Store last error for debugging
        std_after = float(np.mean(self.transition_model['delta_std'][action]))
        std_before = float(np.mean(current_std))
        boost_active = volatility_ratio > 0.3
        boost_factor_used = max(2.0, 1.5 / (volatility_ratio ** 2 + 0.01)) if boost_active else 1.0
        self._last_transition_error = {
            'action': action,
            'error_mean': float(np.mean(np.abs(error))),
            'error_vec': [float(e) for e in error[:4]],  # First 4 dims for debugging
            'actual_delta': [float(d) for d in actual_delta[:4]],
            'predicted_delta': [float(d) for d in predicted_delta[:4]],
            'volatility_ratio': float(volatility_ratio),
            'volatility_boost': bool(boost_active),
            'boost_factor': float(boost_factor_used),
            'lr_used': float(np.mean(lr) if hasattr(lr, '__len__') else lr),
            'lr_std_used': float(lr_std),
            'std_before': std_before,
            'std_after': std_after,
            'std_change_pct': float((std_after - std_before) / (std_before + 1e-6) * 100),
        }

    def update_from_replay(self, action: int, obs_delta: np.ndarray, learning_rate: float = 0.1):
        """
        Update transition model from memory replay during consolidation.

        v4.1: Sleep consolidation에서 호출
        - 저장된 에피소드의 delta를 사용하여 transition model 재학습
        - 반복된 패턴은 delta_std를 줄여 ambiguity 감소

        Args:
            action: Action that was taken
            obs_delta: Observed change in observations [delta_energy, delta_pain, ...]
            learning_rate: Learning rate for consolidation (usually lower than online learning)
        """
        if action < 0 or action >= self.n_actions:
            return

        # Ensure obs_delta matches transition model dimensions
        obs_dim = self.transition_model['delta_mean'].shape[1]
        if len(obs_delta) < obs_dim:
            # Pad with zeros
            padded = np.zeros(obs_dim)
            padded[:len(obs_delta)] = obs_delta
            obs_delta = padded
        elif len(obs_delta) > obs_dim:
            obs_delta = obs_delta[:obs_dim]

        # Current prediction
        predicted_delta = self.transition_model['delta_mean'][action]
        current_std = self.transition_model['delta_std'][action]

        # Update mean toward replayed value
        error = obs_delta - predicted_delta
        self.transition_model['delta_mean'][action] += learning_rate * error

        # Reduce std (consolidation should increase confidence)
        # Key insight: repeated patterns → lower uncertainty
        error_magnitude = np.abs(error)

        # If prediction was close, reduce std
        # If prediction was off, increase std slightly
        std_update = learning_rate * (error_magnitude - current_std)
        new_std = current_std + 0.5 * std_update  # Damped update

        # Consolidation should generally reduce std (increase confidence)
        # Apply a small reduction bias
        new_std = new_std * 0.98  # 2% reduction per replay

        self.transition_model['delta_std'][action] = np.clip(new_std, 0.01, 1.0)

    def select_action(self,
                      Q_s: Optional[np.ndarray] = None,
                      current_obs: Optional[np.ndarray] = None,
                      deterministic: bool = False
                      ) -> ActionResult:
        """
        Select action by minimizing G.

        v2.4.3: Adaptive rollout with enhanced decision uncertainty
        - decision_entropy = H(softmax(-G)) - 스케일 불변
        - 분위수 기반 threshold (상대 기준)
        - ambiguity 높으면 spread 커도 rollout
        - rollout이 선택을 바꿨는지 로그

        If use_rollout is True (always-on), uses multi-step temporal planning.
        If adaptive_rollout is True, uses decision_entropy to decide when to rollout.
        """
        if Q_s is None:
            Q_s = self.model.Q_s

        # Reset THINK state
        self._last_think_selected = False
        self._last_think_reason = ""
        self._physical_action_after_think = None

        # === STEP 1: Compute 1-step G for PHYSICAL actions (0-4) ===
        G_all = self.compute_G(Q_s, current_obs)
        n_physical = 5  # stay, up, down, left, right
        G_values_physical = np.array([G_all[a].G for a in range(n_physical)])
        action_1step_physical = int(np.argmin(G_values_physical))
        self._last_1step_action = action_1step_physical

        # v4.4: Store G_pred for counterfactual (regret) calculation
        if self.regret_enabled:
            self._last_G_pred = {a: G_all[a].G for a in range(n_physical)}

        # === STEP 2: Compute decision uncertainty for physical actions ===
        sorted_G = np.sort(G_values_physical)
        G_best = sorted_G[0]
        G_second_best = sorted_G[1] if len(sorted_G) > 1 else sorted_G[0]
        G_spread = G_second_best - G_best
        self._last_G_spread = G_spread

        # Softmax probabilities and entropy (for physical actions)
        log_probs = -G_values_physical / self.temperature
        log_probs = log_probs - np.max(log_probs)
        probs = np.exp(log_probs)
        probs = probs / (probs.sum() + 1e-10)

        p_best = np.max(probs)
        self._last_p_best = p_best

        # Decision entropy: H = -Σ p log p
        decision_entropy = -np.sum(probs * np.log(probs + 1e-10))
        self._last_decision_entropy = decision_entropy

        max_ambiguity = max(G_all[a].ambiguity for a in range(n_physical))
        self._last_max_ambiguity = max_ambiguity

        # === STEP 2.5: v3.4.1 THINK action (최적화됨) ===
        # 2단계 게이트: 쿨다운 → entropy/G_spread 체크 → G(THINK) 계산
        self._think_gate_passed = False

        if self.think_enabled and current_obs is not None:
            # 1) 쿨다운 체크
            if self._think_cooldown_counter > 0:
                self._think_cooldown_counter -= 1
                # 쿨다운 중이면 THINK 평가 스킵
            else:
                # 2) 게이트 체크: 정말 애매할 때만 THINK 평가
                entropy_gate = decision_entropy >= self.think_entropy_threshold
                spread_gate = G_spread <= self.think_G_spread_threshold

                if entropy_gate or spread_gate:
                    self._think_gate_passed = True

                    # 3) 게이트 통과 → G(THINK) 계산
                    G_think = self.compute_G_think(G_all, current_obs, decision_entropy)
                    G_all[self.THINK_ACTION] = G_think

                    # THINK가 최선인지 확인
                    if G_think.G < G_best:
                        self._last_think_selected = True
                        self._last_think_reason = (
                            f"gate={'entropy' if entropy_gate else 'spread'}, "
                            f"entropy={decision_entropy:.2f}, spread={G_spread:.3f}, "
                            f"G_think={G_think.G:.3f} < G_best={G_best:.3f}"
                        )
                        self._think_count += 1
                        # 쿨다운 시작
                        self._think_cooldown_counter = self.think_cooldown

        # === STEP 3: Decide whether to rollout ===
        should_rollout = False
        self._rollout_triggered = False
        self._last_action_changed = False
        self._last_change_reason = ""

        # v3.4: THINK가 선택되면 무조건 rollout
        if self._last_think_selected:
            should_rollout = True
            self._rollout_triggered = True
        elif self.use_rollout:
            # Always-on mode
            should_rollout = True
        elif self.adaptive_rollout and current_obs is not None:
            # Update entropy history
            self._entropy_history.append(decision_entropy)
            if len(self._entropy_history) > self._entropy_history_size:
                self._entropy_history.pop(0)

            # Compute threshold from history (quantile-based)
            if len(self._entropy_history) >= 10:
                # rollout_quantile=0.3 means "top 30% entropy → rollout"
                # So threshold is at (1 - quantile) percentile
                threshold = np.percentile(self._entropy_history, (1 - self.rollout_quantile) * 100)
            else:
                # Bootstrap phase: use fixed threshold based on max entropy
                # Max entropy for 5 actions = log(5) ≈ 1.61
                threshold = 1.2  # ~75% of max entropy

            # Condition 1: High decision entropy (애매한 선택)
            entropy_trigger = decision_entropy >= threshold

            # Condition 2: High ambiguity (전이 불확실성)
            # 상위 20% ambiguity면 spread 커도 rollout
            ambiguity_trigger = max_ambiguity > 0.4  # 기본 ambiguity가 0.3 근처

            if entropy_trigger or ambiguity_trigger:
                should_rollout = True
                self._rollout_triggered = True

        # === STEP 4: Execute rollout if needed ===
        if should_rollout and current_obs is not None:
            # v3.4.1: THINK용 rollout은 예산 제한
            if self._last_think_selected:
                use_horizon = self.think_rollout_horizon
                use_samples = self.think_rollout_samples
            else:
                use_horizon = self.rollout_horizon
                use_samples = self.rollout_n_samples

            # Lazy initialization of temporal planner
            if self.temporal_planner is None:
                self.temporal_planner = TemporalPlanner(
                    self,
                    horizon=use_horizon,
                    discount=self.rollout_discount,
                    n_samples=use_samples,
                    complexity_decay=self.rollout_complexity_decay
                )
            else:
                self.temporal_planner.set_horizon(use_horizon)
                self.temporal_planner.set_discount(self.rollout_discount)
                self.temporal_planner.n_samples = use_samples
                self.temporal_planner.complexity_decay = self.rollout_complexity_decay

            # Rollout 기반 행동 선택 (물리 행동만)
            rollout_action, rollout_results = self.temporal_planner.select_action_with_rollout(current_obs)
            self._last_rollout_results = rollout_results
            self._last_rollout_action = rollout_action

            # Rollout에서 얻은 total_G를 사용 (물리 행동만)
            G_values = np.array([rollout_results[a].total_G for a in range(n_physical)])

            # === v3.4: THINK 선택 처리 ===
            if self._last_think_selected:
                # THINK가 선택됨: 환경에 THINK를 반환
                # rollout으로 찾은 물리 행동은 저장 (환경이 다음에 사용)
                action = self.THINK_ACTION
                self._physical_action_after_think = rollout_action
                self._last_change_reason = (
                    f"THINK selected, deliberation chose action {rollout_action}"
                )
                self._last_action_changed = (rollout_action != action_1step_physical)
            else:
                # 일반 rollout (THINK 없이)
                action = rollout_action
                if action != action_1step_physical:
                    self._last_action_changed = True
                    G_1step = G_all[action_1step_physical]
                    G_rollout_chosen = rollout_results[action]
                    G_rollout_original = rollout_results[action_1step_physical]

                    if G_rollout_original.total_G > G_rollout_chosen.total_G:
                        self._last_change_reason = f"rollout: {action_1step_physical}→{action} (future risk/ambiguity)"
                    else:
                        self._last_change_reason = f"rollout: {action_1step_physical}→{action} (marginal)"

        else:
            # 1-step 방식 사용
            G_values = G_values_physical
            action = action_1step_physical
            self._last_rollout_results = None

        # Action probabilities: P(a) ∝ exp(-G(a) / temperature)
        log_probs = -G_values / self.temperature
        log_probs = log_probs - np.max(log_probs)
        probs = np.exp(log_probs)
        probs = probs / probs.sum()

        # Deterministic selection
        if not self.use_rollout:  # Rollout 사용 시 이미 action 선택됨
            action = int(np.argmin(G_values))

        # Track action for novelty
        self._action_history.append(action)
        if len(self._action_history) > 20:
            self._action_history.pop(0)
        self._last_action = action

        # Generate explanations
        why_risk = self._explain_risk(G_all)
        why_ambiguity = self._explain_ambiguity(G_all)

        return ActionResult(
            action=action,
            G_all=G_all,
            probabilities=probs,
            why_risk=why_risk,
            why_ambiguity=why_ambiguity
        )

    def _explain_risk(self, G_all: Dict[int, GDecomposition]) -> str:
        """Explain which actions have high risk (preference violation)."""
        risks = [(a, G_all[a].risk) for a in G_all.keys()]
        risks.sort(key=lambda x: x[1], reverse=True)

        high_risk = risks[0]
        low_risk = risks[-1]

        if high_risk[1] > 0.5:
            return f"Action {high_risk[0]} has high risk ({high_risk[1]:.2f}) - avoiding"
        else:
            return "All actions have similar risk"

    def _explain_ambiguity(self, G_all: Dict[int, GDecomposition]) -> str:
        """Explain ambiguity differences."""
        ambiguities = [(a, G_all[a].ambiguity) for a in G_all.keys()]
        ambiguities.sort(key=lambda x: x[1])

        low_ambig = ambiguities[0]
        high_ambig = ambiguities[-1]

        diff = high_ambig[1] - low_ambig[1]
        if diff > 0.3:
            return f"Action {low_ambig[0]} reduces uncertainty most"
        else:
            return "Similar ambiguity across actions"

    def _explain_complexity(self, G_all: Dict[int, GDecomposition]) -> str:
        """Explain complexity differences."""
        complexities = [(a, G_all[a].complexity) for a in G_all.keys()]
        complexities.sort(key=lambda x: x[1])

        low_complex = complexities[0]
        high_complex = complexities[-1]

        diff = high_complex[1] - low_complex[1]
        if diff > 0.2:
            return f"Action {low_complex[0]} keeps belief close to preferred states"
        else:
            return "Similar complexity across actions"

    def get_detailed_log(self, result: ActionResult) -> Dict:
        """
        Get detailed decomposition log.

        This is the key output that shows WHY behavior happens
        without using emotion labels.
        """
        action = result.action
        G = result.G_all[action]

        # Determine dominant factor (now includes complexity)
        factors = {'risk': G.risk, 'ambiguity': G.ambiguity, 'complexity': G.complexity}
        max_factor = max(factors, key=factors.get)

        if max_factor == 'risk' and G.risk > G.ambiguity * 1.5 and G.risk > G.complexity * 1.5:
            dominant = "risk_avoidance"  # Observer might call this "fear response"
        elif max_factor == 'ambiguity' and G.ambiguity > G.risk * 1.5 and G.ambiguity > G.complexity * 1.5:
            dominant = "ambiguity_reduction"  # Observer might call this "curiosity"
        elif max_factor == 'complexity' and G.complexity > G.risk * 1.5 and G.complexity > G.ambiguity * 1.5:
            dominant = "complexity_avoidance"  # Observer might call this "cognitive conservatism"
        else:
            dominant = "balanced"

        # Find best/worst actions by each component
        actions_in_result = list(result.G_all.keys())
        min_risk_action = min(actions_in_result, key=lambda a: result.G_all[a].risk)
        min_ambig_action = min(actions_in_result, key=lambda a: result.G_all[a].ambiguity)
        min_complexity_action = min(actions_in_result, key=lambda a: result.G_all[a].complexity)

        return {
            # The decomposition that explains everything
            'selected_action': action,
            'G': G.G,
            'risk': G.risk,           # High = avoiding preference violation
            'ambiguity': G.ambiguity, # High = uncertain observations
            'complexity': G.complexity,  # High = belief diverges from preferred states
            'action_cost': G.action_cost,

            # What's driving behavior
            'dominant_factor': dominant,

            # Comparison
            'min_risk_action': min_risk_action,
            'min_ambiguity_action': min_ambig_action,
            'min_complexity_action': min_complexity_action,

            # All actions for visualization
            'all_G': {a: result.G_all[a].G for a in result.G_all.keys()},
            'all_risk': {a: result.G_all[a].risk for a in result.G_all.keys()},
            'all_ambiguity': {a: result.G_all[a].ambiguity for a in result.G_all.keys()},
            'all_complexity': {a: result.G_all[a].complexity for a in result.G_all.keys()},

            # Natural language (for observers only)
            'why_risk': result.why_risk,
            'why_ambiguity': result.why_ambiguity,

            # Mapping to what observers might say
            '_observer_note': {
                'risk_avoidance': "looks like fear/avoidance",
                'ambiguity_reduction': "looks like curiosity/exploration",
                'complexity_avoidance': "looks like cognitive conservatism",
                'balanced': "no dominant drive visible"
            }
        }

    def update_precision(self, actual_obs: np.ndarray, action: int):
        """
        Precision 업데이트 (관측 후).

        Args:
            actual_obs: 실제 관측된 값
            action: 수행한 행동
        """
        if self._predicted_obs is not None and len(actual_obs) >= 6:
            self.precision_learner.update(
                predicted_obs=self._predicted_obs,
                actual_obs=actual_obs,
                action=action
            )

    def get_precision_state(self) -> PrecisionState:
        """현재 precision 상태 반환"""
        return self.precision_learner.get_state()

    def get_precision_interpretation(self) -> Dict:
        """Precision 상태의 인간 친화적 해석"""
        return self.precision_learner.interpret_state()

    def reset_transition_model(self):
        """Reset the learned transition model to initial state."""
        self.transition_model = {
            'delta_mean': np.zeros((self.n_actions, 8)),  # v2.5: 8차원
            'delta_std': np.ones((self.n_actions, 8)) * 0.2,
            'count': np.ones((self.n_actions,))
        }
        # Internal states (energy, pain)는 높은 불확실성으로 시작
        self.transition_model['delta_std'][:, 6:8] = 0.5
        self._last_obs = None
        self._action_history = []
        self._last_transition_error = None  # v4.2: Volatility tracking
        self.precision_learner.reset()  # Precision도 리셋

    def set_temperature(self, temp: float):
        """Higher = more random, lower = more deterministic."""
        self.temperature = max(0.01, temp)

    def adapt_temperature(self, F_current: float, F_baseline: float = 1.0):
        """Adapt temperature based on current F."""
        if F_current > F_baseline * 2.0:
            # High stress → explore more
            self.temperature = min(3.0, self.temperature * 1.05)
        elif F_current < F_baseline * 0.3:
            # Low stress → exploit more
            self.temperature = max(0.2, self.temperature * 0.95)

    # === TEMPORAL DEPTH CONTROLS ===

    def enable_rollout(self, horizon: int = 3, discount: float = 0.9,
                       n_samples: int = 3, complexity_decay: float = 0.7):
        """
        Multi-step rollout 활성화 (Monte Carlo + Complexity Decay)

        Args:
            horizon: 몇 스텝 앞까지 볼 것인가 (1-5 권장)
            discount: 미래 가치 할인율 (0.8-0.99)
            n_samples: Monte Carlo 샘플 수 (2-5 권장)
            complexity_decay: 미래 complexity 감쇠율 (0.5-0.9)
                - 1.0 = 모든 스텝 동일 가중치
                - 0.7 = step t에서 complexity *= 0.7^t
        """
        self.use_rollout = True
        self.rollout_horizon = max(1, min(10, horizon))
        self.rollout_discount = max(0.5, min(0.99, discount))
        self.rollout_n_samples = max(1, min(10, n_samples))
        self.rollout_complexity_decay = max(0.3, min(1.0, complexity_decay))
        self.temporal_planner = None  # Reset to apply new params

    def disable_rollout(self):
        """1-step 모드로 복귀"""
        self.use_rollout = False
        self.adaptive_rollout = False
        self._last_rollout_results = None

    # === v3.4: THINK ACTION CONTROLS ===

    def enable_think(self, energy_cost: float = 0.003):
        """
        THINK action 활성화 (v3.4 Metacognition)

        THINK가 활성화되면:
        - 에이전트가 "생각할지 말지"를 G로 선택
        - THINK 선택 시 rollout 실행 → 더 나은 행동 발견
        - 비용: 시간이 흐름 (energy 감소)

        Args:
            energy_cost: THINK 시 energy 감소량 (기본 0.003, 환경과 동일)
        """
        self.think_enabled = True
        self.think_energy_cost = energy_cost
        self._think_count = 0

        # THINK를 위한 rollout 파라미터 설정 (없으면 기본값)
        if self.temporal_planner is None:
            self.rollout_horizon = 3
            self.rollout_discount = 0.9
            self.rollout_n_samples = 3
            self.rollout_complexity_decay = 0.7

    def disable_think(self):
        """THINK action 비활성화"""
        self.think_enabled = False

    def get_think_status(self) -> Dict:
        """THINK 상태 조회 (v3.4.1 최적화 정보 포함)"""
        return {
            'enabled': self.think_enabled,
            'think_count': self._think_count,
            'last_think_selected': self._last_think_selected,
            'last_think_reason': self._last_think_reason,
            'last_expected_improvement': self._last_expected_improvement,
            'physical_action_after_think': self._physical_action_after_think,
            'energy_cost': self.think_energy_cost,
            # v3.4.1 최적화 정보
            'gate_passed': self._think_gate_passed,
            'cooldown_remaining': self._think_cooldown_counter,
            'entropy_threshold': self.think_entropy_threshold,
            'G_spread_threshold': self.think_G_spread_threshold,
            'rollout_horizon': self.think_rollout_horizon,
            'rollout_samples': self.think_rollout_samples,
            'cooldown': self.think_cooldown,
        }

    def get_physical_action_after_think(self) -> Optional[int]:
        """
        THINK 후 실행할 물리 행동 반환.
        환경이 THINK를 처리한 후 이 행동을 실행함.
        """
        return self._physical_action_after_think

    def enable_adaptive_rollout(self, quantile: float = 0.3,
                                 horizon: int = 3, discount: float = 0.9,
                                 n_samples: int = 3, complexity_decay: float = 0.7):
        """
        Adaptive rollout 활성화 (v2.4.3)

        decision_entropy 기반 + 분위수 threshold + ambiguity 보완.
        "결정이 애매하거나 전이가 불확실할 때만 깊이 생각"

        Args:
            quantile: rollout 비율 (0.0-1.0)
                - 0.3 = 상위 30% 불확실한 상황에서만 rollout
                - 0.0 = 항상 rollout
                - 1.0 = 절대 안함
            horizon, discount, n_samples, complexity_decay: rollout 파라미터
        """
        self.use_rollout = False  # always-on 비활성화
        self.adaptive_rollout = True
        self.rollout_quantile = max(0.0, min(1.0, quantile))
        self.rollout_horizon = max(1, min(10, horizon))
        self.rollout_discount = max(0.5, min(0.99, discount))
        self.rollout_n_samples = max(1, min(10, n_samples))
        self.rollout_complexity_decay = max(0.3, min(1.0, complexity_decay))
        self._entropy_history = []  # Reset history
        self.temporal_planner = None  # Reset to apply new params

    def get_rollout_info(self) -> Optional[Dict]:
        """마지막 rollout 결과 요약 반환"""
        if self._last_rollout_results is None:
            return None

        if self.temporal_planner is None:
            return None

        return self.temporal_planner.get_rollout_summary(self._last_rollout_results)

    def get_temporal_config(self) -> Dict:
        """현재 temporal depth 설정 반환 (v2.4.3)"""
        return {
            # 모드 설정
            'enabled': self.use_rollout,
            'adaptive': self.adaptive_rollout,
            'quantile': self.rollout_quantile,
            'horizon': self.rollout_horizon,
            'discount': self.rollout_discount,
            'n_samples': self.rollout_n_samples,
            'complexity_decay': self.rollout_complexity_decay,
            'rollout_count': self.temporal_planner.rollout_count if self.temporal_planner else 0,

            # 결정 불확실성 측정 (v2.4.3)
            'last_G_spread': self._last_G_spread,
            'last_decision_entropy': self._last_decision_entropy,
            'last_p_best': self._last_p_best,
            'last_max_ambiguity': self._last_max_ambiguity,

            # rollout 트리거 정보
            'last_rollout_triggered': self._rollout_triggered,
            'entropy_history_size': len(self._entropy_history),

            # 선택 변경 로그
            'last_1step_action': self._last_1step_action,
            'last_rollout_action': self._last_rollout_action,
            'last_action_changed': self._last_action_changed,
            'last_change_reason': self._last_change_reason
        }

    # === HIERARCHICAL MODELS CONTROLS (v3.0) ===

    def enable_hierarchy(self,
                         K: int = 4,
                         update_interval: int = 10,
                         transition_self: float = 0.95,
                         ema_alpha: float = 0.1):
        """
        계층적 처리 활성화 (v3.0)

        Slow layer가 fast layer의 precision을 조절.

        Args:
            K: Context 수 (추천: 4)
            update_interval: Slow layer 업데이트 주기 (step)
            transition_self: P(c_t = c_{t-1}) - 자기 유지 확률
            ema_alpha: 관측 통계 EMA alpha
        """
        self.hierarchy_controller = HierarchicalController(
            K=K,
            update_interval=update_interval,
            transition_self=transition_self,
            ema_alpha=ema_alpha,
        )

    def disable_hierarchy(self):
        """계층적 처리 비활성화"""
        self.hierarchy_controller = None

    def update_hierarchy(self, pred_error: float, ambiguity: float,
                         complexity: float, observation: np.ndarray,
                         action: int = None):
        """
        Slow layer 업데이트 (매 step 호출)

        Args:
            pred_error: Fast layer의 prediction error
            ambiguity: Fast layer의 ambiguity
            complexity: Fast layer의 complexity
            observation: 현재 관측 (8차원)
            action: 직전 행동 (v3.2: 전이 학습용)
        """
        if self.hierarchy_controller is not None:
            self.hierarchy_controller.update(
                pred_error, ambiguity, complexity, observation, action
            )

    def get_hierarchy_status(self) -> Dict:
        """현재 계층 상태 반환"""
        if self.hierarchy_controller is None:
            return {'enabled': False}

        state = self.hierarchy_controller.get_state()
        if state is None:
            return {'enabled': True, 'initialized': False}

        from .hierarchy import infer_post_hoc_label

        # Get modulation and convert numpy arrays to lists for JSON serialization
        mod = self.hierarchy_controller.get_modulation()
        mod_serializable = {
            'goal_precision_mult': mod['goal_precision_mult'],
            'sensory_mod': mod['sensory_mod'].tolist() if hasattr(mod['sensory_mod'], 'tolist') else mod['sensory_mod'],
            'internal_pref_weight': mod['internal_pref_weight'],
            'rollout_budget': mod['rollout_budget'],
        }

        # v3.1: context_expectations 학습 상태도 포함
        context_exp = self.hierarchy_controller.inference.context_expectations.tolist()

        # v3.2: context별 전이 예측 오차
        context_trans_err = self.hierarchy_controller.inference.context_transition_error.tolist()

        return {
            'enabled': True,
            'initialized': True,
            'step_count': state.step_count,
            'last_update_step': state.last_update_step,
            'context_belief': state.Q_context.tolist(),
            'dominant_context': state.dominant_context,
            'dwell_time': state.dwell_time,
            'slow_F': state.slow_F,
            'modulation': mod_serializable,
            'observer_label': infer_post_hoc_label(state),
            'obs_stats': state.obs_stats,
            'context_expectations': context_exp,  # v3.1: 학습된 기대값
            'context_transition_error': context_trans_err,  # v3.2: 전이 예측 오차
            # v3.3.1: 안정화된 blending 설정
            'alpha_external': self.context_transition_alpha_external,
            'alpha_internal': self.context_transition_alpha_internal,
            'delta_ctx_clamp': self.delta_ctx_clamp,
            'use_confidence_alpha': self.use_confidence_alpha,
            'delta_debug': self._last_delta_debug,
        }

    # === TRUE FEP v3.5: ONLINE PREFERENCE LEARNING ===

    def enable_preference_learning(self,
                                   mode_lr: float = 0.02,
                                   concentration_lr: float = 0.01):
        """
        온라인 선호 학습 활성화 (v3.5)

        내부 선호(energy, pain)의 Beta 파라미터를 경험에서 학습.

        핵심 원리:
        - G가 낮았으면 (좋은 결과) → 현재 내부 상태를 더 선호하도록 mode 이동
        - G가 높았으면 (나쁜 결과) → 현재 내부 상태에서 멀어지도록 mode 이동
        - 일관된 경험 → concentration 증가 (확신)
        - 불일치 경험 → concentration 감소 (불확실)

        Args:
            mode_lr: mode 학습률 (기본 0.02)
            concentration_lr: concentration 학습률 (기본 0.01)
        """
        self.preference_learner = PreferenceLearner(
            mode_lr=mode_lr,
            concentration_lr=concentration_lr
        )
        self.preference_learning_enabled = True

    def disable_preference_learning(self):
        """온라인 선호 학습 비활성화"""
        self.preference_learning_enabled = False
        # preference_learner는 유지 (상태 보존)

    def update_preference_learning(self,
                                   current_obs: np.ndarray,
                                   G_value: float,
                                   prediction_error: float = 0.0):
        """
        선호 분포 업데이트 (매 step 호출)

        Args:
            current_obs: 현재 관측 (8차원)
            G_value: 현재 스텝의 G 값
            prediction_error: 예측 오차
        """
        if self.preference_learner is not None and self.preference_learning_enabled:
            self.preference_learner.update(current_obs, G_value, prediction_error)

            # 학습된 선호를 PreferenceDistributions에 반영
            self._apply_learned_preferences()

    def _apply_learned_preferences(self):
        """
        학습된 Beta 파라미터를 PreferenceDistributions에 적용.

        energy_pref와 pain_pref만 업데이트 (내부 선호).
        """
        if self.preference_learner is None:
            return

        # 학습된 energy 선호 적용
        learned_energy = self.preference_learner.get_energy_beta()
        self.preferences.energy_pref = learned_energy

        # 학습된 pain 선호 적용
        learned_pain = self.preference_learner.get_pain_beta()
        self.preferences.pain_pref = learned_pain

    def get_preference_learning_status(self) -> Dict:
        """현재 선호 학습 상태 반환"""
        if self.preference_learner is None:
            return {
                'enabled': False,
                'initialized': False
            }

        status = self.preference_learner.get_status()
        status['enabled'] = self.preference_learning_enabled
        status['initialized'] = True

        # 현재 적용된 선호 분포 정보 추가
        status['applied_to_risk'] = {
            'energy_alpha': self.preferences.energy_pref.alpha,
            'energy_beta': self.preferences.energy_pref.beta,
            'pain_alpha': self.preferences.pain_pref.alpha,
            'pain_beta': self.preferences.pain_pref.beta,
        }

        return status

    def reset_preference_learning(self):
        """선호 학습 리셋 (초기값으로)"""
        if self.preference_learner is not None:
            self.preference_learner.reset()
            self._apply_learned_preferences()

    # === TRUE FEP v4.3: UNCERTAINTY/CONFIDENCE CONTROLS ===

    def enable_uncertainty(self,
                           belief_weight: float = 0.25,
                           action_weight: float = 0.30,
                           model_weight: float = 0.20,
                           surprise_weight: float = 0.25,
                           sensitivity: float = 1.0):
        """
        불확실성 추적 활성화 (v4.3)

        불확실성이 "자동으로" 행동을 바꾸게 만듦:
        - THINK 선택 확률/비용 연동
        - Precision 메타-조절
        - 탐색/회피 균형
        - 기억 저장 게이트 (v4.0 준비)

        Args:
            belief_weight: context belief 엔트로피 가중치
            action_weight: action 선택 엔트로피 가중치
            model_weight: 전이 모델 불확실성 가중치
            surprise_weight: 예측 오차 가중치
            sensitivity: 조절 민감도 (0.0~2.0)
        """
        self.uncertainty_tracker = UncertaintyTracker(
            belief_weight=belief_weight,
            action_weight=action_weight,
            model_weight=model_weight,
            surprise_weight=surprise_weight,
            modulation_sensitivity=sensitivity
        )
        self.uncertainty_enabled = True

    def disable_uncertainty(self):
        """불확실성 추적 비활성화"""
        self.uncertainty_enabled = False
        # tracker는 유지 (상태 보존)

    def update_uncertainty(self,
                           prediction_error: float = 0.0,
                           transition_std: float = 0.2):
        """
        불확실성 업데이트 (매 step 호출)

        Args:
            prediction_error: 예측 오차
            transition_std: 평균 전이 불확실성
        """
        if self.uncertainty_tracker is None or not self.uncertainty_enabled:
            return

        # 1. Context entropy (hierarchy 있으면)
        context_entropy = None
        if self.hierarchy_controller is not None:
            Q_ctx = self.hierarchy_controller.get_context_belief()
            if Q_ctx is not None:
                context_entropy = compute_context_entropy(Q_ctx)

        # 2. Decision entropy (이미 계산됨)
        decision_entropy = self._last_decision_entropy

        # 3. Update tracker
        self._last_uncertainty_state = self.uncertainty_tracker.update(
            decision_entropy=decision_entropy,
            context_entropy=context_entropy,
            transition_std=transition_std,
            prediction_error=prediction_error
        )

        # 4. Get modulation (다음 action selection에 사용)
        self._last_uncertainty_modulation = self.uncertainty_tracker.get_modulation()

    def get_uncertainty_status(self) -> Dict:
        """현재 불확실성 상태 반환"""
        if self.uncertainty_tracker is None:
            return {
                'enabled': False,
                'initialized': False
            }

        status = self.uncertainty_tracker.get_status()
        status['enabled'] = self.uncertainty_enabled
        status['initialized'] = True

        # 마지막 상태 추가
        if self._last_uncertainty_state is not None:
            status['last_state'] = self._last_uncertainty_state.to_dict()
        if self._last_uncertainty_modulation is not None:
            status['last_modulation'] = self._last_uncertainty_modulation.to_dict()

        return status

    def reset_uncertainty(self):
        """불확실성 상태 리셋"""
        if self.uncertainty_tracker is not None:
            self.uncertainty_tracker.reset()
            self._last_uncertainty_state = None
            self._last_uncertainty_modulation = None

    def get_memory_gate(self) -> float:
        """
        기억 저장 게이트 값 반환 (v4.0 Memory 준비용)

        0.0 = 저장 안 함 (안정적, 반복적)
        1.0 = 강하게 저장 (놀라움, 불확실)

        v4.4: regret-based memory_gate_boost 추가
        - regret 큰 사건 = 저장 우선순위 ↑
        """
        base_gate = 0.5  # 기본값
        if self._last_uncertainty_modulation is not None:
            base_gate = self._last_uncertainty_modulation.memory_gate

        # v4.4: Regret boost
        # v4.6: regret_modulation_enabled로 분리 제어
        if self.regret_enabled and self.regret_modulation_enabled:
            regret_mod = self.get_regret_modulation()
            memory_gate_boost = regret_mod.get('memory_gate_boost', 0.0)
            # Clamp to [0, 1]
            return min(1.0, base_gate + memory_gate_boost)

        return base_gate

    # === TRUE FEP v4.0: LONG-TERM MEMORY CONTROLS ===

    def enable_memory(self,
                     max_episodes: int = 1000,
                     store_threshold: float = 0.5,
                     store_sharpness: float = 5.0,
                     similarity_threshold: float = 0.95,
                     recall_top_k: int = 5,
                     store_enabled: bool = True,
                     recall_enabled: bool = True):
        """
        장기 기억 활성화 (v4.0)

        기억 = 미래 F/G를 줄이는 압축 모델
        - 저장: memory_gate > threshold일 때 확률적 저장
        - 회상: 유사 상황 검색 → G bias (행동 직접 지시 X)
        - 불확실할 때 기억에 더 의존

        Args:
            max_episodes: 최대 저장 에피소드 수
            store_threshold: 저장 임계값 (0.5 = memory_gate 50%에서 50% 확률)
            store_sharpness: sigmoid 기울기 (클수록 임계값 근처에서 급격히 변함)
            similarity_threshold: 중복 억제 유사도 (0.95 = 95% 유사하면 병합)
            recall_top_k: 회상 시 상위 k개 에피소드 사용
            store_enabled: v4.6 분해 실험 - 저장 허용 여부
            recall_enabled: v4.6 분해 실험 - 회상 bias 적용 여부
        """
        self.ltm_store = LTMStore(
            max_episodes=max_episodes,
            store_threshold=store_threshold,
            store_sharpness=store_sharpness,
            similarity_threshold=similarity_threshold,
            recall_top_k=recall_top_k,
            n_actions=self.n_actions
        )
        self.memory_enabled = True
        # v4.6: 분해 실험용 세분화 제어
        self.memory_store_enabled = store_enabled
        self.memory_recall_enabled = recall_enabled

    def set_memory_mode(self, store_enabled: bool = True, recall_enabled: bool = True):
        """
        v4.6: 메모리 분해 실험 모드 설정

        - store_only: store_enabled=True, recall_enabled=False
        - recall_only: store_enabled=False, recall_enabled=True
        - full: store_enabled=True, recall_enabled=True
        """
        self.memory_store_enabled = store_enabled
        self.memory_recall_enabled = recall_enabled

    def enable_drift_suppression(self, spike_threshold: float = 2.0, recovery_rate: float = 0.05,
                                   use_regret: bool = True):
        """
        v4.6: Drift-aware Recall Suppression 활성화

        예측 오차가 급증하면 (drift 감지) recall weight를 억제.
        시간이 지나면 점진적으로 회복.

        v4.6.2: Regret spike 결합
        - transition error spike + regret spike → 더 강한/빠른 억제
        - regret은 "회상 강화"가 아니라 "억제 보조 신호"로 사용

        Args:
            spike_threshold: baseline 대비 몇 배면 spike로 간주 (기본 2.0)
            recovery_rate: 매 step 회복률 (기본 0.05)
            use_regret: regret spike를 억제 트리거로 사용할지 (기본 True)
        """
        self.drift_aware_suppression = True
        self._spike_threshold = spike_threshold
        self._suppression_recovery_rate = recovery_rate
        self._recall_suppression_factor = 1.0
        self._use_regret_for_suppression = use_regret
        self._regret_spike_boost = 0.0  # regret spike로 인한 추가 억제

    def disable_drift_suppression(self):
        """Drift-aware suppression 비활성화"""
        self.drift_aware_suppression = False
        self._recall_suppression_factor = 1.0
        self._regret_spike_boost = 0.0

    def update_drift_suppression(self, prediction_error: float = None, regret_spike: bool = False):
        """
        v4.6: 예측 오차 기반 suppression factor 업데이트

        호출 시점: 매 step, 관측 후

        v4.6.1 변경: transition model error 사용
        - state.prediction_error는 연속 관측과 호환 안됨 (항상 0)
        - _last_transition_error['error_mean']이 실제 drift 감지에 적합

        v4.6.2 변경: Regret spike 결합
        - transition error spike + regret spike → 더 강한 억제
        - regret spike만 있으면 약한 추가 억제

        Args:
            prediction_error: 무시됨 (하위 호환성)
            regret_spike: regret이 spike인지 (v4.6.2)
        """
        if not self.drift_aware_suppression:
            return

        # v4.6.1: Use transition model error instead
        # This accurately reflects drift (intended action vs actual outcome mismatch)
        if self._last_transition_error is None:
            return

        trans_error = self._last_transition_error.get('error_mean', 0.0)

        # Update EMA baseline (slow) - tracks normal error level
        alpha_baseline = 0.02
        self._prediction_error_baseline = (
            (1 - alpha_baseline) * self._prediction_error_baseline +
            alpha_baseline * trans_error
        )

        # Update current EMA (fast) - responds quickly to spikes
        alpha_current = 0.3
        self._prediction_error_ema = (
            (1 - alpha_current) * self._prediction_error_ema +
            alpha_current * trans_error
        )

        # Detect spike: current >> baseline
        spike_ratio = self._prediction_error_ema / max(0.01, self._prediction_error_baseline)
        trans_spike = spike_ratio > self._spike_threshold

        # v4.6.2: Regret spike 처리
        # regret spike는 "회상 강화"가 아니라 "억제 보조 신호"
        if self._use_regret_for_suppression and regret_spike:
            if trans_spike:
                # transition error spike + regret spike = 강한 억제 (drift 확실)
                self._regret_spike_boost = 0.3  # 추가 30% 억제
            else:
                # regret spike만 = 약한 추가 억제 (다른 원인일 수 있음)
                self._regret_spike_boost = min(0.15, self._regret_spike_boost + 0.05)
        else:
            # regret spike 없으면 점진적 감소
            self._regret_spike_boost = max(0.0, self._regret_spike_boost - 0.02)

        if trans_spike:
            # Transition error spike detected → suppress recall
            # suppression = 1 - (spike_ratio - threshold) / threshold, clamped to [0.1, 1.0]
            base_suppression = 1.0 - (spike_ratio - self._spike_threshold) / self._spike_threshold
            # v4.6.2: regret spike boost 적용
            suppression = base_suppression - self._regret_spike_boost
            self._recall_suppression_factor = max(0.1, min(1.0, suppression))
        elif self._regret_spike_boost > 0.1:
            # v4.6.2: transition spike 없어도 regret spike 강하면 약간 억제
            self._recall_suppression_factor = max(0.5, 1.0 - self._regret_spike_boost)
        else:
            # No spike → gradually recover
            self._recall_suppression_factor = min(
                1.0,
                self._recall_suppression_factor + self._suppression_recovery_rate
            )

    def get_drift_suppression_status(self) -> dict:
        """Drift suppression 상태 조회"""
        return {
            'enabled': self.drift_aware_suppression,
            'suppression_factor': self._recall_suppression_factor,
            'prediction_error_ema': self._prediction_error_ema,
            'prediction_error_baseline': self._prediction_error_baseline,
            'is_suppressed': self._recall_suppression_factor < 0.9,
            # v4.6.2
            'use_regret': getattr(self, '_use_regret_for_suppression', False),
            'regret_spike_boost': getattr(self, '_regret_spike_boost', 0.0),
        }

    def disable_memory(self):
        """장기 기억 비활성화 (저장소는 유지)"""
        self.memory_enabled = False
        self.regime_memory_enabled = False

    # === TRUE FEP v4.7: REGIME-TAGGED MEMORY CONTROLS ===

    def enable_regime_memory(self,
                            n_regimes: int = 2,
                            max_episodes_per_regime: int = 500,
                            store_threshold: float = 0.5,
                            store_sharpness: float = 5.0,
                            similarity_threshold: float = 0.95,
                            recall_top_k: int = 5,
                            spike_threshold: float = 2.0,
                            persistence_required: int = 5,
                            grace_period_length: int = 15):
        """
        v4.7 Regime-tagged Memory 활성화.

        핵심: 레짐별 메모리 분리로 "wrong confidence" 문제 해결.
        - pre-drift 기억이 post-drift에서 독이 되지 않음
        - 현재 레짐 뱅크에서만 회상 (MVP)

        Args:
            n_regimes: 레짐 수 (기본 2: pre/post drift)
            max_episodes_per_regime: 레짐당 최대 에피소드 수
            spike_threshold: 레짐 전환 트리거 임계값
            persistence_required: 연속 spike 필요 횟수
            grace_period_length: 전환 후 grace period (회상 추가 억제)
            기타: LTMStore 파라미터와 동일
        """
        # 레짐 트래커 설정
        config = RegimeConfig(
            K=n_regimes,
            spike_threshold=spike_threshold,
            persistence_required=persistence_required,
            grace_period_length=grace_period_length
        )
        self.regime_tracker = RegimeTracker(config)

        # 레짐별 메모리 뱅크
        self.regime_ltm = RegimeLTMStore(
            n_regimes=n_regimes,
            max_episodes_per_regime=max_episodes_per_regime,
            store_threshold=store_threshold,
            store_sharpness=store_sharpness,
            similarity_threshold=similarity_threshold,
            recall_top_k=recall_top_k,
            n_actions=self.n_actions
        )

        self.regime_memory_enabled = True

        # 기존 memory도 활성화 상태로 (API 호환성)
        self.memory_enabled = True
        self.memory_store_enabled = True
        self.memory_recall_enabled = True

    def update_regime(self) -> Optional[Dict]:
        """
        v4.7: 레짐 트래커 업데이트 (매 step 호출).

        transition error를 사용해 레짐 변화 감지.
        호출 시점: step 완료 후 (transition_error 업데이트 이후)

        Returns:
            레짐 업데이트 결과 or None
        """
        if not self.regime_memory_enabled or self.regime_tracker is None:
            return None

        # transition error 가져오기
        trans_error = 0.0
        volatility = 0.0
        if self._last_transition_error is not None:
            trans_error = self._last_transition_error.get('error_mean', 0.0)
            volatility = self._last_transition_error.get('volatility_ratio', 0.0)

        # 레짐 업데이트
        self._last_regime_update = self.regime_tracker.update(trans_error, volatility)
        return self._last_regime_update

    def get_regime_status(self) -> Dict:
        """v4.7: 현재 레짐 상태 조회"""
        if not self.regime_memory_enabled or self.regime_tracker is None:
            return {
                'enabled': False,
                'initialized': False
            }

        status = self.regime_tracker.get_status()
        status['memory_stats'] = self.regime_ltm.get_stats() if self.regime_ltm else {}
        return status

    def disable_regime_memory(self):
        """v4.7: Regime-tagged memory 비활성화"""
        self.regime_memory_enabled = False
        # 레짐 트래커와 뱅크는 유지 (상태 보존)

    def reset_regime_memory(self):
        """v4.7: Regime memory 완전 초기화"""
        if self.regime_tracker:
            self.regime_tracker.reset()
        if self.regime_ltm:
            self.regime_ltm.reset()
        self._last_regime_update = None

    def recall_from_memory(self, current_obs: np.ndarray) -> Optional[RecallResult]:
        """
        현재 상태에서 기억 회상.

        compute_G 전에 호출하여 _last_recall_result 설정.
        G 계산에서 memory_bias가 적용됨.

        v4.7: regime_memory_enabled면 현재 레짐 뱅크에서만 회상.

        Args:
            current_obs: 현재 관측 (8차원)

        Returns:
            RecallResult or None
        """
        # v4.7: Regime-tagged memory 사용
        if self.regime_memory_enabled and self.regime_ltm is not None and self.regime_tracker is not None:
            # Context 정보 가져오기
            context_id = 0
            if self.hierarchy_controller is not None:
                Q_ctx = self.hierarchy_controller.get_context_belief()
                if Q_ctx is not None:
                    context_id = int(np.argmax(Q_ctx))

            # 현재 불확실성
            current_uncertainty = 0.5
            if self._last_uncertainty_state is not None:
                current_uncertainty = self._last_uncertainty_state.global_uncertainty

            # 현재 레짐
            current_regime = self.regime_tracker.get_current_regime()

            # 레짐 기반 recall weight modifier
            # grace period나 낮은 confidence면 회상 가중치 하향
            recall_weight_modifier = self.regime_tracker.get_recall_weight_modifier()

            # v4.6 drift suppression도 함께 적용
            if self.drift_aware_suppression:
                recall_weight_modifier *= self._recall_suppression_factor

            # 레짐별 뱅크에서 회상
            self._last_recall_result = self.regime_ltm.recall(
                current_obs=current_obs,
                current_context_id=context_id,
                current_uncertainty=current_uncertainty,
                current_regime=current_regime,
                recall_weight_modifier=recall_weight_modifier
            )

            return self._last_recall_result

        # 기존 LTM 사용 (v4.0~v4.6)
        if self.ltm_store is None or not self.memory_enabled:
            self._last_recall_result = None
            return None

        # Context 정보 가져오기
        context_id = 0
        if self.hierarchy_controller is not None:
            Q_ctx = self.hierarchy_controller.get_context_belief()
            if Q_ctx is not None:
                context_id = int(np.argmax(Q_ctx))

        # 현재 불확실성
        current_uncertainty = 0.5
        if self._last_uncertainty_state is not None:
            current_uncertainty = self._last_uncertainty_state.global_uncertainty

        # 회상 실행
        self._last_recall_result = self.ltm_store.recall(
            current_obs=current_obs,
            current_context_id=context_id,
            current_uncertainty=current_uncertainty
        )

        return self._last_recall_result

    def prepare_episode(self,
                       t: int,
                       obs_before: np.ndarray,
                       action: int,
                       G_before: float):
        """
        에피소드 저장 준비 (행동 선택 후, 결과 전).

        행동 후 결과(obs_after, G_after)를 알게 되면 store_episode로 완료.

        Args:
            t: 타임스텝
            obs_before: 행동 전 관측
            action: 선택한 행동
            G_before: 행동 전 최소 G값
        """
        context_id = 0
        context_confidence = 1.0
        if self.hierarchy_controller is not None:
            Q_ctx = self.hierarchy_controller.get_context_belief()
            if Q_ctx is not None:
                context_id = int(np.argmax(Q_ctx))
                context_confidence = float(Q_ctx[context_id])

        uncertainty_before = 0.5
        if self._last_uncertainty_state is not None:
            uncertainty_before = self._last_uncertainty_state.global_uncertainty

        self._pending_episode_data = {
            't': t,
            'context_id': context_id,
            'context_confidence': context_confidence,
            'obs_before': obs_before.copy(),
            'action': action,
            'G_before': G_before,
            'uncertainty_before': uncertainty_before,
        }

    def store_episode(self,
                     obs_after: np.ndarray,
                     G_after: float) -> Optional[Dict]:
        """
        에피소드 저장 시도 (결과 관측 후).

        prepare_episode가 먼저 호출되어야 함.
        memory_gate로 저장 확률 결정.

        Args:
            obs_after: 행동 후 관측
            G_after: 행동 후 최소 G값

        Returns:
            저장 결과 dict or None
        """
        # v4.6: memory_store_enabled로 저장 분리 제어
        # v4.7: regime_memory_enabled도 확인
        has_memory = (self.ltm_store is not None) or (self.regime_ltm is not None)
        if not has_memory or not self.memory_enabled or not self.memory_store_enabled:
            return None

        if self._pending_episode_data is None:
            return None

        data = self._pending_episode_data
        self._pending_episode_data = None

        # Delta 계산
        obs_before = data['obs_before']
        delta_energy = float(obs_after[6] - obs_before[6]) if len(obs_after) > 6 else 0.0
        delta_pain = float(obs_after[7] - obs_before[7]) if len(obs_after) > 7 else 0.0

        # Uncertainty 변화
        uncertainty_after = 0.5
        if self._last_uncertainty_state is not None:
            uncertainty_after = self._last_uncertainty_state.global_uncertainty
        delta_uncertainty = uncertainty_after - data['uncertainty_before']

        # Surprise (prediction error)
        prediction_error = 0.0
        if self._last_uncertainty_state is not None:
            prediction_error = self._last_uncertainty_state.components.surprise
        delta_surprise = prediction_error  # 현재 surprise 값 사용

        # Outcome score 계산
        outcome_score = compute_outcome_score(
            G_before=data['G_before'],
            G_after=G_after,
            delta_energy=delta_energy,
            delta_pain=delta_pain
        )

        # Episode 생성
        episode = Episode(
            t=data['t'],
            context_id=data['context_id'],
            context_confidence=data['context_confidence'],
            obs_summary=obs_before,  # 행동 전 상태 저장
            action=data['action'],
            delta_energy=delta_energy,
            delta_pain=delta_pain,
            delta_uncertainty=delta_uncertainty,
            delta_surprise=delta_surprise,
            outcome_score=outcome_score,
        )

        # 저장 시도
        memory_gate = self.get_memory_gate()

        # v4.7: Regime-tagged memory 사용
        if self.regime_memory_enabled and self.regime_ltm is not None and self.regime_tracker is not None:
            # 저장 가능 여부와 레짐 확인
            should_store, regime_id = self.regime_tracker.should_store_memory()

            if not should_store:
                # grace period 초기에는 저장 스킵
                self._last_store_result = {
                    'attempted': True,
                    'stored': False,
                    'merged': False,
                    'reason': 'grace_period_skip',
                    'regime_id': regime_id
                }
                return self._last_store_result

            self._last_store_result = self.regime_ltm.store(episode, memory_gate, regime_id)
            return self._last_store_result

        # 기존 LTM 사용 (v4.0~v4.6)
        self._last_store_result = self.ltm_store.store(episode, memory_gate)

        return self._last_store_result

    def get_memory_status(self) -> Dict:
        """기억 상태 조회"""
        if self.ltm_store is None:
            return {
                'enabled': False,
                'initialized': False
            }

        status = {
            'enabled': self.memory_enabled,
            'initialized': True,
            # v4.6: 분해 실험용 플래그
            'store_enabled': self.memory_store_enabled,
            'recall_enabled': self.memory_recall_enabled,
            'stats': self.ltm_store.get_stats(),
        }

        if self._last_recall_result is not None:
            status['last_recall'] = self._last_recall_result.to_dict()

        if self._last_store_result is not None:
            status['last_store'] = self._last_store_result

        return status

    def get_recent_episodes(self, limit: int = 10) -> List[Dict]:
        """최근 에피소드 목록"""
        if self.ltm_store is None:
            return []
        return self.ltm_store.get_episodes_summary(limit)

    def reset_memory(self):
        """기억 저장소 초기화"""
        if self.ltm_store is not None:
            self.ltm_store.reset()
            self._last_recall_result = None
            self._last_store_result = None
            self._pending_episode_data = None

    # ==================== v4.1: Memory Consolidation ====================

    def enable_consolidation(self,
                             similarity_threshold: float = 0.9,
                             min_cluster_size: int = 2,
                             replay_batch_size: int = 20,
                             auto_trigger: bool = True):
        """
        v4.1 Memory Consolidation 활성화

        Args:
            similarity_threshold: 프로토타입 생성 유사도 기준
            min_cluster_size: 클러스터링 최소 에피소드 수
            replay_batch_size: replay 시 처리할 에피소드 수
            auto_trigger: 자동 sleep 트리거 여부
        """
        self.consolidator = MemoryConsolidator(
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            replay_batch_size=replay_batch_size
        )
        self.consolidation_enabled = True
        self._consolidation_auto = auto_trigger

        # Memory도 활성화 필요
        if not self.memory_enabled:
            self.enable_memory()

    def disable_consolidation(self):
        """v4.1 Memory Consolidation 비활성화"""
        self.consolidation_enabled = False
        # consolidator는 유지 (통계 보존)

    def update_consolidation_trigger(self, surprise: float, merged: bool, context_id: int):
        """
        Sleep 트리거 상태 업데이트

        Args:
            surprise: 현재 스텝의 예측 오차
            merged: 이번 스텝에서 에피소드가 병합되었는지
            context_id: 현재 context
        """
        if self.consolidator is not None and self.consolidation_enabled:
            self.consolidator.update_trigger(surprise, merged, context_id)

    def check_and_consolidate(self) -> Optional[ConsolidationResult]:
        """
        Sleep 조건 확인 및 통합 실행

        Returns:
            ConsolidationResult if sleep occurred, None otherwise
        """
        if not self.consolidation_enabled or self.consolidator is None:
            return None

        if not self._consolidation_auto:
            return None

        should_sleep, signals = self.consolidator.check_sleep_trigger()

        if not should_sleep:
            return None

        return self.force_consolidate()

    def force_consolidate(self) -> Optional[ConsolidationResult]:
        """
        강제 통합 실행 (수동 트리거)

        Returns:
            ConsolidationResult
        """
        if self.consolidator is None or self.ltm_store is None:
            return None

        result = self.consolidator.consolidate(
            ltm_store=self.ltm_store,
            transition_model=self,  # self has update_from_replay and transition_model
            hierarchy_controller=self.hierarchy_controller
        )

        self._last_consolidation_result = result
        return result

    def get_consolidation_status(self) -> Dict:
        """통합 시스템 상태"""
        if not self.consolidation_enabled or self.consolidator is None:
            return {
                'enabled': False,
                'initialized': False
            }

        status = self.consolidator.get_status()
        status['auto_trigger'] = self._consolidation_auto

        # Add transition_std info
        status['current_transition_std'] = float(np.mean(self.transition_model['delta_std']))

        return status

    def get_prototype_bias(self, current_obs: np.ndarray, context_id: int) -> np.ndarray:
        """프로토타입 기반 G bias 계산"""
        if self.consolidator is None:
            return np.zeros(6)
        return self.consolidator.get_prototype_bias(current_obs, context_id)

    def reset_consolidation(self):
        """통합 시스템 초기화"""
        if self.consolidator is not None:
            self.consolidator.reset()
        self._last_consolidation_result = None

    # === TRUE FEP v4.4: COUNTERFACTUAL + REGRET CONTROLS ===

    def enable_regret(self, modulation_enabled: bool = True):
        """
        Counterfactual + Regret 활성화 (v4.4)

        후회(regret)는 "감정 변수"가 아니라,
        선택한 행동이 대안 행동보다 얼마나 더 큰 G를 초래했는지에 대한 '사후 EFE 차이'.

        연결 방식 (FEP스럽게):
        - 정책 직접 변경 X
        - memory_gate, lr_boost, THINK 비용/편익 조정 O
        → "후회가 '학습/추론 자원 배분'을 바꾸는 구조"

        Args:
            modulation_enabled: v4.6 분해 실험 - modulation 적용 여부
        """
        self.counterfactual_engine = CounterfactualEngine(
            action_selector=self,
            n_actions=self.N_PHYSICAL_ACTIONS  # THINK 제외
        )
        self.counterfactual_engine.enable()
        self.regret_enabled = True
        # v4.6: 분해 실험용 - 계산만 하고 modulation 끄기
        self.regret_modulation_enabled = modulation_enabled
        self._last_cf_result = None
        self._last_G_pred = {}

    def set_regret_mode(self, modulation_enabled: bool = True):
        """
        v4.6: Regret 분해 실험 모드 설정

        - calc_only: modulation_enabled=False (계산만, 적용 안함)
        - full: modulation_enabled=True (계산 + 적용)
        """
        self.regret_modulation_enabled = modulation_enabled

    def disable_regret(self):
        """Counterfactual + Regret 비활성화"""
        self.regret_enabled = False
        if self.counterfactual_engine is not None:
            self.counterfactual_engine.disable()

    def reset_regret(self):
        """Regret 상태 초기화"""
        if self.counterfactual_engine is not None:
            self.counterfactual_engine = CounterfactualEngine(
                action_selector=self,
                n_actions=self.N_PHYSICAL_ACTIONS
            )
            self.counterfactual_engine.enable()
        self._last_cf_result = None
        self._last_G_pred = {}

    def compute_counterfactual(self,
                               chosen_action: int,
                               obs_before: np.ndarray,
                               obs_after: np.ndarray) -> Optional[CounterfactualResult]:
        """
        Counterfactual 계산 (행동 후 호출)

        매 step에서:
        1. 선택 시점의 G_pred 저장 (action_selection에서 저장됨)
        2. 관측 후 G_post 계산 (전이 모델 기반 반사실 추론)
        3. Regret 신호 계산

        Args:
            chosen_action: 실제로 선택한 행동
            obs_before: 행동 전 관측
            obs_after: 행동 후 관측 (실제 결과)

        Returns:
            CounterfactualResult with regret signals
        """
        if not self.regret_enabled or self.counterfactual_engine is None:
            return None

        if not self._last_G_pred:
            # G_pred가 없으면 계산 불가
            return None

        result = self.counterfactual_engine.compute_counterfactual(
            chosen_action=chosen_action,
            G_pred=self._last_G_pred,
            obs_before=obs_before,
            obs_after=obs_after
        )

        self._last_cf_result = result
        return result

    def get_regret_modulation(self) -> Dict:
        """
        Regret 기반 조절 파라미터 반환

        연결 대상:
        1. memory_gate_boost: regret 큰 사건 = 저장 우선순위 ↑
        2. lr_boost_factor: regret + surprise = 모델 재학습 필요
        3. think_benefit_boost: regret 누적 = 메타인지 강화 합리적
        """
        if not self.regret_enabled or self.counterfactual_engine is None:
            return {
                'memory_gate_boost': 0.0,
                'lr_boost_factor': 1.0,
                'think_benefit_boost': 0.0,
                'regret_real': 0.0,
                'regret_pred': 0.0,
                'is_spike': False,
            }

        return self.counterfactual_engine.get_regret_modulation()

    def get_regret_status(self) -> Dict:
        """Regret 상태 반환"""
        if not self.regret_enabled or self.counterfactual_engine is None:
            return {
                'enabled': False,
                'modulation_enabled': False,
                'engine_status': None
            }

        return {
            'enabled': True,
            # v4.6: 분해 실험용 플래그
            'modulation_enabled': self.regret_modulation_enabled,
            'engine_status': self.counterfactual_engine.get_status()
        }
