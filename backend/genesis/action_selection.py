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
from .preference_distributions import PreferenceDistributions, StatePreferenceDistribution
from .precision import PrecisionLearner, PrecisionState
from .temporal import TemporalPlanner, RolloutResult
from .hierarchy import HierarchicalController


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

        # THINK 선택 로그
        self._last_think_selected = False
        self._last_think_reason = ""
        self._last_expected_improvement = 0.0
        self._think_count = 0
        self._physical_action_after_think = None

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

                total_delta_food = delta_blended_ext[0]
                total_delta_danger = delta_blended_ext[1]

                # Predicted observation
                Q_o = current_obs.copy()
                Q_o[0] = np.clip(food_prox + total_delta_food, 0.0, 1.0)
                Q_o[1] = np.clip(danger_prox + total_delta_danger, 0.0, 1.0)

                # v3.3.1: 내부 상태는 더 보수적인 alpha_internal 사용
                if alpha_int > 0:
                    delta_blended_int = alpha_int * delta_ctx[6:8]  # physics는 0이라 가정
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

            risk = self.preferences.compute_risk(Q_o, q_uncertainty, risk_weights)

            # === AMBIGUITY: E_{Q(s'|a)}[H[P(o|s')]] with PRECISION weighting ===
            # FEP 정의: 전이 모델의 불확실성만 사용
            # Precision이 높으면 이 행동의 불확실성에 더 민감
            transition_std = self.transition_model['delta_std'][a]
            avg_uncertainty = np.mean(transition_std[:2])  # proximity 차원
            ambiguity_weight = self.precision_learner.get_ambiguity_weight(a)
            ambiguity = self.preferences.compute_ambiguity(avg_uncertainty) * ambiguity_weight

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

        G_think = G_best - potential_improvement + deliberation_cost

        # Risk/Ambiguity/Complexity 분해 (THINK 전용)
        # - Risk: deliberation_cost (시간 비용)
        # - Ambiguity: 현재 결정 불확실성 (역수로, 불확실할수록 THINK가 줄여줌)
        # - Complexity: 0 (THINK는 믿음을 직접 바꾸지 않음)

        risk_component = deliberation_cost
        ambiguity_component = -potential_improvement  # 음수: THINK가 불확실성을 줄임

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

        Online learning:
        - Observe actual delta = obs_after - obs_before
        - Update delta_mean toward actual
        - Update delta_std based on prediction error

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

        # Update count
        self.transition_model['count'][action] += 1
        count = self.transition_model['count'][action]

        # Adaptive learning rate (decays with experience)
        lr = self.transition_lr / np.sqrt(count)

        # Update mean (move toward actual)
        self.transition_model['delta_mean'][action] += lr * error

        # Update std (move toward observed variance)
        # Simple: std = running average of |error|
        self.transition_model['delta_std'][action] = (
            (1 - lr) * self.transition_model['delta_std'][action] +
            lr * np.abs(error)
        )

        # Clip std to reasonable range
        self.transition_model['delta_std'][action] = np.clip(
            self.transition_model['delta_std'][action], 0.01, 1.0
        )

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

        # === STEP 2.5: v3.4 THINK action ===
        # G(THINK)를 계산하고 물리 행동과 비교
        if self.think_enabled and current_obs is not None:
            G_think = self.compute_G_think(G_all, current_obs, decision_entropy)
            G_all[self.THINK_ACTION] = G_think

            # THINK가 최선인지 확인
            if G_think.G < G_best:
                self._last_think_selected = True
                self._last_think_reason = (
                    f"entropy={decision_entropy:.2f}, "
                    f"improvement={self._last_expected_improvement:.3f}, "
                    f"G_think={G_think.G:.3f} < G_best={G_best:.3f}"
                )
                self._think_count += 1

                # THINK 선택 시: rollout 실행하고 물리 행동 재선택
                # (아래 should_rollout 로직으로 넘어감)

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
            # Lazy initialization of temporal planner
            if self.temporal_planner is None:
                self.temporal_planner = TemporalPlanner(
                    self,
                    horizon=self.rollout_horizon,
                    discount=self.rollout_discount,
                    n_samples=self.rollout_n_samples,
                    complexity_decay=self.rollout_complexity_decay
                )
            else:
                self.temporal_planner.set_horizon(self.rollout_horizon)
                self.temporal_planner.set_discount(self.rollout_discount)
                self.temporal_planner.n_samples = self.rollout_n_samples
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
        """THINK 상태 조회"""
        return {
            'enabled': self.think_enabled,
            'think_count': self._think_count,
            'last_think_selected': self._last_think_selected,
            'last_think_reason': self._last_think_reason,
            'last_expected_improvement': self._last_expected_improvement,
            'physical_action_after_think': self._physical_action_after_think,
            'energy_cost': self.think_energy_cost,
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
