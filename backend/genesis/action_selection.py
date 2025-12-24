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
from .preference_distributions import PreferenceDistributions


@dataclass
class GDecomposition:
    """Complete decomposition of Expected Free Energy."""
    G: float              # Total expected free energy
    risk: float           # KL[Q(o|a) || P(o)] - preference violation
    ambiguity: float      # E[H[P(o|s')]] - observation uncertainty
    action_cost: float    # Small cost for movement


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

        # === PARAMETERS ===
        self.temperature = 0.3  # Lower = more deterministic

        # REMOVED: action_cost_weight - absorbed into P(o)
        # REMOVED: novelty_bonus - was causing oscillation

        # Track last action and observations for transition learning
        self._last_action = 0
        self._last_obs = None
        self._action_history = []

        # === TRANSITION MODEL (learned, not hardcoded) ===
        # P(o'|o, a): 각 행동이 관측을 어떻게 바꾸는지 학습
        # Shape: (n_actions, n_obs, n_obs) - 단순화: 각 차원별 delta mean/std
        self.transition_model = {
            'delta_mean': np.zeros((n_actions, 6)),  # 예측된 평균 변화
            'delta_std': np.ones((n_actions, 6)) * 0.2,  # 변화의 불확실성
            'count': np.ones((n_actions,))  # 행동별 경험 횟수
        }

        # Learning rate for transition model
        self.transition_lr = 0.1

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

        for a in range(self.n_actions):
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

                # Physics prior only - learning disabled for now
                # TODO: Re-enable learning when browser interference is resolved
                # The browser calls /step ~20x per second, corrupting the transition model
                total_delta_food = delta_food_prox
                total_delta_danger = delta_danger_prox

                # Predicted observation
                Q_o = current_obs.copy()
                Q_o[0] = np.clip(food_prox + total_delta_food, 0.0, 1.0)
                Q_o[1] = np.clip(danger_prox + total_delta_danger, 0.0, 1.0)

                # Uncertainty combines physics uncertainty with learned uncertainty
                q_uncertainty = delta_std.copy()
                q_uncertainty[0] = max(0.05, delta_std[0])  # Minimum uncertainty
                q_uncertainty[1] = max(0.05, delta_std[1])

            else:
                Q_o = np.zeros(6)
                q_uncertainty = np.ones(6) * 0.5

            # === RISK: KL[Q(o|a) || P(o)] ===
            risk = self.preferences.compute_risk(Q_o, q_uncertainty)

            # === AMBIGUITY: E_{Q(s'|a)}[H[P(o|s')]] ===
            # FEP 정의: 전이 모델의 불확실성만 사용
            # delta_std는 학습을 통해 자연스럽게 감소 (휴리스틱 아님)
            transition_std = self.transition_model['delta_std'][a]
            avg_uncertainty = np.mean(transition_std[:2])  # proximity 차원
            ambiguity = self.preferences.compute_ambiguity(avg_uncertainty)

            # === TOTAL G ===
            G = risk + ambiguity

            results[a] = GDecomposition(
                G=G,
                risk=risk,
                ambiguity=ambiguity,
                action_cost=0.0
            )

        return results

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
        """
        if Q_s is None:
            Q_s = self.model.Q_s

        # Compute G decomposition for each action
        G_all = self.compute_G(Q_s, current_obs)

        # Extract G values for softmax
        G_values = np.array([G_all[a].G for a in range(self.n_actions)])

        # Action probabilities: P(a) ∝ exp(-G(a) / temperature)
        log_probs = -G_values / self.temperature
        log_probs = log_probs - np.max(log_probs)
        probs = np.exp(log_probs)
        probs = probs / probs.sum()

        # Always use deterministic for now (debugging)
        # TODO: Add exploration later
        if True:  # deterministic:
            action = int(np.argmin(G_values))
        else:
            action = int(np.random.choice(self.n_actions, p=probs))

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
        risks = [(a, G_all[a].risk) for a in range(self.n_actions)]
        risks.sort(key=lambda x: x[1], reverse=True)

        high_risk = risks[0]
        low_risk = risks[-1]

        if high_risk[1] > 0.5:
            return f"Action {high_risk[0]} has high risk ({high_risk[1]:.2f}) - avoiding"
        else:
            return "All actions have similar risk"

    def _explain_ambiguity(self, G_all: Dict[int, GDecomposition]) -> str:
        """Explain ambiguity differences."""
        ambiguities = [(a, G_all[a].ambiguity) for a in range(self.n_actions)]
        ambiguities.sort(key=lambda x: x[1])

        low_ambig = ambiguities[0]
        high_ambig = ambiguities[-1]

        diff = high_ambig[1] - low_ambig[1]
        if diff > 0.3:
            return f"Action {low_ambig[0]} reduces uncertainty most"
        else:
            return "Similar ambiguity across actions"

    def get_detailed_log(self, result: ActionResult) -> Dict:
        """
        Get detailed decomposition log.

        This is the key output that shows WHY behavior happens
        without using emotion labels.
        """
        action = result.action
        G = result.G_all[action]

        # Determine dominant factor
        if G.risk > G.ambiguity * 1.5:
            dominant = "risk_avoidance"  # Observer might call this "fear response"
        elif G.ambiguity > G.risk * 1.5:
            dominant = "ambiguity_reduction"  # Observer might call this "curiosity"
        else:
            dominant = "balanced"

        # Find best/worst actions by each component
        min_risk_action = min(range(self.n_actions), key=lambda a: result.G_all[a].risk)
        min_ambig_action = min(range(self.n_actions), key=lambda a: result.G_all[a].ambiguity)

        return {
            # The decomposition that explains everything
            'selected_action': action,
            'G': G.G,
            'risk': G.risk,           # High = avoiding preference violation
            'ambiguity': G.ambiguity, # High = uncertain observations
            'action_cost': G.action_cost,

            # What's driving behavior
            'dominant_factor': dominant,

            # Comparison
            'min_risk_action': min_risk_action,
            'min_ambiguity_action': min_ambig_action,

            # All actions for visualization
            'all_G': {a: result.G_all[a].G for a in range(self.n_actions)},
            'all_risk': {a: result.G_all[a].risk for a in range(self.n_actions)},
            'all_ambiguity': {a: result.G_all[a].ambiguity for a in range(self.n_actions)},

            # Natural language (for observers only)
            'why_risk': result.why_risk,
            'why_ambiguity': result.why_ambiguity,

            # Mapping to what observers might say
            '_observer_note': {
                'risk_avoidance': "looks like fear/avoidance",
                'ambiguity_reduction': "looks like curiosity/exploration",
                'balanced': "no dominant drive visible"
            }
        }

    def reset_transition_model(self):
        """Reset the learned transition model to initial state."""
        self.transition_model = {
            'delta_mean': np.zeros((self.n_actions, 6)),
            'delta_std': np.ones((self.n_actions, 6)) * 0.2,
            'count': np.ones((self.n_actions,))
        }
        self._last_obs = None
        self._action_history = []

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
