"""
Genesis Agent - The Complete Free Energy Minimizing System

This is the unified agent that:
1. Perceives by minimizing F (inference)
2. Acts by minimizing expected F (action selection)
3. Learns by updating the generative model

There are NO explicit:
- Emotion labels
- Goal variables
- Reward signals
- Behavior rules

Everything emerges from F minimization.

What observers might interpret as:
- "Fear" → Predicted future F is high (approaching absorbing state)
- "Curiosity" → High epistemic value (information gain possible)
- "Boredom" → Low information rate (model not updating)
- "Satisfaction" → F is decreasing
- "Goal pursuit" → High pragmatic value (moving toward preferences)

But the agent doesn't have these labels.
It just minimizes Free Energy.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .free_energy import FreeEnergyEngine, FreeEnergyState
from .generative_model import GenerativeModel
from .inference import InferenceEngine, InferenceResult
from .action_selection import ActionSelector, ActionResult, GDecomposition


@dataclass
class AgentState:
    """Complete state of the agent for visualization."""
    # Core quantities
    F: float                  # Current Free Energy
    dF_dt: float             # Rate of change of F
    Q_s: np.ndarray          # Belief over states

    # Action-related
    action: int              # Selected action
    G_decomposition: Dict[int, GDecomposition]  # Full G decomposition per action

    # Key components of G for selected action
    risk: float              # KL[Q(o|a) || P(o)] - preference violation
    ambiguity: float         # E[H[P(o|s')]] - observation uncertainty
    complexity: float        # KL[Q(s'|a) || P(s')] - belief divergence from preferred states

    # Inference-related
    prediction_error: float
    belief_update: float

    # Derived quantities (for observers)
    information_rate: float   # How fast is model learning
    preference_divergence: float  # How far from preferred states

    # History
    F_history: List[float]

    # Explanation
    dominant_factor: str     # 'risk_avoidance', 'ambiguity_reduction', 'complexity_avoidance', or 'balanced'


class GenesisAgent:
    """
    A Free Energy minimizing agent.

    The agent has:
    - A generative model of the world
    - Inference to update beliefs from observations
    - Action selection to minimize expected Free Energy

    That's it. Everything else emerges.
    """

    def __init__(self,
                 n_states: int,
                 n_observations: int,
                 n_actions: int,
                 preferred_observations: Optional[np.ndarray] = None):
        """
        Initialize the agent.

        Args:
            n_states: Number of hidden states in world model
            n_observations: Dimension of observations
            n_actions: Number of possible actions
            preferred_observations: Prior preferences (homeostatic setpoints)
        """
        self.n_states = n_states
        self.n_obs = n_observations
        self.n_actions = n_actions

        # === CORE COMPONENTS ===
        self.model = GenerativeModel(n_states, n_observations, n_actions)
        self.fe = FreeEnergyEngine(n_states, n_observations)
        self.inference = InferenceEngine(self.model, self.fe)
        self.action_selector = ActionSelector(self.model, self.fe, n_actions)

        # === SET PREFERENCES ===
        if preferred_observations is not None:
            self.model.set_preference(preferred_observations)

        # === TRACKING ===
        self._step_count = 0
        self._F_history = []
        self._info_rate_history = []
        self._last_Q_s = self.model.Q_s.copy()

    def step(self, observation: np.ndarray) -> AgentState:
        """
        Perform one step of the agent.

        1. Infer hidden state from observation (perception)
        2. Select action to minimize expected F (action)
        3. Update model (learning)

        Args:
            observation: Current sensory observation

        Returns:
            AgentState with all relevant information
        """
        # Store previous state for learning
        Q_s_prev = self.model.Q_s.copy()

        # === PERCEPTION: Minimize F by updating beliefs ===
        inference_result = self.inference.infer(observation)

        # === ACTION: Minimize expected F by selecting action ===
        # Pass current observation for direct prediction
        action_result = self.action_selector.select_action(current_obs=observation)

        # === LEARNING: Update model from experience ===
        # (This happens after action is taken and new observation received)
        # For now, we update based on current observation
        self.model.update_model(
            observation,
            action_result.action,
            Q_s_prev,
            inference_result.Q_s
        )

        # === COMPUTE DERIVED QUANTITIES ===

        # Information rate: How much are beliefs changing?
        belief_change = np.abs(inference_result.Q_s - self._last_Q_s).sum()
        self._info_rate_history.append(belief_change)
        if len(self._info_rate_history) > 20:
            self._info_rate_history.pop(0)
        information_rate = np.mean(self._info_rate_history)

        # Preference divergence: How far from preferred observations?
        preference_divergence = self.model.get_preference_divergence(observation)

        # Store for next step
        self._last_Q_s = inference_result.Q_s.copy()
        self._F_history.append(inference_result.F.F)
        if len(self._F_history) > 100:
            self._F_history.pop(0)

        self._step_count += 1

        # Get decomposition for selected action
        selected_G = action_result.G_all[action_result.action]

        # Determine dominant factor (for observer interpretation only)
        factors = {'risk': selected_G.risk, 'ambiguity': selected_G.ambiguity, 'complexity': selected_G.complexity}
        max_factor = max(factors, key=factors.get)

        if max_factor == 'risk' and selected_G.risk > selected_G.ambiguity * 1.5 and selected_G.risk > selected_G.complexity * 1.5:
            dominant = 'risk_avoidance'  # Looks like "fear"
        elif max_factor == 'ambiguity' and selected_G.ambiguity > selected_G.risk * 1.5 and selected_G.ambiguity > selected_G.complexity * 1.5:
            dominant = 'ambiguity_reduction'  # Looks like "curiosity"
        elif max_factor == 'complexity' and selected_G.complexity > selected_G.risk * 1.5 and selected_G.complexity > selected_G.ambiguity * 1.5:
            dominant = 'complexity_avoidance'  # Looks like "cognitive conservatism"
        else:
            dominant = 'balanced'

        return AgentState(
            F=inference_result.F.F,
            dF_dt=inference_result.F.dF_dt,
            Q_s=inference_result.Q_s,
            action=action_result.action,
            G_decomposition=action_result.G_all,
            risk=selected_G.risk,
            ambiguity=selected_G.ambiguity,
            complexity=selected_G.complexity,
            prediction_error=inference_result.prediction_error,
            belief_update=inference_result.belief_update,
            information_rate=information_rate,
            preference_divergence=preference_divergence,
            F_history=self._F_history.copy(),
            dominant_factor=dominant
        )

    def step_with_action(self,
                         observation: np.ndarray,
                         previous_action: int
                         ) -> AgentState:
        """
        Perform step knowing the previous action.

        This allows better inference using the transition model.

        Args:
            observation: Current observation
            previous_action: Action taken before this observation

        Returns:
            AgentState
        """
        Q_s_prev = self.model.Q_s.copy()

        # Infer with action context
        inference_result = self.inference.infer_with_action(
            observation, previous_action, Q_s_prev
        )

        # Select next action with current observation
        action_result = self.action_selector.select_action(current_obs=observation)

        # Learn
        self.model.update_model(
            observation,
            previous_action,
            Q_s_prev,
            inference_result.Q_s
        )

        # Compute derived quantities
        belief_change = np.abs(inference_result.Q_s - self._last_Q_s).sum()
        self._info_rate_history.append(belief_change)
        if len(self._info_rate_history) > 20:
            self._info_rate_history.pop(0)
        information_rate = np.mean(self._info_rate_history)

        preference_divergence = self.model.get_preference_divergence(observation)

        self._last_Q_s = inference_result.Q_s.copy()
        self._F_history.append(inference_result.F.F)
        if len(self._F_history) > 100:
            self._F_history.pop(0)

        self._step_count += 1

        # Get decomposition for selected action
        selected_G = action_result.G_all[action_result.action]

        # Determine dominant factor
        factors = {'risk': selected_G.risk, 'ambiguity': selected_G.ambiguity, 'complexity': selected_G.complexity}
        max_factor = max(factors, key=factors.get)

        if max_factor == 'risk' and selected_G.risk > selected_G.ambiguity * 1.5 and selected_G.risk > selected_G.complexity * 1.5:
            dominant = 'risk_avoidance'
        elif max_factor == 'ambiguity' and selected_G.ambiguity > selected_G.risk * 1.5 and selected_G.ambiguity > selected_G.complexity * 1.5:
            dominant = 'ambiguity_reduction'
        elif max_factor == 'complexity' and selected_G.complexity > selected_G.risk * 1.5 and selected_G.complexity > selected_G.ambiguity * 1.5:
            dominant = 'complexity_avoidance'
        else:
            dominant = 'balanced'

        return AgentState(
            F=inference_result.F.F,
            dF_dt=inference_result.F.dF_dt,
            Q_s=inference_result.Q_s,
            action=action_result.action,
            G_decomposition=action_result.G_all,
            risk=selected_G.risk,
            ambiguity=selected_G.ambiguity,
            complexity=selected_G.complexity,
            prediction_error=inference_result.prediction_error,
            belief_update=inference_result.belief_update,
            information_rate=information_rate,
            preference_divergence=preference_divergence,
            F_history=self._F_history.copy(),
            dominant_factor=dominant
        )

    def get_explanation(self, state: AgentState) -> Dict:
        """
        Generate explanation of current behavior.

        This is for HUMAN OBSERVERS only.
        The agent doesn't use these labels internally.

        Args:
            state: Current agent state

        Returns:
            Human-readable explanation
        """
        # Determine what an observer might call this state
        # Based on F dynamics and information flow

        F = state.F
        dF = state.dF_dt
        info_rate = state.information_rate
        pref_div = state.preference_divergence

        # F trajectory interpretation
        if len(state.F_history) > 10:
            F_trend = np.mean(state.F_history[-5:]) - np.mean(state.F_history[-10:-5])
        else:
            F_trend = 0

        # Compute what might be called "emotional state"
        # (Again, these are observer labels, not internal states)

        interpretations = []

        # Get G decomposition for selected action
        selected_G = state.G_decomposition[state.action]

        # High F + increasing = distress/anxiety
        if F > 2.0 and F_trend > 0.1:
            interpretations.append("distress (F high and rising)")

        # High risk = avoiding preference violation (looks like "fear")
        if state.risk > 0.5:
            interpretations.append("risk-averse (avoiding preference violation)")

        # F decreasing = satisfaction/relief
        if F_trend < -0.1:
            interpretations.append("satisfaction (F decreasing)")

        # Low info rate + low F = boredom
        if info_rate < 0.1 and F < 1.0:
            interpretations.append("boredom-like (no information, low F)")

        # Low ambiguity seeking = curiosity (reducing observation uncertainty)
        min_ambiguity = min(g.ambiguity for g in state.G_decomposition.values())
        if state.ambiguity < min_ambiguity + 0.1:
            # Chose action with low ambiguity - seeking clear information
            interpretations.append("information-seeking (reducing ambiguity)")

        # Dominant factor interpretation
        if state.dominant_factor == 'risk_avoidance':
            interpretations.append("dominant: avoiding bad outcomes")
        elif state.dominant_factor == 'ambiguity_reduction':
            interpretations.append("dominant: seeking clarity")
        elif state.dominant_factor == 'complexity_avoidance':
            interpretations.append("dominant: maintaining cognitive stability")

        return {
            'F': F,
            'F_trend': 'increasing' if F_trend > 0.1 else 'decreasing' if F_trend < -0.1 else 'stable',
            'information_rate': info_rate,
            'preference_divergence': pref_div,
            'action': state.action,
            'action_G': selected_G.G,
            'observer_interpretations': interpretations,

            # Risk/Ambiguity/Complexity decomposition (the mechanical explanation)
            'risk': state.risk,
            'ambiguity': state.ambiguity,
            'complexity': state.complexity,
            'dominant_factor': state.dominant_factor,

            # All actions G decomposition
            'all_G': {a: state.G_decomposition[a].G for a in state.G_decomposition},
            'all_risk': {a: state.G_decomposition[a].risk for a in state.G_decomposition},
            'all_ambiguity': {a: state.G_decomposition[a].ambiguity for a in state.G_decomposition},
            'all_complexity': {a: state.G_decomposition[a].complexity for a in state.G_decomposition},

            # The only "ground truth"
            'why_this_action': f"Action {state.action}: G={selected_G.G:.3f} (risk={selected_G.risk:.3f}, ambiguity={selected_G.ambiguity:.3f}, complexity={selected_G.complexity:.3f})"
        }

    def reset(self):
        """Reset agent to initial state."""
        self.model.reset()
        self.fe.reset()
        self.action_selector.reset_transition_model()  # Reset learned transitions
        self._step_count = 0
        self._F_history = []
        self._info_rate_history = []
        self._last_Q_s = self.model.Q_s.copy()
