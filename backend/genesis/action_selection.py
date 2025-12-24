"""
Action Selection - Choosing Actions to Minimize Expected Free Energy

Action is not about maximizing reward.
Action is about minimizing EXPECTED Free Energy.

G(a) = E_Q(o,s'|a)[F(o,s')]

This naturally decomposes into:
    G(a) = Epistemic Value + Pragmatic Value

    Epistemic Value: "Will this action reduce my uncertainty?"
        = Expected information gain
        = How much will I learn?

    Pragmatic Value: "Will this action bring me to preferred states?"
        = Expected log preference
        = How close to my setpoints?

Why is this better than reward maximization?
    1. No external reward signal needed
    2. Exploration emerges naturally (epistemic value)
    3. Goal-directed behavior emerges from preferences
    4. Everything explained by one principle

What observers call "curiosity" is high epistemic value.
What observers call "goal pursuit" is high pragmatic value.
The agent just minimizes G.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .free_energy import FreeEnergyEngine
from .generative_model import GenerativeModel


@dataclass
class ActionResult:
    """Result of action selection."""
    action: int               # Selected action
    G: Dict[int, float]       # Expected free energy for each action
    epistemic: Dict[int, float]   # Epistemic value for each action
    pragmatic: Dict[int, float]   # Pragmatic value for each action
    probabilities: np.ndarray     # Action probabilities (softmax of -G)


class ActionSelector:
    """
    Selects actions by minimizing Expected Free Energy.

    G(a) = E_Q[log Q(s'|a) - log P(s',o|a)]

    This can be decomposed as:
        G(a) = -Information Gain - Expected Log Preference

    Where:
        Information Gain = H[Q(s'|a)] - E_Q(o|a)[H[Q(s'|o,a)]]
            "How much uncertainty will be resolved?"

        Expected Log Preference = E_Q(o|a)[log P(o)]
            "How preferred are the expected observations?"

    Actions are selected probabilistically:
        P(a) = softmax(-G(a) / temperature)

    Temperature controls exploration:
        High temperature = more random (early learning)
        Low temperature = more deterministic (exploit knowledge)
    """

    def __init__(self,
                 model: GenerativeModel,
                 free_energy: FreeEnergyEngine,
                 n_actions: int):
        """
        Initialize action selector.

        Args:
            model: The generative model
            free_energy: The free energy engine
            n_actions: Number of possible actions
        """
        self.model = model
        self.fe = free_energy
        self.n_actions = n_actions

        # === PARAMETERS ===
        self.temperature = 1.0      # Softmax temperature
        self.planning_horizon = 1   # How many steps to look ahead

    def compute_expected_free_energy(self,
                                     Q_s: Optional[np.ndarray] = None
                                     ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """
        Compute Expected Free Energy G(a) for each action.

        Returns:
            Tuple of (G, epistemic, pragmatic) dicts
        """
        if Q_s is None:
            Q_s = self.model.Q_s

        G = {}
        epistemic = {}
        pragmatic = {}

        for a in range(self.n_actions):
            # === PREDICT NEXT STATE ===
            # Q(s'|a) = sum_s P(s'|s,a) Q(s)
            Q_s_next = self.model.predict_next_state(a, Q_s)

            # === PREDICT OBSERVATIONS ===
            # Q(o|a) = sum_s' P(o|s') Q(s'|a)
            Q_o = self.model.A.T @ Q_s_next
            Q_o = np.clip(Q_o, 1e-10, 1.0)
            Q_o = Q_o / Q_o.sum()

            # === EPISTEMIC VALUE ===
            # Information gain: How much will we learn?
            # IG = H[Q(s'|a)] - E_Q(o|a)[H[Q(s'|o,a)]]

            # Prior entropy over next states
            H_prior = -np.sum(Q_s_next * np.log(Q_s_next + 1e-10))

            # Expected posterior entropy (after observing o)
            H_posterior_expected = 0.0
            for o in range(self.model.n_obs):
                if Q_o[o] > 1e-10:
                    # Posterior via Bayes: Q(s'|o,a) ∝ P(o|s') Q(s'|a)
                    Q_s_given_o = self.model.A[:, o] * Q_s_next
                    if Q_s_given_o.sum() > 1e-10:
                        Q_s_given_o = Q_s_given_o / Q_s_given_o.sum()
                        H_post = -np.sum(Q_s_given_o * np.log(Q_s_given_o + 1e-10))
                        H_posterior_expected += Q_o[o] * H_post

            information_gain = H_prior - H_posterior_expected
            epistemic[a] = information_gain

            # === PRAGMATIC VALUE ===
            # How close are expected observations to preferences?
            # E_Q(o|a)[log P(o)] where P(o) is encoded in C
            expected_log_preference = np.sum(Q_o * self.model.C)
            pragmatic[a] = expected_log_preference

            # === EXPECTED FREE ENERGY ===
            # G(a) = -epistemic + (-pragmatic)
            # Lower G is better
            # Epistemic: higher IG = lower G (we want to learn)
            # Pragmatic: higher log P(o) = lower G (we want preferred observations)
            G[a] = -information_gain - expected_log_preference

        return G, epistemic, pragmatic

    def select_action(self,
                      Q_s: Optional[np.ndarray] = None,
                      deterministic: bool = False
                      ) -> ActionResult:
        """
        Select an action by minimizing Expected Free Energy.

        Args:
            Q_s: Current belief over states
            deterministic: If True, always select lowest G

        Returns:
            ActionResult with selected action and details
        """
        if Q_s is None:
            Q_s = self.model.Q_s

        # Compute G for each action
        G, epistemic, pragmatic = self.compute_expected_free_energy(Q_s)

        # Convert to arrays for softmax
        G_values = np.array([G[a] for a in range(self.n_actions)])

        # Action probabilities: P(a) = softmax(-G(a) / temperature)
        # We negate G because lower G is better
        log_probs = -G_values / self.temperature
        log_probs = log_probs - np.max(log_probs)  # Numerical stability
        probs = np.exp(log_probs)
        probs = probs / probs.sum()

        if deterministic:
            action = np.argmin(G_values)
        else:
            action = np.random.choice(self.n_actions, p=probs)

        return ActionResult(
            action=action,
            G=G,
            epistemic=epistemic,
            pragmatic=pragmatic,
            probabilities=probs
        )

    def explain_action(self, result: ActionResult) -> Dict:
        """
        Explain why an action was selected.

        This is for observation/debugging only.
        The agent doesn't "know" these labels.

        Returns:
            Human-readable explanation
        """
        action = result.action

        # Get values for selected action
        G_selected = result.G[action]
        epistemic_selected = result.epistemic[action]
        pragmatic_selected = result.pragmatic[action]

        # Determine dominant drive
        # High epistemic = exploring/curious behavior
        # High pragmatic = goal-directed/exploiting behavior
        epistemic_ratio = epistemic_selected / (abs(epistemic_selected) + abs(pragmatic_selected) + 1e-10)

        if epistemic_ratio > 0.6:
            dominant_drive = "epistemic"  # Observer might call this "curiosity"
        elif epistemic_ratio < 0.4:
            dominant_drive = "pragmatic"  # Observer might call this "goal pursuit"
        else:
            dominant_drive = "balanced"

        # F trajectory
        F_trajectory = self.fe.get_F_trajectory()

        return {
            'action': action,
            'G': G_selected,
            'epistemic_value': epistemic_selected,
            'pragmatic_value': pragmatic_selected,
            'dominant_drive': dominant_drive,
            'probability': result.probabilities[action],
            'F_trend': F_trajectory['trend'],

            # What observers might call these states
            # (These are NOT internal labels - just for human understanding)
            '_observer_interpretation': {
                'high_epistemic': "exploring, curious",
                'high_pragmatic': "goal-directed, exploiting",
                'F_increasing': "distressed, anxious",
                'F_decreasing': "satisfied, relieved"
            }
        }

    def set_temperature(self, temperature: float):
        """
        Set exploration temperature.

        Higher temperature = more random exploration.
        Lower temperature = more deterministic exploitation.
        """
        self.temperature = max(0.01, temperature)

    def adapt_temperature(self, F_current: float, F_baseline: float = 1.0):
        """
        Adapt temperature based on Free Energy.

        When F is high (stressed), be more random (try anything).
        When F is low (comfortable), be more deterministic (exploit knowledge).
        """
        if F_current > F_baseline * 1.5:
            # High F - increase exploration
            self.temperature = min(5.0, self.temperature * 1.1)
        elif F_current < F_baseline * 0.5:
            # Low F - decrease exploration
            self.temperature = max(0.1, self.temperature * 0.9)
