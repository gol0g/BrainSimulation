"""
Free Energy Engine - The One Principle

F = D_KL[Q(s) || P(s|o)] - log P(o)

This is the ONLY quantity we minimize.
Everything else is derived from this.

In practice, for discrete states:
    F = sum_s Q(s) * [log Q(s) - log P(s,o)]
      = sum_s Q(s) * log Q(s)           # -H[Q] (negative entropy of belief)
        - sum_s Q(s) * log P(s,o)       # -E_Q[log P(s,o)] (expected log joint)

    F = -H[Q(s)] + E_Q[-log P(o|s)] + E_Q[-log P(s)]
      = -entropy + energy

Where:
    - entropy: how uncertain our beliefs are
    - energy: how unlikely our observations given beliefs

Minimizing F means:
    1. Maximize entropy (spread beliefs) - but constrained by...
    2. Minimize energy (beliefs must explain observations)

This is variational inference.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FreeEnergyState:
    """Current state of the free energy computation."""
    F: float                    # Total free energy
    prediction_error: float     # E_Q[-log P(o|s)]
    complexity: float           # D_KL[Q(s) || P(s)]
    entropy: float              # H[Q(s)]

    # Derivatives for understanding dynamics
    dF_dt: float = 0.0         # Rate of change
    F_expected: float = 0.0    # Expected future F (for action selection)


class FreeEnergyEngine:
    """
    The core engine that computes Free Energy.

    This is the foundation. All behavior emerges from minimizing F.

    Why does a system minimize F?
    - F is an upper bound on surprise (negative log evidence)
    - Systems that don't minimize surprise don't persist
    - Therefore, any persisting system minimizes F

    This is not a goal we impose. It's a consequence of existence.
    """

    def __init__(self, state_dim: int, obs_dim: int):
        """
        Initialize the engine.

        Args:
            state_dim: Number of possible hidden states
            obs_dim: Number of observation dimensions
        """
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # History for computing derivatives
        self._F_history = []
        self._max_history = 100

    def compute(self,
                Q_s: np.ndarray,           # Belief over states Q(s), shape: (state_dim,)
                log_P_o_given_s: np.ndarray,  # Log likelihood log P(o|s), shape: (state_dim,)
                log_P_s: np.ndarray           # Log prior log P(s), shape: (state_dim,)
                ) -> FreeEnergyState:
        """
        Compute Free Energy.

        F = -H[Q(s)] - E_Q[log P(o|s)] - E_Q[log P(s)]
          = -H[Q(s)] + E_Q[-log P(o|s)] + E_Q[-log P(s)]
          = -entropy + prediction_error + complexity_prior

        Or equivalently:
        F = E_Q[log Q(s) - log P(o,s)]
          = E_Q[log Q(s) - log P(o|s) - log P(s)]
        """
        # Ensure valid probability distribution
        Q_s = np.clip(Q_s, 1e-10, 1.0)
        Q_s = Q_s / Q_s.sum()

        # Entropy of belief: H[Q(s)] = -sum Q(s) log Q(s)
        entropy = -np.sum(Q_s * np.log(Q_s + 1e-10))

        # Prediction error: E_Q[-log P(o|s)] = -sum Q(s) log P(o|s)
        prediction_error = -np.sum(Q_s * log_P_o_given_s)

        # Complexity (KL from prior): E_Q[log Q(s) - log P(s)]
        # This penalizes beliefs that deviate from prior
        complexity = np.sum(Q_s * (np.log(Q_s + 1e-10) - log_P_s))

        # Total Free Energy
        # F = -entropy + prediction_error + complexity_prior
        # But complexity already includes the entropy term in KL form
        # So: F = prediction_error + complexity
        F = prediction_error + complexity

        # Compute rate of change
        dF_dt = 0.0
        if len(self._F_history) > 0:
            dF_dt = F - self._F_history[-1]

        # Store history
        self._F_history.append(F)
        if len(self._F_history) > self._max_history:
            self._F_history.pop(0)

        return FreeEnergyState(
            F=F,
            prediction_error=prediction_error,
            complexity=complexity,
            entropy=entropy,
            dF_dt=dF_dt
        )

    def compute_expected(self,
                         Q_s: np.ndarray,
                         P_o_given_s_a: np.ndarray,  # P(o|s,a) for each action
                         P_s_given_s_a: np.ndarray,  # P(s'|s,a) transition
                         log_P_s_prior: np.ndarray,
                         C: np.ndarray               # Prior preference over observations
                         ) -> Dict[int, float]:
        """
        Compute Expected Free Energy G(a) for each action.

        G(a) = E_Q(o,s|a)[log Q(s|a) - log P(o,s|a)]
             = E_Q[log Q(s|a) - log Q(s|o,a) - log P(o|a)]  # (rearranging)

        This decomposes into:
        G(a) = -E_Q[H[Q(s|o,a)]]     # Epistemic value (information gain)
             - E_Q[log P(o)]         # Pragmatic value (reaching preferred states)

        Actions that minimize G:
        1. Reduce uncertainty about states (epistemic)
        2. Lead to preferred observations (pragmatic)

        Args:
            Q_s: Current belief over states
            P_o_given_s_a: Observation model P(o|s,a), shape: (n_actions, state_dim, obs_dim)
            P_s_given_s_a: Transition model P(s'|s,a), shape: (n_actions, state_dim, state_dim)
            log_P_s_prior: Log prior over states
            C: Prior preference over observations (log scale), shape: (obs_dim,)

        Returns:
            Dict mapping action index to expected free energy G(a)
        """
        n_actions = P_o_given_s_a.shape[0]
        G = {}

        for a in range(n_actions):
            # Predict next state distribution: Q(s'|a) = sum_s P(s'|s,a) Q(s)
            Q_s_next = P_s_given_s_a[a].T @ Q_s  # (state_dim,)
            Q_s_next = np.clip(Q_s_next, 1e-10, 1.0)
            Q_s_next = Q_s_next / Q_s_next.sum()

            # Predict observation distribution: Q(o|a) = sum_s' P(o|s',a) Q(s'|a)
            Q_o = P_o_given_s_a[a].T @ Q_s_next  # (obs_dim,)
            Q_o = np.clip(Q_o, 1e-10, 1.0)
            Q_o = Q_o / Q_o.sum()

            # Epistemic value: expected information gain
            # This is the expected reduction in uncertainty about states
            # after observing o
            # IG = H[Q(s'|a)] - E_Q(o|a)[H[Q(s'|o,a)]]
            H_s_prior = -np.sum(Q_s_next * np.log(Q_s_next + 1e-10))

            # Expected conditional entropy (approximation)
            # For each possible observation, compute posterior entropy
            H_s_posterior = 0.0
            for o_idx in range(self.obs_dim):
                if Q_o[o_idx] > 1e-10:
                    # Posterior Q(s'|o,a) via Bayes
                    likelihood = P_o_given_s_a[a, :, o_idx]  # P(o|s',a)
                    Q_s_given_o = likelihood * Q_s_next
                    if Q_s_given_o.sum() > 1e-10:
                        Q_s_given_o = Q_s_given_o / Q_s_given_o.sum()
                        H_o = -np.sum(Q_s_given_o * np.log(Q_s_given_o + 1e-10))
                        H_s_posterior += Q_o[o_idx] * H_o

            information_gain = H_s_prior - H_s_posterior
            epistemic_value = -information_gain  # Negative because we want to maximize IG

            # Pragmatic value: expected log preference
            # How much do the expected observations match our preferences?
            pragmatic_value = -np.sum(Q_o * C)  # Negative log preference

            # Expected Free Energy
            G[a] = epistemic_value + pragmatic_value

        return G

    def get_F_trajectory(self) -> Dict:
        """Get trajectory of F over time."""
        if len(self._F_history) < 2:
            return {
                'current': self._F_history[-1] if self._F_history else 0.0,
                'trend': 'unknown',
                'mean': 0.0
            }

        recent = self._F_history[-10:]
        older = self._F_history[-20:-10] if len(self._F_history) >= 20 else self._F_history[:10]

        mean_recent = np.mean(recent)
        mean_older = np.mean(older) if older else mean_recent

        if mean_recent < mean_older * 0.9:
            trend = 'decreasing'  # F is going down - good
        elif mean_recent > mean_older * 1.1:
            trend = 'increasing'  # F is going up - bad
        else:
            trend = 'stable'

        return {
            'current': self._F_history[-1],
            'trend': trend,
            'mean': mean_recent,
            'history': list(self._F_history[-20:])
        }

    def reset(self):
        """Reset history."""
        self._F_history = []
