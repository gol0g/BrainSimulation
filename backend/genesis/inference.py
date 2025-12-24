"""
Inference Engine - Perception as Free Energy Minimization

Perception is not passive reception of data.
Perception is ACTIVE INFERENCE - updating beliefs to minimize Free Energy.

Q*(s) = argmin_Q F(Q)
      = argmin_Q { -H[Q(s)] + E_Q[-log P(o|s)] + E_Q[-log P(s)] }

The solution is:
    Q*(s) ∝ P(o|s) × P(s)  (Bayes' theorem)

But we don't compute this directly.
We iteratively update Q(s) to minimize F.
This is what the brain does - gradient descent on prediction error.

Why iterative instead of direct Bayes?
    1. The brain doesn't have access to P(o|s) directly
    2. The brain must learn P(o|s) from experience
    3. Iterative update allows integration of temporal dynamics
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .free_energy import FreeEnergyEngine, FreeEnergyState
from .generative_model import GenerativeModel


@dataclass
class InferenceResult:
    """Result of inference step."""
    Q_s: np.ndarray           # Updated belief over states
    F: FreeEnergyState        # Free energy after update
    prediction_error: float   # Surprise of observation
    belief_update: float      # How much belief changed
    converged: bool           # Did inference converge?


class InferenceEngine:
    """
    Updates beliefs to minimize Free Energy.

    This is PERCEPTION.

    When an observation arrives:
    1. Compute prediction error (observation vs. expectation)
    2. Update beliefs to reduce prediction error
    3. But also consider prior beliefs (don't update too much)

    The balance between "trusting observations" and "trusting prior"
    is determined by PRECISION.
    """

    def __init__(self,
                 model: GenerativeModel,
                 free_energy: FreeEnergyEngine):
        """
        Initialize inference engine.

        Args:
            model: The generative model
            free_energy: The free energy engine
        """
        self.model = model
        self.fe = free_energy

        # === INFERENCE PARAMETERS ===
        self.n_iterations = 16      # Max gradient steps
        self.learning_rate = 0.5    # Step size
        self.convergence_threshold = 0.001  # When to stop

    def infer(self,
              observation: np.ndarray,
              Q_s_prior: Optional[np.ndarray] = None
              ) -> InferenceResult:
        """
        Perform inference given an observation.

        Update Q(s) to minimize F given the observation.

        Args:
            observation: The sensory observation
            Q_s_prior: Prior belief (uses model's current belief if None)

        Returns:
            InferenceResult with updated beliefs
        """
        if Q_s_prior is None:
            Q_s_prior = self.model.Q_s.copy()

        Q_s = Q_s_prior.copy()
        initial_Q_s = Q_s.copy()

        # Get log likelihood P(o|s) for each state
        log_P_o_given_s = self.model.get_log_likelihood(observation)

        # Get log prior P(s)
        log_P_s = np.log(self.model.D + 1e-10)

        # Iterative Free Energy minimization
        converged = False
        for iteration in range(self.n_iterations):
            # Compute current F
            F = self.fe.compute(Q_s, log_P_o_given_s, log_P_s)

            # Compute gradient of F with respect to Q(s)
            # dF/dQ(s) = log Q(s) + 1 - log P(o|s) - log P(s)
            # (Using natural gradient for probability simplex)
            gradient = np.log(Q_s + 1e-10) - log_P_o_given_s - log_P_s

            # Update Q(s) in the direction that reduces F
            # Use softmax to stay on probability simplex
            log_Q_s = np.log(Q_s + 1e-10) - self.learning_rate * gradient
            Q_s_new = np.exp(log_Q_s - np.max(log_Q_s))  # Numerical stability
            Q_s_new = Q_s_new / Q_s_new.sum()

            # Check convergence
            update_size = np.abs(Q_s_new - Q_s).max()
            Q_s = Q_s_new

            if update_size < self.convergence_threshold:
                converged = True
                break

        # Store updated belief
        self.model.Q_s = Q_s

        # Compute final F
        F_final = self.fe.compute(Q_s, log_P_o_given_s, log_P_s)

        # Compute how much belief changed
        belief_update = np.abs(Q_s - initial_Q_s).sum()

        # Compute prediction error
        prediction_error = self.model.compute_prediction_error(observation)

        return InferenceResult(
            Q_s=Q_s,
            F=F_final,
            prediction_error=prediction_error,
            belief_update=belief_update,
            converged=converged
        )

    def infer_with_action(self,
                          observation: np.ndarray,
                          action: int,
                          Q_s_prev: np.ndarray
                          ) -> InferenceResult:
        """
        Infer state after taking an action.

        This incorporates the transition model.

        Q(s') ∝ P(o|s') × sum_s P(s'|s,a) Q(s)

        Args:
            observation: Observation after action
            action: Action that was taken
            Q_s_prev: Belief before action

        Returns:
            InferenceResult with updated beliefs
        """
        # Predict next state based on action
        Q_s_predicted = self.model.predict_next_state(action, Q_s_prev)

        # Use this as prior for inference
        return self.infer(observation, Q_s_prior=Q_s_predicted)

    def get_precision_weighted_prediction_error(self,
                                                 observation: np.ndarray) -> float:
        """
        Compute precision-weighted prediction error.

        High precision = "trust this error"
        Low precision = "ignore this error"

        Args:
            observation: The observation

        Returns:
            Precision-weighted prediction error
        """
        pe = self.model.compute_prediction_error(observation)
        return pe * self.model.precision_A

    def update_precision(self,
                         prediction_error: float,
                         expected_error: float = 0.5):
        """
        Update precision based on prediction error history.

        If errors are consistently low, increase precision (trust observations more).
        If errors are consistently high, decrease precision (trust observations less).

        Args:
            prediction_error: Current prediction error
            expected_error: Expected level of error
        """
        # Simple precision update rule
        # precision += learning_rate × (expected_error - actual_error)
        lr = 0.05
        precision_change = lr * (expected_error - prediction_error)
        self.model.precision_A = np.clip(
            self.model.precision_A + precision_change,
            0.1, 10.0
        )
