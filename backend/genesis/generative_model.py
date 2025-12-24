"""
Generative Model - The Agent's Model of the World

P(o, s, s', a) = P(o|s) × P(s'|s,a) × P(s) × P(a)

Components:
    - P(o|s): Observation model (likelihood)
        "If the world is in state s, what would I observe?"

    - P(s'|s,a): Transition model (dynamics)
        "If I do action a in state s, what state will I be in?"

    - P(s): Prior over states
        "What states do I expect to be in?" (homeostatic setpoint)

    - P(a): Prior over actions
        (derived from Expected Free Energy)

The generative model is learned through experience.
It is NOT the actual world - it's the agent's BELIEF about the world.
Prediction errors arise when the model doesn't match reality.

Key insight:
    The agent doesn't know the "true" state of the world.
    It only has observations and must INFER the hidden state.
    This inference is what we call perception.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Observation:
    """Sensory observation from the environment."""
    raw: np.ndarray           # Raw observation vector
    timestamp: int = 0


@dataclass
class ModelState:
    """Internal state of the generative model."""
    # Probability distributions (beliefs)
    Q_s: np.ndarray          # Belief over current state Q(s)
    Q_s_prev: np.ndarray     # Previous belief (for computing prediction error)

    # Model parameters (learned)
    A: np.ndarray            # Observation model P(o|s)
    B: np.ndarray            # Transition model P(s'|s,a)
    C: np.ndarray            # Prior preference over observations
    D: np.ndarray            # Prior over initial state

    # Precision (confidence in model components)
    precision_A: float = 1.0  # Confidence in observation model
    precision_B: float = 1.0  # Confidence in transition model


class GenerativeModel:
    """
    The agent's internal model of the world.

    This model generates predictions about:
    1. What I will observe given my beliefs about the world
    2. How the world will change given my actions

    The mismatch between predictions and reality is the prediction error.
    Minimizing Free Energy means either:
    - Updating beliefs to match observations (perception)
    - Acting to make observations match predictions (action)
    """

    def __init__(self,
                 n_states: int,
                 n_observations: int,
                 n_actions: int):
        """
        Initialize the generative model.

        Args:
            n_states: Number of hidden states
            n_observations: Dimension of observation space
            n_actions: Number of possible actions
        """
        self.n_states = n_states
        self.n_obs = n_observations
        self.n_actions = n_actions

        # === OBSERVATION MODEL: P(o|s) ===
        # "What would I observe if the world is in state s?"
        # Shape: (n_states, n_observations)
        # Initially uniform (no knowledge)
        self.A = np.ones((n_states, n_observations)) / n_observations

        # === TRANSITION MODEL: P(s'|s,a) ===
        # "If I do action a in state s, what state will I be in?"
        # Shape: (n_actions, n_states, n_states)
        # Initially identity (actions don't change state - will be learned)
        self.B = np.zeros((n_actions, n_states, n_states))
        for a in range(n_actions):
            self.B[a] = np.eye(n_states)

        # === PRIOR PREFERENCE: P(o) or C ===
        # "What observations do I prefer?"
        # This encodes homeostatic setpoints and goals
        # Shape: (n_observations,)
        # Log scale: C[o] = log P(o) where P(o) is preferred distribution
        # Uniform means no preference
        self.C = np.zeros(n_observations)  # log(1/n) for all = uniform

        # === INITIAL STATE PRIOR: P(s_0) or D ===
        # "What state do I expect to start in?"
        # Shape: (n_states,)
        self.D = np.ones(n_states) / n_states  # Uniform initially

        # === CURRENT BELIEF: Q(s) ===
        # This is updated by inference
        self.Q_s = np.ones(n_states) / n_states

        # === PRECISION (CONFIDENCE) ===
        # How confident am I in my model components?
        self.precision_A = 1.0  # Observation model confidence
        self.precision_B = 1.0  # Transition model confidence

        # === LEARNING RATE ===
        self.learning_rate = 0.1

        # === HISTORY ===
        self._observation_history = []
        self._action_history = []
        self._state_history = []

    def predict_observation(self, Q_s: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict observation distribution given current belief.

        P(o) = sum_s P(o|s) Q(s)

        Returns:
            Predicted observation distribution, shape: (n_observations,)
        """
        if Q_s is None:
            Q_s = self.Q_s

        # Weighted sum over states
        P_o = self.A.T @ Q_s  # (n_obs,)
        P_o = np.clip(P_o, 1e-10, 1.0)
        return P_o / P_o.sum()

    def predict_next_state(self, action: int, Q_s: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict next state distribution given action.

        Q(s') = sum_s P(s'|s,a) Q(s)

        Args:
            action: Action taken
            Q_s: Current belief (uses self.Q_s if None)

        Returns:
            Predicted next state distribution, shape: (n_states,)
        """
        if Q_s is None:
            Q_s = self.Q_s

        Q_s_next = self.B[action].T @ Q_s
        Q_s_next = np.clip(Q_s_next, 1e-10, 1.0)
        return Q_s_next / Q_s_next.sum()

    def compute_prediction_error(self, observation: np.ndarray) -> float:
        """
        Compute prediction error for an observation.

        PE = -log P(o) where P(o) = sum_s P(o|s) Q(s)

        This is the "surprise" of the observation given our beliefs.

        Args:
            observation: Observed values (can be continuous or one-hot)

        Returns:
            Prediction error (negative log likelihood)
        """
        # Predict observation distribution
        P_o = self.predict_observation()

        # If observation is one-hot (discrete)
        if len(observation.shape) == 1 and observation.sum() == 1:
            o_idx = np.argmax(observation)
            return -np.log(P_o[o_idx] + 1e-10)

        # If observation is continuous (treat as log-likelihood)
        # Use Gaussian likelihood centered on prediction
        predicted_mean = np.argmax(P_o)
        actual = np.argmax(observation) if observation.max() > 0 else 0
        error = abs(predicted_mean - actual) / self.n_obs
        return error * self.precision_A

    def get_log_likelihood(self, observation: np.ndarray) -> np.ndarray:
        """
        Get log P(o|s) for each state.

        Args:
            observation: Observed values

        Returns:
            Log likelihood for each state, shape: (n_states,)
        """
        # If observation is one-hot
        if len(observation.shape) == 1 and np.isclose(observation.sum(), 1.0):
            o_idx = np.argmax(observation)
            return np.log(self.A[:, o_idx] + 1e-10) * self.precision_A

        # If observation is soft distribution
        log_lik = np.zeros(self.n_states)
        for s in range(self.n_states):
            # Cross-entropy between observation and model prediction
            P_o_s = self.A[s]
            log_lik[s] = np.sum(observation * np.log(P_o_s + 1e-10))
        return log_lik * self.precision_A

    def update_model(self,
                     observation: np.ndarray,
                     action: int,
                     Q_s_prev: np.ndarray,
                     Q_s_curr: np.ndarray):
        """
        Learn from experience.

        Update A and B matrices based on prediction errors.

        Args:
            observation: What was observed
            action: What action was taken
            Q_s_prev: Belief before action
            Q_s_curr: Belief after observation (posterior)
        """
        lr = self.learning_rate

        # === UPDATE OBSERVATION MODEL A ===
        # If we observed o and believe we're in state s,
        # increase P(o|s)
        for s in range(self.n_states):
            # Weight by belief in state s
            weight = Q_s_curr[s]
            # Move A[s] toward observation
            self.A[s] = (1 - lr * weight) * self.A[s] + lr * weight * observation
            # Normalize
            self.A[s] = np.clip(self.A[s], 1e-10, 1.0)
            self.A[s] = self.A[s] / self.A[s].sum()

        # === UPDATE TRANSITION MODEL B ===
        # If we believed s_prev, took action a, and now believe s_curr,
        # update P(s_curr|s_prev, a)
        for s_prev in range(self.n_states):
            for s_curr in range(self.n_states):
                weight = Q_s_prev[s_prev] * Q_s_curr[s_curr]
                self.B[action, s_prev, s_curr] = (
                    (1 - lr * weight) * self.B[action, s_prev, s_curr] +
                    lr * weight
                )
            # Normalize
            row_sum = self.B[action, s_prev].sum()
            if row_sum > 0:
                self.B[action, s_prev] /= row_sum

    def set_preference(self, preferred_observations: np.ndarray):
        """
        Set prior preference over observations.

        This encodes what observations the agent "prefers" -
        i.e., its homeostatic setpoints.

        Args:
            preferred_observations: Desired observation distribution
        """
        preferred_observations = np.clip(preferred_observations, 1e-10, 1.0)
        preferred_observations = preferred_observations / preferred_observations.sum()
        self.C = np.log(preferred_observations)

    def get_preference_divergence(self, observation: np.ndarray) -> float:
        """
        How far is this observation from our preferences?

        This is related to "pragmatic value" in EFE.

        Args:
            observation: Current observation

        Returns:
            Divergence from preferred observations (higher = worse)
        """
        # KL divergence from observation to preference
        P_o = observation / (observation.sum() + 1e-10)
        P_pref = np.exp(self.C)
        P_pref = P_pref / P_pref.sum()

        kl = np.sum(P_o * (np.log(P_o + 1e-10) - np.log(P_pref + 1e-10)))
        return max(0.0, kl)

    def reset(self):
        """Reset beliefs to prior."""
        self.Q_s = self.D.copy()
        self._observation_history = []
        self._action_history = []
        self._state_history = []

    def get_state(self) -> ModelState:
        """Get current model state."""
        return ModelState(
            Q_s=self.Q_s.copy(),
            Q_s_prev=self._state_history[-1] if self._state_history else self.Q_s.copy(),
            A=self.A.copy(),
            B=self.B.copy(),
            C=self.C.copy(),
            D=self.D.copy(),
            precision_A=self.precision_A,
            precision_B=self.precision_B
        )
