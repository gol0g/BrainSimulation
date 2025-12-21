"""
Self-Agency Module for Consciousness Simulation

This module implements the Forward Model and Agency Detection system.
The core idea: "Did I cause this change, or did something external cause it?"

Components:
1. Forward Model: Predicts next sensory state from (current state + action)
2. Comparator: Computes prediction error (predicted vs actual)
3. Agency Integrator: Converts prediction error to agency signal

When prediction error is LOW → HIGH agency ("I caused this")
When prediction error is HIGH → LOW agency ("External force")
"""

import math
from typing import Dict, Optional, List
from collections import deque


class ForwardModel:
    """
    Learns to predict: current_sensory + action → next_sensory

    Uses simple associative learning (Hebbian-like):
    - When (state, action) → outcome is observed, strengthen that association
    - Prediction is weighted sum of learned associations
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

        # Prediction weights: weights[action][sensory_input][predicted_sensory]
        # e.g., "If I go UP when food is RIGHT, food will be UP-RIGHT"
        self.directions = ['up', 'down', 'left', 'right']

        # Initialize prediction matrix
        # For each action, learn what sensory change to expect
        # weights[action][current_dir] = expected_change per direction
        self.weights: Dict[str, Dict[str, Dict[str, float]]] = {}

        for action in self.directions:
            self.weights[action] = {}
            for curr_dir in self.directions:
                self.weights[action][curr_dir] = {}
                for next_dir in self.directions:
                    # Initial prediction: action moves you in that direction
                    # So if food is UP and you go UP, food might be NONE (you reached it)
                    # If food is UP and you go DOWN, food is still UP (maybe more UP)
                    self.weights[action][curr_dir][next_dir] = 0.0

        # History for learning
        self.last_sensory: Optional[Dict[str, float]] = None
        self.last_action: Optional[str] = None
        self.last_prediction: Optional[Dict[str, float]] = None

        # Agency tracking
        self.agency_history: deque = deque(maxlen=100)
        self.current_agency: float = 1.0  # Start with high agency

    def predict(self, current_sensory: Dict[str, float], action: str) -> Dict[str, float]:
        """
        Given current sensory state and action, predict next sensory state.

        Args:
            current_sensory: {direction: intensity} e.g., {'up': 0.8, 'down': 0.0, ...}
            action: 'up', 'down', 'left', 'right', or 'stay'

        Returns:
            Predicted sensory state
        """
        if action == 'stay' or action not in self.directions:
            # Staying = no change predicted
            return current_sensory.copy()

        predicted = {d: 0.0 for d in self.directions}

        # Simple prediction model:
        # For each current sensory input, weight contributes to prediction
        for curr_dir, intensity in current_sensory.items():
            if curr_dir not in self.directions:
                continue
            for next_dir in self.directions:
                weight = self.weights[action][curr_dir][next_dir]
                predicted[next_dir] += intensity * weight

        # Normalize predictions to [0, 1]
        max_pred = max(abs(v) for v in predicted.values()) if predicted else 1.0
        if max_pred > 0:
            predicted = {k: max(0, min(1, v / max_pred)) for k, v in predicted.items()}

        # Store for learning
        self.last_sensory = current_sensory.copy()
        self.last_action = action
        self.last_prediction = predicted.copy()

        return predicted

    def learn(self, actual_sensory: Dict[str, float]) -> float:
        """
        Learn from the actual outcome. Returns prediction error.

        Args:
            actual_sensory: The actual sensory state that occurred

        Returns:
            Prediction error (0 = perfect prediction, higher = worse)
        """
        if self.last_prediction is None or self.last_action is None:
            return 0.0

        if self.last_action not in self.directions:
            return 0.0

        # Compute prediction error
        error = 0.0
        for direction in self.directions:
            predicted = self.last_prediction.get(direction, 0.0)
            actual = actual_sensory.get(direction, 0.0)
            error += (predicted - actual) ** 2
        error = math.sqrt(error / len(self.directions))  # RMSE

        # Update weights using delta rule
        for curr_dir, intensity in self.last_sensory.items():
            if curr_dir not in self.directions or intensity < 0.1:
                continue
            for next_dir in self.directions:
                predicted = self.last_prediction.get(next_dir, 0.0)
                actual = actual_sensory.get(next_dir, 0.0)
                delta = actual - predicted

                # Hebbian update: strengthen associations that predict correctly
                self.weights[self.last_action][curr_dir][next_dir] += \
                    self.learning_rate * delta * intensity

        # Clear history
        self.last_sensory = None
        self.last_action = None
        self.last_prediction = None

        return error


class AgencyDetector:
    """
    Detects whether changes are self-caused or externally-caused.

    Uses the Forward Model's prediction error:
    - Low error = high agency (I predicted this, so I caused it)
    - High error = low agency (unexpected change = external cause)

    Also detects "external pushes" by comparing intended action with actual movement.

    NEW: Self-state conditioning for self-explanation capability:
    - When uncertainty is high, expect higher prediction errors
    - "I am confused, so errors are expected" → less surprising
    """

    def __init__(self):
        self.forward_model = ForwardModel(learning_rate=0.15)

        # Agency state
        self.agency_level: float = 1.0  # 0 = external control, 1 = full agency
        self.agency_history: deque = deque(maxlen=50)

        # Prediction tracking
        self.prediction_errors: deque = deque(maxlen=20)

        # External push detection
        self.was_pushed: bool = False
        self.push_direction: Optional[str] = None

        # Thresholds
        self.HIGH_AGENCY_THRESHOLD = 0.3  # Error below this = high agency
        self.LOW_AGENCY_THRESHOLD = 0.7   # Error above this = low agency

        # === Self-State Conditioning (for self-explanation) ===
        # When uncertainty is high, we expect higher prediction errors
        # This creates "self-knowledge": "I know I'm confused, so errors are normal"
        self.expected_error_base: float = 0.2  # Baseline expected error
        self.uncertainty_error_scale: float = 0.4  # How much uncertainty adds to expected error
        self.effort_error_scale: float = 0.2  # How much effort/fatigue adds to expected error

        # Track self-explanation events
        self.last_expected_error: float = 0.0
        self.last_adjusted_error: float = 0.0
        self.self_explanation_active: bool = False

    def on_action_taken(self, current_sensory: Dict[str, float], action: str):
        """Called when the agent takes an action. Stores prediction."""
        self.forward_model.predict(current_sensory, action)
        self.was_pushed = False
        self.push_direction = None

    def on_external_push(self, push_direction: str):
        """Called when agent is pushed by external force."""
        self.was_pushed = True
        self.push_direction = push_direction
        # This should cause high prediction error (agent didn't predict this)

    def update(self, actual_sensory: Dict[str, float],
               self_state: Optional[Dict[str, float]] = None) -> Dict:
        """
        Update agency detection with actual sensory outcome.

        NEW: Self-state conditioning for self-explanation:
        - uncertainty ↑ → expected_error ↑ → adjusted_error ↓
        - "I am confused, so this error level is expected"

        Args:
            actual_sensory: The actual sensory state that occurred
            self_state: Optional self-model state for self-explanation
                       {'uncertainty': float, 'effort': float, ...}

        Returns:
            Dict with agency info:
            - agency_level: 0.0 to 1.0
            - prediction_error: float (raw)
            - adjusted_error: float (after self-state conditioning)
            - expected_error: float (what we expected given our state)
            - was_pushed: bool
            - interpretation: str
            - self_explanation: str (if self-state reduced perceived error)
        """
        # Get raw prediction error from forward model
        raw_error = self.forward_model.learn(actual_sensory)
        self.prediction_errors.append(raw_error)

        # If externally pushed, add extra "surprise"
        if self.was_pushed:
            raw_error = min(1.0, raw_error + 0.5)

        # === Self-State Conditioning: Compute Expected Error ===
        # "Given my current state, how much error do I expect?"
        self.self_explanation_active = False
        self_explanation = ""

        if self_state:
            uncertainty = self_state.get('uncertainty', 0.0)
            effort = self_state.get('effort', 0.0)

            # Expected error increases with uncertainty and effort
            # "When I'm confused (high uncertainty), I expect to make more errors"
            # "When I'm tired (high effort), my predictions are less accurate"
            self.last_expected_error = (
                self.expected_error_base +
                uncertainty * self.uncertainty_error_scale +
                effort * self.effort_error_scale
            )

            # Adjusted error: how surprising is this error given my self-state?
            # If raw_error <= expected_error: "This is normal for my state"
            # If raw_error > expected_error: "This is surprising even for my state"
            self.last_adjusted_error = max(0.0, raw_error - self.last_expected_error * 0.5)

            # Generate self-explanation if self-state significantly reduced perceived error
            error_reduction = raw_error - self.last_adjusted_error
            if error_reduction > 0.1:
                self.self_explanation_active = True
                if uncertainty > effort:
                    self_explanation = f"Error expected: I'm confused (u={uncertainty:.2f})"
                else:
                    self_explanation = f"Error expected: I'm fatigued (e={effort:.2f})"

            # Use adjusted error for agency calculation
            effective_error = self.last_adjusted_error
        else:
            self.last_expected_error = self.expected_error_base
            self.last_adjusted_error = raw_error
            effective_error = raw_error

        # Update agency level with exponential smoothing (using effective_error)
        if effective_error < self.HIGH_AGENCY_THRESHOLD:
            # High agency - I predicted this correctly (or error was expected)
            target_agency = 1.0
        elif effective_error > self.LOW_AGENCY_THRESHOLD:
            # Low agency - unexpectedly high error
            target_agency = 0.2
        else:
            # Middle range - interpolate
            t = (effective_error - self.HIGH_AGENCY_THRESHOLD) / \
                (self.LOW_AGENCY_THRESHOLD - self.HIGH_AGENCY_THRESHOLD)
            target_agency = 1.0 - 0.8 * t

        # Smooth transition
        self.agency_level = 0.7 * self.agency_level + 0.3 * target_agency
        self.agency_history.append(self.agency_level)

        # Interpretation
        if self.was_pushed:
            interpretation = "EXTERNAL_PUSH"
        elif self.agency_level > 0.7:
            interpretation = "SELF_CAUSED"
        elif self.agency_level < 0.4:
            interpretation = "EXTERNAL_CAUSE"
        else:
            interpretation = "UNCERTAIN"

        return {
            "agency_level": round(self.agency_level, 3),
            "prediction_error": round(raw_error, 3),
            "adjusted_error": round(self.last_adjusted_error, 3),
            "expected_error": round(self.last_expected_error, 3),
            "was_pushed": self.was_pushed,
            "interpretation": interpretation,
            "self_explanation": self_explanation,
            "self_explanation_active": self.self_explanation_active,
            "avg_error": round(sum(self.prediction_errors) / len(self.prediction_errors), 3) \
                        if self.prediction_errors else 0.0
        }

    def get_dopamine_modifier(self) -> float:
        """
        Returns a modifier for dopamine signal based on agency.

        High agency = full dopamine effect (I earned this reward)
        Low agency = reduced dopamine effect (I didn't cause this)
        """
        # Scale from 0.3 (low agency) to 1.0 (high agency)
        return 0.3 + 0.7 * self.agency_level

    def to_dict(self) -> Dict:
        """Get current agency state for API response."""
        return {
            "agency_level": round(self.agency_level, 3),
            "interpretation": "SELF_CAUSED" if self.agency_level > 0.7 else \
                            "EXTERNAL_CAUSE" if self.agency_level < 0.4 else "UNCERTAIN",
            "recent_errors": [round(e, 3) for e in list(self.prediction_errors)[-5:]],
            "was_pushed": self.was_pushed,
            # Self-explanation info
            "expected_error": round(self.last_expected_error, 3),
            "adjusted_error": round(self.last_adjusted_error, 3),
            "self_explanation_active": self.self_explanation_active
        }
