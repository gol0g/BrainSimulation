"""
World Model - Prediction and Error System

Core Concept: Predictive Processing
- The brain is fundamentally a prediction machine
- It constantly predicts what will happen next
- Errors between prediction and reality drive:
  - Learning (update the model)
  - Action (change the world to match predictions)

Two Types of Prediction Error:
1. External (Exteroceptive): "The world didn't match my prediction"
   - Sensory prediction errors
   - Drives attention, curiosity, learning

2. Internal (Interoceptive): "My body state didn't match my setpoint"
   - Viability deviations
   - Drives homeostatic behavior, "emotions"

Key insight:
- Emotions are not labels, they're patterns of prediction error
- Fear = predicted future viability is low
- Satisfaction = prediction errors decreasing + viability improving
- Curiosity = high uncertainty, low threat, information gain possible
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
from collections import deque


@dataclass
class Prediction:
    """A prediction about the next state."""
    # External predictions
    food_direction: Optional[str] = None  # Expected food direction
    threat_direction: Optional[str] = None  # Expected threat direction
    expected_sensory: Dict[str, float] = field(default_factory=dict)

    # Internal predictions
    expected_energy: float = 0.0
    expected_integrity: float = 0.0
    expected_threat_exposure: float = 0.0

    # Action outcome predictions
    action: Optional[str] = None
    expected_reward: float = 0.0
    expected_p_absorb: float = 0.0

    # Confidence in predictions
    confidence: float = 0.5


@dataclass
class PredictionError:
    """Discrepancy between prediction and reality."""
    # External errors
    sensory_error: float = 0.0  # How wrong were sensory predictions
    action_error: float = 0.0   # Did action outcome match expectation

    # Internal errors
    energy_error: float = 0.0      # Deviation from energy setpoint
    integrity_error: float = 0.0   # Deviation from integrity setpoint
    threat_error: float = 0.0      # Unexpected threat level

    # Aggregate metrics
    total_external: float = 0.0
    total_internal: float = 0.0

    # Precision-weighted (how much to trust this error)
    precision: float = 1.0


class WorldModel:
    """
    Maintains predictions about world and self, computes errors.

    This is the heart of the predictive processing framework:
    - Predictions are generated before each action
    - After action, errors are computed
    - Errors drive learning and future behavior

    The model learns from experience:
    - Which directions tend to have food/threats
    - What outcomes actions lead to
    - How the internal state evolves
    """

    def __init__(self):
        # === SPATIAL PREDICTIONS ===
        # Expected value of each direction (food, threat, wall)
        self.direction_values = {
            'up': {'food': 0.0, 'threat': 0.0, 'wall': 0.0},
            'down': {'food': 0.0, 'threat': 0.0, 'wall': 0.0},
            'left': {'food': 0.0, 'threat': 0.0, 'wall': 0.0},
            'right': {'food': 0.0, 'threat': 0.0, 'wall': 0.0}
        }

        # === INTERNAL SETPOINTS ===
        # Where the system "wants" to be (viability targets)
        self.setpoints = {
            'energy': 0.7,      # Target energy level
            'integrity': 0.9,   # Target integrity
            'threat_exposure': 0.1  # Target threat exposure
        }

        # === PREDICTION HISTORY ===
        self._last_prediction: Optional[Prediction] = None
        self._error_history = deque(maxlen=50)
        self._confidence = 0.5

        # === LEARNING PARAMETERS ===
        self._learning_rate = 0.1
        self._error_decay = 0.95

        # === UNCERTAINTY TRACKING ===
        # High uncertainty = model is unreliable
        self._uncertainty = 0.5
        self._surprise_history = deque(maxlen=20)

    def predict(self,
                current_sensory: Dict[str, float],
                current_viability: Dict,
                intended_action: str) -> Prediction:
        """
        Generate predictions for the next timestep.

        This happens BEFORE action execution.
        """
        pred = Prediction()
        pred.action = intended_action
        pred.confidence = self._confidence

        # === SENSORY PREDICTIONS ===
        # Based on current sensory + learned direction values
        pred.expected_sensory = {}
        for direction, sensors in [
            ('up', 'food_up'), ('down', 'food_down'),
            ('left', 'food_left'), ('right', 'food_right')
        ]:
            base = current_sensory.get(sensors.replace('food_', ''), 0)
            learned = self.direction_values[direction]['food']
            pred.expected_sensory[direction] = base * 0.7 + learned * 0.3

        # === INTERNAL PREDICTIONS ===
        state = current_viability.get('state', {})
        rates = current_viability.get('rates', {})

        # Predict based on current rates
        pred.expected_energy = state.get('energy', 0.5) + rates.get('d_energy', 0)
        pred.expected_integrity = state.get('integrity', 1.0) + rates.get('d_integrity', 0)
        pred.expected_threat_exposure = state.get('threat_exposure', 0.0)

        # === ACTION OUTCOME PREDICTION ===
        if intended_action in self.direction_values:
            dir_vals = self.direction_values[intended_action]
            # Expected reward based on food probability
            pred.expected_reward = dir_vals['food'] * 10 - dir_vals['threat'] * 5
            # Expected threat based on danger
            pred.expected_p_absorb = current_viability.get('p_absorb', 0) + dir_vals['threat'] * 0.2

        # Save for later comparison
        self._last_prediction = pred
        return pred

    def compute_error(self,
                      actual_sensory: Dict[str, float],
                      actual_viability: Dict,
                      actual_reward: float,
                      hit_wall: bool = False,
                      took_damage: bool = False) -> PredictionError:
        """
        Compute prediction errors after observing outcomes.

        This is where the magic happens:
        - Large errors → something unexpected → attention + learning
        - Consistent errors → model needs updating
        - Prediction errors ARE the signal for everything
        """
        err = PredictionError()

        if self._last_prediction is None:
            return err

        pred = self._last_prediction

        # === SENSORY PREDICTION ERROR ===
        sensory_errors = []
        for direction in ['up', 'down', 'left', 'right']:
            expected = pred.expected_sensory.get(direction, 0)
            actual = actual_sensory.get(f'food_{direction}', 0)
            sensory_errors.append(abs(expected - actual))

        err.sensory_error = sum(sensory_errors) / len(sensory_errors) if sensory_errors else 0

        # === ACTION OUTCOME ERROR ===
        err.action_error = abs(pred.expected_reward - actual_reward) / 10.0

        # Unexpected wall hit
        if hit_wall and self.direction_values.get(pred.action, {}).get('wall', 0) < 0.5:
            err.action_error += 0.5

        # Unexpected damage
        if took_damage and pred.expected_p_absorb < 0.3:
            err.action_error += 0.8

        # === INTERNAL PREDICTION ERRORS ===
        state = actual_viability.get('state', {})

        # Error = deviation from setpoint (what system "wants")
        err.energy_error = max(0, self.setpoints['energy'] - state.get('energy', 0.5))
        err.integrity_error = max(0, self.setpoints['integrity'] - state.get('integrity', 1.0))
        err.threat_error = max(0, state.get('threat_exposure', 0) - self.setpoints['threat_exposure'])

        # === AGGREGATE ERRORS ===
        err.total_external = (err.sensory_error + err.action_error) / 2
        err.total_internal = (err.energy_error + err.integrity_error + err.threat_error) / 3

        # === PRECISION (how much to trust this error) ===
        # Low when highly uncertain
        err.precision = 1.0 - self._uncertainty * 0.5

        # === UPDATE UNCERTAINTY ===
        surprise = err.total_external + err.total_internal
        self._surprise_history.append(surprise)
        avg_surprise = sum(self._surprise_history) / len(self._surprise_history)
        self._uncertainty = min(0.9, avg_surprise)

        # === UPDATE CONFIDENCE ===
        if err.total_external < 0.2 and err.total_internal < 0.2:
            self._confidence = min(0.95, self._confidence + 0.02)
        else:
            self._confidence = max(0.1, self._confidence - 0.05)

        # === LEARN FROM ERROR ===
        self._update_model(actual_sensory, actual_viability, actual_reward,
                          hit_wall, took_damage)

        self._error_history.append(err)
        return err

    def _update_model(self,
                      sensory: Dict,
                      viability: Dict,
                      reward: float,
                      hit_wall: bool,
                      took_damage: bool):
        """Update internal model based on observations."""
        if self._last_prediction is None:
            return

        action = self._last_prediction.action
        if action not in self.direction_values:
            return

        lr = self._learning_rate

        # Update food expectations
        if reward > 5:  # Found food
            self.direction_values[action]['food'] += lr * (1.0 - self.direction_values[action]['food'])
        else:
            self.direction_values[action]['food'] *= (1 - lr * 0.1)

        # Update threat expectations
        if took_damage:
            self.direction_values[action]['threat'] += lr * (1.0 - self.direction_values[action]['threat'])
        else:
            self.direction_values[action]['threat'] *= (1 - lr * 0.05)

        # Update wall expectations
        if hit_wall:
            self.direction_values[action]['wall'] += lr * (1.0 - self.direction_values[action]['wall'])

    def get_prediction_error_summary(self) -> Dict:
        """Get summary of recent prediction errors."""
        if not self._error_history:
            return {
                'avg_external': 0.0,
                'avg_internal': 0.0,
                'uncertainty': self._uncertainty,
                'confidence': self._confidence
            }

        recent = list(self._error_history)[-10:]
        return {
            'avg_external': sum(e.total_external for e in recent) / len(recent),
            'avg_internal': sum(e.total_internal for e in recent) / len(recent),
            'uncertainty': self._uncertainty,
            'confidence': self._confidence
        }

    def get_emergent_state(self, viability_metrics: Dict) -> Dict:
        """
        Compute "emotional" state as emergent from prediction errors.

        This is NOT labeling emotions - it's computing policy parameters
        that LOOK LIKE emotions to an observer.

        What we're computing:
        - fear_like: High p_absorb + rising threat → narrow attention, avoidance
        - curiosity_like: High uncertainty + low threat → exploration bias
        - satisfaction_like: Errors decreasing + viability improving
        - distress_like: High internal errors + low control
        """
        p_absorb = viability_metrics.get('p_absorb', 0)
        urgencies = viability_metrics.get('urgencies', {})

        # Get recent error trends
        error_summary = self.get_prediction_error_summary()
        ext_error = error_summary['avg_external']
        int_error = error_summary['avg_internal']
        uncertainty = error_summary['uncertainty']

        # === FEAR-LIKE STATE ===
        # High when: p_absorb high OR threat urgency high
        # This WILL narrow attention and bias toward avoidance
        fear_like = max(p_absorb, urgencies.get('threat', 0)) ** 0.8

        # === CURIOSITY-LIKE STATE ===
        # High when: uncertainty high AND threat low AND energy ok
        # This WILL bias toward exploration
        safety = 1.0 - urgencies.get('threat', 0)
        energy_ok = 1.0 - urgencies.get('energy', 0)
        curiosity_like = uncertainty * safety * energy_ok

        # === SATISFACTION-LIKE STATE ===
        # High when: errors decreasing AND viability margin good
        margin = viability_metrics.get('viability_margin', 0)
        low_error = 1.0 - (ext_error + int_error) / 2
        satisfaction_like = margin * low_error * (1.0 - fear_like)

        # === DISTRESS-LIKE STATE ===
        # High when: internal errors high AND external errors high
        # This is "nothing is working" state
        distress_like = int_error * (1.0 + ext_error) * (1.0 - self._confidence)

        return {
            'fear_like': min(1.0, fear_like),
            'curiosity_like': min(1.0, max(0.0, curiosity_like)),
            'satisfaction_like': min(1.0, max(0.0, satisfaction_like)),
            'distress_like': min(1.0, max(0.0, distress_like)),
            'uncertainty': uncertainty,
            'confidence': self._confidence
        }

    def reset(self):
        """Reset model to initial state."""
        for d in self.direction_values:
            self.direction_values[d] = {'food': 0.0, 'threat': 0.0, 'wall': 0.0}
        self._last_prediction = None
        self._error_history.clear()
        self._surprise_history.clear()
        self._confidence = 0.5
        self._uncertainty = 0.5

    def get_visualization_data(self) -> Dict:
        """Data for frontend."""
        error_summary = self.get_prediction_error_summary()
        return {
            'direction_values': self.direction_values,
            'confidence': self._confidence,
            'uncertainty': self._uncertainty,
            'avg_external_error': error_summary['avg_external'],
            'avg_internal_error': error_summary['avg_internal'],
            'setpoints': self.setpoints
        }
