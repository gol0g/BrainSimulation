"""
Unified Value Function - One Principle to Explain All Behavior

Core Concept:
ALL behavior is explained by minimizing a single cost function.
This is not multiple goals competing - it's ONE evaluation that has multiple terms.

The cost has four components:
1. Viability Cost: Distance from absorbing state (death)
2. Prediction Error Cost: Mismatch between model and reality
3. Control Cost: Loss of agency / externality
4. Information Cost: Uncertainty that could be reduced

Why one function?
- "Fear" is not a goal, it's when viability cost is high
- "Curiosity" is not a goal, it's when information cost can be reduced
- "Satisfaction" is not a state, it's when total cost is decreasing

All behavior becomes explainable by: "I did X because it minimizes expected future cost"

Mathematical Form:
    C(s,a) = w_v * C_viability(s,a)
           + w_p * C_prediction(s,a)
           + w_c * C_control(s,a)
           + w_i * C_information(s,a)

Where weights are NOT preferences but importance based on current state:
- Threat high → w_v dominates
- Stable & curious → w_i matters more
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class CostBreakdown:
    """Detailed breakdown of cost components."""
    viability_cost: float = 0.0    # Risk of absorbing state
    prediction_cost: float = 0.0   # Model mismatch
    control_cost: float = 0.0      # Loss of agency
    information_cost: float = 0.0  # Reducible uncertainty

    total: float = 0.0

    # Weights used
    w_viability: float = 1.0
    w_prediction: float = 1.0
    w_control: float = 1.0
    w_information: float = 1.0


class UnifiedValue:
    """
    Computes a single cost/value for any state-action pair.

    This replaces:
    - Separate goal system (SAFE/FEED/REST)
    - Emotion-based biases
    - Hardcoded preferences

    Everything becomes "which action minimizes future cost?"
    """

    def __init__(self):
        # Base weights (can be modulated by state)
        self.base_weights = {
            'viability': 3.0,    # Survival matters most
            'prediction': 1.0,   # Model accuracy matters
            'control': 0.5,      # Agency matters
            'information': 0.3   # Information has value
        }

        # Temporal discount (future costs matter less)
        self.gamma = 0.9

        # Track cost history for "satisfaction" detection
        self._cost_history = []
        self._max_history = 50

    def compute_cost(self,
                     viability_metrics: Dict,
                     prediction_errors: Dict,
                     emergent_state: Dict,
                     action_context: Dict = None) -> CostBreakdown:
        """
        Compute total cost for current state.

        This is the central evaluation that drives all behavior.
        """
        result = CostBreakdown()

        # === DYNAMIC WEIGHT MODULATION ===
        # Weights change based on state, not as "goals" but as importance
        weights = self._compute_weights(viability_metrics, emergent_state)
        result.w_viability = weights['viability']
        result.w_prediction = weights['prediction']
        result.w_control = weights['control']
        result.w_information = weights['information']

        # === VIABILITY COST ===
        # How close to absorbing state?
        p_absorb = viability_metrics.get('p_absorb', 0)
        margin = viability_metrics.get('viability_margin', 1)
        urgencies = viability_metrics.get('urgencies', {})

        # Cost rises sharply as p_absorb increases
        # This is why "fear" emerges near death
        result.viability_cost = (
            p_absorb ** 0.5 * 5.0 +  # Direct death risk
            sum(urgencies.values()) / 3.0 +  # Channel urgencies
            (1.0 - margin) * 2.0  # Distance from safety
        )

        # === PREDICTION ERROR COST ===
        # Mismatch between model and reality
        ext_error = prediction_errors.get('avg_external', 0)
        int_error = prediction_errors.get('avg_internal', 0)

        result.prediction_cost = (
            ext_error * 2.0 +  # External surprises
            int_error * 3.0    # Internal deviations (homeostatic)
        )

        # === CONTROL COST ===
        # Loss of agency / externality
        confidence = prediction_errors.get('confidence', 0.5)
        # Low confidence = low perceived control
        result.control_cost = (1.0 - confidence) * 2.0

        # === INFORMATION COST ===
        # Uncertainty that could potentially be reduced
        uncertainty = prediction_errors.get('uncertainty', 0.5)
        fear_like = emergent_state.get('fear_like', 0)

        # Information cost only matters when safe enough to explore
        safety_factor = 1.0 - fear_like
        result.information_cost = uncertainty * safety_factor

        # === TOTAL COST ===
        result.total = (
            result.w_viability * result.viability_cost +
            result.w_prediction * result.prediction_cost +
            result.w_control * result.control_cost -
            result.w_information * result.information_cost  # Information REDUCES cost
        )

        # Track history
        self._cost_history.append(result.total)
        if len(self._cost_history) > self._max_history:
            self._cost_history.pop(0)

        return result

    def _compute_weights(self,
                         viability_metrics: Dict,
                         emergent_state: Dict) -> Dict[str, float]:
        """
        Dynamically adjust weights based on current state.

        This is NOT goal switching. It's how the same cost function
        naturally emphasizes different terms based on context.
        """
        weights = dict(self.base_weights)

        p_absorb = viability_metrics.get('p_absorb', 0)
        fear_like = emergent_state.get('fear_like', 0)
        curiosity_like = emergent_state.get('curiosity_like', 0)

        # === CRISIS MODE ===
        # When near death, viability dominates everything
        if p_absorb > 0.3 or fear_like > 0.5:
            crisis_factor = max(p_absorb, fear_like)
            weights['viability'] *= (1 + crisis_factor * 3)
            weights['information'] *= (1 - crisis_factor * 0.8)  # No time for curiosity

        # === EXPLORATION MODE ===
        # When safe and curious, information matters more
        if curiosity_like > 0.3 and fear_like < 0.2:
            weights['information'] *= (1 + curiosity_like)
            weights['prediction'] *= (1 + curiosity_like * 0.5)  # Want to learn

        # === STABLE MODE ===
        # When everything is fine, maintain balance
        margin = viability_metrics.get('viability_margin', 0)
        if margin > 0.5 and fear_like < 0.1:
            # Reduce urgency of viability when very safe
            weights['viability'] *= 0.7

        return weights

    def evaluate_action(self,
                        action: str,
                        predicted_viability: Dict,
                        predicted_errors: Dict,
                        emergent_state: Dict,
                        depth: int = 1) -> Tuple[float, CostBreakdown]:
        """
        Evaluate the expected cost of taking an action.

        This is for action selection:
        "If I do X, what will the future cost be?"

        Returns (expected_cost, breakdown)
        """
        # Compute immediate cost
        immediate = self.compute_cost(
            predicted_viability,
            predicted_errors,
            emergent_state,
            {'action': action}
        )

        # For now, just return immediate cost
        # (Full rollout would recursively evaluate future states)
        return immediate.total, immediate

    def get_cost_trajectory(self) -> Dict:
        """
        Analyze cost trajectory over time.

        This is where "satisfaction" comes from:
        - Cost decreasing = things are getting better
        - Cost stable and low = comfortable
        - Cost rising = something is wrong
        """
        if len(self._cost_history) < 5:
            return {
                'trend': 'unknown',
                'current': self._cost_history[-1] if self._cost_history else 0,
                'average': 0,
                'is_improving': False
            }

        recent = self._cost_history[-10:]
        older = self._cost_history[-20:-10] if len(self._cost_history) >= 20 else self._cost_history[:10]

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older) if older else avg_recent

        if avg_recent < avg_older * 0.8:
            trend = 'improving'
            is_improving = True
        elif avg_recent > avg_older * 1.2:
            trend = 'worsening'
            is_improving = False
        else:
            trend = 'stable'
            is_improving = avg_recent < 3.0  # Low absolute cost

        return {
            'trend': trend,
            'current': self._cost_history[-1],
            'average': avg_recent,
            'is_improving': is_improving,
            'history': list(self._cost_history[-20:])
        }

    def get_dominant_concern(self,
                             viability_metrics: Dict,
                             prediction_errors: Dict,
                             emergent_state: Dict) -> str:
        """
        What is currently driving behavior most?

        Not a "goal" - but an explanation of what the cost function cares about now.
        """
        cost = self.compute_cost(viability_metrics, prediction_errors, emergent_state)

        weighted_costs = [
            (cost.w_viability * cost.viability_cost, 'survival'),
            (cost.w_prediction * cost.prediction_cost, 'understanding'),
            (cost.w_control * cost.control_cost, 'control'),
            (cost.w_information * cost.information_cost, 'exploration')
        ]

        weighted_costs.sort(key=lambda x: x[0], reverse=True)
        return weighted_costs[0][1]

    def reset(self):
        """Reset cost history."""
        self._cost_history = []

    def get_visualization_data(self,
                               viability_metrics: Dict,
                               prediction_errors: Dict,
                               emergent_state: Dict) -> Dict:
        """Data for frontend."""
        cost = self.compute_cost(viability_metrics, prediction_errors, emergent_state)
        trajectory = self.get_cost_trajectory()

        return {
            'total_cost': cost.total,
            'breakdown': {
                'viability': cost.viability_cost,
                'prediction': cost.prediction_cost,
                'control': cost.control_cost,
                'information': cost.information_cost
            },
            'weights': {
                'viability': cost.w_viability,
                'prediction': cost.w_prediction,
                'control': cost.w_control,
                'information': cost.w_information
            },
            'trajectory': trajectory,
            'dominant_concern': self.get_dominant_concern(
                viability_metrics, prediction_errors, emergent_state
            )
        }
