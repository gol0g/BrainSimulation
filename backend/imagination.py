"""
Internal Simulation System (Forward Model)

Core Concept: "Think before you act"
- Agent imagines outcomes of each possible action
- Combines predictions with emotions/drives to evaluate options
- Creates the foundation for conscious deliberation

This is NOT complex world modeling - just 1-step prediction.
But it's enough to create:
- Hesitation (conflicting predictions)
- Anticipation (expecting reward)
- Dread (expecting pain)
- Relief (avoiding predicted danger)

Key Insight:
When the agent "imagines" going toward the predator and predicts pain,
then chooses NOT to go there - that's the beginning of conscious choice.
"""

from typing import Dict, List, Tuple, Optional


class ImaginationSystem:
    """
    Simulates outcomes of actions before executing them.

    For each direction (UP, DOWN, LEFT, RIGHT):
    1. Predict what would happen (Δenergy, Δpain, Δsafety, etc.)
    2. Calculate expected utility based on current drives/emotions
    3. Return scores to inform decision-making
    """

    def __init__(self):
        # === Imagination Weights ===
        # How much each predicted change matters
        self.weights = {
            'energy': 1.0,      # Gaining energy is good
            'pain': -3.0,       # Pain is very bad (high weight)
            'safety': 1.5,      # Safety is important
            'food_proximity': 0.8,  # Getting closer to food
            'predator_proximity': -2.0,  # Getting closer to predator is bad
        }

        # === Imagination State ===
        self.last_imagination = {}  # Scores for each direction
        self.chosen_reason = ""     # Why this action was chosen
        self.imagination_count = 0

        # === Confidence in Predictions ===
        # Starts low (infant doesn't know outcomes)
        # Increases with experience
        self.prediction_confidence = 0.1

        # === Experience-based learning ===
        # Track actual outcomes to improve predictions
        self.outcome_history = {
            'toward_food': [],      # What happened when moving toward food
            'toward_predator': [],  # What happened when moving toward predator
            'random': []            # Random movements
        }

    def imagine_action(self,
                       direction: str,
                       current_state: Dict,
                       world_info: Dict) -> Dict:
        """
        Imagine what would happen if we take this action.

        Args:
            direction: 'up', 'down', 'left', 'right'
            current_state: Current agent state (energy, safety, etc.)
            world_info: World information (food_pos, predator_pos, agent_pos)

        Returns:
            Predicted changes and utility score
        """
        agent_pos = world_info.get('agent_pos', [7, 7])
        food_pos = world_info.get('food_pos', [7, 7])
        predator_pos = world_info.get('predator_pos', None)
        grid_size = world_info.get('grid_size', [15, 15])

        # Current distances
        food_dist = abs(agent_pos[0] - food_pos[0]) + abs(agent_pos[1] - food_pos[1])
        pred_dist = None
        if predator_pos:
            pred_dist = abs(agent_pos[0] - predator_pos[0]) + abs(agent_pos[1] - predator_pos[1])

        # Simulate new position
        new_pos = agent_pos.copy()
        if direction == 'up':
            new_pos[1] = max(0, agent_pos[1] - 1)
        elif direction == 'down':
            new_pos[1] = min(grid_size[1] - 1, agent_pos[1] + 1)
        elif direction == 'left':
            new_pos[0] = max(0, agent_pos[0] - 1)
        elif direction == 'right':
            new_pos[0] = min(grid_size[0] - 1, agent_pos[0] + 1)

        # Check if hitting wall
        hit_wall = (new_pos == agent_pos) and direction != 'stay'

        # New distances
        new_food_dist = abs(new_pos[0] - food_pos[0]) + abs(new_pos[1] - food_pos[1])
        new_pred_dist = None
        if predator_pos:
            new_pred_dist = abs(new_pos[0] - predator_pos[0]) + abs(new_pos[1] - predator_pos[1])

        # === Predict Changes ===
        predictions = {
            'delta_food_dist': new_food_dist - food_dist,  # Negative = getting closer
            'delta_pred_dist': (new_pred_dist - pred_dist) if pred_dist else 0,
            'hit_wall': hit_wall,
            'reach_food': new_food_dist == 0,
            'reach_predator': new_pred_dist == 0 if new_pred_dist is not None else False,
        }

        # === Predict Outcomes ===
        predicted = {
            'energy_change': 0.0,
            'pain_change': 0.0,
            'safety_change': 0.0,
        }

        # Food prediction
        if predictions['reach_food']:
            predicted['energy_change'] = 15.0  # Food gives energy
        elif predictions['delta_food_dist'] < 0:
            predicted['energy_change'] = 0.5   # Getting closer is slightly good

        # Predator prediction (only if we've learned to fear)
        learned_fear = current_state.get('learned_fear', 0.0)
        if predictions['reach_predator'] and learned_fear > 0:
            predicted['pain_change'] = 1.0 * learned_fear  # Expect pain
            predicted['safety_change'] = -0.5 * learned_fear
        elif predictions['delta_pred_dist'] is not None and predictions['delta_pred_dist'] < 0:
            # Getting closer to predator
            if learned_fear > 0.3:
                predicted['pain_change'] = 0.3 * learned_fear
                predicted['safety_change'] = -0.2 * learned_fear
        elif predictions['delta_pred_dist'] is not None and predictions['delta_pred_dist'] > 0:
            # Getting away from predator
            if learned_fear > 0.3:
                predicted['safety_change'] = 0.2 * learned_fear

        # Wall prediction
        if predictions['hit_wall']:
            predicted['energy_change'] -= 0.1  # Wasted effort

        return {
            'direction': direction,
            'predictions': predictions,
            'predicted_outcomes': predicted,
            'new_pos': new_pos,
        }

    def evaluate_all_actions(self,
                            current_state: Dict,
                            world_info: Dict,
                            homeostasis: Dict,
                            emotions: Dict) -> Dict:
        """
        Imagine all possible actions and score them.

        Args:
            current_state: Agent's current state
            world_info: World information
            homeostasis: Current homeostasis state (drives)
            emotions: Current emotional state

        Returns:
            Scores for each direction and reasoning
        """
        self.imagination_count += 1

        directions = ['up', 'down', 'left', 'right']
        scores = {}
        imaginations = {}

        # Get current drives/emotions for weighting
        hunger_drive = homeostasis.get('hunger_drive', 0.0)
        safety_drive = homeostasis.get('safety_drive', 0.0)
        fear = emotions.get('fear', 0.0)
        curiosity = emotions.get('curiosity', 0.0)
        learned_fear = current_state.get('learned_fear', 0.0)

        # Dynamic weight adjustment based on state
        dynamic_weights = self.weights.copy()

        # Hungry? Value food more
        if hunger_drive > 0.3:
            dynamic_weights['energy'] *= (1 + hunger_drive)
            dynamic_weights['food_proximity'] *= (1 + hunger_drive * 0.5)

        # Afraid? Avoid predator more strongly
        if fear > 0.3 or learned_fear > 0.5:
            dynamic_weights['pain'] *= (1 + fear + learned_fear)
            dynamic_weights['predator_proximity'] *= (1 + fear * 0.5)

        # Curious? Explore more
        if curiosity > 0.3:
            dynamic_weights['food_proximity'] *= 0.8  # Less focused on known goals

        # Imagine each action
        for direction in directions:
            imagination = self.imagine_action(direction, current_state, world_info)
            imaginations[direction] = imagination

            pred = imagination['predicted_outcomes']
            predictions = imagination['predictions']

            # Calculate utility score
            score = 0.0

            # Energy gain/loss
            score += pred['energy_change'] * dynamic_weights['energy']

            # Pain avoidance (very important!)
            score += pred['pain_change'] * dynamic_weights['pain']

            # Safety
            score += pred['safety_change'] * dynamic_weights['safety']

            # Food proximity
            if predictions['delta_food_dist'] != 0:
                score += -predictions['delta_food_dist'] * dynamic_weights['food_proximity']

            # Predator proximity
            if predictions['delta_pred_dist'] is not None and learned_fear > 0.2:
                score += predictions['delta_pred_dist'] * abs(dynamic_weights['predator_proximity']) * learned_fear

            # Wall penalty
            if predictions['hit_wall']:
                score -= 0.5

            scores[direction] = round(score, 3)

        # Find best action and reason
        best_dir = max(scores, key=scores.get)
        worst_dir = min(scores, key=scores.get)

        # Generate reasoning
        reason = self._generate_reason(best_dir, imaginations[best_dir],
                                       scores, hunger_drive, fear, learned_fear)

        self.last_imagination = {
            'scores': scores,
            'imaginations': imaginations,
            'best_action': best_dir,
            'worst_action': worst_dir,
            'reason': reason,
            'confidence': self.prediction_confidence,
        }
        self.chosen_reason = reason

        return self.last_imagination

    def _generate_reason(self, best_dir: str, imagination: Dict,
                        scores: Dict, hunger: float, fear: float,
                        learned_fear: float) -> str:
        """Generate human-readable reason for choice."""
        pred = imagination['predictions']
        outcomes = imagination['predicted_outcomes']

        reasons = []

        # Food-related reasons
        if pred['reach_food']:
            reasons.append("음식 도달!")
        elif pred['delta_food_dist'] < 0 and hunger > 0.2:
            reasons.append("음식에 접근")

        # Predator-related reasons
        if pred['reach_predator'] and learned_fear > 0.3:
            reasons.append("위험! 포식자")
        elif pred['delta_pred_dist'] is not None:
            if pred['delta_pred_dist'] > 0 and learned_fear > 0.3:
                reasons.append("위험 회피")
            elif pred['delta_pred_dist'] < 0 and learned_fear > 0.3:
                reasons.append("위험 접근...")

        # Wall
        if pred['hit_wall']:
            reasons.append("벽 충돌")

        # Safety
        if outcomes['safety_change'] > 0:
            reasons.append("안전↑")
        elif outcomes['safety_change'] < 0:
            reasons.append("안전↓")

        if not reasons:
            reasons.append("탐색")

        return " / ".join(reasons)

    def update_from_outcome(self,
                           predicted_action: str,
                           actual_outcome: Dict):
        """
        Learn from actual outcomes to improve predictions.

        Called after action is executed with actual results.
        """
        # Track if prediction was accurate
        if predicted_action in self.last_imagination.get('imaginations', {}):
            predicted = self.last_imagination['imaginations'][predicted_action]

            # Compare predicted vs actual
            pred_food = predicted['predictions'].get('reach_food', False)
            actual_food = actual_outcome.get('got_food', False)

            pred_pain = predicted['predicted_outcomes'].get('pain_change', 0) > 0
            actual_pain = actual_outcome.get('got_hurt', False)

            # Increase confidence if predictions were correct
            correct = (pred_food == actual_food) and (pred_pain == actual_pain)
            if correct:
                self.prediction_confidence = min(1.0, self.prediction_confidence + 0.01)
            else:
                self.prediction_confidence = max(0.1, self.prediction_confidence - 0.02)

    def get_visualization_data(self) -> Dict:
        """Get data for frontend visualization."""
        if not self.last_imagination:
            return {
                'scores': {'up': 0, 'down': 0, 'left': 0, 'right': 0},
                'best_action': None,
                'reason': '',
                'confidence': self.prediction_confidence,
                'active': False,
            }

        return {
            'scores': self.last_imagination.get('scores', {}),
            'best_action': self.last_imagination.get('best_action'),
            'worst_action': self.last_imagination.get('worst_action'),
            'reason': self.last_imagination.get('reason', ''),
            'confidence': round(self.prediction_confidence, 2),
            'active': True,
        }

    def get_action_recommendation(self) -> Tuple[str, float]:
        """
        Get the recommended action based on imagination.

        Returns:
            (best_direction, confidence)
        """
        if not self.last_imagination:
            return None, 0.0

        return (
            self.last_imagination.get('best_action'),
            self.prediction_confidence
        )
