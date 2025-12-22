"""
Goal-Directed Behavior System (v1: Simple Subgoal Switching)

Core Concept: "What should I focus on right now?"

Instead of complex planning, this starts simple:
- Two subgoals: SAFE and FEED
- Switch based on current state
- Modulate imagination/action scores based on active goal

This creates "goal-directed" behavior without complex planning:
- When in danger → prioritize safety (SAFE)
- When safe but hungry → prioritize feeding (FEED)

Later can be extended to:
- More goals (REST, EXPLORE, SOCIAL)
- Goal hierarchies and conflicts
- Explicit goal representations

Key Insight:
The agent already has imagination (predicting outcomes) and memory.
Goals simply bias which outcomes matter more right now.
"""

from typing import Dict, Tuple, Optional
from enum import Enum


class Goal(Enum):
    """Available subgoals."""
    SAFE = "safe"      # 안전 확보: 포식자로부터 거리 벌리기
    FEED = "feed"      # 먹이 확보: 에너지 회복
    IDLE = "idle"      # 특별한 목표 없음 (안정 상태)


class GoalSystem:
    """
    Simple goal switching based on internal state.

    Design:
    - No complex planning, just state → goal mapping
    - Goals bias action selection, not replace it
    - Smooth transitions with hysteresis
    """

    def __init__(self):
        self.current_goal = Goal.IDLE
        self.goal_duration = 0  # How long in current goal

        # Thresholds for goal switching
        self.safety_critical = 0.4     # Below this → SAFE mode
        self.safety_comfortable = 0.7  # Above this → can leave SAFE
        self.energy_critical = 0.3     # Below this → FEED mode
        self.energy_comfortable = 0.6  # Above this → can leave FEED
        self.predator_threat_radius = 4  # Predator closer than this → SAFE

        # Minimum duration before switching (hysteresis)
        self.min_goal_duration = 10  # Steps

        # Goal-specific action biases (applied to imagination scores)
        # These modulate what the agent cares about
        self.goal_biases = {
            Goal.SAFE: {
                'predator_proximity': -3.0,  # Strong avoidance
                'pain': -4.0,                # Very important to avoid
                'safety': 2.0,               # Seek safety
                'food_proximity': 0.5,       # Still somewhat interested
                'energy': 0.5,               # Less priority
            },
            Goal.FEED: {
                'predator_proximity': -1.5,  # Still avoid, but less
                'pain': -2.0,                # Still avoid
                'safety': 1.0,               # Some importance
                'food_proximity': 2.5,       # Prioritize food
                'energy': 2.0,               # Very important
            },
            Goal.IDLE: {
                # Balanced, no special bias
                'predator_proximity': -2.0,
                'pain': -3.0,
                'safety': 1.5,
                'food_proximity': 1.0,
                'energy': 1.0,
            }
        }

        # Statistics
        self.goal_switches = 0
        self.time_in_goals = {Goal.SAFE: 0, Goal.FEED: 0, Goal.IDLE: 0}

    def update(self,
               safety: float,
               energy: float,
               predator_distance: Optional[float] = None,
               pain_level: float = 0.0) -> Goal:
        """
        Update current goal based on internal state.

        Args:
            safety: Current safety level (0-1)
            energy: Current energy level (0-1)
            predator_distance: Distance to nearest predator (None if no predator)
            pain_level: Current pain level (0-1)

        Returns:
            Current active goal
        """
        self.goal_duration += 1
        self.time_in_goals[self.current_goal] += 1

        # Determine desired goal
        desired_goal = self._evaluate_desired_goal(
            safety, energy, predator_distance, pain_level
        )

        # Check if we should switch (with hysteresis)
        if desired_goal != self.current_goal:
            if self._should_switch(desired_goal, safety, energy, predator_distance, pain_level):
                old_goal = self.current_goal
                self.current_goal = desired_goal
                self.goal_duration = 0
                self.goal_switches += 1
                print(f"[GOAL] Switch: {old_goal.value} → {desired_goal.value}")

        return self.current_goal

    def _evaluate_desired_goal(self,
                               safety: float,
                               energy: float,
                               predator_distance: Optional[float],
                               pain_level: float) -> Goal:
        """Determine what goal we SHOULD be pursuing based on state."""

        # Priority 1: Immediate danger → SAFE
        if pain_level > 0.3:  # Recent pain
            return Goal.SAFE

        if predator_distance is not None and predator_distance <= self.predator_threat_radius:
            return Goal.SAFE

        if safety < self.safety_critical:
            return Goal.SAFE

        # Priority 2: Low energy and safe → FEED
        if energy < self.energy_critical and safety > self.safety_comfortable:
            return Goal.FEED

        # Priority 3: Moderate hunger and very safe → FEED
        if energy < 0.5 and safety > 0.8:
            return Goal.FEED

        # Default: IDLE (balanced state)
        return Goal.IDLE

    def _should_switch(self,
                       new_goal: Goal,
                       safety: float,
                       energy: float,
                       predator_distance: Optional[float],
                       pain_level: float) -> bool:
        """
        Check if we should actually switch goals (hysteresis).

        Immediate switches for emergencies, gradual for others.
        """
        # Emergency: Always switch immediately to SAFE
        if new_goal == Goal.SAFE:
            if pain_level > 0.5:  # Strong pain
                return True
            if predator_distance is not None and predator_distance <= 2:  # Very close
                return True
            if safety < 0.2:  # Critical safety
                return True

        # Non-emergency: Require minimum duration
        if self.goal_duration < self.min_goal_duration:
            return False

        # Check exit conditions for current goal
        if self.current_goal == Goal.SAFE:
            # Exit SAFE only if truly safe
            if safety > self.safety_comfortable and \
               (predator_distance is None or predator_distance > self.predator_threat_radius + 2):
                return True

        elif self.current_goal == Goal.FEED:
            # Exit FEED if satisfied or in danger
            if energy > self.energy_comfortable:
                return True
            if new_goal == Goal.SAFE:
                return True  # Danger overrides feeding

        elif self.current_goal == Goal.IDLE:
            # Exit IDLE more freely
            return True

        return False

    def get_action_bias(self, direction: str, imagination_details: Dict,
                        predator_distance: Optional[float] = None) -> float:
        """
        Get goal-specific bias for a direction based on imagination predictions.

        Args:
            direction: 'up', 'down', 'left', 'right'
            imagination_details: Predictions for this direction from imagination system
            predator_distance: Current distance to predator (for distance-based penalty)

        Returns:
            Score adjustment based on current goal
        """
        biases = self.goal_biases[self.current_goal]

        # Extract predictions from imagination
        delta_pred_dist = imagination_details.get('delta_pred_dist', 0)
        reach_predator = imagination_details.get('reach_predator', False)
        reach_food = imagination_details.get('reach_food', False)
        delta_food_dist = imagination_details.get('delta_food_dist', 0)

        adjustment = 0.0

        # === v1.1: Distance-based predator penalty ===
        # 고정값보다 거리 기반: 가까울수록 강한 패널티
        # danger(distance) = 1/distance (멀면 약함, 가까우면 강함)
        if predator_distance is not None and predator_distance > 0:
            danger_factor = min(1.0, 3.0 / predator_distance)  # 거리 3에서 1.0, 거리 6에서 0.5
        else:
            danger_factor = 0.3  # 포식자 없으면 기본 낮은 위험

        # Predator proximity adjustment (이제 거리 기반)
        if delta_pred_dist < 0:  # Getting closer
            # 가까이 갈수록, 현재 거리가 가까울수록 더 큰 패널티
            adjustment += biases['predator_proximity'] * abs(delta_pred_dist) * danger_factor
        elif delta_pred_dist > 0:  # Getting farther
            adjustment -= biases['predator_proximity'] * 0.2 * danger_factor

        # Pain/danger adjustment (also distance-scaled)
        if reach_predator:
            adjustment += biases['pain'] * 2.0 * danger_factor

        # Food proximity adjustment
        if reach_food:
            adjustment += biases['food_proximity'] * 3.0
        elif delta_food_dist < 0:  # Getting closer to food
            adjustment += biases['food_proximity'] * abs(delta_food_dist) * 0.5

        return adjustment

    def get_visualization_data(self) -> Dict:
        """Get data for frontend visualization."""
        return {
            'current_goal': self.current_goal.value,
            'goal_duration': self.goal_duration,
            'goal_switches': self.goal_switches,
            'time_in_goals': {
                goal.value: self.time_in_goals[goal]
                for goal in Goal
            },
            'biases': {
                k: round(v, 2)
                for k, v in self.goal_biases[self.current_goal].items()
            }
        }

    def get_goal_description(self) -> str:
        """Human-readable description of current goal."""
        if self.current_goal == Goal.SAFE:
            return "안전 확보 중"
        elif self.current_goal == Goal.FEED:
            return "먹이 탐색 중"
        else:
            return "탐험 중"

    def evaluate_success(self,
                         delta_energy: float,
                         delta_pain: float,
                         delta_safety: float,
                         delta_predator_dist: float) -> bool:
        """
        Evaluate if the current action contributed to goal success.

        FEED success: energy increased
        SAFE success: pain decreased OR predator distance increased OR safety increased
        IDLE: always "success" (no specific goal to fail)

        Args:
            delta_energy: Change in energy
            delta_pain: Change in pain
            delta_safety: Change in safety
            delta_predator_dist: Change in predator distance (positive = farther)

        Returns:
            True if action helped achieve current goal
        """
        if self.current_goal == Goal.FEED:
            # FEED 성공: 에너지가 증가함
            return delta_energy > 0.01

        elif self.current_goal == Goal.SAFE:
            # SAFE 성공: 통증 감소 OR 포식자와 거리 증가 OR 안전 증가
            pain_reduced = delta_pain < -0.01
            escaped = delta_predator_dist > 0.5
            safer = delta_safety > 0.05
            return pain_reduced or escaped or safer

        else:  # IDLE
            # IDLE은 특별한 목표가 없으므로 항상 "성공"
            return True
