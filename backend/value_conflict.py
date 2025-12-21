"""
Value Conflict Module for Consciousness Simulation

Core Concept: Decision-making under competing values
- Immediate small reward vs Delayed large reward
- Creates "hesitation" and "conflict" states
- Enables regret/satisfaction after choice

Key Properties:
1. Conflict Level: How strongly two options compete
2. Temporal Discounting: Future rewards worth less than immediate
3. Hesitation: Behavioral slowdown during conflict
4. Regret Signal: Post-decision evaluation

Biological Basis:
- Prefrontal cortex vs limbic system competition
- Dopamine prediction for future vs immediate reward
- ACC (Anterior Cingulate Cortex) conflict monitoring

Integration:
- High conflict → Attention broadens (evaluate both)
- High conflict → Agency feels uncertain
- Regret → Memory updates (learn from mistake)
"""

from typing import Dict, Optional, Tuple, List
from collections import deque
import math


class ValueConflictSystem:
    """
    Detects and manages value conflicts in decision-making.

    Two food types:
    1. SMALL: Close, immediate, low reward (e.g., +5)
    2. LARGE: Far, delayed, high reward (e.g., +15)

    Conflict arises when:
    - Both are visible
    - Agent must choose direction
    - Temporal discounting makes them "feel" similar
    """

    def __init__(self,
                 discount_rate: float = 0.15,      # How much future is discounted per step
                 conflict_threshold: float = 0.3,  # Min difference to NOT be in conflict
                 hesitation_decay: float = 0.9):   # How quickly hesitation fades

        # Temporal discounting: future_value = value * (1 - discount_rate)^steps
        self.discount_rate = discount_rate
        self.conflict_threshold = conflict_threshold
        self.hesitation_decay = hesitation_decay

        # Current state
        self.conflict_level: float = 0.0      # 0 = no conflict, 1 = maximum conflict
        self.hesitation: float = 0.0          # Accumulated hesitation
        self.in_conflict: bool = False        # Binary conflict state

        # Option tracking
        self.small_reward_info: Optional[Dict] = None  # {direction, distance, value}
        self.large_reward_info: Optional[Dict] = None  # {direction, distance, value}

        # Decision tracking
        self.last_choice: Optional[str] = None  # 'small', 'large', or None
        self.regret_signal: float = 0.0         # Post-decision regret (-1 to 1)
        self.satisfaction: float = 0.0          # Post-decision satisfaction

        # History
        self.conflict_history: deque = deque(maxlen=50)
        self.choice_history: deque = deque(maxlen=20)

        # Statistics
        self.total_conflicts = 0
        self.chose_small = 0
        self.chose_large = 0
        self.total_regret = 0.0

    def compute_subjective_value(self, reward: float, distance: int) -> float:
        """
        Compute subjective value with temporal discounting.

        Closer rewards feel more valuable (hyperbolic discounting).
        distance=0 → full value
        distance=5 → significantly discounted
        """
        if distance <= 0:
            return reward

        # Hyperbolic discounting: V = R / (1 + k*D)
        # More realistic than exponential for biological systems
        k = self.discount_rate * 2  # Steeper discounting
        subjective = reward / (1 + k * distance)

        return subjective

    def update(self,
               small_food: Optional[Dict] = None,
               large_food: Optional[Dict] = None,
               chosen_direction: Optional[str] = None) -> Dict:
        """
        Update conflict state based on current food positions.

        Args:
            small_food: {'direction': 'up', 'distance': 2, 'reward': 5}
            large_food: {'direction': 'right', 'distance': 6, 'reward': 15}
            chosen_direction: The direction agent chose (if any)

        Returns:
            Conflict state dictionary
        """
        self.small_reward_info = small_food
        self.large_reward_info = large_food

        # Decay hesitation
        self.hesitation *= self.hesitation_decay

        # Check if both options exist and are in different directions
        if small_food and large_food:
            small_dir = small_food.get('direction')
            large_dir = large_food.get('direction')

            # Conflict only if different directions
            if small_dir != large_dir:
                self._compute_conflict(small_food, large_food)
            else:
                # Same direction = no conflict, go for larger
                self.conflict_level = 0.0
                self.in_conflict = False
        else:
            # Only one option or none = no conflict
            self.conflict_level = 0.0
            self.in_conflict = False

        # Process choice if made
        if chosen_direction and self.in_conflict:
            self._process_choice(chosen_direction)

        # Record history
        self.conflict_history.append(self.conflict_level)

        return self.get_state()

    def _compute_conflict(self, small_food: Dict, large_food: Dict):
        """
        Compute conflict level between two options.

        DISABLED: Conflict detection was causing performance degradation.
        Now just tracks subjective values for the boost calculation.
        """
        # Get subjective values (discounted) - still needed for get_large_food_advantage()
        small_subj = self.compute_subjective_value(
            small_food.get('reward', 5),
            small_food.get('distance', 1)
        )
        large_subj = self.compute_subjective_value(
            large_food.get('reward', 15),
            large_food.get('distance', 5)
        )

        # DISABLED: No conflict detection - it was hurting performance
        # Just record values for logging/visualization
        self.conflict_level = 0.0
        self.in_conflict = False
        self.hesitation = 0.0

        # Optional: Log when large food is significantly better
        if large_subj > small_subj * 1.2:
            # Don't print every step, too noisy
            pass

    def _process_choice(self, chosen_direction: str):
        """
        Process the agent's choice and compute regret/satisfaction.
        """
        if not self.small_reward_info or not self.large_reward_info:
            return

        small_dir = self.small_reward_info.get('direction')
        large_dir = self.large_reward_info.get('direction')

        # Determine what was chosen
        if chosen_direction == small_dir:
            self.last_choice = 'small'
            self.chose_small += 1

            # Regret: could have gotten more
            # Higher regret if large was much better
            small_subj = self.compute_subjective_value(
                self.small_reward_info.get('reward', 5),
                self.small_reward_info.get('distance', 1)
            )
            large_subj = self.compute_subjective_value(
                self.large_reward_info.get('reward', 15),
                self.large_reward_info.get('distance', 5)
            )

            if large_subj > small_subj:
                self.regret_signal = (large_subj - small_subj) / large_subj
                self.satisfaction = -self.regret_signal * 0.5
                self.total_regret += self.regret_signal
                print(f"[CHOICE] Chose SMALL → Regret: {self.regret_signal:.2f}")
            else:
                # Actually made the right choice!
                self.regret_signal = 0
                self.satisfaction = 0.3

        elif chosen_direction == large_dir:
            self.last_choice = 'large'
            self.chose_large += 1

            # Satisfaction from delayed gratification
            self.regret_signal = 0
            self.satisfaction = 0.5
            print(f"[CHOICE] Chose LARGE → Satisfaction: {self.satisfaction:.2f}")

        # Record choice
        self.choice_history.append({
            'choice': self.last_choice,
            'conflict_level': self.conflict_level,
            'regret': self.regret_signal
        })

        # Reset conflict after choice
        self.in_conflict = False

    def on_reward_received(self, reward: float, was_large: bool):
        """
        Called when agent actually receives the reward.
        Updates satisfaction based on outcome.
        """
        if was_large:
            # Delayed gratification paid off!
            self.satisfaction = min(1.0, self.satisfaction + 0.3)
            print(f"[VALUE] Large reward received! Satisfaction: {self.satisfaction:.2f}")
        else:
            # Quick reward, but was it worth it?
            # Satisfaction based on whether regret was low
            if self.regret_signal < 0.2:
                self.satisfaction = 0.2
            else:
                self.satisfaction = -0.1  # Slight disappointment

    def get_conflict_modulation(self) -> Dict[str, float]:
        """
        Get modulation signals for other systems.

        Returns multipliers/adjustments for:
        - attention_width: broaden during conflict
        - agency_confidence: reduce during conflict
        - decision_threshold: raise during hesitation

        Note: Reduced negative effects to prevent performance degradation
        """
        return {
            'attention_width_boost': self.conflict_level * 0.2,  # Slightly broaden
            'agency_reduction': self.conflict_level * 0.05,      # Minimal reduction
            'decision_slowdown': self.hesitation * 0.1,          # Much less slowdown
            'exploration_boost': 0.0                             # No extra exploration
        }

    def get_large_food_advantage(self) -> Tuple[float, Optional[str]]:
        """
        Get how much better large food is than small food.

        Returns:
            (advantage_ratio, direction)
            - advantage_ratio: 0.0 = equal, 1.0+ = large much better, negative = small better
            - direction: direction to large food, or None
        """
        if not self.small_reward_info or not self.large_reward_info:
            return 0.0, None

        small_subj = self.compute_subjective_value(
            self.small_reward_info.get('reward', 10),
            self.small_reward_info.get('distance', 1)
        )
        large_subj = self.compute_subjective_value(
            self.large_reward_info.get('reward', 25),
            self.large_reward_info.get('distance', 5)
        )

        if small_subj <= 0:
            return 1.0, self.large_reward_info.get('direction')

        # Advantage = (large - small) / small
        # 0 = equal, 0.5 = large is 50% better, 1.0 = large is 2x better
        advantage = (large_subj - small_subj) / small_subj

        return advantage, self.large_reward_info.get('direction')

    def get_state(self) -> Dict:
        """Get current conflict state."""
        return {
            'conflict_level': round(self.conflict_level, 3),
            'in_conflict': self.in_conflict,
            'hesitation': round(self.hesitation, 3),
            'last_choice': self.last_choice,
            'regret': round(self.regret_signal, 3),
            'satisfaction': round(self.satisfaction, 3)
        }

    def get_visualization_data(self) -> Dict:
        """Get data for frontend visualization."""
        return {
            'conflict': round(self.conflict_level, 3),
            'hesitation': round(self.hesitation, 3),
            'in_conflict': self.in_conflict,
            'regret': round(self.regret_signal, 3),
            'satisfaction': round(self.satisfaction, 3),
            'last_choice': self.last_choice,
            'stats': {
                'total_conflicts': self.total_conflicts,
                'chose_small': self.chose_small,
                'chose_large': self.chose_large,
                'avg_regret': round(self.total_regret / max(1, self.chose_small), 3)
            }
        }

    def to_dict(self) -> Dict:
        """Full state for API."""
        return {
            **self.get_state(),
            'small_reward_info': self.small_reward_info,
            'large_reward_info': self.large_reward_info,
            'conflict_history': list(self.conflict_history),
            'statistics': {
                'total_conflicts': self.total_conflicts,
                'chose_small': self.chose_small,
                'chose_large': self.chose_large,
                'total_regret': round(self.total_regret, 3),
                'delayed_gratification_rate': round(
                    self.chose_large / max(1, self.chose_small + self.chose_large), 3
                )
            }
        }

    def clear(self):
        """Reset conflict state."""
        self.conflict_level = 0.0
        self.hesitation = 0.0
        self.in_conflict = False
        self.small_reward_info = None
        self.large_reward_info = None
        self.last_choice = None
        self.regret_signal = 0.0
        self.satisfaction = 0.0
