"""
Predator System v1

A threat entity that affects the agent's safety.

Design Philosophy:
- Extensible: Can later have its own SNN brain
- Simple v1: Random movement with basic chase behavior
- Interface ready: move() can be replaced with neural output

Future Plans:
- Attach Network to predator
- Predator learns to chase agent
- Multiple predators with different behaviors
"""

import random
from typing import Dict, List, Optional, Tuple


class Predator:
    """
    A predator entity that threatens the agent.

    v1: Simple random movement with optional chase tendency
    Future: Can have its own neural network for decision making
    """

    def __init__(self,
                 world_width: int = 10,
                 world_height: int = 10,
                 chase_probability: float = 0.3,  # Chance to move toward agent
                 move_probability: float = 0.5,   # Chance to move each step
                 threat_radius: int = 3):         # Distance at which agent feels threatened

        self.world_width = world_width
        self.world_height = world_height
        self.chase_probability = chase_probability
        self.move_probability = move_probability
        self.threat_radius = threat_radius

        # Position
        self.pos: List[int] = [0, 0]
        self._place_random()

        # State tracking
        self.steps_taken: int = 0
        self.times_caught_agent: int = 0

        # === For future neural network integration ===
        self.network = None  # Will be a Network instance later
        self.last_action: Optional[str] = None
        self.sensory_input: Dict[str, float] = {}

    def _place_random(self, avoid_pos: Optional[List[int]] = None, min_distance: int = 3):
        """Place predator at random position, avoiding agent."""
        attempts = 0
        while attempts < 100:
            self.pos = [
                random.randint(0, self.world_width - 1),
                random.randint(0, self.world_height - 1)
            ]
            if avoid_pos is None:
                return
            dist = abs(self.pos[0] - avoid_pos[0]) + abs(self.pos[1] - avoid_pos[1])
            if dist >= min_distance:
                return
            attempts += 1

    def get_sensory_input(self, agent_pos: List[int]) -> Dict[str, float]:
        """
        Get predator's view of the world.

        For v1: Just direction to agent
        Future: Could include walls, other entities, etc.
        """
        dx = agent_pos[0] - self.pos[0]
        dy = agent_pos[1] - self.pos[1]
        dist = abs(dx) + abs(dy)

        # Signal strength based on distance (closer = stronger)
        strength = max(0.1, 1.0 - dist * 0.1)

        self.sensory_input = {
            'agent_up': strength if dy < 0 else 0.0,
            'agent_down': strength if dy > 0 else 0.0,
            'agent_left': strength if dx < 0 else 0.0,
            'agent_right': strength if dx > 0 else 0.0,
            'agent_distance': dist,
        }
        return self.sensory_input

    def decide_action(self, agent_pos: List[int]) -> int:
        """
        Decide next action.

        v1: Random with chase tendency
        Future: Use neural network output

        Returns: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        """
        # === Future: Neural network decision ===
        # if self.network:
        #     return self._neural_decide(agent_pos)

        # === v1: Random + Chase behavior ===
        if random.random() > self.move_probability:
            return 0  # Stay

        if random.random() < self.chase_probability:
            # Chase: Move toward agent
            return self._get_chase_action(agent_pos)
        else:
            # Random movement
            return random.randint(1, 4)

    def _get_chase_action(self, agent_pos: List[int]) -> int:
        """Get action that moves toward agent."""
        dx = agent_pos[0] - self.pos[0]
        dy = agent_pos[1] - self.pos[1]

        # Prioritize larger distance axis
        if abs(dx) > abs(dy):
            return 4 if dx > 0 else 3  # Right or Left
        elif abs(dy) > 0:
            return 2 if dy > 0 else 1  # Down or Up
        else:
            return 0  # At same position

    def move(self, action: int) -> None:
        """
        Execute movement action.

        Action: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        """
        self.steps_taken += 1

        if action == 0:
            self.last_action = 'stay'
            return

        new_pos = self.pos.copy()

        if action == 1:  # Up
            new_pos[1] = max(0, new_pos[1] - 1)
            self.last_action = 'up'
        elif action == 2:  # Down
            new_pos[1] = min(self.world_height - 1, new_pos[1] + 1)
            self.last_action = 'down'
        elif action == 3:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)
            self.last_action = 'left'
        elif action == 4:  # Right
            new_pos[0] = min(self.world_width - 1, new_pos[0] + 1)
            self.last_action = 'right'

        self.pos = new_pos

    def step(self, agent_pos: List[int]) -> Dict:
        """
        Full step: sense, decide, move.

        Returns info about this step.
        """
        # Sense
        sensory = self.get_sensory_input(agent_pos)

        # Decide
        action = self.decide_action(agent_pos)

        # Move
        self.move(action)

        # Check if caught agent
        caught = self.pos == agent_pos
        if caught:
            self.times_caught_agent += 1

        return {
            'position': self.pos.copy(),
            'action': self.last_action,
            'caught_agent': caught,
            'distance_to_agent': sensory['agent_distance']
        }

    def get_threat_level(self, agent_pos: List[int]) -> float:
        """
        Calculate threat level based on distance.

        Returns 0.0 (no threat) to 1.0 (maximum threat)
        Threat increases as predator gets closer.
        """
        dist = abs(self.pos[0] - agent_pos[0]) + abs(self.pos[1] - agent_pos[1])

        if dist >= self.threat_radius:
            return 0.0
        elif dist == 0:
            return 1.0  # Caught!
        else:
            # Linear interpolation: closer = more threat
            return 1.0 - (dist / self.threat_radius)

    def reset(self, agent_pos: Optional[List[int]] = None):
        """Reset predator to new random position."""
        self._place_random(avoid_pos=agent_pos, min_distance=3)
        self.last_action = None

    def to_dict(self) -> Dict:
        """Get full state for API/debugging."""
        return {
            'position': self.pos.copy(),
            'last_action': self.last_action,
            'steps_taken': self.steps_taken,
            'times_caught': self.times_caught_agent,
            'threat_radius': self.threat_radius,
            'chase_probability': self.chase_probability,
            'has_network': self.network is not None
        }

    def get_visualization_data(self) -> Dict:
        """Get data for frontend visualization."""
        return {
            'position': self.pos.copy(),
            'threat_radius': self.threat_radius,
            'last_action': self.last_action
        }


# === Future: Multiple predators ===
class PredatorManager:
    """
    Manages multiple predators.
    Future: Different predator types, spawn/despawn logic, etc.
    """

    def __init__(self, world_width: int = 10, world_height: int = 10):
        self.world_width = world_width
        self.world_height = world_height
        self.predators: List[Predator] = []

    def add_predator(self, **kwargs) -> Predator:
        """Add a new predator to the world."""
        predator = Predator(
            world_width=self.world_width,
            world_height=self.world_height,
            **kwargs
        )
        self.predators.append(predator)
        return predator

    def step_all(self, agent_pos: List[int]) -> List[Dict]:
        """Step all predators."""
        return [p.step(agent_pos) for p in self.predators]

    def get_max_threat(self, agent_pos: List[int]) -> float:
        """Get maximum threat level from all predators."""
        if not self.predators:
            return 0.0
        return max(p.get_threat_level(agent_pos) for p in self.predators)

    def reset_all(self, agent_pos: Optional[List[int]] = None):
        """Reset all predators."""
        for p in self.predators:
            p.reset(agent_pos)

    def to_dict(self) -> Dict:
        """Get all predator states."""
        return {
            'count': len(self.predators),
            'predators': [p.to_dict() for p in self.predators]
        }
