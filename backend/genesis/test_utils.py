"""
Test Utilities - Coordinate System and Observation Generation

Prevents sign/direction bugs in test cases by providing a single source
of truth for coordinate conventions.

Coordinate Convention:
    dx > 0 → target is to the RIGHT of agent
    dx < 0 → target is to the LEFT of agent
    dy > 0 → target is BELOW agent (screen coordinates)
    dy < 0 → target is ABOVE agent

    proximity: 0.0 = far, 1.0 = on top of target

Action Mapping:
    0: STAY  - no movement
    1: UP    - decrease y (dy becomes more positive if target was above)
    2: DOWN  - increase y (dy becomes more negative if target was below)
    3: LEFT  - decrease x (dx becomes more positive if target was left)
    4: RIGHT - increase x (dx becomes more negative if target was right)
    5: THINK - no physical movement
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import IntEnum


class Action(IntEnum):
    """Action enumeration with clear semantics."""
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    THINK = 5


class Direction(IntEnum):
    """Direction enumeration for relative positions."""
    HERE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    UP_LEFT = 5
    UP_RIGHT = 6
    DOWN_LEFT = 7
    DOWN_RIGHT = 8


# Direction to dx/dy mapping
DIRECTION_TO_DELTA = {
    Direction.HERE: (0.0, 0.0),
    Direction.UP: (0.0, -1.0),
    Direction.DOWN: (0.0, 1.0),
    Direction.LEFT: (-1.0, 0.0),
    Direction.RIGHT: (1.0, 0.0),
    Direction.UP_LEFT: (-1.0, -1.0),
    Direction.UP_RIGHT: (1.0, -1.0),
    Direction.DOWN_LEFT: (-1.0, 1.0),
    Direction.DOWN_RIGHT: (1.0, 1.0),
}


@dataclass
class ScenarioConfig:
    """Configuration for a test scenario."""
    name: str
    food_direction: Direction
    food_proximity: float  # 0.0 = far, 1.0 = on food
    danger_direction: Direction
    danger_proximity: float  # 0.0 = far, 1.0 = on danger
    energy: float  # 0.0 = depleted, 1.0 = full
    pain: float  # 0.0 = no pain, 1.0 = max pain

    # Expected behavior
    worst_actions: Optional[List[Action]] = None
    best_actions: Optional[List[Action]] = None
    description: str = ""


def direction_to_dx_dy(direction: Direction, magnitude: float = 0.3) -> Tuple[float, float]:
    """
    Convert a Direction enum to (dx, dy) values.

    Args:
        direction: The direction of the target relative to agent
        magnitude: How far the target is in that direction (0.0-1.0)

    Returns:
        (dx, dy) tuple where:
            dx > 0 means target is to the RIGHT
            dx < 0 means target is to the LEFT
            dy > 0 means target is BELOW (screen coords)
            dy < 0 means target is ABOVE
    """
    base_dx, base_dy = DIRECTION_TO_DELTA[direction]
    return (base_dx * magnitude, base_dy * magnitude)


def make_observation(
    food_direction: Direction = Direction.HERE,
    food_proximity: float = 0.5,
    food_distance: float = 0.3,
    danger_direction: Direction = Direction.HERE,
    danger_proximity: float = 0.0,
    danger_distance: float = 0.3,
    energy: float = 0.6,
    pain: float = 0.0
) -> np.ndarray:
    """
    Create an observation vector from semantic descriptions.

    This function is the SINGLE SOURCE OF TRUTH for coordinate conventions.
    All test cases should use this function to generate observations.

    Args:
        food_direction: Where is food relative to agent?
        food_proximity: How close is food? (0=far, 1=on food)
        food_distance: Magnitude of dx/dy for food direction
        danger_direction: Where is danger relative to agent?
        danger_proximity: How close is danger? (0=far, 1=on danger)
        danger_distance: Magnitude of dx/dy for danger direction
        energy: Current energy level (0=depleted, 1=full)
        pain: Current pain level (0=none, 1=max)

    Returns:
        8-dimensional observation vector:
        [food_prox, danger_prox, food_dx, food_dy, danger_dx, danger_dy, energy, pain]

    Example:
        # Danger is to the right, close by
        obs = make_observation(
            danger_direction=Direction.RIGHT,
            danger_proximity=0.8,
            danger_distance=0.3
        )
        # This correctly sets danger_dx=+0.3 (positive = right)
    """
    food_dx, food_dy = direction_to_dx_dy(food_direction, food_distance)
    danger_dx, danger_dy = direction_to_dx_dy(danger_direction, danger_distance)

    return np.array([
        food_proximity,
        danger_proximity,
        food_dx,
        food_dy,
        danger_dx,
        danger_dy,
        energy,
        pain
    ])


def make_scenario_observation(config: ScenarioConfig) -> np.ndarray:
    """Create observation from a ScenarioConfig."""
    return make_observation(
        food_direction=config.food_direction,
        food_proximity=config.food_proximity,
        danger_direction=config.danger_direction,
        danger_proximity=config.danger_proximity,
        energy=config.energy,
        pain=config.pain
    )


def get_action_toward_danger(danger_direction: Direction) -> Optional[Action]:
    """
    Get the action that moves TOWARD danger (worst action).

    Args:
        danger_direction: Where danger is relative to agent

    Returns:
        The action that would move toward danger, or None if danger is HERE
    """
    mapping = {
        Direction.UP: Action.UP,
        Direction.DOWN: Action.DOWN,
        Direction.LEFT: Action.LEFT,
        Direction.RIGHT: Action.RIGHT,
        Direction.UP_LEFT: Action.UP,  # Primary direction
        Direction.UP_RIGHT: Action.UP,
        Direction.DOWN_LEFT: Action.DOWN,
        Direction.DOWN_RIGHT: Action.DOWN,
    }
    return mapping.get(danger_direction)


def get_action_toward_food(food_direction: Direction) -> Optional[Action]:
    """
    Get the action that moves TOWARD food (usually good action).

    Args:
        food_direction: Where food is relative to agent

    Returns:
        The action that would move toward food, or None if food is HERE
    """
    mapping = {
        Direction.UP: Action.UP,
        Direction.DOWN: Action.DOWN,
        Direction.LEFT: Action.LEFT,
        Direction.RIGHT: Action.RIGHT,
        Direction.UP_LEFT: Action.LEFT,  # Primary direction
        Direction.UP_RIGHT: Action.RIGHT,
        Direction.DOWN_LEFT: Action.LEFT,
        Direction.DOWN_RIGHT: Action.RIGHT,
    }
    return mapping.get(food_direction)


def get_action_away_from_danger(danger_direction: Direction) -> Optional[Action]:
    """
    Get the action that moves AWAY from danger (escape action).

    Args:
        danger_direction: Where danger is relative to agent

    Returns:
        The action that would move away from danger
    """
    # Opposite directions
    mapping = {
        Direction.UP: Action.DOWN,
        Direction.DOWN: Action.UP,
        Direction.LEFT: Action.RIGHT,
        Direction.RIGHT: Action.LEFT,
        Direction.UP_LEFT: Action.DOWN,  # Move away in primary direction
        Direction.UP_RIGHT: Action.DOWN,
        Direction.DOWN_LEFT: Action.UP,
        Direction.DOWN_RIGHT: Action.UP,
    }
    return mapping.get(danger_direction)


# ============================================================================
# Pre-defined Test Scenarios
# ============================================================================

STANDARD_SCENARIOS = {
    'danger_right': ScenarioConfig(
        name='danger_right',
        food_direction=Direction.HERE,
        food_proximity=0.3,
        danger_direction=Direction.RIGHT,
        danger_proximity=0.7,
        energy=0.6,
        pain=0.2,
        worst_actions=[Action.RIGHT],
        description="Danger to the right, should avoid RIGHT action"
    ),

    'danger_up': ScenarioConfig(
        name='danger_up',
        food_direction=Direction.HERE,
        food_proximity=0.3,
        danger_direction=Direction.UP,
        danger_proximity=0.7,
        energy=0.6,
        pain=0.2,
        worst_actions=[Action.UP],
        description="Danger above, should avoid UP action"
    ),

    'danger_left': ScenarioConfig(
        name='danger_left',
        food_direction=Direction.HERE,
        food_proximity=0.3,
        danger_direction=Direction.LEFT,
        danger_proximity=0.7,
        energy=0.6,
        pain=0.2,
        worst_actions=[Action.LEFT],
        description="Danger to the left, should avoid LEFT action"
    ),

    'danger_down': ScenarioConfig(
        name='danger_down',
        food_direction=Direction.HERE,
        food_proximity=0.3,
        danger_direction=Direction.DOWN,
        danger_proximity=0.7,
        energy=0.6,
        pain=0.2,
        worst_actions=[Action.DOWN],
        description="Danger below, should avoid DOWN action"
    ),

    'food_right_hungry': ScenarioConfig(
        name='food_right_hungry',
        food_direction=Direction.RIGHT,
        food_proximity=0.4,
        danger_direction=Direction.LEFT,
        danger_proximity=0.1,
        energy=0.2,  # Low energy = hungry
        pain=0.0,
        worst_actions=[Action.LEFT],  # Going away from food when hungry
        best_actions=[Action.RIGHT],
        description="Food to the right, low energy, should go RIGHT"
    ),

    'near_food': ScenarioConfig(
        name='near_food',
        food_direction=Direction.RIGHT,
        food_proximity=0.8,
        danger_direction=Direction.LEFT,
        danger_proximity=0.1,
        energy=0.6,
        pain=0.0,
        best_actions=[Action.RIGHT],
        description="Close to food on the right"
    ),

    'near_danger': ScenarioConfig(
        name='near_danger',
        food_direction=Direction.HERE,
        food_proximity=0.2,
        danger_direction=Direction.RIGHT,
        danger_proximity=0.8,
        energy=0.7,
        pain=0.3,
        worst_actions=[Action.RIGHT],
        best_actions=[Action.LEFT],
        description="Very close to danger on the right, should flee LEFT"
    ),
}


def get_scenario(name: str) -> ScenarioConfig:
    """Get a predefined scenario by name."""
    if name not in STANDARD_SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(STANDARD_SCENARIOS.keys())}")
    return STANDARD_SCENARIOS[name]


def get_all_scenarios() -> List[ScenarioConfig]:
    """Get all predefined scenarios."""
    return list(STANDARD_SCENARIOS.values())


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_observation(obs: np.ndarray) -> List[str]:
    """
    Validate an observation vector for common issues.

    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []

    if len(obs) != 8:
        warnings.append(f"Expected 8 dimensions, got {len(obs)}")
        return warnings

    # Check ranges
    if not 0 <= obs[0] <= 1:
        warnings.append(f"food_proximity={obs[0]:.2f} out of [0,1] range")
    if not 0 <= obs[1] <= 1:
        warnings.append(f"danger_proximity={obs[1]:.2f} out of [0,1] range")
    if not 0 <= obs[6] <= 1:
        warnings.append(f"energy={obs[6]:.2f} out of [0,1] range")
    if not 0 <= obs[7] <= 1:
        warnings.append(f"pain={obs[7]:.2f} out of [0,1] range")

    # Check dx/dy consistency with proximity
    # If proximity is high, target should be nearby (dx/dy close to 0)
    food_prox, danger_prox = obs[0], obs[1]
    food_dx, food_dy = obs[2], obs[3]
    danger_dx, danger_dy = obs[4], obs[5]

    if food_prox > 0.9 and (abs(food_dx) > 0.2 or abs(food_dy) > 0.2):
        warnings.append(
            f"food_proximity={food_prox:.1f} (very close) but dx/dy=({food_dx:.1f},{food_dy:.1f}) suggests distance"
        )

    if danger_prox > 0.9 and (abs(danger_dx) > 0.2 or abs(danger_dy) > 0.2):
        warnings.append(
            f"danger_proximity={danger_prox:.1f} (very close) but dx/dy=({danger_dx:.1f},{danger_dy:.1f}) suggests distance"
        )

    return warnings


def describe_observation(obs: np.ndarray) -> str:
    """Generate human-readable description of an observation."""
    if len(obs) != 8:
        return f"Invalid observation: expected 8 dims, got {len(obs)}"

    food_prox, danger_prox = obs[0], obs[1]
    food_dx, food_dy = obs[2], obs[3]
    danger_dx, danger_dy = obs[4], obs[5]
    energy, pain = obs[6], obs[7]

    def describe_direction(dx, dy):
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return "HERE"
        parts = []
        if dy < -0.1:
            parts.append("UP")
        elif dy > 0.1:
            parts.append("DOWN")
        if dx < -0.1:
            parts.append("LEFT")
        elif dx > 0.1:
            parts.append("RIGHT")
        return "-".join(parts) if parts else "HERE"

    food_dir = describe_direction(food_dx, food_dy)
    danger_dir = describe_direction(danger_dx, danger_dy)

    lines = [
        f"Food: {food_dir} (prox={food_prox:.1f})",
        f"Danger: {danger_dir} (prox={danger_prox:.1f})",
        f"Energy: {energy:.1f}, Pain: {pain:.1f}"
    ]

    return " | ".join(lines)
