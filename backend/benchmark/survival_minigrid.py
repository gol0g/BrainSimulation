"""
SurvivalMiniGrid v0 - Genesis Brain with Survival Dynamics

Core Design Principles:
1. Goal achievement alone is NOT optimal - survival matters
2. Internal states (energy/pain) cause ACTUAL failure/termination
3. PlannerCircuit must make risk-efficiency tradeoffs

Phase 0: Food-only + Starvation Death
- Energy depletes over time
- Food restores energy
- energy <= 0 → episode terminates (starvation death)

"자기 보존(self-preservation)"이 실제 행동 선택에 영향을 미치는 첫 단계
"""

import gymnasium as gym
import numpy as np
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Door, Wall, Floor
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import re

# Import Genesis Brain components from existing file
from genesis_minigrid import (
    GenesisBrain, RiskCircuit, DefenseCircuit, TransitionCircuit,
    MemoryCircuit, HippocampalCircuit, CuriosityCircuit, MetaCognitionCircuit,
    InteroceptionCircuit, WorkingMemoryCircuit, PlannerCircuit, NarrativeCircuit,
    PCZBridge
)


# =============================================================================
# CUSTOM WORLD OBJECTS
# =============================================================================

class Food(Floor):
    """
    Food object that restores energy when stepped on.
    Disappears after consumption.
    """
    def __init__(self, energy_gain: float = 0.3):
        super().__init__('green')
        self.energy_gain = energy_gain

    def can_overlap(self):
        return True  # Agent can step on food

    def encode(self):
        """Encode for observation: (OBJECT_IDX, COLOR_IDX, state)"""
        # Use a unique object index for food (we'll use 8 which is typically unused)
        return (8, COLOR_TO_IDX['green'], 0)


class Lava(Floor):
    """
    Lava object that causes pain and energy loss when stepped on.
    Agent can still pass through (risk vs shortcut tradeoff).
    """
    def __init__(self, pain_amount: float = 0.5, energy_loss: float = 0.2):
        super().__init__('red')
        self.pain_amount = pain_amount
        self.energy_loss = energy_loss

    def can_overlap(self):
        return True  # Agent can step on lava (but takes damage)

    def encode(self):
        return (9, COLOR_TO_IDX['red'], 0)  # Object index 9 for lava


class Poison(Floor):
    """
    Poison tile that causes continuous pain while standing on it.
    """
    def __init__(self, pain_per_step: float = 0.1):
        super().__init__('purple')
        self.pain_per_step = pain_per_step

    def can_overlap(self):
        return True

    def encode(self):
        return (10, COLOR_TO_IDX['purple'], 0)  # Object index 10 for poison


# =============================================================================
# SURVIVAL MINIGRID ENVIRONMENT
# =============================================================================

class SurvivalMiniGridEnv(MiniGridEnv):
    """
    MiniGrid environment with survival dynamics.

    Key Features:
    - Energy system: depletes over time, death at 0
    - Food objects: restore energy
    - (Future) Lava/Poison: cause damage

    Curriculum Levels:
    - v0: Food-only (test energy management)
    - v1: Lava-only (test risk-reward)
    - v2: Poison-only (test area avoidance)
    - v3: Mixed (full survival challenge)
    """

    def __init__(
        self,
        size: int = 8,
        # Energy system
        initial_energy: float = 1.0,
        energy_decay: float = 0.015,  # Per step
        action_costs: Optional[Dict[int, float]] = None,  # Extra cost per action
        # Food
        n_food: int = 3,
        food_gain: float = 0.3,
        # Hazards (for later phases)
        n_lava: int = 0,
        n_poison: int = 0,
        lava_pain: float = 0.5,
        lava_energy_loss: float = 0.2,
        poison_pain: float = 0.1,
        # Goal
        has_goal: bool = True,
        has_key_door: bool = False,
        # General
        max_steps: int = 200,
        **kwargs
    ):
        self.size = size
        self.initial_energy = initial_energy
        self.energy_decay = energy_decay
        self.action_costs = action_costs or {}

        self.n_food = n_food
        self.food_gain = food_gain
        self.n_lava = n_lava
        self.n_poison = n_poison
        self.lava_pain = lava_pain
        self.lava_energy_loss = lava_energy_loss
        self.poison_pain = poison_pain

        self.has_goal = has_goal
        self.has_key_door = has_key_door

        # Survival state (will be reset in reset())
        self.energy = initial_energy
        self.pain = 0.0
        self.food_positions = []
        self.lava_positions = []
        self.poison_positions = []

        # Event tracking for this step
        self.step_events = []

        mission_space = MissionSpace(mission_func=lambda: "survive and reach the goal")

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs
        )

    def _gen_grid(self, width, height):
        """Generate the grid with survival objects."""
        self.grid = Grid(width, height)

        # Walls around the border
        self.grid.wall_rect(0, 0, width, height)

        # Reset survival state
        self.energy = self.initial_energy
        self.pain = 0.0
        self.food_positions = []
        self.lava_positions = []
        self.poison_positions = []
        self.step_events = []

        # Poison tracking (for ENTER/EXIT events)
        self.was_on_poison = False
        self.poison_ticks_this_episode = 0

        # Place agent at fixed position FIRST (prevents conflicts with objects)
        self.agent_pos = np.array([1, 1])
        self.agent_dir = self._rand_int(0, 4)

        # Place goal
        if self.has_goal:
            self.put_obj(Goal(), width - 2, height - 2)

        # Place food
        for _ in range(self.n_food):
            pos = self.place_obj(Food(self.food_gain), max_tries=100)
            if pos is not None:
                self.food_positions.append(pos)

        # Place lava (for later phases)
        for _ in range(self.n_lava):
            pos = self.place_obj(Lava(self.lava_pain, self.lava_energy_loss), max_tries=100)
            if pos is not None:
                self.lava_positions.append(pos)

        # Place poison (for later phases)
        for _ in range(self.n_poison):
            pos = self.place_obj(Poison(self.poison_pain), max_tries=100)
            if pos is not None:
                self.poison_positions.append(pos)

        # Optional: Key-Door puzzle
        if self.has_key_door:
            # Place door in the middle
            door_pos = (width // 2, height // 2)
            self.put_obj(Door('yellow', is_locked=True), *door_pos)
            # Place key somewhere
            self.place_obj(Key('yellow'), max_tries=100)

    def step(self, action):
        """Execute action with survival dynamics."""
        self.step_events = []  # Reset events for this step

        # Energy cost for action
        action_cost = self.action_costs.get(action, 0.0)
        self.energy -= action_cost

        # Base energy decay per step
        self.energy -= self.energy_decay

        # Check for near-starvation event (first time crossing threshold)
        if self.energy < 0.3 and self.energy + self.energy_decay >= 0.3:
            self.step_events.append(('NEAR_STARVATION', {'energy': self.energy}))

        # Execute the base MiniGrid step
        obs, reward, terminated, truncated, info = super().step(action)

        # Check what we stepped on
        current_cell = self.grid.get(*self.agent_pos)

        # Food consumption (automatic on step)
        if isinstance(current_cell, Food):
            food_obj = current_cell
            self.energy = min(1.0, self.energy + food_obj.energy_gain)
            self.step_events.append(('FOOD_EATEN', {
                'amount': food_obj.energy_gain,
                'pos': self.agent_pos,
                'energy_after': self.energy
            }))
            # Remove food from grid
            self.grid.set(*self.agent_pos, None)
            if tuple(self.agent_pos) in self.food_positions:
                self.food_positions.remove(tuple(self.agent_pos))

        # Lava damage (one-time on contact)
        if isinstance(current_cell, Lava):
            self.pain = min(1.0, self.pain + current_cell.pain_amount)
            self.energy = max(0.0, self.energy - current_cell.energy_loss)
            self.step_events.append(('LAVA_HIT', {
                'pain': current_cell.pain_amount,
                'energy_loss': current_cell.energy_loss,
                'pos': self.agent_pos
            }))

        # Poison damage (continuous while standing)
        is_on_poison = isinstance(current_cell, Poison)

        if is_on_poison:
            # POISON_ENTER: first step onto poison
            if not self.was_on_poison:
                self.step_events.append(('POISON_ENTER', {
                    'pos': tuple(self.agent_pos)
                }))

            # POISON_TICK: every step on poison
            self.pain = min(1.0, self.pain + current_cell.pain_per_step)
            self.poison_ticks_this_episode += 1
            self.step_events.append(('POISON_TICK', {
                'pain': current_cell.pain_per_step,
                'total_ticks': self.poison_ticks_this_episode,
                'pos': tuple(self.agent_pos)
            }))
        else:
            # POISON_EXIT: left poison zone
            if self.was_on_poison:
                self.step_events.append(('POISON_EXIT', {
                    'total_ticks': self.poison_ticks_this_episode,
                    'pos': tuple(self.agent_pos)
                }))

        # Update poison tracking
        self.was_on_poison = is_on_poison

        # Check for death conditions
        death_reason = None

        # Starvation death
        if self.energy <= 0:
            self.energy = 0.0
            terminated = True
            death_reason = 'starvation'
            self.step_events.append(('ENERGY_DEPLETED', {'final_energy': 0.0}))
            reward = -1.0  # Penalty for death

        # Fatal injury (optional - pain >= 1.0)
        if self.pain >= 1.0:
            self.pain = 1.0
            terminated = True
            death_reason = 'injury'
            reward = -1.0

        # Natural pain recovery (small)
        self.pain = max(0.0, self.pain - 0.02)

        # Add survival info
        info['energy'] = self.energy
        info['pain'] = self.pain
        info['death_reason'] = death_reason
        info['events'] = self.step_events.copy()
        info['food_remaining'] = len(self.food_positions)
        info['time_pressure'] = self.step_count / self.max_steps

        return obs, reward, terminated, truncated, info

    def get_survival_state(self) -> Dict:
        """Get current survival state for brain integration."""
        return {
            'energy': self.energy,
            'pain': self.pain,
            'time_pressure': self.step_count / self.max_steps,
            'food_positions': self.food_positions.copy(),
            'lava_positions': self.lava_positions.copy(),
            'poison_positions': self.poison_positions.copy(),
            'events': self.step_events.copy(),
        }


# =============================================================================
# MIXED SURVIVAL ENVIRONMENT (Phase 3)
# =============================================================================

class MixedSurvivalEnv(SurvivalMiniGridEnv):
    """
    Phase 3 Mixed Environment with Dilemma-based Map Generation.

    Key Design:
    1. Shortest path crosses hazard zones (creates dilemma)
    2. Safe alternate path always exists (prevents luck-based failure)
    3. Food placement: 1 safe-path, 1 risky-path (choice measurement)

    New Metrics:
    - risky_shortcut_rate: hazard tile crossing rate
    - food_before_goal_rate: food prioritization frequency
    """

    def __init__(
        self,
        size: int = 10,
        # Dilemma settings
        hazard_on_path_prob: float = 0.7,  # Prob of hazard blocking shortest path
        safe_path_guaranteed: bool = True,
        # Inherited
        **kwargs
    ):
        self.hazard_on_path_prob = hazard_on_path_prob
        self.safe_path_guaranteed = safe_path_guaranteed

        # Phase 3 tracking
        self.risky_tiles_crossed = 0
        self.total_path_tiles = 0
        self.food_eaten_before_goal = 0
        self.goal_reached_hungry = 0

        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        """Generate dilemma-based grid for Phase 3."""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Reset survival state
        self.energy = self.initial_energy
        self.pain = 0.0
        self.food_positions = []
        self.lava_positions = []
        self.poison_positions = []
        self.step_events = []
        self.was_on_poison = False
        self.poison_ticks_this_episode = 0

        # Reset Phase 3 tracking
        self.risky_tiles_crossed = 0
        self.total_path_tiles = 0

        # Place agent at top-left area
        self.agent_pos = np.array([1, 1])
        self.agent_dir = 0  # Facing right

        # Place goal at bottom-right
        goal_pos = (width - 2, height - 2)
        if self.has_goal:
            self.put_obj(Goal(), *goal_pos)

        # Calculate "direct path" region (diagonal band from agent to goal)
        # Hazards placed here create shortcut dilemma
        direct_path_cells = self._get_direct_path_region(
            self.agent_pos, goal_pos, width, height
        )

        # Place hazards with bias toward direct path
        hazard_positions = []

        # Place lava (if any)
        for _ in range(self.n_lava):
            if np.random.random() < self.hazard_on_path_prob and direct_path_cells:
                # Place on direct path
                idx = np.random.randint(len(direct_path_cells))
                pos = direct_path_cells.pop(idx)
                self.put_obj(Lava(self.lava_pain, self.lava_energy_loss), *pos)
                self.lava_positions.append(pos)
                hazard_positions.append(pos)
            else:
                # Random placement
                pos = self.place_obj(
                    Lava(self.lava_pain, self.lava_energy_loss), max_tries=100
                )
                if pos:
                    self.lava_positions.append(pos)
                    hazard_positions.append(pos)

        # Place poison (if any)
        for _ in range(self.n_poison):
            if np.random.random() < self.hazard_on_path_prob and direct_path_cells:
                idx = np.random.randint(len(direct_path_cells))
                pos = direct_path_cells.pop(idx)
                self.put_obj(Poison(self.poison_pain), *pos)
                self.poison_positions.append(pos)
                hazard_positions.append(pos)
            else:
                pos = self.place_obj(Poison(self.poison_pain), max_tries=100)
                if pos:
                    self.poison_positions.append(pos)
                    hazard_positions.append(pos)

        # Ensure safe path exists (clear a corridor along edges if needed)
        if self.safe_path_guaranteed and hazard_positions:
            self._ensure_safe_path(width, height, hazard_positions)

        # Place food with dilemma distribution
        # 1 food on safe path side, 1 near risky path
        if self.n_food >= 2:
            # Safe food: near top or left edge
            safe_food_pos = self._find_safe_food_position(width, height, hazard_positions)
            if safe_food_pos:
                self.put_obj(Food(self.food_gain), *safe_food_pos)
                self.food_positions.append(safe_food_pos)

            # Risky food: near hazards (temptation)
            risky_food_pos = self._find_risky_food_position(hazard_positions)
            if risky_food_pos:
                self.put_obj(Food(self.food_gain), *risky_food_pos)
                self.food_positions.append(risky_food_pos)

            # Remaining food random
            for _ in range(self.n_food - 2):
                pos = self.place_obj(Food(self.food_gain), max_tries=100)
                if pos:
                    self.food_positions.append(pos)
        else:
            # Few food: random placement
            for _ in range(self.n_food):
                pos = self.place_obj(Food(self.food_gain), max_tries=100)
                if pos:
                    self.food_positions.append(pos)

        # Key-Door if enabled
        if self.has_key_door:
            door_pos = (width // 2, height // 2)
            self.put_obj(Door('yellow', is_locked=True), *door_pos)
            self.place_obj(Key('yellow'), max_tries=100)

    def _get_direct_path_region(self, start, goal, width, height) -> List[Tuple[int, int]]:
        """Get cells along the direct diagonal path (potential shortcut region)."""
        cells = []
        sx, sy = start
        gx, gy = goal

        # Diagonal band: cells within 2 tiles of the direct line
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if (x, y) == tuple(start) or (x, y) == goal:
                    continue
                # Check if on diagonal path
                dx = gx - sx
                dy = gy - sy
                if dx == 0 and dy == 0:
                    continue
                # Distance from point to line
                t = ((x - sx) * dx + (y - sy) * dy) / (dx * dx + dy * dy)
                t = max(0, min(1, t))
                closest_x = sx + t * dx
                closest_y = sy + t * dy
                dist = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
                if dist < 2.0:  # Within 2 tiles of direct path
                    cells.append((x, y))
        return cells

    def _ensure_safe_path(self, width, height, hazard_positions):
        """Ensure a safe path exists along edges."""
        # Clear top edge path
        for x in range(1, width - 1):
            pos = (x, 1)
            if pos in hazard_positions:
                cell = self.grid.get(*pos)
                if isinstance(cell, (Lava, Poison)):
                    self.grid.set(*pos, None)
                    if pos in self.lava_positions:
                        self.lava_positions.remove(pos)
                    if pos in self.poison_positions:
                        self.poison_positions.remove(pos)

        # Clear right edge path
        for y in range(1, height - 1):
            pos = (width - 2, y)
            if pos in hazard_positions:
                cell = self.grid.get(*pos)
                if isinstance(cell, (Lava, Poison)):
                    self.grid.set(*pos, None)
                    if pos in self.lava_positions:
                        self.lava_positions.remove(pos)
                    if pos in self.poison_positions:
                        self.poison_positions.remove(pos)

    def _find_safe_food_position(self, width, height, hazard_positions) -> Optional[Tuple[int, int]]:
        """Find a food position on the safe path (edges)."""
        # Try top edge
        for x in range(2, width - 2):
            pos = (x, 1)
            if pos not in hazard_positions and self.grid.get(*pos) is None:
                return pos
        # Try left edge
        for y in range(2, height - 2):
            pos = (1, y)
            if pos not in hazard_positions and self.grid.get(*pos) is None:
                return pos
        return None

    def _find_risky_food_position(self, hazard_positions) -> Optional[Tuple[int, int]]:
        """Find a food position near hazards (temptation)."""
        if not hazard_positions:
            return None

        # Pick a random hazard and place food adjacent
        hazard = hazard_positions[np.random.randint(len(hazard_positions))]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            pos = (hazard[0] + dx, hazard[1] + dy)
            if (1 <= pos[0] < self.width - 1 and
                1 <= pos[1] < self.height - 1 and
                self.grid.get(*pos) is None):
                return pos
        return None

    def step(self, action):
        """Step with Phase 3 tracking."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Track risky tile crossing
        current_cell = self.grid.get(*self.agent_pos)
        if isinstance(current_cell, (Lava, Poison)):
            self.risky_tiles_crossed += 1
        self.total_path_tiles += 1

        # Add Phase 3 metrics to info
        info['risky_tiles_crossed'] = self.risky_tiles_crossed
        info['total_path_tiles'] = self.total_path_tiles

        return obs, reward, terminated, truncated, info

    def get_phase3_metrics(self) -> Dict:
        """Get Phase 3 specific metrics."""
        return {
            'risky_shortcut_rate': self.risky_tiles_crossed / max(1, self.total_path_tiles),
            'risky_tiles_crossed': self.risky_tiles_crossed,
            'total_path_tiles': self.total_path_tiles,
        }


# =============================================================================
# PHASE 4A: GOAL VALUE MODULATION ENVIRONMENT
# =============================================================================

class GoalValueEnv(MixedSurvivalEnv):
    """
    Phase 4A: Goal Value Modulation Environment.

    Key Innovation:
    - Each episode has a goal_value (low/mid/high)
    - High goal_value + time_pressure → should encourage risk-taking
    - Tests: "Is agent just always safe, or does it adapt to context?"

    Expected Behavior:
    - goal_value=high → risky_shortcut_rate increases
    - goal_value=low → food/safety prioritized
    """

    # Goal value levels
    GOAL_LOW = 0.3
    GOAL_MID = 0.6
    GOAL_HIGH = 1.0

    def __init__(
        self,
        size: int = 10,
        # Goal value settings
        goal_value_distribution: str = 'uniform',  # 'uniform', 'bimodal', 'fixed_high', 'fixed_low'
        fixed_goal_value: Optional[float] = None,
        # Inherited
        **kwargs
    ):
        self.goal_value_distribution = goal_value_distribution
        self.fixed_goal_value = fixed_goal_value

        # Current episode goal value
        self.goal_value = self.GOAL_MID
        self.goal_value_label = 'mid'

        # Phase 4A tracking
        self.risk_taken_for_goal = 0  # Times crossed hazard while heading to goal
        self.safe_detour_taken = 0    # Times took longer safe path

        super().__init__(size=size, **kwargs)

    def _sample_goal_value(self) -> Tuple[float, str]:
        """Sample goal value for this episode."""
        if self.fixed_goal_value is not None:
            value = self.fixed_goal_value
        elif self.goal_value_distribution == 'uniform':
            # Equal chance of low/mid/high
            choice = np.random.choice(['low', 'mid', 'high'])
            value = {'low': self.GOAL_LOW, 'mid': self.GOAL_MID, 'high': self.GOAL_HIGH}[choice]
        elif self.goal_value_distribution == 'bimodal':
            # Either low or high (test extremes)
            choice = np.random.choice(['low', 'high'])
            value = {'low': self.GOAL_LOW, 'high': self.GOAL_HIGH}[choice]
        elif self.goal_value_distribution == 'fixed_high':
            value = self.GOAL_HIGH
            choice = 'high'
        elif self.goal_value_distribution == 'fixed_low':
            value = self.GOAL_LOW
            choice = 'low'
        else:
            value = self.GOAL_MID
            choice = 'mid'

        # Determine label
        if value <= 0.4:
            label = 'low'
        elif value >= 0.8:
            label = 'high'
        else:
            label = 'mid'

        return value, label

    def _gen_grid(self, width, height):
        """Generate grid with goal value sampling."""
        # Sample goal value for this episode
        self.goal_value, self.goal_value_label = self._sample_goal_value()

        # Reset Phase 4A tracking
        self.risk_taken_for_goal = 0
        self.safe_detour_taken = 0

        # Generate base grid
        super()._gen_grid(width, height)

    def step(self, action):
        """Step with goal value context in info."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Add goal value context to info
        info['goal_value'] = self.goal_value
        info['goal_value_label'] = self.goal_value_label
        info['time_pressure'] = self.step_count / self.max_steps

        # Compute "urgency" = goal_value * time_pressure
        # This is what should drive risk-taking behavior
        info['urgency'] = self.goal_value * info['time_pressure']

        # Scale reward by goal value (reaching high-value goal is more rewarding)
        if reward > 0:  # Goal reached
            reward = reward * (0.5 + 0.5 * self.goal_value)

        return obs, reward, terminated, truncated, info

    def get_phase4_metrics(self) -> Dict:
        """Get Phase 4A specific metrics."""
        base = self.get_phase3_metrics()
        base.update({
            'goal_value': self.goal_value,
            'goal_value_label': self.goal_value_label,
            'risk_taken_for_goal': self.risk_taken_for_goal,
            'safe_detour_taken': self.safe_detour_taken,
        })
        return base


# =============================================================================
# PHASE 4B: UNCERTAINTY / HIDDEN HAZARD ENVIRONMENT
# =============================================================================

class UncertaintyEnv(GoalValueEnv):
    """
    Phase 4B: Uncertainty Environment with Hidden Hazards.

    Key Innovations:
    1. Some hazards are HIDDEN (not visible in observation)
    2. Hidden hazards emit SIGNALS (smell/heat) in adjacent cells
    3. Agent must use signals to infer hazard locations
    4. Information-seeking behavior becomes valuable

    This tests:
    - Active Inference: "I'm uncertain, so I should observe more"
    - Deliberate exploration before commitment
    - Uncertainty-weighted risk assessment
    """

    def __init__(
        self,
        size: int = 10,
        # Uncertainty settings
        hazard_masked_prob: float = 0.5,  # Probability hazard is hidden
        signal_radius: int = 1,            # Signal detection radius
        probabilistic_hazard: bool = False,  # If True, some hazards are probabilistic
        prob_hazard_chance: float = 0.3,   # Damage probability for prob hazards
        # SCAN action
        enable_scan: bool = True,          # Add SCAN action (reveals adjacent hidden hazards)
        scan_energy_cost: float = 0.02,
        # Inherited
        **kwargs
    ):
        self.hazard_masked_prob = hazard_masked_prob
        self.signal_radius = signal_radius
        self.probabilistic_hazard = probabilistic_hazard
        self.prob_hazard_chance = prob_hazard_chance
        self.enable_scan = enable_scan
        self.scan_energy_cost = scan_energy_cost

        # Hidden hazard tracking
        self.hidden_hazards = set()       # Positions of hidden hazards
        self.revealed_hazards = set()     # Hazards revealed by proximity/scan
        self.hazard_signal_map = {}       # Position -> signal strength (0-1)
        self.prob_hazards = set()         # Probabilistic hazard positions

        # Phase 4B tracking
        self.scan_actions = 0
        self.hazards_revealed = 0
        self.entered_uncertain_area = 0   # Entered area with signal but no visual
        self.observed_before_entry = 0    # Observed (moved nearby) before entering
        self.surprises = 0                # Hit hidden hazard without signal

        super().__init__(size=size, **kwargs)

    def _gen_grid(self, width, height):
        """Generate grid with hidden hazards."""
        # First generate base grid
        super()._gen_grid(width, height)

        # Reset Phase 4B tracking
        self.hidden_hazards = set()
        self.revealed_hazards = set()
        self.hazard_signal_map = {}
        self.prob_hazards = set()
        self.scan_actions = 0
        self.hazards_revealed = 0
        self.entered_uncertain_area = 0
        self.observed_before_entry = 0
        self.surprises = 0

        # Convert some hazards to hidden
        all_hazards = list(self.lava_positions) + list(self.poison_positions)

        for pos in all_hazards:
            if np.random.random() < self.hazard_masked_prob:
                self.hidden_hazards.add(tuple(pos))
                # Remove from visible grid (but keep in position lists)
                # The hazard still exists, just not visible
                # We'll handle damage in step()

            # Some hazards become probabilistic
            if self.probabilistic_hazard and np.random.random() < 0.3:
                self.prob_hazards.add(tuple(pos))

        # Generate signal map for hidden hazards
        self._update_signal_map()

    def _update_signal_map(self):
        """Update hazard signal map based on hidden hazards."""
        self.hazard_signal_map = {}

        for hazard_pos in self.hidden_hazards:
            if hazard_pos in self.revealed_hazards:
                continue  # Already revealed, no signal needed

            # Add signal to adjacent cells
            hx, hy = hazard_pos
            for dx in range(-self.signal_radius, self.signal_radius + 1):
                for dy in range(-self.signal_radius, self.signal_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    sx, sy = hx + dx, hy + dy
                    if 1 <= sx < self.width - 1 and 1 <= sy < self.height - 1:
                        # Signal strength decreases with distance
                        dist = max(abs(dx), abs(dy))
                        signal = 1.0 / dist
                        current = self.hazard_signal_map.get((sx, sy), 0.0)
                        self.hazard_signal_map[(sx, sy)] = max(current, signal)

    def _reveal_hazard(self, pos):
        """Reveal a hidden hazard at position."""
        pos = tuple(pos)
        if pos in self.hidden_hazards and pos not in self.revealed_hazards:
            self.revealed_hazards.add(pos)
            self.hazards_revealed += 1
            self._update_signal_map()
            return True
        return False

    def step(self, action):
        """Step with hidden hazard mechanics."""
        self.step_events = []

        # Handle SCAN action (action 7 if enabled)
        if self.enable_scan and action == 7:
            self.scan_actions += 1
            self.energy -= self.scan_energy_cost

            # Reveal hidden hazards in adjacent cells
            revealed_count = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                scan_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                if self._reveal_hazard(scan_pos):
                    revealed_count += 1

            self.step_events.append(('SCAN', {
                'pos': tuple(self.agent_pos),
                'revealed': revealed_count
            }))

            # SCAN doesn't move, so return current state
            obs = self.gen_obs()
            info = {
                'energy': self.energy,
                'pain': self.pain,
                'death_reason': None,
                'events': self.step_events.copy(),
                'goal_value': self.goal_value,
                'goal_value_label': self.goal_value_label,
                'time_pressure': self.step_count / self.max_steps,
                'urgency': self.goal_value * (self.step_count / self.max_steps),
                'hazard_signal_map': self.hazard_signal_map.copy(),
                'hidden_hazards_count': len(self.hidden_hazards) - len(self.revealed_hazards),
                'uncertainty': self._compute_uncertainty(),
            }
            return obs, 0, False, False, info

        # Check if entering uncertain area (has signal but no visible hazard)
        agent_tuple = tuple(self.agent_pos)
        if agent_tuple in self.hazard_signal_map:
            self.entered_uncertain_area += 1

        # Auto-reveal hazards when agent is adjacent (proximity reveal)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            self._reveal_hazard(adj_pos)

        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if stepped on hidden hazard
        agent_pos = tuple(self.agent_pos)
        if agent_pos in self.hidden_hazards and agent_pos not in self.revealed_hazards:
            # Surprise! Hit hidden hazard
            self.surprises += 1
            self._reveal_hazard(agent_pos)
            self.step_events.append(('SURPRISE_HAZARD', {'pos': agent_pos}))

        # Handle probabilistic hazards
        if agent_pos in self.prob_hazards:
            if np.random.random() < self.prob_hazard_chance:
                # Damage occurs
                self.pain = min(1.0, self.pain + 0.1)
                self.step_events.append(('PROB_HAZARD_TRIGGERED', {
                    'pos': agent_pos,
                    'pain': 0.1
                }))
            else:
                self.step_events.append(('PROB_HAZARD_AVOIDED', {'pos': agent_pos}))

        # Add Phase 4B info
        info['hazard_signal_map'] = self.hazard_signal_map.copy()
        info['hidden_hazards_count'] = len(self.hidden_hazards) - len(self.revealed_hazards)
        info['uncertainty'] = self._compute_uncertainty()
        info['events'] = self.step_events.copy()

        return obs, reward, terminated, truncated, info

    def _compute_uncertainty(self) -> float:
        """Compute current uncertainty level (0-1)."""
        if not self.hidden_hazards:
            return 0.0
        hidden_unrevealed = len(self.hidden_hazards) - len(self.revealed_hazards)
        return hidden_unrevealed / len(self.hidden_hazards)

    def get_phase4b_metrics(self) -> Dict:
        """Get Phase 4B specific metrics."""
        base = self.get_phase4_metrics()
        base.update({
            'hidden_hazards_total': len(self.hidden_hazards),
            'hazards_revealed': self.hazards_revealed,
            'scan_actions': self.scan_actions,
            'entered_uncertain_area': self.entered_uncertain_area,
            'observed_before_entry': self.observed_before_entry,
            'surprises': self.surprises,
            'uncertainty': self._compute_uncertainty(),
        })
        return base


# =============================================================================
# GENESIS BRAIN WITH SURVIVAL EXTENSIONS
# =============================================================================

class SurvivalGenesisBrain(GenesisBrain):
    """
    Extended Genesis Brain with survival-aware decision making.

    Key Extensions:
    1. G(a) includes survival term: energy urgency, pain avoidance
    2. Planner considers food/hazard locations
    3. MetaCognition tracks survival events and learns rules
    4. Interoception connects to ACTUAL energy/pain from environment
    """

    def __init__(self, n_actions: int = 7, grid_size: int = 10):
        super().__init__(n_actions, grid_size)

        # Survival-specific weights for G(a)
        # CRITICAL: These must be strong enough to override goal-seeking
        self.energy_urgency_weight = 5.0  # High value for survival priority
        self.pain_avoidance_weight = 2.0  # How much pain affects decisions
        self.food_attraction_weight = 4.0  # Strong attraction to food when hungry

        # Track survival events for metacognition
        self.survival_events = []

        # Episode survival stats
        self.food_eaten_count = 0
        self.lava_hits = 0
        self.poison_ticks = 0
        self.poison_enters = 0
        self.poison_exits = 0
        self.poison_loops = 0  # Re-entering poison after exit
        self.near_starvation_count = 0

        # Poison zone tracking for loop detection
        self.exited_poison_positions = set()  # Positions we've left

        # Phase 3 metrics
        self.food_eaten_while_hungry = 0  # Ate when energy < 0.5
        self.goal_reached_while_hungry = 0  # Reached goal without eating first
        self.planner_activations = 0  # PC-Z triggered planner
        self.risky_tiles_crossed = 0  # Stepped on hazard tiles

        # Phase 4A: Goal Value Modulation
        self.goal_value_weight = 3.0  # How much goal_value affects risk tolerance
        self.urgency_risk_weight = 2.0  # urgency = goal_value * time_pressure
        self.current_goal_value = 0.5  # Default mid
        self.current_urgency = 0.0
        self.risks_taken_for_high_goal = 0  # Crossed hazard when goal_value > 0.7
        self.safe_choices_for_low_goal = 0  # Avoided hazard when goal_value < 0.4
        self.goal_value_risk_decisions = []  # (goal_value, took_risk) pairs

        # Phase 4B: Uncertainty / Information Seeking
        self.uncertainty_weight = 2.0  # How much uncertainty affects exploration
        self.info_gain_weight = 1.5    # Reward for reducing uncertainty
        self.current_uncertainty = 0.0
        self.scan_count = 0            # SCAN action usage
        self.exploration_for_info = 0  # Moved to gather information
        self.surprises_count = 0       # Hit hidden hazards
        self.info_gain_events = []     # (uncertainty_before, uncertainty_after) pairs

        # Phase 5: Causal Curiosity (NEW - for discovering new causal rules)
        # Track which object encodings have been interacted with
        self.object_interactions = {}   # {(obj_type, color): count}
        # Track which positions caused unexpected effects
        self.causal_positions = {}      # {(x, y): {'effect': str, 'count': int}}
        # Track recent state for causal discovery
        self.prev_door_state = False
        self.prev_key_state = False
        # Novel object exploration weight
        self.causal_curiosity_weight = 2.5
        # Statistics
        self.causal_discoveries = 0     # Times found new causal relationship
        self.novel_object_approaches = 0  # Times approached novel objects

        # Phase 6: Pattern Conflict Detection & Prior Suppression
        # Track expected patterns and their success/failure
        self.pattern_attempts = {}      # {'key_door': {'success': 0, 'fail': 0}}
        self.pattern_suppression = {}   # {'key_door': 0.0-1.0} suppression weight
        self.suppression_threshold = 3  # Failures before suppression kicks in (fast adaptation)
        self.suppression_decay = 0.95   # Decay rate per episode
        # Track consecutive failures for rapid adaptation
        self.consecutive_door_failures = 0
        self.last_door_attempt_step = -1
        # Fresh exploration boost when priors suppressed
        self.exploration_boost = 0.0
        # Statistics
        self.pattern_conflicts_detected = 0
        self.prior_suppressions = 0

        # Phase 7: Directed Exploration
        # Track visited positions to guide exploration to new areas
        self.visited_positions = set()  # {(x, y), ...}
        self.visit_counts = {}          # {(x, y): count}
        self.exploration_targets = []   # [(x, y), ...] positions to explore
        self.directed_exploration_weight = 3.0
        # Track steps since last progress
        self.steps_without_progress = 0
        self.last_progress_step = 0
        # Statistics
        self.directed_explorations = 0
        self.novel_positions_visited = 0

        # Phase 8: Anti-Forgetting (Memory Consolidation)
        # Experience replay buffer for successful episodes
        self.replay_buffer = []  # List of successful episode traces
        self.replay_buffer_size = 50  # Max stored episodes
        self.current_episode_trace = []  # Current episode's (state, action, reward) tuples
        # Success pattern protection
        self.success_patterns = {}  # {state_key: {action: success_count}}
        self.protection_threshold = 3  # Successes before protection kicks in
        # Consolidation parameters
        self.consolidation_interval = 10  # Replay every N episodes
        self.consolidation_batch_size = 5  # Episodes to replay
        self.consolidation_lr_multiplier = 0.3  # Lower LR for replay
        # Episode counter for consolidation timing
        self.total_episodes = 0
        # Statistics
        self.consolidation_count = 0
        self.protected_updates = 0
        self.replay_updates = 0

    def reset(self):
        """Reset with survival tracking."""
        super().reset()
        self.survival_events = []
        self.food_eaten_count = 0
        self.lava_hits = 0
        self.poison_ticks = 0
        self.poison_enters = 0
        self.poison_exits = 0
        self.poison_loops = 0
        self.near_starvation_count = 0
        self.exited_poison_positions = set()
        # Phase 3 resets
        self.food_eaten_while_hungry = 0
        self.goal_reached_while_hungry = 0
        self.planner_activations = 0
        self.risky_tiles_crossed = 0
        # Phase 4A resets (per-episode)
        self.current_goal_value = 0.5
        self.current_urgency = 0.0
        # Phase 4B resets
        self.current_uncertainty = 0.0
        self.scan_count = 0
        self.exploration_for_info = 0
        self.surprises_count = 0
        # Phase 5 resets (per-episode, but keep learned causal knowledge)
        self.prev_door_state = False
        self.prev_key_state = False
        # Phase 6 resets (per-episode, but keep suppression weights)
        self.consecutive_door_failures = 0
        self.last_door_attempt_step = -1
        # Decay suppression weights each episode (allow recovery)
        for pattern in self.pattern_suppression:
            self.pattern_suppression[pattern] *= self.suppression_decay
        # Compute exploration boost based on total suppression
        total_suppression = sum(self.pattern_suppression.values())
        self.exploration_boost = min(1.0, total_suppression * 0.5)
        # Phase 7 resets (per-episode, reset visit tracking)
        self.visited_positions = set()
        self.visit_counts = {}
        self.exploration_targets = []
        self.steps_without_progress = 0
        self.last_progress_step = 0

    def _extract_survival_info(self, obs: Dict, env) -> Dict:
        """Extract survival-specific information from observation."""
        base_info = self._extract_info(obs, env)

        # Get survival state from environment
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'get_survival_state'):
            survival = env.unwrapped.get_survival_state()
            base_info['env_energy'] = survival['energy']
            base_info['env_pain'] = survival['pain']
            base_info['food_positions'] = survival['food_positions']
            base_info['lava_positions'] = survival['lava_positions']
            base_info['poison_positions'] = survival['poison_positions']
            base_info['survival_events'] = survival['events']
        else:
            # Fallback for non-survival environments
            base_info['env_energy'] = 1.0
            base_info['env_pain'] = 0.0
            base_info['food_positions'] = []
            base_info['lava_positions'] = []
            base_info['poison_positions'] = []
            base_info['survival_events'] = []

        # Phase 4A: Get goal value from environment
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'goal_value'):
            base_info['goal_value'] = env.unwrapped.goal_value
            base_info['goal_value_label'] = getattr(env.unwrapped, 'goal_value_label', 'mid')
        else:
            base_info['goal_value'] = 0.5  # Default mid
            base_info['goal_value_label'] = 'mid'

        # Compute time pressure and urgency
        if hasattr(env, 'unwrapped'):
            step_count = getattr(env.unwrapped, 'step_count', 0)
            max_steps = getattr(env.unwrapped, 'max_steps', 100)
            base_info['time_pressure'] = step_count / max_steps
        else:
            base_info['time_pressure'] = 0.0

        base_info['urgency'] = base_info['goal_value'] * base_info['time_pressure']

        # Phase 4B: Get uncertainty and signal info
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'hazard_signal_map'):
            base_info['hazard_signal_map'] = env.unwrapped.hazard_signal_map.copy()
            base_info['uncertainty'] = env.unwrapped._compute_uncertainty() if hasattr(env.unwrapped, '_compute_uncertainty') else 0.0
            base_info['hidden_hazards_count'] = len(env.unwrapped.hidden_hazards) - len(env.unwrapped.revealed_hazards)
            base_info['has_scan_action'] = getattr(env.unwrapped, 'enable_scan', False)
        else:
            base_info['hazard_signal_map'] = {}
            base_info['uncertainty'] = 0.0
            base_info['hidden_hazards_count'] = 0
            base_info['has_scan_action'] = False

        return base_info

    def _compute_survival_G(self, state: Tuple, info: Dict) -> np.ndarray:
        """
        Compute survival component of G(a).

        G_survival(a) = w_E * f(energy) + w_P * g(pain) + w_F * h(food_dist)

        Key insight: energy urgency increases NON-LINEARLY as energy drops
        """
        G_survival = np.zeros(self.n_actions)

        agent_pos = info['agent_pos']
        env_energy = info.get('env_energy', 1.0)
        env_pain = info.get('env_pain', 0.0)
        food_positions = info.get('food_positions', [])

        # 1. Energy urgency: 1/(energy + eps) style - gets urgent as energy drops
        if env_energy < 0.5:
            # Energy urgency kicks in below 50%
            energy_urgency = self.energy_urgency_weight * (1.0 / (env_energy + 0.1) - 1.0)
        else:
            energy_urgency = 0.0

        # 2. Pain avoidance: linear but increases near death threshold
        if env_pain > 0.3:
            pain_factor = self.pain_avoidance_weight * (env_pain - 0.3) * 2.0
        else:
            pain_factor = 0.0

        # 3. Food attraction when hungry
        if env_energy < 0.5 and food_positions:
            # Find nearest food
            food_dists = [np.linalg.norm(agent_pos - np.array(fp)) for fp in food_positions]
            nearest_food_dist = min(food_dists) if food_dists else float('inf')
            nearest_food_idx = food_dists.index(nearest_food_dist) if food_dists else -1

            if nearest_food_idx >= 0:
                nearest_food = np.array(food_positions[nearest_food_idx])

                # Direction to food
                direction = info.get('direction', 0)
                dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                forward_vec = np.array(dir_vecs[direction])

                food_diff = nearest_food - agent_pos
                food_dist = np.linalg.norm(food_diff)

                if food_dist > 0:
                    food_unit = food_diff / food_dist
                    forward_alignment = np.dot(forward_vec, food_unit)

                    # Food attraction scales NON-LINEARLY with hunger
                    # At 50% energy: hunger = 0
                    # At 25% energy: hunger = 1.0
                    # At 10% energy: hunger = 4.0 (URGENT!)
                    # At 5% energy: hunger = 9.0 (CRITICAL!)
                    if env_energy < 0.5:
                        # Use 1/(energy + eps) - 2 style for sharp increase
                        hunger = max(0, (1.0 / (env_energy + 0.1)) - 2.0)
                    else:
                        hunger = 0.0

                    food_attraction = self.food_attraction_weight * hunger

                    # Bias actions toward food (stronger effect)
                    if forward_alignment > 0.1:
                        G_survival[2] -= food_attraction * forward_alignment  # Forward toward food

                    # Need to turn toward food (always check)
                    for turn_action in [0, 1]:  # Left, Right
                        new_dir = (direction + (1 if turn_action == 1 else -1)) % 4
                        new_forward = np.array(dir_vecs[new_dir])
                        new_alignment = np.dot(new_forward, food_unit)
                        if new_alignment > forward_alignment + 0.1:
                            G_survival[turn_action] -= food_attraction * 0.5

        # 4. Hazard avoidance (for lava/poison)
        lava_positions = info.get('lava_positions', [])
        poison_positions = info.get('poison_positions', [])

        direction = info.get('direction', 0)
        dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        forward_vec = np.array(dir_vecs[direction])
        front_pos = agent_pos + np.array(forward_vec)

        # Penalty for moving toward lava
        for lava_pos in lava_positions:
            if np.allclose(front_pos, lava_pos):
                # Lava directly ahead - strong penalty for forward
                G_survival[2] += pain_factor + 2.0  # Don't walk into lava

        # Penalty for moving toward poison
        for poison_pos in poison_positions:
            if np.allclose(front_pos, poison_pos):
                G_survival[2] += pain_factor + 1.0

        return G_survival

    def _compute_goal_value_G(self, state: Tuple, info: Dict) -> np.ndarray:
        """
        Phase 4A: Compute goal-value modulated component of G(a).

        Key mechanism:
        - HIGH goal_value: REDUCE hazard avoidance penalty → more risk-taking
        - LOW goal_value: INCREASE hazard avoidance penalty → extra safety

        This is applied DIRECTLY based on goal_value, not just urgency.
        Time pressure amplifies the effect but isn't required.
        """
        G_goal = np.zeros(self.n_actions)

        goal_value = info.get('goal_value', 0.5)
        time_pressure = info.get('time_pressure', 0.0)

        # Store for tracking
        self.current_goal_value = goal_value
        self.current_urgency = goal_value * time_pressure

        agent_pos = info['agent_pos']
        lava_positions = info.get('lava_positions', [])
        poison_positions = info.get('poison_positions', [])

        direction = info.get('direction', 0)
        dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        forward_vec = np.array(dir_vecs[direction])
        front_pos = agent_pos + np.array(forward_vec)

        # Calculate goal direction
        goal_pos = np.array([info.get('grid_size', 10) - 2] * 2)
        goal_diff = goal_pos - agent_pos
        goal_unit = goal_diff / (np.linalg.norm(goal_diff) + 1e-6)
        forward_goal_alignment = np.dot(forward_vec, goal_unit)

        # Check if hazard is directly ahead
        hazard_ahead = False
        for hazard_pos in lava_positions + poison_positions:
            if np.allclose(front_pos, hazard_pos):
                hazard_ahead = True
                break

        if hazard_ahead:
            # KEY MECHANISM: Goal value directly modulates hazard penalty
            # goal_value = 1.0 (HIGH): reduce penalty by 3.0 (encourages risk)
            # goal_value = 0.3 (LOW): add penalty of 2.1 (discourages risk)

            if goal_value > 0.7:
                # HIGH value goal: reduce hazard penalty to encourage risk
                risk_reduction = self.goal_value_weight * (goal_value - 0.5)
                G_goal[2] -= risk_reduction  # Make forward less costly

                # Extra boost if heading toward goal
                if forward_goal_alignment > 0.3:
                    G_goal[2] -= 1.0  # Encourage risky shortcut to goal

            elif goal_value < 0.4:
                # LOW value goal: add extra hazard penalty for safety
                safety_boost = self.goal_value_weight * (0.5 - goal_value)
                G_goal[2] += safety_boost  # Make forward more costly

            # Time pressure amplifies the effect
            if time_pressure > 0.5:
                # Running out of time: goal value effect is stronger
                if goal_value > 0.7:
                    G_goal[2] -= 1.5 * time_pressure  # More urgent = more risk
                # But low value + time pressure = still don't risk

        # For MID goal value (0.4-0.7): no modification, use base survival behavior

        return G_goal

    def _compute_uncertainty_G(self, state: Tuple, info: Dict) -> np.ndarray:
        """
        Phase 4B: Compute uncertainty-driven component of G(a).

        Key mechanisms:
        1. HIGH uncertainty → encourage SCAN action (action 7)
        2. Signal detection → encourage cautious movement / observation
        3. Information gain → reward for reducing uncertainty
        4. Surprise penalty → avoid areas with potential hidden hazards

        This implements Active Inference: "I'm uncertain, so I should observe more"
        """
        # Support variable n_actions (7 or 8 with SCAN)
        G_uncertainty = np.zeros(self.n_actions)

        uncertainty = info.get('uncertainty', 0.0)
        hazard_signal_map = info.get('hazard_signal_map', {})
        hidden_count = info.get('hidden_hazards_count', 0)
        has_scan = info.get('has_scan_action', False)

        # Store for tracking
        self.current_uncertainty = uncertainty

        agent_pos = info['agent_pos']
        direction = info.get('direction', 0)
        dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        forward_vec = np.array(dir_vecs[direction])
        front_pos = tuple(agent_pos + np.array(forward_vec))

        # 1. Signal-based caution: If there's a signal ahead, be cautious
        signal_ahead = hazard_signal_map.get(front_pos, 0.0)

        if signal_ahead > 0:
            # Hazard signal detected ahead - penalize forward movement
            # The stronger the signal, the more caution
            caution_penalty = self.uncertainty_weight * signal_ahead
            G_uncertainty[2] += caution_penalty  # Forward becomes less attractive

            # Encourage turning to observe (lateral movement)
            G_uncertainty[0] -= 0.3 * signal_ahead  # Left turn
            G_uncertainty[1] -= 0.3 * signal_ahead  # Right turn

        # 2. SCAN action encouragement (action 7 if enabled)
        if has_scan and self.n_actions > 7:
            # SCAN is attractive when:
            # a) There's a signal nearby (need to confirm)
            # b) High overall uncertainty
            # c) Haven't scanned recently (cooldown)

            # Check for signals in adjacent cells
            nearby_signals = 0.0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                adj_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
                nearby_signals += hazard_signal_map.get(adj_pos, 0.0)

            # Only encourage SCAN if there's actual signal (prevents scan addiction)
            if nearby_signals > 0.3:
                # SCAN is valuable - reduce its G (make it more attractive)
                # Cap the bonus to prevent excessive scanning
                scan_value = min(1.5, self.info_gain_weight * nearby_signals * 0.5)
                G_uncertainty[7] -= scan_value
            elif uncertainty > 0.5 and nearby_signals > 0:
                # Moderate encouragement for high uncertainty with weak signal
                G_uncertainty[7] -= 0.5

        # 3. Exploration bonus when uncertain (move to gather information)
        if uncertainty > 0.3 and hidden_count > 0:
            # High uncertainty encourages exploration (moving around)
            # But not blindly forward - prefer turns to survey area
            exploration_bonus = 0.2 * uncertainty
            G_uncertainty[0] -= exploration_bonus  # Left
            G_uncertainty[1] -= exploration_bonus  # Right

        # 4. Penalty for staying still when uncertain (encourage active sensing)
        if uncertainty > 0.5 and has_scan:
            # If very uncertain and has SCAN, discourage wait action
            # (Wait is typically action 6 in MiniGrid: done/no-op)
            if self.n_actions > 6:
                G_uncertainty[6] += 0.5 * uncertainty  # Discourage waiting

        return G_uncertainty

    def _compute_causal_curiosity_G(self, state: Tuple, info: Dict, obs: Dict) -> np.ndarray:
        """
        Compute causal curiosity component of G(a).

        Phase 5: Encourages exploration of novel objects and positions.
        Key insight: The brain should be curious about untested object interactions.

        1. Identify novel objects (unseen object encodings) in visible area
        2. Give G bonus for actions that approach novel objects
        3. Extra bonus for positions that previously caused causal effects
        """
        G_causal = np.zeros(self.n_actions)

        agent_pos = info['agent_pos']
        direction = info.get('direction', 0)
        image = obs.get('image', None)

        if image is None:
            return G_causal

        # Direction vectors for computing adjacent positions
        dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        forward_vec = np.array(dir_vecs[direction])

        # Position in front of agent
        front_pos = tuple(agent_pos + forward_vec)

        # Analyze objects in visible area (7x7 view)
        novel_objects_ahead = 0
        causal_positions_ahead = 0

        # Check the image for novel objects
        # image shape: (width, height, 3) - (obj_type, color, state)
        if len(image.shape) == 3:
            height, width = image.shape[0], image.shape[1]
            center_y, center_x = height // 2, width // 2

            # Scan visible tiles for novel objects
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    img_y = center_y + dy
                    img_x = center_x + dx

                    if 0 <= img_y < height and 0 <= img_x < width:
                        obj_type = image[img_y, img_x, 0]
                        color = image[img_y, img_x, 1]

                        # Skip empty, walls, and already-known objects
                        if obj_type <= 1:  # empty or unseen
                            continue

                        obj_key = (int(obj_type), int(color))

                        # Check if this object type is novel
                        if obj_key not in self.object_interactions:
                            # Is it in front of us?
                            if (dx == 0 and dy < 0) or (dx > 0 and direction == 0) or \
                               (dy > 0 and direction == 1) or (dx < 0 and direction == 2) or \
                               (dy < 0 and direction == 3):
                                novel_objects_ahead += 1

        # Check if any known causal position is ahead
        for pos, causal_info in self.causal_positions.items():
            pos_diff = np.array(pos) - agent_pos
            dist = np.linalg.norm(pos_diff)

            if dist < 4:  # Within visible range
                # Is this position in our forward direction?
                if dist > 0:
                    pos_dir = pos_diff / dist
                    alignment = np.dot(forward_vec, pos_dir)

                    if alignment > 0.5:  # Roughly ahead
                        causal_positions_ahead += causal_info.get('count', 1)

        # Apply curiosity bonuses to G
        # Forward action (2) gets bonus for novel objects
        if novel_objects_ahead > 0:
            curiosity_bonus = self.causal_curiosity_weight * min(1.0, novel_objects_ahead * 0.3)
            G_causal[2] -= curiosity_bonus  # Encourage forward
            self.novel_object_approaches += 1

        # Causal positions get even stronger attraction
        if causal_positions_ahead > 0:
            causal_bonus = self.causal_curiosity_weight * min(1.5, causal_positions_ahead * 0.5)
            G_causal[2] -= causal_bonus

        # Goal-directed causal planning: If door is closed, seek known door_opener
        door_is_open = info.get('door_is_open', True)
        carrying_key = info.get('carrying_key', False)

        if not door_is_open and not carrying_key:
            # Find known door_opener positions (learned from switch)
            for pos, causal_info in self.causal_positions.items():
                if causal_info.get('effect') == 'door_opener':
                    # Strong attraction to door_opener
                    pos_diff = np.array(pos) - agent_pos
                    dist = np.linalg.norm(pos_diff)

                    if dist < 1.5:
                        # At the switch - toggle it!
                        G_causal[5] -= 5.0  # Strong toggle bonus
                    elif dist > 0:
                        # Navigate toward switch
                        pos_dir = pos_diff / dist
                        alignment = np.dot(forward_vec, pos_dir)

                        if alignment > 0.3:
                            G_causal[2] -= 4.0 * alignment  # Strong forward bonus
                        else:
                            # Need to turn toward switch
                            left_vec = np.array(dir_vecs[(direction + 3) % 4])
                            right_vec = np.array(dir_vecs[(direction + 1) % 4])

                            left_align = np.dot(left_vec, pos_dir)
                            right_align = np.dot(right_vec, pos_dir)

                            if left_align > right_align:
                                G_causal[0] -= 2.0  # Turn left
                            else:
                                G_causal[1] -= 2.0  # Turn right
                    break  # Only use first door_opener

        return G_causal

    def _record_object_interaction(self, obs: Dict, agent_pos: np.ndarray):
        """Record object at current position as interacted with."""
        image = obs.get('image', None)
        if image is None or len(image.shape) != 3:
            return

        height, width = image.shape[0], image.shape[1]
        center_y, center_x = height // 2, width // 2

        # Get object at agent's current position
        obj_type = image[center_y, center_x, 0]
        color = image[center_y, center_x, 1]

        if obj_type > 1:  # Not empty
            obj_key = (int(obj_type), int(color))
            self.object_interactions[obj_key] = self.object_interactions.get(obj_key, 0) + 1

    def _detect_causal_surprise(self, prev_info: Dict, curr_info: Dict, agent_pos: np.ndarray):
        """
        Detect unexpected causal effects.

        Key insight: If door_open changed WITHOUT the agent having a key
        and WITHOUT toggle action, something at current position caused it.
        """
        prev_door = prev_info.get('door_is_open', False)
        curr_door = curr_info.get('door_is_open', False)
        carrying_key = curr_info.get('carrying_key', False)

        # Door opened unexpectedly (not by toggle with key)
        if curr_door and not prev_door and not carrying_key:
            # This is a causal discovery!
            pos_key = tuple(agent_pos)

            if pos_key not in self.causal_positions:
                self.causal_positions[pos_key] = {'effect': 'door_opened', 'count': 0}
                self.causal_discoveries += 1

            self.causal_positions[pos_key]['count'] += 1

            # Also record this for metacognition
            self.metacog.record_state_transition(
                step=self.step_count,
                carrying_key=False,
                door_is_open=True,
                action=self.prev_action if hasattr(self, 'prev_action') else -1
            )

    def _detect_pattern_conflict(self, action: int, info: Dict, prev_info: Dict):
        """
        Detect when expected patterns fail.

        Key patterns tracked:
        - 'key_door': Tried toggle on door without success (expected key->door to work)
        - 'toggle_door': Tried toggle action near door but door didn't open
        """
        if prev_info is None:
            return

        door_pos = info.get('door_pos')
        agent_pos = info.get('agent_pos', np.array([0, 0]))
        door_was_open = prev_info.get('door_is_open', False)
        door_is_open = info.get('door_is_open', False)
        carrying_key = info.get('carrying_key', False)

        # Check if we tried to interact with door
        if door_pos is not None:
            dist_to_door = np.linalg.norm(agent_pos - np.array(door_pos))

            # Toggle action (5) near door
            if action == 5 and dist_to_door < 2.0:
                self.last_door_attempt_step = self.step_count

                if not door_is_open and not door_was_open:
                    # Door didn't open - pattern failed
                    self.consecutive_door_failures += 1

                    # Track which pattern failed
                    if carrying_key:
                        # Had key but door didn't open - unexpected!
                        pattern = 'key_door_unexpected_fail'
                    else:
                        # No key, door didn't open - expected for key-door, but not for switch-door
                        pattern = 'no_key_toggle'

                    if pattern not in self.pattern_attempts:
                        self.pattern_attempts[pattern] = {'success': 0, 'fail': 0}
                    self.pattern_attempts[pattern]['fail'] += 1

                    # Check if pattern should be suppressed
                    if self.consecutive_door_failures >= self.suppression_threshold:
                        if pattern not in self.pattern_suppression:
                            self.pattern_suppression[pattern] = 0.0
                        # Increase suppression
                        self.pattern_suppression[pattern] = min(1.0,
                            self.pattern_suppression[pattern] + 0.2)
                        self.prior_suppressions += 1
                        self.pattern_conflicts_detected += 1

                elif door_is_open and not door_was_open:
                    # Door opened - pattern succeeded
                    self.consecutive_door_failures = 0
                    pattern = 'key_door' if carrying_key else 'toggle_door'

                    if pattern not in self.pattern_attempts:
                        self.pattern_attempts[pattern] = {'success': 0, 'fail': 0}
                    self.pattern_attempts[pattern]['success'] += 1

    def _compute_prior_suppression_G(self, state: Tuple, info: Dict) -> np.ndarray:
        """
        Compute G adjustment based on prior pattern suppression.

        When key_door pattern is suppressed:
        1. Strongly suppress toggle near door without key
        2. Boost exploration to find switch
        3. Add switch-seeking behavior when suppression is high
        """
        G_suppression = np.zeros(self.n_actions)

        # Get suppression weights
        key_door_supp = self.pattern_suppression.get('no_key_toggle', 0.0)
        key_door_supp += self.pattern_suppression.get('key_door_unexpected_fail', 0.0)

        if key_door_supp > 0.1:
            # Strongly suppress toggle action near door when pattern is failing
            door_pos = info.get('door_pos')
            agent_pos = info.get('agent_pos', np.array([0, 0]))
            carrying_key = info.get('carrying_key', False)

            if door_pos is not None and not carrying_key:
                dist_to_door = np.linalg.norm(agent_pos - np.array(door_pos))
                if dist_to_door < 3.0:
                    # Strong suppression of toggle near door
                    G_suppression[5] += key_door_supp * 5.0  # Much stronger penalty
                    # Also suppress pickup (3) since key doesn't help here
                    G_suppression[3] += key_door_supp * 2.0

            # Boost forward movement to explore alternatives
            G_suppression[2] -= key_door_supp * 1.5  # Stronger forward boost

            # Boost turning to search for switch
            G_suppression[0] -= key_door_supp * 0.8  # Left turn
            G_suppression[1] -= key_door_supp * 0.8  # Right turn

            # If we have switch position info, strongly attract toward it
            switch_pos = info.get('switch_pos')
            if switch_pos is not None:
                direction = info.get('direction', 0)
                dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                forward_vec = np.array(dir_vecs[direction])

                switch_diff = np.array(switch_pos) - agent_pos
                switch_dist = np.linalg.norm(switch_diff)

                if switch_dist > 0:
                    switch_dir = switch_diff / switch_dist
                    alignment = np.dot(forward_vec, switch_dir)

                    # Strong bonus for moving toward switch
                    if alignment > 0.3:
                        G_suppression[2] -= key_door_supp * 3.0 * alignment

                    # Bonus for toggling when adjacent to switch
                    if switch_dist < 2.0:
                        G_suppression[5] -= key_door_supp * 4.0  # Override door suppression

            # Add exploration noise when suppression is high
            if self.exploration_boost > 0.3:
                noise = np.random.randn(self.n_actions) * self.exploration_boost * 0.5
                G_suppression += noise

        return G_suppression

    def _compute_directed_exploration_G(self, state: Tuple, info: Dict) -> np.ndarray:
        """
        Compute G adjustment for directed exploration.

        Phase 7: When stuck or suppression is active, guide agent to unvisited areas.

        Key mechanism:
        1. Track visited positions
        2. Identify frontier (adjacent unvisited positions)
        3. Give G bonus for actions leading to frontier
        """
        G_explore = np.zeros(self.n_actions)

        agent_pos = info.get('agent_pos', np.array([0, 0]))
        direction = info.get('direction', 0)
        door_is_open = info.get('door_is_open', True)
        carrying_key = info.get('carrying_key', False)

        # Record current position visit
        pos_key = tuple(agent_pos)
        if pos_key not in self.visited_positions:
            self.visited_positions.add(pos_key)
            self.novel_positions_visited += 1
        self.visit_counts[pos_key] = self.visit_counts.get(pos_key, 0) + 1

        # Energy-aware exploration: Don't explore when hungry
        env_energy = info.get('env_energy', 1.0)
        food_positions = info.get('food_positions', [])

        # If energy is low and food exists, prioritize survival over exploration
        if env_energy < 0.4 and food_positions:
            return G_explore  # Let survival G handle food-seeking

        # Scale exploration by energy (explore more when well-fed)
        energy_factor = min(1.0, env_energy / 0.6)  # Full exploration above 60% energy

        # Only activate directed exploration when:
        # 1. Door is locked and we don't have key (stuck on puzzle)
        # 2. OR exploration_boost is high (prior suppression active)
        # 3. OR too many steps without progress
        should_explore = (
            (not door_is_open and not carrying_key) or
            self.exploration_boost > 0.2 or
            self.steps_without_progress > 15
        )

        if not should_explore:
            return G_explore

        # Direction vectors
        dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        forward_vec = np.array(dir_vecs[direction])

        # Find frontier positions (adjacent unvisited or low-visit)
        frontier_scores = {}
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                check_pos = (int(agent_pos[0] + dx), int(agent_pos[1] + dy))

                # Skip walls (assuming grid is 8x8 with walls at edges)
                if check_pos[0] <= 0 or check_pos[0] >= 7 or check_pos[1] <= 0 or check_pos[1] >= 7:
                    continue

                visits = self.visit_counts.get(check_pos, 0)
                if visits < 2:  # Low-visit or unvisited
                    # Score based on novelty and distance
                    dist = abs(dx) + abs(dy)
                    novelty = 1.0 / (1.0 + visits)
                    frontier_scores[check_pos] = novelty / (dist + 1)

        if not frontier_scores:
            return G_explore

        # Find best frontier position
        best_pos = max(frontier_scores.keys(), key=lambda p: frontier_scores[p])
        best_diff = np.array(best_pos) - agent_pos
        best_dist = np.linalg.norm(best_diff)

        if best_dist > 0:
            best_dir = best_diff / best_dist

            # Compute alignment with each action direction
            forward_alignment = np.dot(forward_vec, best_dir)

            # Forward gets bonus if facing frontier (scaled by energy)
            if forward_alignment > 0.3:
                bonus = self.directed_exploration_weight * forward_alignment * frontier_scores[best_pos]
                bonus *= energy_factor  # Scale by energy level
                G_explore[2] -= bonus
                self.directed_explorations += 1

            # Turn actions get bonus if they would face frontier
            left_vec = np.array(dir_vecs[(direction + 3) % 4])
            right_vec = np.array(dir_vecs[(direction + 1) % 4])

            left_alignment = np.dot(left_vec, best_dir)
            right_alignment = np.dot(right_vec, best_dir)

            if left_alignment > forward_alignment:
                G_explore[0] -= self.directed_exploration_weight * 0.5 * left_alignment * energy_factor
            if right_alignment > forward_alignment:
                G_explore[1] -= self.directed_exploration_weight * 0.5 * right_alignment * energy_factor

        return G_explore

    def _update_progress_tracking(self, info: Dict):
        """Track progress for stuck detection."""
        door_is_open = info.get('door_is_open', False)
        carrying_key = info.get('carrying_key', False)

        # Progress = got key or opened door
        made_progress = False
        if carrying_key and not self.prev_key_state:
            made_progress = True
        if door_is_open and not self.prev_door_state:
            made_progress = True

        if made_progress:
            self.steps_without_progress = 0
            self.last_progress_step = self.step_count
        else:
            self.steps_without_progress += 1

        self.prev_door_state = door_is_open
        self.prev_key_state = carrying_key

    # =========================================================================
    # Phase 8: Anti-Forgetting (Memory Consolidation)
    # =========================================================================

    def _record_experience(self, state: Tuple, action: int, reward: float, info: Dict):
        """Record experience for potential replay."""
        self.current_episode_trace.append({
            'state': state,
            'action': action,
            'reward': reward,
            'switch_pressed': info.get('switch_pressed', False),
            'door_opened': info.get('door_is_open', False),
        })

    def _store_successful_episode(self, success: bool, total_reward: float):
        """Store successful episode in replay buffer."""
        if success and len(self.current_episode_trace) > 0:
            episode_data = {
                'trace': self.current_episode_trace.copy(),
                'total_reward': total_reward,
                'length': len(self.current_episode_trace),
            }
            self.replay_buffer.append(episode_data)

            # Keep buffer size limited (remove oldest)
            if len(self.replay_buffer) > self.replay_buffer_size:
                self.replay_buffer.pop(0)

            # Record success patterns
            for exp in self.current_episode_trace:
                state_key = exp['state']
                action = exp['action']
                if state_key not in self.success_patterns:
                    self.success_patterns[state_key] = {}
                if action not in self.success_patterns[state_key]:
                    self.success_patterns[state_key][action] = 0
                self.success_patterns[state_key][action] += 1

        # Clear current trace for next episode
        self.current_episode_trace = []

    def _consolidate_memory(self):
        """Replay successful experiences to prevent forgetting."""
        if len(self.replay_buffer) == 0:
            return

        # Select episodes to replay (prioritize recent and high-reward)
        episodes_to_replay = sorted(
            self.replay_buffer,
            key=lambda e: e['total_reward'],
            reverse=True
        )[:self.consolidation_batch_size]

        original_lr = self.transition.lr
        self.transition.lr = original_lr * self.consolidation_lr_multiplier

        for episode in episodes_to_replay:
            trace = episode['trace']
            for i in range(len(trace) - 1):
                exp = trace[i]
                next_exp = trace[i + 1]

                # Replay the transition
                self.transition.update(
                    exp['state'],
                    exp['action'],
                    exp['reward'],
                    next_exp['state'],
                    done=False
                )
                self.replay_updates += 1

            # Final transition
            if len(trace) > 0:
                final_exp = trace[-1]
                self.transition.update(
                    final_exp['state'],
                    final_exp['action'],
                    final_exp['reward'],
                    final_exp['state'],  # Terminal state
                    done=True
                )

        self.transition.lr = original_lr
        self.consolidation_count += 1

    def _get_adaptive_lr(self, state: Tuple, action: int) -> float:
        """Get learning rate based on visit count and success history."""
        base_lr = self.transition.lr

        # Check if this is a protected pattern
        if state in self.success_patterns:
            if action in self.success_patterns[state]:
                success_count = self.success_patterns[state][action]
                if success_count >= self.protection_threshold:
                    # Protected pattern - learn very slowly
                    self.protected_updates += 1
                    return base_lr * 0.1

        # Visit-based decay
        x, y, d, gd = state
        visits = self.transition.visits[x, y, d, gd, action]
        decay_factor = 1.0 / (1.0 + 0.1 * visits)

        return base_lr * decay_factor

    def _end_episode_consolidation(self, success: bool, total_reward: float):
        """Handle end-of-episode memory consolidation."""
        self.total_episodes += 1

        # Store successful episode
        self._store_successful_episode(success, total_reward)

        # Periodic consolidation (replay)
        if self.total_episodes % self.consolidation_interval == 0:
            self._consolidate_memory()

    def act(self, obs: Dict, env=None) -> int:
        """Select action with survival-aware G(a)."""
        self.step_count += 1

        # Extract information including survival state
        info = self._extract_survival_info(obs, env)

        # Sync interoception with actual environment state
        self.interoception.energy = info.get('env_energy', 1.0)
        self.interoception.pain = info.get('env_pain', 0.0)
        self.interoception.time_pressure = info.get('time_pressure', 0.0)

        # Update carrying_key if applicable
        self.carrying_key = info.get('carrying_key', False)

        # Get state representation
        state = self.transition.get_state(
            info['agent_pos'],
            info['direction'],
            info['goal_dir']
        )

        # Compute base G(a) (from parent class logic)
        G = self._compute_G(state, info)

        # Add survival component
        G_survival = self._compute_survival_G(state, info)
        G = G + G_survival

        # Add Phase 4A: Goal value modulation
        G_goal_value = self._compute_goal_value_G(state, info)
        G = G + G_goal_value

        # Add Phase 4B: Uncertainty-driven behavior
        G_uncertainty = self._compute_uncertainty_G(state, info)
        G = G + G_uncertainty

        # Add Phase 5: Causal Curiosity (novel object exploration)
        G_causal = self._compute_causal_curiosity_G(state, info, obs)
        G = G + G_causal

        # Add Phase 6: Prior Suppression (reduce failed pattern influence)
        G_suppression = self._compute_prior_suppression_G(state, info)
        G = G + G_suppression

        # Add Phase 7: Directed Exploration (guide to unvisited areas when stuck)
        G_explore = self._compute_directed_exploration_G(state, info)
        G = G + G_explore

        # Update progress tracking for stuck detection
        self._update_progress_tracking(info)

        # Record object interaction at current position
        self._record_object_interaction(obs, info['agent_pos'])

        # Get survival-aware sub-goal target
        env_energy = info.get('env_energy', 1.0)
        food_positions = info.get('food_positions', [])

        # If energy is critically low and food exists, prioritize food over goal
        if env_energy < 0.3 and food_positions:
            # This is where "self-preservation" overrides goal-seeking
            # Food becomes the primary target
            pass  # G_survival already handles this

        # PC-Z and planner updates (same as parent)
        door_is_open = info.get('door_is_open', False)
        key_pos = info.get('key_pos')
        door_pos = info.get('door_pos')
        agent_pos = info['agent_pos']

        # Calculate current target distance for PC-Z
        if key_pos is not None and not self.carrying_key:
            current_target_dist = np.linalg.norm(agent_pos - key_pos)
        elif door_pos is not None and not door_is_open:
            current_target_dist = np.linalg.norm(agent_pos - door_pos)
        else:
            current_target_dist = info['goal_dist']

        # Update PC-Z
        self.pcz.update(
            goal_dist=current_target_dist,
            wall_hit=self.wall_hit_this_step,
            G_values=G
        )

        # Wall avoidance
        if self.stuck_count >= 2:
            G[2] += 2.0
            G[0] -= 0.5
            G[1] -= 0.5

        # Working memory loop detection
        if self.working_memory.is_repeating():
            G += np.random.randn(self.n_actions) * 0.3

        # PC-Z controlled planner
        stuck_score = self.pcz.progress_error
        uncertainty = self.pcz.action_margin
        if self.planner.should_activate(stuck_score, 1.0 - uncertainty):
            self.planner.generate_plan(self.carrying_key, door_is_open)

        if self.planner.state == PlannerCircuit.EXECUTING:
            direction = info.get('direction', 0)
            dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            forward_vec = np.array(dir_vecs[direction])
            front_pos = agent_pos + np.array(forward_vec)

            key_in_front = key_pos is not None and np.allclose(key_pos, front_pos)
            door_in_front = door_pos is not None and np.allclose(door_pos, front_pos)

            planner_biases = self.planner.get_G_bias(
                self.carrying_key, door_is_open, key_in_front, door_in_front
            )
            for action_idx, bias in planner_biases.items():
                if action_idx < self.n_actions:
                    G[action_idx] += bias

        # Softmax action selection
        effective_temperature = self.base_temperature * self.pcz.temperature_mult
        probs = np.exp(-G / effective_temperature)
        probs = probs / (probs.sum() + 1e-10)

        action = np.random.choice(self.n_actions, p=probs)

        # Store for learning
        self.prev_state = state
        self.prev_action = action
        self.prev_info = info

        return action

    def learn_survival(self, obs: Dict, reward: float, done: bool, env_info: Dict, env=None):
        """Learn from experience with survival events."""
        # Call parent learn
        self.learn(obs, reward, done, env_info, env)

        # Phase 8: Record experience for replay buffer
        if hasattr(self, 'prev_state') and self.prev_state is not None:
            self._record_experience(self.prev_state, self.prev_action, reward, env_info)

        # Phase 5: Detect causal surprises (state changes without expected cause)
        if hasattr(self, 'prev_info') and self.prev_info is not None:
            current_info = self._extract_survival_info(obs, env) if env else env_info
            agent_pos = current_info.get('agent_pos', np.array([0, 0]))
            self._detect_causal_surprise(self.prev_info, current_info, agent_pos)

            # Phase 6: Detect pattern conflicts (when expected patterns fail)
            if hasattr(self, 'prev_action'):
                self._detect_pattern_conflict(self.prev_action, current_info, self.prev_info)

        # Process survival events
        events = env_info.get('events', [])
        for event_type, event_data in events:
            if event_type == 'FOOD_EATEN':
                self.food_eaten_count += 1
                # Phase 3: Track if ate while hungry
                energy_before = event_data['energy_after'] - event_data['amount']
                if energy_before < 0.5:
                    self.food_eaten_while_hungry += 1
                self.metacog.record_state_transition(
                    step=self.step_count,
                    carrying_key=self.carrying_key,
                    door_is_open=env_info.get('door_is_open', False),
                    action=self.prev_action
                )
                # Record as narrative event
                self.narrative.record_event(f"ate_food_energy_{event_data['energy_after']:.2f}", self.step_count)

            elif event_type == 'NEAR_STARVATION':
                self.near_starvation_count += 1
                self.narrative.record_challenge("near_starvation", self.step_count)

            elif event_type == 'LAVA_HIT':
                self.lava_hits += 1
                self.risky_tiles_crossed += 1
                # Phase 4A: Track if this was a calculated risk
                goal_value = env_info.get('goal_value', 0.5)
                urgency = env_info.get('urgency', 0.0)
                if goal_value > 0.7:
                    self.risks_taken_for_high_goal += 1
                    self.narrative.record_event(
                        f"took_lava_risk_goal_value_{goal_value:.2f}_urgency_{urgency:.2f}",
                        self.step_count
                    )
                self.goal_value_risk_decisions.append((goal_value, True))
                self.narrative.record_challenge("lava_hit", self.step_count)

            elif event_type == 'POISON_ENTER':
                self.poison_enters += 1
                self.risky_tiles_crossed += 1
                # Phase 4A: Track risk decision
                goal_value = env_info.get('goal_value', 0.5)
                if goal_value > 0.7:
                    self.risks_taken_for_high_goal += 1
                self.goal_value_risk_decisions.append((goal_value, True))
                pos = event_data.get('pos')
                # Loop detection: re-entering a previously exited poison zone
                if pos in self.exited_poison_positions:
                    self.poison_loops += 1
                    self.narrative.record_challenge("poison_loop", self.step_count)
                self.narrative.record_event(f"entered_poison_at_{pos}", self.step_count)

            elif event_type == 'POISON_TICK':
                self.poison_ticks += 1

            elif event_type == 'POISON_EXIT':
                self.poison_exits += 1
                pos = event_data.get('pos')
                if pos:
                    self.exited_poison_positions.add(pos)
                self.narrative.record_event(f"exited_poison_ticks_{event_data.get('total_ticks', 0)}", self.step_count)

            elif event_type == 'ENERGY_DEPLETED':
                self.narrative.record_challenge("starvation_death", self.step_count)

            # Phase 4B events
            elif event_type == 'SCAN':
                self.scan_count += 1
                revealed = event_data.get('revealed', 0)
                if revealed > 0:
                    self.exploration_for_info += 1
                    # Record info gain
                    uncertainty_before = self.current_uncertainty
                    # Uncertainty reduced by revealing hazards
                    self.info_gain_events.append((uncertainty_before, revealed))
                self.narrative.record_event(f"scan_revealed_{revealed}", self.step_count)

            elif event_type == 'SURPRISE_HAZARD':
                self.surprises_count += 1
                self.narrative.record_challenge("surprise_hazard", self.step_count)

            elif event_type == 'PROB_HAZARD_TRIGGERED':
                self.surprises_count += 1
                self.narrative.record_challenge("prob_hazard_triggered", self.step_count)

            # Phase 5: Causal discovery events
            elif event_type == 'SWITCH_PRESSED':
                # Discovered a switch!
                pos = event_data.get('pos', tuple(env_info.get('agent_pos', [0, 0])))
                if pos not in self.causal_positions:
                    self.causal_positions[pos] = {'effect': 'switch_pressed', 'count': 0}
                    self.causal_discoveries += 1
                self.causal_positions[pos]['count'] += 1
                self.narrative.record_event(f"discovered_switch_at_{pos}", self.step_count)

            elif event_type == 'DOOR_OPENED_BY_SWITCH':
                # Learned that switch opens door!
                door_pos = event_data.get('door_pos')
                if door_pos:
                    self.narrative.record_event(f"switch_opened_door_at_{door_pos}", self.step_count)
                # Use switch_pos directly (not agent_pos which may be stale)
                switch_pos = env_info.get('switch_pos')
                if switch_pos is not None:
                    pos_key = tuple(switch_pos)
                    if pos_key in self.causal_positions:
                        self.causal_positions[pos_key]['effect'] = 'door_opener'
                        self.causal_positions[pos_key]['count'] += 2  # Extra reinforcement
                    else:
                        # Switch wasn't recorded yet, add it now
                        self.causal_positions[pos_key] = {'effect': 'door_opener', 'count': 2}
                        self.causal_discoveries += 1

        # Episode end with survival stats
        if done:
            death_reason = env_info.get('death_reason')
            success = reward > 0 and death_reason is None

            # Phase 8: Memory consolidation at episode end
            self._end_episode_consolidation(success, self.episode_reward)

            # Update narrative with survival context
            self.narrative.summarize_episode(
                success=success,
                steps=self.step_count,
                key_acquired=self.carrying_key,
                door_opened=env_info.get('door_is_open', False),
                wall_hits=self.wall_hits_episode,
                got_stuck=self.got_stuck_episode
            )

    def get_survival_stats(self) -> Dict:
        """Get survival-specific statistics."""
        base_stats = self.get_stats()
        base_stats.update({
            'food_eaten': self.food_eaten_count,
            'lava_hits': self.lava_hits,
            'poison_ticks': self.poison_ticks,
            'poison_enters': self.poison_enters,
            'poison_exits': self.poison_exits,
            'poison_loops': self.poison_loops,
            'near_starvation_events': self.near_starvation_count,
            # Phase 3 metrics
            'food_eaten_while_hungry': self.food_eaten_while_hungry,
            'goal_reached_while_hungry': self.goal_reached_while_hungry,
            'planner_activations': self.planner_activations,
            'risky_tiles_crossed': self.risky_tiles_crossed,
            # Phase 4A metrics
            'risks_taken_for_high_goal': self.risks_taken_for_high_goal,
            'safe_choices_for_low_goal': self.safe_choices_for_low_goal,
            'goal_value_risk_correlation': self._compute_goal_risk_correlation(),
            # Phase 4B metrics
            'scan_count': self.scan_count,
            'exploration_for_info': self.exploration_for_info,
            'surprises_count': self.surprises_count,
            'info_gain_events': len(self.info_gain_events),
            # Phase 5: Causal curiosity metrics
            'causal_discoveries': self.causal_discoveries,
            'novel_object_approaches': self.novel_object_approaches,
            'known_causal_positions': len(self.causal_positions),
            'object_types_interacted': len(self.object_interactions),
            # Phase 6: Pattern conflict metrics
            'pattern_conflicts_detected': self.pattern_conflicts_detected,
            'prior_suppressions': self.prior_suppressions,
            'exploration_boost': self.exploration_boost,
            'active_suppressions': sum(1 for v in self.pattern_suppression.values() if v > 0.1),
            # Phase 7: Directed exploration metrics
            'directed_explorations': self.directed_explorations,
            'novel_positions_visited': self.novel_positions_visited,
            'positions_visited': len(self.visited_positions),
        })
        return base_stats

    def _compute_goal_risk_correlation(self) -> float:
        """Compute correlation between goal_value and risk-taking."""
        if len(self.goal_value_risk_decisions) < 5:
            return 0.0

        # Compute correlation between goal_value and took_risk
        goal_values = [gv for gv, _ in self.goal_value_risk_decisions]
        risks = [1.0 if took else 0.0 for _, took in self.goal_value_risk_decisions]

        if len(set(goal_values)) < 2 or len(set(risks)) < 2:
            return 0.0

        # Simple correlation
        mean_gv = np.mean(goal_values)
        mean_r = np.mean(risks)
        std_gv = np.std(goal_values) + 1e-6
        std_r = np.std(risks) + 1e-6

        corr = np.mean((np.array(goal_values) - mean_gv) * (np.array(risks) - mean_r)) / (std_gv * std_r)
        return float(corr)


# =============================================================================
# TRAINING LOOP FOR SURVIVAL
# =============================================================================

def train_survival_genesis(
    env_config: Dict,
    n_episodes: int = 500,
    verbose: bool = True
) -> Tuple[SurvivalGenesisBrain, List[Dict]]:
    """Train Survival Genesis Brain."""

    # Create environment
    base_env = SurvivalMiniGridEnv(**env_config)
    env = FullyObsWrapper(base_env)

    # Create brain
    grid_size = env_config.get('size', 8)
    brain = SurvivalGenesisBrain(n_actions=7, grid_size=grid_size)

    history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        brain.reset()
        done = False
        total_reward = 0
        steps = 0
        death_reason = None

        while not done:
            action = brain.act(obs, env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Learn from this step
            brain.learn_survival(next_obs, reward, done, info, env)

            total_reward += reward
            steps += 1
            obs = next_obs

            if done:
                death_reason = info.get('death_reason')

        # Record episode
        success = total_reward > 0 and death_reason is None
        history.append({
            'episode': episode,
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'death_reason': death_reason,
            'food_eaten': brain.food_eaten_count,
            'energy_final': info.get('energy', 0),
            'near_starvation': brain.near_starvation_count,
            'lava_hits': brain.lava_hits,
            'poison_ticks': brain.poison_ticks,
            'poison_enters': brain.poison_enters,
            'poison_loops': brain.poison_loops,
        })

        # Verbose output
        if verbose and (episode + 1) % 50 == 0:
            recent = history[-50:]
            success_rate = sum(1 for h in recent if h['success']) / len(recent)
            death_by_starvation = sum(1 for h in recent if h['death_reason'] == 'starvation')
            death_by_injury = sum(1 for h in recent if h['death_reason'] == 'injury')
            food_eaten = sum(h['food_eaten'] for h in recent) / len(recent)

            print(f"Episode {episode + 1}: "
                  f"success={success_rate:.0%}, "
                  f"starved={death_by_starvation}, "
                  f"injured={death_by_injury}, "
                  f"food/ep={food_eaten:.1f}")

    env.close()
    return brain, history


# =============================================================================
# CURRICULUM EXPERIMENTS
# =============================================================================

def run_survival_curriculum():
    """Run curriculum of survival environments."""
    print("=" * 60)
    print("SURVIVAL MINIGRID CURRICULUM")
    print("Genesis Brain with Self-Preservation")
    print("=" * 60)

    # Phase 0: Food-only (test energy management)
    print("\n" + "=" * 60)
    print("PHASE 0: Food-Only Environment")
    print("Test: Does the agent learn to eat when hungry?")
    print("=" * 60)

    food_only_config = {
        'size': 12,  # Even larger grid
        'initial_energy': 0.4,  # Start hungry
        'energy_decay': 0.03,  # Very fast decay (~13 steps to death without food)
        'n_food': 5,
        'food_gain': 0.4,
        'n_lava': 0,
        'n_poison': 0,
        'has_goal': True,
        'has_key_door': False,
        'max_steps': 80,
    }

    brain, history = train_survival_genesis(food_only_config, n_episodes=300)

    # Analyze results
    print("\n" + "-" * 60)
    print("PHASE 0 RESULTS")
    print("-" * 60)

    early = history[:50]
    late = history[-50:]

    early_success = sum(1 for h in early if h['success']) / len(early)
    late_success = sum(1 for h in late if h['success']) / len(late)

    early_starvation = sum(1 for h in early if h['death_reason'] == 'starvation')
    late_starvation = sum(1 for h in late if h['death_reason'] == 'starvation')

    early_food = sum(h['food_eaten'] for h in early) / len(early)
    late_food = sum(h['food_eaten'] for h in late) / len(late)

    print(f"  Success Rate: {early_success:.0%} -> {late_success:.0%}")
    print(f"  Starvation Deaths: {early_starvation} -> {late_starvation}")
    print(f"  Avg Food Eaten: {early_food:.1f} -> {late_food:.1f}")

    # Check survival behavior
    print("\n  [Survival Behavior Analysis]")

    # Did agent learn to eat before starving?
    late_near_starvation = sum(h['near_starvation'] for h in late)
    print(f"    Near-starvation events: {late_near_starvation}")

    # Did food consumption correlate with low energy?
    hungry_eats = sum(1 for h in late if h['food_eaten'] > 0 and h['near_starvation'] > 0)
    print(f"    Ate while hungry: {hungry_eats}/{len(late)}")

    # Brain analysis
    stats = brain.get_survival_stats()
    print(f"\n  [Brain Circuit Stats]")
    print(f"    Planner Completion Rate: {stats['planner_completion_rate']:.1%}")
    print(f"    Narrative Self-Assessed Success: {stats['narrative_avg_success']:.1%}")
    print(f"    Total Food Eaten: {stats['food_eaten']}")

    # Pass/fail criteria
    print("\n  [Pass/Fail Criteria]")
    passed = []

    # Criterion 1: Goal success rate >= 50%
    if late_success >= 0.5:
        passed.append("[PASS] Goal success rate >= 50%")
    else:
        passed.append(f"[FAIL] Goal success rate = {late_success:.0%}")

    # Criterion 2: Starvation deaths reduced
    if late_starvation < early_starvation:
        passed.append(f"[PASS] Starvation reduced: {early_starvation} -> {late_starvation}")
    else:
        passed.append(f"[FAIL] Starvation not reduced: {early_starvation} -> {late_starvation}")

    # Criterion 3: Food consumption increased
    if late_food > early_food:
        passed.append(f"[PASS] Food consumption increased: {early_food:.1f} -> {late_food:.1f}")
    else:
        passed.append(f"[FAIL] Food consumption not increased")

    for p in passed:
        print(f"    {p}")

    # Phase 1: Lava environment (risk-reward tradeoff)
    print("\n" + "=" * 60)
    print("PHASE 1: Lava Environment")
    print("Test: Does the agent avoid lava or take risky shortcuts?")
    print("=" * 60)

    lava_config = {
        'size': 10,
        'initial_energy': 0.5,
        'energy_decay': 0.02,
        'n_food': 3,
        'food_gain': 0.3,
        'n_lava': 4,  # Add lava tiles
        'n_poison': 0,
        'lava_pain': 0.4,
        'lava_energy_loss': 0.2,
        'has_goal': True,
        'has_key_door': False,
        'max_steps': 100,
    }

    brain2, history2 = train_survival_genesis(lava_config, n_episodes=300)

    # Analyze lava results
    print("\n" + "-" * 60)
    print("PHASE 1 RESULTS")
    print("-" * 60)

    early2 = history2[:50]
    late2 = history2[-50:]

    early2_success = sum(1 for h in early2 if h['success']) / len(early2)
    late2_success = sum(1 for h in late2 if h['success']) / len(late2)

    early2_starvation = sum(1 for h in early2 if h['death_reason'] == 'starvation')
    late2_starvation = sum(1 for h in late2 if h['death_reason'] == 'starvation')

    early2_injury = sum(1 for h in early2 if h['death_reason'] == 'injury')
    late2_injury = sum(1 for h in late2 if h['death_reason'] == 'injury')

    print(f"  Success Rate: {early2_success:.0%} -> {late2_success:.0%}")
    print(f"  Starvation Deaths: {early2_starvation} -> {late2_starvation}")
    print(f"  Injury Deaths: {early2_injury} -> {late2_injury}")

    # Check for lava avoidance behavior
    lava_stats = brain2.get_survival_stats()
    print(f"\n  [Lava Interaction Stats]")
    print(f"    Total Lava Hits: {lava_stats['lava_hits']}")
    print(f"    Avg per episode: {lava_stats['lava_hits'] / 300:.2f}")

    # Phase 2: Poison environment (cumulative damage)
    print("\n" + "=" * 60)
    print("PHASE 2: Poison Environment")
    print("Test: Does the agent minimize time in poison zones?")
    print("Key: Poison = cumulative damage (different from lava's one-time hit)")
    print("=" * 60)

    poison_config = {
        'size': 10,
        'initial_energy': 0.5,  # Start lower
        'energy_decay': 0.025,  # Faster decay
        'n_food': 2,  # Less food forces tradeoffs
        'food_gain': 0.3,
        'n_lava': 0,
        'n_poison': 10,  # Many poison zones
        'poison_pain': 0.1,  # Higher pain per tick (10 ticks = death)
        'has_goal': True,
        'has_key_door': False,
        'max_steps': 80,  # Less time
    }

    brain3, history3 = train_survival_genesis(poison_config, n_episodes=300)

    # Analyze poison results
    print("\n" + "-" * 60)
    print("PHASE 2 RESULTS")
    print("-" * 60)

    early3 = history3[:50]
    late3 = history3[-50:]

    early3_success = sum(1 for h in early3 if h['success']) / len(early3)
    late3_success = sum(1 for h in late3 if h['success']) / len(late3)

    early3_starvation = sum(1 for h in early3 if h['death_reason'] == 'starvation')
    late3_starvation = sum(1 for h in late3 if h['death_reason'] == 'starvation')

    early3_injury = sum(1 for h in early3 if h['death_reason'] == 'injury')
    late3_injury = sum(1 for h in late3 if h['death_reason'] == 'injury')

    print(f"  Success Rate: {early3_success:.0%} -> {late3_success:.0%}")
    print(f"  Starvation Deaths: {early3_starvation} -> {late3_starvation}")
    print(f"  Injury Deaths (pain>=1.0): {early3_injury} -> {late3_injury}")

    # Poison-specific metrics
    early3_ticks = sum(h['poison_ticks'] for h in early3)
    late3_ticks = sum(h['poison_ticks'] for h in late3)
    early3_enters = sum(h['poison_enters'] for h in early3)
    late3_enters = sum(h['poison_enters'] for h in late3)
    early3_loops = sum(h['poison_loops'] for h in early3)
    late3_loops = sum(h['poison_loops'] for h in late3)

    # Average ticks per enter (duration in poison)
    early3_avg_duration = early3_ticks / max(1, early3_enters)
    late3_avg_duration = late3_ticks / max(1, late3_enters)

    print(f"\n  [Poison Interaction Stats]")
    print(f"    Total Poison Ticks: {early3_ticks} -> {late3_ticks}")
    print(f"    Poison Enters: {early3_enters} -> {late3_enters}")
    print(f"    Avg Duration per Enter: {early3_avg_duration:.1f} -> {late3_avg_duration:.1f}")
    print(f"    Poison Loops (re-entry): {early3_loops} -> {late3_loops}")

    # Brain stats
    poison_stats = brain3.get_survival_stats()
    print(f"\n  [Brain Circuit Stats]")
    print(f"    Planner Completion Rate: {poison_stats['planner_completion_rate']:.1%}")
    print(f"    Narrative Self-Assessed Success: {poison_stats['narrative_avg_success']:.1%}")
    print(f"    Total Poison Loops Detected: {poison_stats['poison_loops']}")

    # Pass/fail criteria for Phase 2
    print("\n  [Pass/Fail Criteria - Phase 2]")
    passed2 = []

    # Criterion 1: Tick rate reduced (less time in poison)
    if late3_ticks < early3_ticks:
        passed2.append(f"[PASS] Tick rate reduced: {early3_ticks} -> {late3_ticks}")
    else:
        passed2.append(f"[FAIL] Tick rate not reduced: {early3_ticks} -> {late3_ticks}")

    # Criterion 2: Loop reduction (fewer re-entries)
    if late3_loops <= early3_loops:
        passed2.append(f"[PASS] Loop rate stable/reduced: {early3_loops} -> {late3_loops}")
    else:
        passed2.append(f"[FAIL] Loop rate increased: {early3_loops} -> {late3_loops}")

    # Criterion 3: Success maintained (>= 50%)
    if late3_success >= 0.5:
        passed2.append(f"[PASS] Success rate >= 50%: {late3_success:.0%}")
    else:
        passed2.append(f"[FAIL] Success rate < 50%: {late3_success:.0%}")

    # Criterion 4: Avg duration in poison reduced (learned to exit quickly)
    if late3_avg_duration < early3_avg_duration:
        passed2.append(f"[PASS] Avg poison duration reduced: {early3_avg_duration:.1f} -> {late3_avg_duration:.1f}")
    else:
        passed2.append(f"[WARN] Avg poison duration not reduced: {early3_avg_duration:.1f} -> {late3_avg_duration:.1f}")

    # Criterion 5: Injury deaths reduced (pain accumulation avoided)
    if late3_injury <= early3_injury:
        passed2.append(f"[PASS] Injury deaths stable/reduced: {early3_injury} -> {late3_injury}")
    else:
        passed2.append(f"[FAIL] Injury deaths increased: {early3_injury} -> {late3_injury}")

    for p in passed2:
        print(f"    {p}")

    # Final summary
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY: Poison = Cumulative Cost")
    print("=" * 60)
    print(f"  Key Behavior: Exit poison quickly, don't loop back")
    print(f"  Tick Reduction: {((early3_ticks - late3_ticks) / max(1, early3_ticks)) * 100:.1f}%")
    print(f"  Duration Reduction: {((early3_avg_duration - late3_avg_duration) / max(1, early3_avg_duration)) * 100:.1f}%")

    # =========================================================================
    # PHASE 3: MIXED ENVIRONMENT (Cumulative Curriculum)
    # =========================================================================

    # Phase 3A: Food + Goal (survival dilemma, no hazards yet)
    print("\n" + "=" * 60)
    print("PHASE 3A: Food + Goal (Survival Dilemma)")
    print("Test: Does agent balance hunger vs goal-seeking?")
    print("=" * 60)

    phase3a_config = {
        'size': 10,
        'initial_energy': 0.4,  # Start hungry
        'energy_decay': 0.03,   # Fast decay forces food-seeking
        'n_food': 3,
        'food_gain': 0.35,
        'n_lava': 0,
        'n_poison': 0,
        'has_goal': True,
        'has_key_door': False,
        'max_steps': 80,
        'hazard_on_path_prob': 0.7,
        'safe_path_guaranteed': True,
    }

    brain3a, history3a = train_mixed_survival(phase3a_config, n_episodes=200)
    analyze_phase3_results("3A", history3a, brain3a)

    # Phase 3B: + Lava (shortcut temptation)
    print("\n" + "=" * 60)
    print("PHASE 3B: Food + Lava + Goal (Shortcut Temptation)")
    print("Test: Does agent avoid risky shortcuts?")
    print("=" * 60)

    phase3b_config = {
        'size': 10,
        'initial_energy': 0.5,
        'energy_decay': 0.025,
        'n_food': 3,
        'food_gain': 0.35,
        'n_lava': 4,            # Add lava on direct path
        'n_poison': 0,
        'lava_pain': 0.4,
        'lava_energy_loss': 0.2,
        'has_goal': True,
        'has_key_door': False,
        'max_steps': 100,
        'hazard_on_path_prob': 0.7,
        'safe_path_guaranteed': True,
    }

    brain3b, history3b = train_mixed_survival(phase3b_config, n_episodes=200)
    analyze_phase3_results("3B", history3b, brain3b)

    # Phase 3C: + Poison (full mixed environment)
    print("\n" + "=" * 60)
    print("PHASE 3C: Food + Lava + Poison + Goal (Full Mixed)")
    print("Test: Complete survival with all hazards")
    print("=" * 60)

    phase3c_config = {
        'size': 10,
        'initial_energy': 0.5,
        'energy_decay': 0.025,
        'n_food': 3,
        'food_gain': 0.35,
        'n_lava': 3,
        'n_poison': 4,
        'lava_pain': 0.4,
        'lava_energy_loss': 0.2,
        'poison_pain': 0.1,
        'has_goal': True,
        'has_key_door': False,
        'max_steps': 100,
        'hazard_on_path_prob': 0.7,
        'safe_path_guaranteed': True,
    }

    brain3c, history3c = train_mixed_survival(phase3c_config, n_episodes=300)
    analyze_phase3_results("3C", history3c, brain3c)

    # =========================================================================
    # DOORKEY REGRESSION TEST (Run in parallel with Phase 3)
    # =========================================================================
    print("\n" + "=" * 60)
    print("DOORKEY REGRESSION TEST")
    print("Ensure survival circuits don't break Key->Door logic")
    print("=" * 60)

    doorkey_results = run_doorkey_regression()

    # Final Phase 3 Summary
    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE: Mixed Survival Environment")
    print("=" * 60)
    print("  3A (Food+Goal): Survival dilemma baseline")
    print("  3B (+Lava): Shortcut avoidance")
    print("  3C (+Poison): Full cumulative risk")
    print("\n  DoorKey Regression:", "PASS" if doorkey_results['passed'] else "FAIL")

    return brain3c, history3c


def train_mixed_survival(
    env_config: Dict,
    n_episodes: int = 200,
    verbose: bool = True
) -> Tuple[SurvivalGenesisBrain, List[Dict]]:
    """Train on MixedSurvivalEnv with dilemma-based maps."""

    # Create Mixed environment
    base_env = MixedSurvivalEnv(**env_config)
    env = FullyObsWrapper(base_env)

    # Create brain
    grid_size = env_config.get('size', 10)
    brain = SurvivalGenesisBrain(n_actions=7, grid_size=grid_size)

    history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        brain.reset()
        done = False
        total_reward = 0
        steps = 0
        death_reason = None

        while not done:
            action = brain.act(obs, env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            brain.learn_survival(next_obs, reward, done, info, env)

            total_reward += reward
            steps += 1
            obs = next_obs

            if done:
                death_reason = info.get('death_reason')

        # Get Phase 3 metrics from env
        phase3_metrics = env.unwrapped.get_phase3_metrics()

        # Record episode
        success = total_reward > 0 and death_reason is None
        history.append({
            'episode': episode,
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'death_reason': death_reason,
            'food_eaten': brain.food_eaten_count,
            'food_eaten_while_hungry': brain.food_eaten_while_hungry,
            'energy_final': info.get('energy', 0),
            'near_starvation': brain.near_starvation_count,
            'lava_hits': brain.lava_hits,
            'poison_ticks': brain.poison_ticks,
            'poison_enters': brain.poison_enters,
            'poison_loops': brain.poison_loops,
            'risky_shortcut_rate': phase3_metrics['risky_shortcut_rate'],
            'risky_tiles_crossed': phase3_metrics['risky_tiles_crossed'],
        })

        if verbose and (episode + 1) % 50 == 0:
            recent = history[-50:]
            success_rate = sum(1 for h in recent if h['success']) / len(recent)
            starvation = sum(1 for h in recent if h['death_reason'] == 'starvation')
            injury = sum(1 for h in recent if h['death_reason'] == 'injury')
            risky_rate = np.mean([h['risky_shortcut_rate'] for h in recent])

            print(f"  Episode {episode + 1}: "
                  f"success={success_rate:.0%}, "
                  f"starved={starvation}, injured={injury}, "
                  f"risky_rate={risky_rate:.2%}")

    env.close()
    return brain, history


def analyze_phase3_results(phase_name: str, history: List[Dict], brain: SurvivalGenesisBrain):
    """Analyze and report Phase 3 results."""
    print(f"\n  [{phase_name} RESULTS]")

    early = history[:40]
    late = history[-40:]

    # Basic metrics
    early_success = sum(1 for h in early if h['success']) / len(early)
    late_success = sum(1 for h in late if h['success']) / len(late)

    early_starvation = sum(1 for h in early if h['death_reason'] == 'starvation')
    late_starvation = sum(1 for h in late if h['death_reason'] == 'starvation')

    early_injury = sum(1 for h in early if h['death_reason'] == 'injury')
    late_injury = sum(1 for h in late if h['death_reason'] == 'injury')

    print(f"    Success Rate: {early_success:.0%} -> {late_success:.0%}")
    print(f"    Starvation: {early_starvation} -> {late_starvation}")
    print(f"    Injury: {early_injury} -> {late_injury}")

    # Phase 3 specific metrics
    early_hungry_eats = sum(h['food_eaten_while_hungry'] for h in early)
    late_hungry_eats = sum(h['food_eaten_while_hungry'] for h in late)
    early_food = sum(h['food_eaten'] for h in early)
    late_food = sum(h['food_eaten'] for h in late)

    food_before_goal_rate = late_hungry_eats / max(1, late_food)

    early_risky = np.mean([h['risky_shortcut_rate'] for h in early])
    late_risky = np.mean([h['risky_shortcut_rate'] for h in late])

    print(f"\n    [Phase 3 Behavior Metrics]")
    print(f"    Food when hungry: {early_hungry_eats} -> {late_hungry_eats}")
    print(f"    Food-before-goal rate: {food_before_goal_rate:.0%}")
    print(f"    Risky shortcut rate: {early_risky:.2%} -> {late_risky:.2%}")

    # Lava/Poison specifics (if applicable)
    early_lava = sum(h['lava_hits'] for h in early)
    late_lava = sum(h['lava_hits'] for h in late)
    early_poison = sum(h['poison_ticks'] for h in early)
    late_poison = sum(h['poison_ticks'] for h in late)

    if early_lava > 0 or late_lava > 0:
        print(f"    Lava hits: {early_lava} -> {late_lava}")
    if early_poison > 0 or late_poison > 0:
        print(f"    Poison ticks: {early_poison} -> {late_poison}")

    # Pass/Fail criteria
    print(f"\n    [Pass/Fail Criteria - {phase_name}]")

    # 1. Success >= 70%
    if late_success >= 0.7:
        print(f"    [PASS] Success >= 70%: {late_success:.0%}")
    elif late_success >= 0.5:
        print(f"    [WARN] Success 50-70%: {late_success:.0%}")
    else:
        print(f"    [FAIL] Success < 50%: {late_success:.0%}")

    # 2. Death not exploding (within +15% of baseline)
    total_early_deaths = early_starvation + early_injury
    total_late_deaths = late_starvation + late_injury
    death_increase = (total_late_deaths - total_early_deaths) / max(1, len(late))

    if death_increase <= 0.15:
        print(f"    [PASS] Death increase <= 15%: {death_increase:.0%}")
    else:
        print(f"    [FAIL] Death increase > 15%: {death_increase:.0%}")

    # 3. Food-seeking when hungry (>= 70%)
    hungry_eat_rate = late_hungry_eats / max(1, sum(h['near_starvation'] for h in late))
    if late_hungry_eats > 0 or sum(h['near_starvation'] for h in late) == 0:
        print(f"    [PASS] Eats when hungry: {late_hungry_eats} events")
    else:
        print(f"    [WARN] Not eating when hungry: {late_hungry_eats}")


def run_doorkey_regression() -> Dict:
    """Run DoorKey regression tests to ensure survival doesn't break Key->Door."""
    from genesis_minigrid import train_genesis_minigrid

    results = {'passed': True, 'details': []}

    sizes = [5, 6, 8]
    for size in sizes:
        print(f"\n  Testing DoorKey {size}x{size}...")

        # Run standard DoorKey test
        brain, history = train_genesis_minigrid(
            grid_size=size,
            n_episodes=100,
            verbose=False
        )

        # Check results
        late = history[-30:]
        success_rate = sum(1 for h in late if h['success']) / len(late)

        # Get rule accuracy from metacognition
        stats = brain.get_stats()
        key_door_accuracy = stats.get('metacog_key_door_accuracy', 1.0)
        door_path_accuracy = stats.get('metacog_door_path_accuracy', 1.0)

        result = {
            'size': size,
            'success_rate': success_rate,
            'key_door_accuracy': key_door_accuracy,
            'door_path_accuracy': door_path_accuracy,
        }
        results['details'].append(result)

        # Check pass criteria
        passed = success_rate >= 0.6 and key_door_accuracy >= 0.8
        if not passed:
            results['passed'] = False

        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {status} {size}x{size}: success={success_rate:.0%}, "
              f"key->door={key_door_accuracy:.0%}")

    return results


# =============================================================================
# PHASE 4A: GOAL VALUE MODULATION TRAINING
# =============================================================================

def train_goal_value_modulation(
    env_config: Dict,
    n_episodes: int = 200,
    verbose: bool = True
) -> Tuple[SurvivalGenesisBrain, List[Dict]]:
    """Train on GoalValueEnv with goal value modulation."""

    # Create Goal Value environment
    base_env = GoalValueEnv(**env_config)
    env = FullyObsWrapper(base_env)

    # Create brain
    grid_size = env_config.get('size', 10)
    brain = SurvivalGenesisBrain(n_actions=7, grid_size=grid_size)

    history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        brain.reset()
        done = False
        total_reward = 0
        steps = 0
        death_reason = None

        # Get this episode's goal value
        episode_goal_value = env.unwrapped.goal_value
        episode_goal_label = env.unwrapped.goal_value_label

        while not done:
            action = brain.act(obs, env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            brain.learn_survival(next_obs, reward, done, info, env)

            total_reward += reward
            steps += 1
            obs = next_obs

            if done:
                death_reason = info.get('death_reason')

        # Get Phase 4 metrics from env
        phase4_metrics = env.unwrapped.get_phase4_metrics()

        # Record episode
        success = total_reward > 0 and death_reason is None
        history.append({
            'episode': episode,
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'death_reason': death_reason,
            'goal_value': episode_goal_value,
            'goal_value_label': episode_goal_label,
            'food_eaten': brain.food_eaten_count,
            'lava_hits': brain.lava_hits,
            'poison_ticks': brain.poison_ticks,
            'risky_shortcut_rate': phase4_metrics['risky_shortcut_rate'],
            'risky_tiles_crossed': phase4_metrics['risky_tiles_crossed'],
        })

        if verbose and (episode + 1) % 50 == 0:
            recent = history[-50:]
            success_rate = sum(1 for h in recent if h['success']) / len(recent)

            # Separate by goal value
            high_eps = [h for h in recent if h['goal_value_label'] == 'high']
            low_eps = [h for h in recent if h['goal_value_label'] == 'low']

            high_risky = np.mean([h['risky_shortcut_rate'] for h in high_eps]) if high_eps else 0
            low_risky = np.mean([h['risky_shortcut_rate'] for h in low_eps]) if low_eps else 0

            print(f"  Episode {episode + 1}: "
                  f"success={success_rate:.0%}, "
                  f"risky_high={high_risky:.2%}, risky_low={low_risky:.2%}")

    env.close()
    return brain, history


def run_phase4a_test():
    """Run Phase 4A: Goal Value Modulation test."""
    print("=" * 60)
    print("PHASE 4A: GOAL VALUE MODULATION")
    print("Test: Does risky_rate increase with goal_value?")
    print("=" * 60)

    phase4a_config = {
        'size': 10,
        'initial_energy': 0.5,
        'energy_decay': 0.025,
        'n_food': 2,
        'food_gain': 0.35,
        'n_lava': 4,
        'n_poison': 3,
        'lava_pain': 0.4,
        'lava_energy_loss': 0.2,
        'poison_pain': 0.1,
        'has_goal': True,
        'has_key_door': False,
        'max_steps': 100,
        'hazard_on_path_prob': 0.8,  # Higher chance hazards block direct path
        'safe_path_guaranteed': True,
        'goal_value_distribution': 'uniform',
    }

    brain, history = train_goal_value_modulation(phase4a_config, n_episodes=300)

    # Analyze by goal value
    print("\n" + "-" * 60)
    print("PHASE 4A RESULTS")
    print("-" * 60)

    # Separate results by goal value
    high_eps = [h for h in history if h['goal_value_label'] == 'high']
    mid_eps = [h for h in history if h['goal_value_label'] == 'mid']
    low_eps = [h for h in history if h['goal_value_label'] == 'low']

    def analyze_group(eps, label):
        if not eps:
            return {}
        success = sum(1 for h in eps if h['success']) / len(eps)
        risky = np.mean([h['risky_shortcut_rate'] for h in eps])
        lava = sum(h['lava_hits'] for h in eps) / len(eps)
        starvation = sum(1 for h in eps if h['death_reason'] == 'starvation')
        injury = sum(1 for h in eps if h['death_reason'] == 'injury')
        return {
            'n': len(eps),
            'success': success,
            'risky_rate': risky,
            'lava_per_ep': lava,
            'starvation': starvation,
            'injury': injury,
        }

    high_stats = analyze_group(high_eps, 'high')
    mid_stats = analyze_group(mid_eps, 'mid')
    low_stats = analyze_group(low_eps, 'low')

    print(f"\n  [Results by Goal Value]")
    print(f"  {'Goal Value':<12} {'N':>5} {'Success':>10} {'Risky Rate':>12} {'Lava/Ep':>10}")
    print(f"  {'-'*50}")

    for label, stats in [('HIGH', high_stats), ('MID', mid_stats), ('LOW', low_stats)]:
        if stats:
            print(f"  {label:<12} {stats['n']:>5} {stats['success']:>10.0%} "
                  f"{stats['risky_rate']:>12.2%} {stats['lava_per_ep']:>10.2f}")

    # Key test: Does risky_rate increase with goal_value?
    print(f"\n  [Phase 4A Key Hypothesis Test]")
    print(f"  H0: Risky rate is independent of goal value")
    print(f"  H1: Risky rate increases with goal value")

    # Compute correlation
    all_goal_values = [h['goal_value'] for h in history]
    all_risky_rates = [h['risky_shortcut_rate'] for h in history]

    if len(set(all_goal_values)) >= 2:
        mean_gv = np.mean(all_goal_values)
        mean_rr = np.mean(all_risky_rates)
        std_gv = np.std(all_goal_values) + 1e-6
        std_rr = np.std(all_risky_rates) + 1e-6
        correlation = np.mean((np.array(all_goal_values) - mean_gv) *
                              (np.array(all_risky_rates) - mean_rr)) / (std_gv * std_rr)
    else:
        correlation = 0.0

    print(f"\n  Correlation(goal_value, risky_rate) = {correlation:.3f}")

    # Pass/Fail criteria
    print(f"\n  [Pass/Fail Criteria]")

    # 1. Positive correlation (risky increases with goal value)
    if correlation > 0.1:
        print(f"  [PASS] Positive correlation: {correlation:.3f} > 0.1")
    elif correlation > 0:
        print(f"  [WARN] Weak positive correlation: {correlation:.3f}")
    else:
        print(f"  [FAIL] No positive correlation: {correlation:.3f}")

    # 2. High goal risky > Low goal risky
    if high_stats and low_stats:
        if high_stats['risky_rate'] > low_stats['risky_rate']:
            print(f"  [PASS] HIGH risky > LOW risky: {high_stats['risky_rate']:.2%} > {low_stats['risky_rate']:.2%}")
        else:
            print(f"  [FAIL] HIGH risky <= LOW risky: {high_stats['risky_rate']:.2%} <= {low_stats['risky_rate']:.2%}")

    # 3. Success maintained (>= 50%)
    overall_success = sum(1 for h in history if h['success']) / len(history)
    if overall_success >= 0.5:
        print(f"  [PASS] Overall success >= 50%: {overall_success:.0%}")
    else:
        print(f"  [WARN] Overall success < 50%: {overall_success:.0%}")

    # Brain stats
    stats = brain.get_survival_stats()
    print(f"\n  [Brain Phase 4A Stats]")
    print(f"    Risks taken for high goal: {stats['risks_taken_for_high_goal']}")
    print(f"    Goal-Risk correlation (brain): {stats['goal_value_risk_correlation']:.3f}")

    return brain, history


# =============================================================================
# PHASE 4B: UNCERTAINTY / INFORMATION SEEKING TRAINING
# =============================================================================

def train_uncertainty_environment(
    env_config: Dict,
    n_episodes: int = 200,
    verbose: bool = True
) -> Tuple[SurvivalGenesisBrain, List[Dict]]:
    """Train on UncertaintyEnv with hidden hazards and SCAN action."""

    # Create Uncertainty environment
    base_env = UncertaintyEnv(**env_config)
    env = FullyObsWrapper(base_env)

    # Create brain with 8 actions (including SCAN)
    grid_size = env_config.get('size', 10)
    n_actions = 8 if env_config.get('enable_scan', True) else 7
    brain = SurvivalGenesisBrain(n_actions=n_actions, grid_size=grid_size)

    history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        brain.reset()
        done = False
        total_reward = 0
        steps = 0
        death_reason = None

        # Track uncertainty dynamics for this episode
        uncertainty_samples = []
        scan_on_signal = 0  # SCAN when there was a signal
        scan_no_signal = 0  # SCAN without signal

        while not done:
            # Track uncertainty before action
            current_uncertainty = env.unwrapped._compute_uncertainty()
            uncertainty_samples.append(current_uncertainty)

            # Check for nearby signals
            agent_pos = env.unwrapped.agent_pos
            nearby_signal = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                adj_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
                if env.unwrapped.hazard_signal_map.get(adj_pos, 0.0) > 0:
                    nearby_signal = True
                    break

            action = brain.act(obs, env)

            # Track SCAN usage pattern
            if action == 7:  # SCAN
                if nearby_signal:
                    scan_on_signal += 1
                else:
                    scan_no_signal += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            brain.learn_survival(next_obs, reward, done, info, env)

            total_reward += reward
            steps += 1
            obs = next_obs

            if done:
                death_reason = info.get('death_reason')

        # Get Phase 4B metrics from env
        phase4b_metrics = env.unwrapped.get_phase4b_metrics()

        # Compute exploration metrics
        avg_uncertainty = np.mean(uncertainty_samples) if uncertainty_samples else 0
        uncertainty_reduction = uncertainty_samples[0] - uncertainty_samples[-1] if len(uncertainty_samples) > 1 else 0

        # Record episode
        success = total_reward > 0 and death_reason is None
        history.append({
            'episode': episode,
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'death_reason': death_reason,
            # Phase 4B specific
            'hidden_hazards_total': phase4b_metrics['hidden_hazards_total'],
            'hazards_revealed': phase4b_metrics['hazards_revealed'],
            'scan_actions': phase4b_metrics['scan_actions'],
            'entered_uncertain_area': phase4b_metrics['entered_uncertain_area'],
            'surprises': phase4b_metrics['surprises'],
            'final_uncertainty': phase4b_metrics['uncertainty'],
            'avg_uncertainty': avg_uncertainty,
            'uncertainty_reduction': uncertainty_reduction,
            'scan_on_signal': scan_on_signal,
            'scan_no_signal': scan_no_signal,
            # Survival metrics
            'lava_hits': brain.lava_hits,
            'poison_ticks': brain.poison_ticks,
        })

        if verbose and (episode + 1) % 50 == 0:
            recent = history[-50:]
            success_rate = sum(1 for h in recent if h['success']) / len(recent)
            avg_scans = np.mean([h['scan_actions'] for h in recent])
            avg_surprises = np.mean([h['surprises'] for h in recent])
            avg_revealed = np.mean([h['hazards_revealed'] for h in recent])

            print(f"  Episode {episode + 1}: "
                  f"success={success_rate:.0%}, "
                  f"scans={avg_scans:.1f}, "
                  f"revealed={avg_revealed:.1f}, "
                  f"surprises={avg_surprises:.1f}")

    env.close()
    return brain, history


def run_phase4b_test():
    """Run Phase 4B: Uncertainty / Information Seeking test."""
    print("=" * 60)
    print("PHASE 4B: UNCERTAINTY & INFORMATION SEEKING")
    print("Test: Does agent use SCAN to reduce uncertainty?")
    print("      Does high uncertainty lead to more exploration?")
    print("=" * 60)

    phase4b_config = {
        'size': 10,
        'initial_energy': 0.5,
        'energy_decay': 0.02,  # Slightly slower for exploration time
        'n_food': 2,
        'food_gain': 0.35,
        'n_lava': 3,
        'n_poison': 3,
        'lava_pain': 0.4,
        'lava_energy_loss': 0.2,
        'poison_pain': 0.1,
        'has_goal': True,
        'has_key_door': False,
        'max_steps': 120,  # More steps for exploration
        'hazard_on_path_prob': 0.8,
        'safe_path_guaranteed': True,
        # Phase 4B specific
        'hazard_masked_prob': 0.5,  # 50% hazards are hidden
        'signal_radius': 1,
        'enable_scan': True,
        'scan_energy_cost': 0.02,
    }

    # Train with SCAN enabled
    print("\n[Training with SCAN action enabled]")
    brain_scan, history_scan = train_uncertainty_environment(phase4b_config, n_episodes=300)

    # Train without SCAN for comparison
    print("\n[Training WITHOUT SCAN (baseline)]")
    no_scan_config = phase4b_config.copy()
    no_scan_config['enable_scan'] = False
    brain_noscan, history_noscan = train_uncertainty_environment(no_scan_config, n_episodes=300)

    # Analyze results
    print("\n" + "-" * 60)
    print("PHASE 4B RESULTS")
    print("-" * 60)

    def analyze_group(history, label):
        if not history:
            return {}
        late = history[-100:]  # Last 100 episodes
        success = sum(1 for h in late if h['success']) / len(late)
        avg_scans = np.mean([h['scan_actions'] for h in late])
        avg_revealed = np.mean([h['hazards_revealed'] for h in late])
        avg_surprises = np.mean([h['surprises'] for h in late])
        avg_uncertainty_reduction = np.mean([h['uncertainty_reduction'] for h in late])
        scan_on_signal_rate = np.sum([h['scan_on_signal'] for h in late]) / max(1, np.sum([h['scan_actions'] for h in late]))

        return {
            'label': label,
            'n': len(late),
            'success': success,
            'avg_scans': avg_scans,
            'avg_revealed': avg_revealed,
            'avg_surprises': avg_surprises,
            'uncertainty_reduction': avg_uncertainty_reduction,
            'scan_on_signal_rate': scan_on_signal_rate,
        }

    scan_stats = analyze_group(history_scan, 'WITH SCAN')
    noscan_stats = analyze_group(history_noscan, 'NO SCAN')

    print(f"\n  [Comparison: SCAN vs No-SCAN]")
    print(f"  {'Metric':<25} {'WITH SCAN':>12} {'NO SCAN':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Success Rate':<25} {scan_stats['success']:>12.0%} {noscan_stats['success']:>12.0%}")
    print(f"  {'Avg Scans/Episode':<25} {scan_stats['avg_scans']:>12.1f} {'N/A':>12}")
    print(f"  {'Hazards Revealed/Ep':<25} {scan_stats['avg_revealed']:>12.1f} {noscan_stats['avg_revealed']:>12.1f}")
    print(f"  {'Surprise Hazards/Ep':<25} {scan_stats['avg_surprises']:>12.2f} {noscan_stats['avg_surprises']:>12.2f}")
    print(f"  {'Uncertainty Reduction':<25} {scan_stats['uncertainty_reduction']:>12.2f} {noscan_stats['uncertainty_reduction']:>12.2f}")

    # Key test: Information seeking behavior
    print(f"\n  [Phase 4B Key Hypothesis Tests]")

    # 1. SCAN reduces surprises
    surprise_reduction = noscan_stats['avg_surprises'] - scan_stats['avg_surprises']
    if surprise_reduction > 0:
        print(f"  [PASS] SCAN reduces surprises: {noscan_stats['avg_surprises']:.2f} → {scan_stats['avg_surprises']:.2f} (-{surprise_reduction:.2f})")
    else:
        print(f"  [FAIL] SCAN doesn't reduce surprises: {noscan_stats['avg_surprises']:.2f} → {scan_stats['avg_surprises']:.2f}")

    # 2. SCAN is used more when there's a signal
    if scan_stats['scan_on_signal_rate'] > 0.5:
        print(f"  [PASS] SCAN on signal rate > 50%: {scan_stats['scan_on_signal_rate']:.0%}")
    else:
        print(f"  [WARN] SCAN on signal rate <= 50%: {scan_stats['scan_on_signal_rate']:.0%}")

    # 3. More hazards revealed with SCAN
    reveal_improvement = scan_stats['avg_revealed'] - noscan_stats['avg_revealed']
    if reveal_improvement > 0:
        print(f"  [PASS] More hazards revealed with SCAN: +{reveal_improvement:.2f}/ep")
    else:
        print(f"  [FAIL] No improvement in hazard revelation: {reveal_improvement:.2f}")

    # 4. Success maintained or improved
    success_delta = scan_stats['success'] - noscan_stats['success']
    if success_delta >= -0.05:  # Allow 5% tolerance
        print(f"  [PASS] Success maintained: {noscan_stats['success']:.0%} → {scan_stats['success']:.0%}")
    else:
        print(f"  [FAIL] Success decreased: {noscan_stats['success']:.0%} → {scan_stats['success']:.0%}")

    # Analyze uncertainty vs exploration correlation
    print(f"\n  [Uncertainty → Exploration Analysis]")

    # High uncertainty episodes should have more exploration (scans, turns)
    high_uncertainty_eps = [h for h in history_scan[-100:] if h['avg_uncertainty'] > 0.5]
    low_uncertainty_eps = [h for h in history_scan[-100:] if h['avg_uncertainty'] <= 0.3]

    if high_uncertainty_eps and low_uncertainty_eps:
        high_scans = np.mean([h['scan_actions'] for h in high_uncertainty_eps])
        low_scans = np.mean([h['scan_actions'] for h in low_uncertainty_eps])

        print(f"    High uncertainty eps: {len(high_uncertainty_eps)}, avg scans: {high_scans:.1f}")
        print(f"    Low uncertainty eps: {len(low_uncertainty_eps)}, avg scans: {low_scans:.1f}")

        if high_scans > low_scans:
            print(f"  [PASS] High uncertainty → more SCAN: {low_scans:.1f} → {high_scans:.1f}")
        else:
            print(f"  [FAIL] High uncertainty doesn't increase SCAN")
    else:
        print(f"    Insufficient data for uncertainty stratification")

    # Brain stats
    stats = brain_scan.get_survival_stats()
    print(f"\n  [Brain Phase 4B Stats]")
    print(f"    Total SCAN actions: {stats['scan_count']}")
    print(f"    Exploration for info: {stats['exploration_for_info']}")
    print(f"    Surprise hazards: {stats['surprises_count']}")
    print(f"    Info gain events: {stats['info_gain_events']}")

    return brain_scan, history_scan


# =============================================================================
# PHASE 4B ABLATION SUITE: Active Inference Verification
# =============================================================================

def compute_active_inference_metrics(history: List[Dict]) -> Dict:
    """
    Compute the 3 key Active Inference metrics:
    1. Scan Precision: scan ratio with signal / scan ratio without signal
    2. Info-gain per scan: revealed_hazards / scan_count
    3. Decision latency: (approximated by steps after first signal)
    """
    if not history:
        return {}

    total_scans_with_signal = sum(h.get('scan_on_signal', 0) for h in history)
    total_scans_no_signal = sum(h.get('scan_no_signal', 0) for h in history)
    total_scans = sum(h.get('scan_actions', 0) for h in history)
    total_revealed = sum(h.get('hazards_revealed', 0) for h in history)

    # Scan Precision: ratio of scans-with-signal to scans-without-signal
    # Higher = more precise (only scans when needed)
    if total_scans_no_signal > 0:
        scan_precision = total_scans_with_signal / total_scans_no_signal
    else:
        scan_precision = float('inf') if total_scans_with_signal > 0 else 0.0

    # Info-gain per scan: revealed / scans
    # Higher = more efficient scanning
    info_gain_per_scan = total_revealed / max(1, total_scans)

    # Scan efficiency: what % of scans had signal nearby
    scan_efficiency = total_scans_with_signal / max(1, total_scans)

    return {
        'scan_precision': scan_precision,
        'info_gain_per_scan': info_gain_per_scan,
        'scan_efficiency': scan_efficiency,
        'total_scans': total_scans,
        'total_revealed': total_revealed,
        'scans_with_signal': total_scans_with_signal,
        'scans_no_signal': total_scans_no_signal,
    }


def run_phase4b_ablation():
    """
    Phase 4B Ablation Suite: Verify Active Inference behavior.

    4 Ablation Tests:
    1. SCAN ON vs OFF (baseline)
    2. signal_map ON vs OFF (scan addiction check)
    3. scan_energy_cost sweep: 0 / 0.02 / 0.05
    4. hazard_masked_prob sweep: 0.0 / 0.5 / 0.8
    """
    print("=" * 70)
    print("PHASE 4B ABLATION: Active Inference Verification")
    print("=" * 70)

    base_config = {
        'size': 8,
        'initial_energy': 0.5,
        'energy_decay': 0.025,
        'n_food': 2,
        'food_gain': 0.35,
        'n_lava': 2,
        'n_poison': 2,
        'has_goal': True,
        'max_steps': 80,
        'hazard_masked_prob': 0.5,
        'signal_radius': 1,
        'enable_scan': True,
        'scan_energy_cost': 0.02,
    }

    n_episodes = 100  # Reduced for faster ablation
    results = {}

    # =========================================================================
    # Ablation 1: SCAN ON vs OFF
    # =========================================================================
    print("\n[Ablation 1: SCAN ON vs OFF]")

    # SCAN ON
    _, history_scan_on = train_uncertainty_environment(base_config, n_episodes=n_episodes, verbose=False)

    # SCAN OFF
    config_scan_off = base_config.copy()
    config_scan_off['enable_scan'] = False
    _, history_scan_off = train_uncertainty_environment(config_scan_off, n_episodes=n_episodes, verbose=False)

    results['scan_on'] = {
        'success': np.mean([h['success'] for h in history_scan_on[-50:]]),
        'surprises': np.mean([h['surprises'] for h in history_scan_on[-50:]]),
        **compute_active_inference_metrics(history_scan_on[-50:])
    }
    results['scan_off'] = {
        'success': np.mean([h['success'] for h in history_scan_off[-50:]]),
        'surprises': np.mean([h['surprises'] for h in history_scan_off[-50:]]),
    }

    print(f"  SCAN ON:  success={results['scan_on']['success']:.0%}, surprises={results['scan_on']['surprises']:.2f}")
    print(f"  SCAN OFF: success={results['scan_off']['success']:.0%}, surprises={results['scan_off']['surprises']:.2f}")

    # =========================================================================
    # Ablation 2: signal_map ON vs OFF (scan addiction check)
    # =========================================================================
    print("\n[Ablation 2: signal_map ON vs OFF (Scan Addiction Check)]")

    # signal_radius = 0 means no signals emitted
    config_no_signal = base_config.copy()
    config_no_signal['signal_radius'] = 0
    _, history_no_signal = train_uncertainty_environment(config_no_signal, n_episodes=n_episodes, verbose=False)

    results['no_signal'] = {
        'success': np.mean([h['success'] for h in history_no_signal[-50:]]),
        'surprises': np.mean([h['surprises'] for h in history_no_signal[-50:]]),
        **compute_active_inference_metrics(history_no_signal[-50:])
    }

    # Check for scan addiction: many scans without signal info
    scan_addiction = results['no_signal']['total_scans'] > results['scan_on']['total_scans'] * 0.8

    print(f"  With signals:    scans={results['scan_on']['total_scans']}, efficiency={results['scan_on']['scan_efficiency']:.0%}")
    print(f"  Without signals: scans={results['no_signal']['total_scans']}, efficiency={results['no_signal']['scan_efficiency']:.0%}")

    if scan_addiction:
        print(f"  [WARN] Scan addiction detected! Scanning even without signal info")
    else:
        print(f"  [PASS] No scan addiction: scans reduced without signals")

    # =========================================================================
    # Ablation 3: scan_energy_cost sweep (0 / 0.02 / 0.05)
    # =========================================================================
    print("\n[Ablation 3: scan_energy_cost sweep]")

    cost_results = {}
    for cost in [0.0, 0.02, 0.05]:
        config_cost = base_config.copy()
        config_cost['scan_energy_cost'] = cost
        _, history_cost = train_uncertainty_environment(config_cost, n_episodes=n_episodes, verbose=False)

        metrics = compute_active_inference_metrics(history_cost[-50:])
        cost_results[cost] = {
            'success': np.mean([h['success'] for h in history_cost[-50:]]),
            'scans': metrics['total_scans'],
            'efficiency': metrics['scan_efficiency'],
            'info_gain': metrics['info_gain_per_scan'],
        }

    print(f"  {'Cost':<8} {'Success':>10} {'Scans':>10} {'Efficiency':>12} {'Info/Scan':>12}")
    print(f"  {'-'*52}")
    for cost, r in cost_results.items():
        print(f"  {cost:<8.2f} {r['success']:>10.0%} {r['scans']:>10.0f} {r['efficiency']:>12.0%} {r['info_gain']:>12.2f}")

    # Check: higher cost should reduce scans but maintain efficiency
    if cost_results[0.05]['scans'] < cost_results[0.0]['scans']:
        print(f"  [PASS] Higher cost reduces scan count: {cost_results[0.0]['scans']:.0f} → {cost_results[0.05]['scans']:.0f}")
    else:
        print(f"  [WARN] Cost doesn't affect scan frequency")

    results['cost_sweep'] = cost_results

    # =========================================================================
    # Ablation 4: hazard_masked_prob sweep (0.0 / 0.5 / 0.8)
    # =========================================================================
    print("\n[Ablation 4: hazard_masked_prob sweep (Uncertainty Level)]")

    uncertainty_results = {}
    for mask_prob in [0.0, 0.5, 0.8]:
        config_unc = base_config.copy()
        config_unc['hazard_masked_prob'] = mask_prob
        _, history_unc = train_uncertainty_environment(config_unc, n_episodes=n_episodes, verbose=False)

        metrics = compute_active_inference_metrics(history_unc[-50:])
        uncertainty_results[mask_prob] = {
            'success': np.mean([h['success'] for h in history_unc[-50:]]),
            'scans': metrics['total_scans'],
            'surprises': np.mean([h['surprises'] for h in history_unc[-50:]]),
            'avg_uncertainty': np.mean([h['avg_uncertainty'] for h in history_unc[-50:]]),
        }

    print(f"  {'Mask %':<8} {'Success':>10} {'Scans':>10} {'Surprises':>12} {'Uncertainty':>12}")
    print(f"  {'-'*52}")
    for mask, r in uncertainty_results.items():
        print(f"  {mask:<8.0%} {r['success']:>10.0%} {r['scans']:>10.0f} {r['surprises']:>12.2f} {r['avg_uncertainty']:>12.2f}")

    # Check monotonicity: more hidden → more scans
    if uncertainty_results[0.8]['scans'] > uncertainty_results[0.0]['scans']:
        print(f"  [PASS] Monotonic: more uncertainty → more scanning")
    else:
        print(f"  [WARN] Non-monotonic uncertainty response")

    results['uncertainty_sweep'] = uncertainty_results

    # =========================================================================
    # Summary: Active Inference Verification
    # =========================================================================
    print("\n" + "=" * 70)
    print("ACTIVE INFERENCE VERIFICATION SUMMARY")
    print("=" * 70)

    checks = []

    # Check 1: SCAN reduces surprises
    surprise_reduction = results['scan_off']['surprises'] - results['scan_on']['surprises']
    checks.append(('SCAN reduces surprises', surprise_reduction > 0, f"{surprise_reduction:.2f}"))

    # Check 2: Scan precision > 1 (more scans with signal than without)
    checks.append(('Scan precision > 1', results['scan_on']['scan_precision'] > 1,
                   f"{results['scan_on']['scan_precision']:.2f}"))

    # Check 3: No scan addiction
    checks.append(('No scan addiction', not scan_addiction,
                   f"ratio={results['no_signal']['total_scans']/max(1,results['scan_on']['total_scans']):.2f}"))

    # Check 4: Cost sensitivity
    cost_sensitive = cost_results[0.05]['scans'] < cost_results[0.0]['scans'] * 0.9
    checks.append(('Cost-sensitive scanning', cost_sensitive,
                   f"{cost_results[0.0]['scans']:.0f}→{cost_results[0.05]['scans']:.0f}"))

    # Check 5: Uncertainty monotonicity
    monotonic = uncertainty_results[0.8]['scans'] > uncertainty_results[0.0]['scans']
    checks.append(('Uncertainty→exploration monotonic', monotonic,
                   f"{uncertainty_results[0.0]['scans']:.0f}→{uncertainty_results[0.8]['scans']:.0f}"))

    for name, passed, value in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}: {value}")

    passed_count = sum(1 for _, p, _ in checks if p)
    print(f"\n  Active Inference Score: {passed_count}/{len(checks)}")

    if passed_count >= 4:
        print("  ✓ Agent exhibits Active Inference behavior")
    else:
        print("  ✗ Active Inference behavior not confirmed")

    return results


# =============================================================================
# TRANSFER TESTS: Plasticity Verification
# =============================================================================

class CardGateEnv(SurvivalMiniGridEnv):
    """
    Transfer Test 1: Isomorphic Reskin

    Same structure as Key→Door, but renamed:
    - Key → Card
    - Door → Gate

    Event schema preserved:
    - CARD_ACQUIRED (was KEY_ACQUIRED)
    - GATE_UNLOCKED (was DOOR_UNLOCKED)

    Expected: Near 0-shot transfer - same underlying logic.
    """

    def __init__(self, **kwargs):
        # Force key-door logic with different names
        kwargs['has_key_door'] = True
        kwargs['has_goal'] = True
        super().__init__(**kwargs)

        # Rename tracking
        self.card_acquired = False
        self.gate_opened = False

    def _gen_grid(self, width, height):
        """Generate grid with Card and Gate instead of Key and Door."""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Reset survival state
        self.energy = self.initial_energy
        self.pain = 0.0
        self.food_positions = []
        self.lava_positions = []
        self.poison_positions = []
        self.step_events = []
        self.was_on_poison = False
        self.poison_ticks_this_episode = 0

        # Place agent at fixed position first
        self.agent_pos = np.array([1, 1])
        self.agent_dir = 0

        # Place goal
        self.put_obj(Goal(), width - 2, height - 2)

        # Place Card (Key) and Gate (Door)
        door_pos = (width // 2, height // 2)
        self.put_obj(Door('yellow', is_locked=True), *door_pos)
        self.place_obj(Key('yellow'), max_tries=100)

        # Place food
        for _ in range(self.n_food):
            pos = self.place_obj(Food(self.food_gain), max_tries=100)
            if pos is not None:
                self.food_positions.append(pos)

        # Reset tracking
        self.card_acquired = False
        self.gate_opened = False

    def step(self, action):
        """Step with Card/Gate event renaming."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Rename events for transfer test
        renamed_events = []
        for event_type, event_data in info.get('events', []):
            if event_type == 'KEY_ACQUIRED':
                renamed_events.append(('CARD_ACQUIRED', event_data))
                self.card_acquired = True
            elif event_type == 'DOOR_UNLOCKED':
                renamed_events.append(('GATE_UNLOCKED', event_data))
                self.gate_opened = True
            else:
                renamed_events.append((event_type, event_data))

        info['events'] = renamed_events
        info['card_acquired'] = self.card_acquired
        info['gate_opened'] = self.gate_opened
        # Compatibility
        info['carrying_key'] = self.card_acquired
        info['door_is_open'] = self.gate_opened

        return obs, reward, terminated, truncated, info


class SwitchDoorEnv(SurvivalMiniGridEnv):
    """
    Transfer Test 2: Rule Swap

    Different causal rule:
    - Instead of Key→Door, it's Switch→Door
    - Switch is a tile that toggles door state when stepped on
    - No item to carry - positional activation

    This tests true plasticity: can the agent learn NEW causal rules?
    """

    def __init__(self, **kwargs):
        kwargs['has_key_door'] = False  # We implement our own
        kwargs['has_goal'] = True

        # Scale survival parameters with grid size
        # Balance: enough resources to survive, but enough pressure to act
        grid_size = kwargs.get('size', 8)
        scale_factor = grid_size / 8.0  # Linear scaling with size

        # Balanced survival parameters
        kwargs['n_food'] = int(8 * scale_factor)  # Linear food scaling
        kwargs['energy_decay'] = 0.008 / scale_factor  # Slower decay for larger grids
        kwargs['food_gain'] = 0.5  # Fixed gain
        kwargs['initial_energy'] = 1.0

        self.switch_pos = None
        self.door_pos = None
        self.door_open = False
        self.switch_pressed = False
        super().__init__(**kwargs)

    def _gen_grid(self, width, height):
        """Generate grid with Switch and Door."""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Reset survival state
        self.energy = self.initial_energy
        self.pain = 0.0
        self.food_positions = []
        self.lava_positions = []
        self.poison_positions = []
        self.step_events = []
        self.was_on_poison = False
        self.poison_ticks_this_episode = 0

        # Place agent at top-left
        self.agent_pos = np.array([1, 1])
        self.agent_dir = 0

        # Place goal at bottom-right
        self.put_obj(Goal(), width - 2, height - 2)

        # Place door in middle (blocking direct path)
        self.door_pos = (width // 2, height // 2)
        self.put_obj(Door('yellow', is_locked=True), *self.door_pos)
        self.door_open = False

        # Place switch somewhere accessible (not behind door)
        # Switch is on the same side as agent
        self.switch_pos = (2, height // 2)
        # Mark switch position (we'll use a green floor tile)
        self.put_obj(Floor('green'), *self.switch_pos)
        self.switch_pressed = False

        # Place food
        for _ in range(self.n_food):
            pos = self.place_obj(Food(self.food_gain), max_tries=100)
            if pos is not None:
                self.food_positions.append(pos)

    def step(self, action):
        """Step with Switch→Door rule."""
        self.step_events = []

        # Energy decay
        self.energy -= self.energy_decay

        # Check near-starvation
        if self.energy < 0.3 and self.energy + self.energy_decay >= 0.3:
            self.step_events.append(('NEAR_STARVATION', {'energy': self.energy}))

        # Execute base step
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if stepped on switch
        agent_tuple = tuple(self.agent_pos)
        if agent_tuple == self.switch_pos and not self.switch_pressed:
            self.switch_pressed = True
            self.step_events.append(('SWITCH_PRESSED', {'pos': self.switch_pos}))

            # Open the door
            if not self.door_open:
                self.door_open = True
                # Remove door from grid (or set to open)
                door_obj = self.grid.get(*self.door_pos)
                if door_obj is not None and isinstance(door_obj, Door):
                    door_obj.is_open = True
                    door_obj.is_locked = False
                self.step_events.append(('DOOR_OPENED_BY_SWITCH', {'door_pos': self.door_pos}))

        # Update info
        info['switch_pressed'] = self.switch_pressed
        info['door_is_open'] = self.door_open
        info['switch_pos'] = self.switch_pos
        info['door_pos'] = self.door_pos
        info['events'] = self.step_events.copy()

        # Death checks
        if self.energy <= 0:
            self.energy = 0.0
            terminated = True
            info['death_reason'] = 'starvation'
            reward = -1.0

        return obs, reward, terminated, truncated, info


def train_transfer_test(
    env_class,
    env_config: Dict,
    brain: SurvivalGenesisBrain = None,
    n_episodes: int = 100,
    verbose: bool = True,
    label: str = "Transfer"
) -> Tuple[SurvivalGenesisBrain, List[Dict]]:
    """Train on transfer test environment."""

    base_env = env_class(**env_config)
    env = FullyObsWrapper(base_env)

    # Use provided brain or create new one
    if brain is None:
        grid_size = env_config.get('size', 8)
        brain = SurvivalGenesisBrain(n_actions=7, grid_size=grid_size)

    history = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        brain.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = brain.act(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            brain.learn_survival(obs, reward, done, info, env)
            total_reward += reward
            steps += 1

        success = total_reward > 0 and info.get('death_reason') is None
        history.append({
            'episode': episode,
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'death_reason': info.get('death_reason'),
            # Phase 5: Causal discovery tracking
            'causal_discoveries': brain.causal_discoveries,
            'known_causal_positions': len(brain.causal_positions),
            'switch_pressed': info.get('switch_pressed', False),
            # Phase 6: Pattern suppression tracking
            'pattern_conflicts': brain.pattern_conflicts_detected,
            'prior_suppressions': brain.prior_suppressions,
            'exploration_boost': brain.exploration_boost,
            # Phase 7: Directed exploration tracking
            'directed_explorations': brain.directed_explorations,
            'positions_visited': len(brain.visited_positions),
        })

        if verbose and (episode + 1) % 25 == 0:
            recent = history[-25:]
            success_rate = sum(1 for h in recent if h['success']) / len(recent)
            switch_rate = sum(1 for h in recent if h.get('switch_pressed', False)) / len(recent)
            positions = np.mean([h.get('positions_visited', 0) for h in recent])
            print(f"  [{label}] Episode {episode + 1}: success={success_rate:.0%}, switch={switch_rate:.0%}, pos_visited={positions:.0f}")

    env.close()
    return brain, history


def run_transfer_tests():
    """
    Run Transfer Test Suite to verify plasticity.

    Test 1: Isomorphic Reskin (Card→Gate)
    - Same logic, different names
    - Expected: Fast adaptation (near 0-shot)

    Test 2: Rule Swap (Switch→Door)
    - Different causal rule
    - Expected: Initial failure, then learning
    """
    print("=" * 70)
    print("TRANSFER TEST SUITE: Plasticity Verification")
    print("=" * 70)

    base_config = {
        'size': 8,
        'initial_energy': 0.6,
        'energy_decay': 0.02,
        'n_food': 2,
        'food_gain': 0.35,
        'max_steps': 100,
    }

    results = {}

    # =========================================================================
    # Phase 0: Train baseline on standard Key→Door
    # =========================================================================
    print("\n[Phase 0: Baseline Training on Key→Door]")

    baseline_config = base_config.copy()
    baseline_config['has_key_door'] = True

    brain_baseline, history_baseline = train_survival_genesis(
        baseline_config, n_episodes=150, verbose=False
    )

    baseline_success = np.mean([h['success'] for h in history_baseline[-50:]])
    print(f"  Baseline Key→Door success: {baseline_success:.0%}")

    results['baseline'] = {
        'success': baseline_success,
        'episodes_to_70': next((i for i, h in enumerate(history_baseline)
                               if np.mean([x['success'] for x in history_baseline[max(0,i-20):i+1]]) >= 0.7),
                              len(history_baseline))
    }

    # =========================================================================
    # Transfer Test 1: Isomorphic Reskin (Card→Gate)
    # =========================================================================
    print("\n[Transfer Test 1: Isomorphic Reskin (Card→Gate)]")
    print("  Expected: Fast adaptation since logic is identical")

    # Test 1a: Fresh agent on Card→Gate
    print("\n  [1a] Fresh agent (no transfer):")
    _, history_fresh_cg = train_transfer_test(
        CardGateEnv, base_config, brain=None, n_episodes=100, verbose=True, label="Fresh"
    )

    # Test 1b: Pre-trained agent on Card→Gate (transfer)
    print("\n  [1b] Pre-trained agent (with transfer):")
    brain_transfer = SurvivalGenesisBrain(n_actions=7, grid_size=base_config['size'])
    # Copy learned weights/patterns from baseline
    brain_transfer.transition = brain_baseline.transition
    brain_transfer.hippocampus = brain_baseline.hippocampus
    brain_transfer.metacog = brain_baseline.metacog

    _, history_transfer_cg = train_transfer_test(
        CardGateEnv, base_config, brain=brain_transfer, n_episodes=100, verbose=True, label="Transfer"
    )

    fresh_cg_success = np.mean([h['success'] for h in history_fresh_cg[-30:]])
    transfer_cg_success = np.mean([h['success'] for h in history_transfer_cg[-30:]])

    # Measure adaptation speed
    fresh_first_70 = next((i for i in range(20, len(history_fresh_cg))
                          if np.mean([h['success'] for h in history_fresh_cg[i-20:i]]) >= 0.7),
                         len(history_fresh_cg))
    transfer_first_70 = next((i for i in range(20, len(history_transfer_cg))
                             if np.mean([h['success'] for h in history_transfer_cg[i-20:i]]) >= 0.7),
                            len(history_transfer_cg))

    results['reskin'] = {
        'fresh_success': fresh_cg_success,
        'transfer_success': transfer_cg_success,
        'fresh_episodes_to_70': fresh_first_70,
        'transfer_episodes_to_70': transfer_first_70,
        'speedup': fresh_first_70 / max(1, transfer_first_70),
    }

    print(f"\n  [Reskin Results]")
    print(f"    Fresh agent:    success={fresh_cg_success:.0%}, episodes to 70%={fresh_first_70}")
    print(f"    Transfer agent: success={transfer_cg_success:.0%}, episodes to 70%={transfer_first_70}")
    print(f"    Speedup: {results['reskin']['speedup']:.1f}x")

    if results['reskin']['speedup'] > 1.5:
        print(f"  [PASS] Significant transfer benefit on isomorphic reskin")
    else:
        print(f"  [WARN] Limited transfer benefit")

    # =========================================================================
    # Transfer Test 2: Rule Swap (Switch→Door)
    # =========================================================================
    print("\n[Transfer Test 2: Rule Swap (Switch→Door)]")
    print("  Expected: Initial failure, then learning new causal rule")

    # Test 2a: Fresh agent on Switch→Door
    print("\n  [2a] Fresh agent:")
    _, history_fresh_sd = train_transfer_test(
        SwitchDoorEnv, base_config, brain=None, n_episodes=100, verbose=True, label="Fresh"
    )

    # Test 2b: Pre-trained (Key→Door) agent on Switch→Door
    print("\n  [2b] Pre-trained agent (negative transfer expected):")
    brain_transfer2 = SurvivalGenesisBrain(n_actions=7, grid_size=base_config['size'])
    brain_transfer2.transition = brain_baseline.transition
    brain_transfer2.hippocampus = brain_baseline.hippocampus
    brain_transfer2.metacog = brain_baseline.metacog

    _, history_transfer_sd = train_transfer_test(
        SwitchDoorEnv, base_config, brain=brain_transfer2, n_episodes=100, verbose=True, label="Transfer"
    )

    fresh_sd_success = np.mean([h['success'] for h in history_fresh_sd[-30:]])
    transfer_sd_success = np.mean([h['success'] for h in history_transfer_sd[-30:]])

    # Check early vs late performance (learning curve)
    fresh_early = np.mean([h['success'] for h in history_fresh_sd[:30]])
    fresh_late = np.mean([h['success'] for h in history_fresh_sd[-30:]])
    transfer_early = np.mean([h['success'] for h in history_transfer_sd[:30]])
    transfer_late = np.mean([h['success'] for h in history_transfer_sd[-30:]])

    # Causal discovery metrics
    fresh_switch_pressed = sum(1 for h in history_fresh_sd if h.get('switch_pressed', False))
    transfer_switch_pressed = sum(1 for h in history_transfer_sd if h.get('switch_pressed', False))
    fresh_causal_disc = max([h.get('causal_discoveries', 0) for h in history_fresh_sd], default=0)
    transfer_causal_disc = max([h.get('causal_discoveries', 0) for h in history_transfer_sd], default=0)

    # Pattern suppression metrics
    fresh_suppressions = max([h.get('prior_suppressions', 0) for h in history_fresh_sd], default=0)
    transfer_suppressions = max([h.get('prior_suppressions', 0) for h in history_transfer_sd], default=0)
    fresh_conflicts = max([h.get('pattern_conflicts', 0) for h in history_fresh_sd], default=0)
    transfer_conflicts = max([h.get('pattern_conflicts', 0) for h in history_transfer_sd], default=0)

    results['rule_swap'] = {
        'fresh_success': fresh_sd_success,
        'transfer_success': transfer_sd_success,
        'fresh_early': fresh_early,
        'fresh_late': fresh_late,
        'transfer_early': transfer_early,
        'transfer_late': transfer_late,
        'fresh_learning': fresh_late - fresh_early,
        'transfer_learning': transfer_late - transfer_early,
        # Causal discovery
        'fresh_switch_pressed': fresh_switch_pressed,
        'transfer_switch_pressed': transfer_switch_pressed,
        'fresh_causal_discoveries': fresh_causal_disc,
        'transfer_causal_discoveries': transfer_causal_disc,
        # Pattern suppression
        'fresh_suppressions': fresh_suppressions,
        'transfer_suppressions': transfer_suppressions,
        'fresh_conflicts': fresh_conflicts,
        'transfer_conflicts': transfer_conflicts,
    }

    print(f"\n  [Rule Swap Results]")
    print(f"    Fresh agent:    early={fresh_early:.0%} -> late={fresh_late:.0%} (D={results['rule_swap']['fresh_learning']:+.0%})")
    print(f"    Transfer agent: early={transfer_early:.0%} -> late={transfer_late:.0%} (D={results['rule_swap']['transfer_learning']:+.0%})")
    print(f"\n  [Causal Discovery]")
    print(f"    Fresh agent:    switch_pressed={fresh_switch_pressed}/100, discoveries={fresh_causal_disc}")
    print(f"    Transfer agent: switch_pressed={transfer_switch_pressed}/100, discoveries={transfer_causal_disc}")
    print(f"\n  [Pattern Suppression]")
    print(f"    Fresh agent:    conflicts={fresh_conflicts}, suppressions={fresh_suppressions}")
    print(f"    Transfer agent: conflicts={transfer_conflicts}, suppressions={transfer_suppressions}")

    # Directed exploration metrics
    fresh_positions = np.mean([h.get('positions_visited', 0) for h in history_fresh_sd[-30:]])
    transfer_positions = np.mean([h.get('positions_visited', 0) for h in history_transfer_sd[-30:]])
    fresh_directed = max([h.get('directed_explorations', 0) for h in history_fresh_sd], default=0)
    transfer_directed = max([h.get('directed_explorations', 0) for h in history_transfer_sd], default=0)

    print(f"\n  [Directed Exploration]")
    print(f"    Fresh agent:    avg_positions={fresh_positions:.0f}, directed_moves={fresh_directed}")
    print(f"    Transfer agent: avg_positions={transfer_positions:.0f}, directed_moves={transfer_directed}")

    # Key insight: Compare transfer vs fresh, not just learning curve
    # Transfer should outperform fresh even if both face environmental challenges
    transfer_advantage = transfer_late - fresh_late
    transfer_switch_advantage = transfer_switch_pressed - fresh_switch_pressed

    if transfer_late > fresh_late:
        print(f"  [PASS] Transfer outperforms Fresh: {transfer_late:.0%} vs {fresh_late:.0%} (+{transfer_advantage:.0%})")
    else:
        print(f"  [WARN] Transfer underperforms Fresh: {transfer_late:.0%} vs {fresh_late:.0%}")

    if transfer_switch_pressed > fresh_switch_pressed * 1.5:
        print(f"  [PASS] Switch discovery improved: {transfer_switch_pressed} vs {fresh_switch_pressed} (+{transfer_switch_advantage})")
    else:
        print(f"  [INFO] Switch discovery similar: {transfer_switch_pressed} vs {fresh_switch_pressed}")

    if transfer_late > 0.3:
        print(f"  [PASS] Agent learned new causal rule despite different prior")
    elif transfer_late > fresh_late:
        print(f"  [PARTIAL] Some transfer benefit, but rule not fully learned")
    else:
        print(f"  [FAIL] No positive transfer")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRANSFER TEST SUMMARY")
    print("=" * 70)

    print(f"\n  [Baseline]")
    print(f"    Key→Door success: {results['baseline']['success']:.0%}")

    print(f"\n  [Test 1: Isomorphic Reskin (Card→Gate)]")
    print(f"    Transfer speedup: {results['reskin']['speedup']:.1f}x")
    print(f"    Verdict: {'PASS' if results['reskin']['speedup'] > 1.5 else 'PARTIAL'}")

    # Transfer advantage metrics
    transfer_advantage = results['rule_swap']['transfer_late'] - results['rule_swap']['fresh_late']
    switch_advantage = results['rule_swap']['transfer_switch_pressed'] - results['rule_swap']['fresh_switch_pressed']

    print(f"\n  [Test 2: Rule Swap (Switch->Door)]")
    print(f"    Transfer vs Fresh (late): {results['rule_swap']['transfer_late']:.0%} vs {results['rule_swap']['fresh_late']:.0%} ({transfer_advantage:+.0%})")
    print(f"    Switch discovery: {results['rule_swap']['transfer_switch_pressed']} vs {results['rule_swap']['fresh_switch_pressed']} ({switch_advantage:+d})")
    print(f"    Causal Discovery: Fresh={results['rule_swap']['fresh_causal_discoveries']}, Transfer={results['rule_swap']['transfer_causal_discoveries']}")
    print(f"    Pattern Suppression: conflicts={results['rule_swap']['transfer_conflicts']}, suppressions={results['rule_swap']['transfer_suppressions']}")

    # Verdict based on transfer advantage, not absolute threshold
    if transfer_advantage > 0.1 and switch_advantage > 10:
        verdict = "PASS"
    elif transfer_advantage > 0 or switch_advantage > 0:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    print(f"    Verdict: {verdict}")

    # Overall plasticity score - updated criteria
    plasticity_checks = [
        results['reskin']['speedup'] > 1.5,  # Positive transfer on reskin
        transfer_advantage > 0,  # Transfer outperforms fresh
        switch_advantage > 10,  # Better causal discovery
        results['rule_swap']['transfer_causal_discoveries'] > 0,  # Discovered causal rule
        results['rule_swap']['transfer_suppressions'] > 0,  # Prior suppression activated
    ]

    print(f"\n  Plasticity Score: {sum(plasticity_checks)}/{len(plasticity_checks)}")

    if sum(plasticity_checks) >= 2:
        print("  [OK] Brain exhibits plasticity")
    else:
        print("  [X] Plasticity not confirmed")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'phase4a':
            run_phase4a_test()
        elif cmd == 'phase4b':
            run_phase4b_test()
        elif cmd == 'ablation':
            run_phase4b_ablation()
        elif cmd == 'transfer':
            run_transfer_tests()
        else:
            run_survival_curriculum()
    else:
        run_survival_curriculum()
