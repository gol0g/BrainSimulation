"""
Rollout System (v1: 2-3 Step Lookahead)

Core Concept: "What happens if I go this way, then that way?"

Instead of just imagining the next step, the agent now thinks 2-3 steps ahead.
This creates more "deliberative" behavior:
- "If I go left, I'll be closer to food. But then the predator might catch up."
- "If I go up first, then left, I can reach food while staying safe."

Implementation:
- Depth: 2-3 steps (configurable)
- Branching: 4 directions per step (4×4=16 or 4×4×4=64 paths)
- Evaluation: Discounted sum of utilities: U = U(s1) + γU(s2) + γ²U(s3)
- Memory integration: LTM recall at each step (with decay for deeper steps)

Key Insight:
The agent isn't doing "tree search" - it's imagining possible futures
and evaluating them emotionally. More like daydreaming than planning.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class RolloutState:
    """Simulated state during rollout."""
    agent_pos: Tuple[int, int]
    predator_pos: Optional[Tuple[int, int]]
    food_pos: Tuple[int, int]
    energy: float
    safety: float


@dataclass
class RolloutPath:
    """A sequence of actions and their cumulative utility."""
    actions: List[str]           # e.g., ['up', 'left']
    states: List[RolloutState]   # States after each action
    utilities: List[float]       # Utility at each step
    total_utility: float         # Discounted sum
    reasons: List[str]           # Why this path is good/bad


class RolloutSystem:
    """
    Multi-step lookahead imagination system.

    Design Philosophy:
    - Simple state transition (no full world simulation)
    - Predator assumed to maintain direction or chase
    - Focus on "feeling" of future, not exact prediction
    """

    def __init__(self, depth: int = 2, gamma: float = 0.9):
        self.depth = depth  # How many steps to look ahead
        self.gamma = gamma  # Discount factor for future utilities
        self.grid_size = 15  # Must match environment

        # Direction vectors
        self.directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        # Utility weights (same as goal system for consistency)
        self.utility_weights = {
            'food_proximity': 1.5,      # Getting closer to food is good
            'predator_proximity': -2.5, # Getting closer to predator is bad
            'reach_food': 5.0,          # Reaching food is very good
            'reach_predator': -8.0,     # Reaching predator is very bad
            'wall_hit': -0.5,           # Hitting wall is mildly bad
            'energy_low': -1.0,         # Low energy is concerning
        }

        # Statistics
        self.paths_evaluated = 0
        self.best_path_depth = 0

    def rollout(self,
                current_state: Dict,
                goal_biases: Dict = None,
                memory_influence_fn = None) -> Dict:
        """
        Perform multi-step rollout from current state.

        Args:
            current_state: Current world state dict with:
                - agent_pos: (x, y)
                - predator_pos: (x, y) or None
                - food_pos: (x, y)
                - energy: float
                - safety: float
            goal_biases: Optional dict of goal-specific utility adjustments
            memory_influence_fn: Optional function(pos, action) -> utility adjustment

        Returns:
            Dict with:
                - best_action: First action of best path
                - best_path: Full best path info
                - all_paths: All evaluated paths (for visualization)
                - first_step_scores: Scores grouped by first action
                - top_reasons: Why best path was chosen
        """
        # Initialize state
        init_state = RolloutState(
            agent_pos=tuple(current_state['agent_pos']),
            predator_pos=tuple(current_state['predator_pos']) if current_state.get('predator_pos') else None,
            food_pos=tuple(current_state['food_pos']),
            energy=current_state.get('energy', 1.0),
            safety=current_state.get('safety', 1.0)
        )

        # Generate all possible paths
        all_paths = self._generate_paths(init_state, goal_biases, memory_influence_fn)
        self.paths_evaluated = len(all_paths)

        if not all_paths:
            return {
                'best_action': 'up',  # Default
                'best_path': None,
                'all_paths': [],
                'first_step_scores': {},
                'top_reasons': []
            }

        # Find best path
        best_path = max(all_paths, key=lambda p: p.total_utility)
        self.best_path_depth = len(best_path.actions)

        # Aggregate scores by first action
        first_step_scores = {d: [] for d in self.directions}
        for path in all_paths:
            if path.actions:
                first_step_scores[path.actions[0]].append(path.total_utility)

        # Average scores per first action
        avg_first_step = {
            d: sum(scores) / len(scores) if scores else 0.0
            for d, scores in first_step_scores.items()
        }

        return {
            'best_action': best_path.actions[0] if best_path.actions else 'up',
            'best_path': {
                'actions': best_path.actions,
                'total_utility': best_path.total_utility,
                'utilities': best_path.utilities,
                'reasons': best_path.reasons[:3]  # Top 3 reasons
            },
            'all_paths': [
                {
                    'actions': p.actions,
                    'total_utility': p.total_utility
                } for p in all_paths
            ],
            'first_step_scores': avg_first_step,
            'top_reasons': best_path.reasons[:2]
        }

    def _generate_paths(self,
                        init_state: RolloutState,
                        goal_biases: Dict = None,
                        memory_fn = None) -> List[RolloutPath]:
        """Generate all paths up to depth and evaluate them."""
        paths = []

        # Recursive path generation
        def expand(state: RolloutState,
                   actions: List[str],
                   states: List[RolloutState],
                   utilities: List[float],
                   reasons: List[str],
                   depth_remaining: int):

            if depth_remaining == 0:
                # Calculate total discounted utility
                total = 0.0
                for i, u in enumerate(utilities):
                    total += (self.gamma ** i) * u

                paths.append(RolloutPath(
                    actions=actions.copy(),
                    states=states.copy(),
                    utilities=utilities.copy(),
                    total_utility=total,
                    reasons=reasons.copy()
                ))
                return

            # Try each direction
            for direction in self.directions:
                next_state, utility, step_reasons = self._simulate_step(
                    state, direction, goal_biases, memory_fn,
                    depth=self.depth - depth_remaining + 1
                )

                new_actions = actions + [direction]
                new_states = states + [next_state]
                new_utilities = utilities + [utility]
                new_reasons = reasons + step_reasons

                expand(next_state, new_actions, new_states, new_utilities,
                       new_reasons, depth_remaining - 1)

        # Start expansion from initial state
        expand(init_state, [], [], [], [], self.depth)

        return paths

    def _simulate_step(self,
                       state: RolloutState,
                       action: str,
                       goal_biases: Dict = None,
                       memory_fn = None,
                       depth: int = 1) -> Tuple[RolloutState, float, List[str]]:
        """
        Simulate one step and calculate utility.

        Returns:
            (new_state, utility, reasons)
        """
        dx, dy = self.directions[action]
        reasons = []
        utility = 0.0

        # Calculate new agent position
        new_x = state.agent_pos[0] + dx
        new_y = state.agent_pos[1] + dy

        # Wall collision check
        hit_wall = False
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            new_x = state.agent_pos[0]
            new_y = state.agent_pos[1]
            hit_wall = True
            utility += self.utility_weights['wall_hit']
            reasons.append(f"{depth}스텝: 벽 충돌")

        new_agent_pos = (new_x, new_y)

        # Simulate predator movement (simple: chase with 50% probability)
        new_predator_pos = state.predator_pos
        if state.predator_pos:
            # Simple prediction: predator moves toward agent
            pred_x, pred_y = state.predator_pos
            agent_x, agent_y = new_agent_pos

            # Move one step toward agent (simplified)
            if pred_x < agent_x:
                pred_x += 1
            elif pred_x > agent_x:
                pred_x -= 1
            if pred_y < agent_y:
                pred_y += 1
            elif pred_y > agent_y:
                pred_y -= 1

            new_predator_pos = (pred_x, pred_y)

        # Check if reached food
        reached_food = (new_agent_pos == state.food_pos)
        if reached_food:
            utility += self.utility_weights['reach_food']
            reasons.append(f"{depth}스텝: 음식 도달!")
        else:
            # Food proximity
            old_food_dist = self._distance(state.agent_pos, state.food_pos)
            new_food_dist = self._distance(new_agent_pos, state.food_pos)
            delta_food = old_food_dist - new_food_dist
            if delta_food > 0:
                utility += self.utility_weights['food_proximity'] * delta_food

        # Check if reached predator (danger!)
        reached_predator = new_predator_pos and (new_agent_pos == new_predator_pos)
        if reached_predator:
            utility += self.utility_weights['reach_predator']
            reasons.append(f"{depth}스텝: 포식자 충돌!")
        elif new_predator_pos:
            # Predator proximity
            old_pred_dist = self._distance(state.agent_pos, state.predator_pos) if state.predator_pos else 99
            new_pred_dist = self._distance(new_agent_pos, new_predator_pos)
            delta_pred = new_pred_dist - old_pred_dist

            if delta_pred > 0:
                # Getting farther from predator is good
                utility += abs(delta_pred) * 0.5
            elif delta_pred < 0:
                # Getting closer is bad (more penalty if closer)
                danger_factor = min(1.0, 3.0 / max(1, new_pred_dist))
                utility += self.utility_weights['predator_proximity'] * abs(delta_pred) * danger_factor
                if new_pred_dist <= 2:
                    reasons.append(f"{depth}스텝: 포식자 접근 위험")

        # Apply goal biases if provided
        if goal_biases:
            for key, weight in goal_biases.items():
                if key == 'movement_cost' and weight != 0:
                    utility += weight

        # Apply memory influence (with decay for deeper steps)
        if memory_fn:
            memory_decay = 0.7 ** (depth - 1)  # Memory influence decreases with depth
            mem_influence = memory_fn(new_agent_pos, action)
            utility += mem_influence * memory_decay

        # Calculate new state values
        new_energy = state.energy - 0.01  # Small energy cost per step
        if reached_food:
            new_energy = min(1.0, new_energy + 0.3)

        new_safety = state.safety
        if new_predator_pos:
            pred_dist = self._distance(new_agent_pos, new_predator_pos)
            if pred_dist <= 3:
                new_safety = max(0.0, new_safety - 0.2)
            elif pred_dist > 5:
                new_safety = min(1.0, new_safety + 0.1)

        new_state = RolloutState(
            agent_pos=new_agent_pos,
            predator_pos=new_predator_pos,
            food_pos=state.food_pos,  # Food doesn't move (simplification)
            energy=new_energy,
            safety=new_safety
        )

        return new_state, utility, reasons

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_visualization_data(self) -> Dict:
        """Get data for frontend visualization."""
        return {
            'depth': self.depth,
            'gamma': self.gamma,
            'paths_evaluated': self.paths_evaluated,
            'best_path_depth': self.best_path_depth
        }
