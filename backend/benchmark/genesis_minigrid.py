"""
Genesis Brain on MiniGrid - Circuit-based Learning
===================================================
Connect Genesis Brain's modular circuits to MiniGrid environments.

NO LSTM. NO end-to-end black box.
Just modular circuits with learnable parameters.

Core circuits:
1. Risk Estimation (TTC, proximity-based danger)
2. Defense Mode (hysteresis-based mode switching)
3. Predictive Control (transition model learning)
4. Memory (episode-spanning experience)
5. G(a) Computation (FEP-based action selection)
"""

import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time
import re

# MiniGrid object indices
OBJ_EMPTY = 1
OBJ_WALL = 2
OBJ_DOOR = 4
OBJ_KEY = 5
OBJ_BALL = 6
OBJ_BOX = 7
OBJ_GOAL = 8
OBJ_LAVA = 9
OBJ_AGENT = 10


# =============================================================================
# GENESIS CIRCUITS (Learnable Parameters)
# =============================================================================

@dataclass
class RiskCircuit:
    """
    Risk Estimation Circuit.

    Estimates danger level based on proximity and approach.
    Parameters adapt based on experience.
    """
    # Learnable parameters
    ttc_threshold: float = 3.0       # Time-to-collision threshold
    danger_radius: float = 2.0       # Distance considered dangerous
    risk_sensitivity: float = 1.0    # How sensitive to risk

    # Internal state
    risk_raw: float = 0.0
    risk_filtered: float = 0.0       # EMA filtered
    approach_streak: int = 0
    prev_danger_dist: float = 999.0

    # Learning statistics
    false_alarms: int = 0            # Triggered defense but no hit
    missed_dangers: int = 0          # Hit without defense

    def update(self, danger_dist: float, was_hit: bool, was_in_defense: bool):
        """Update risk estimate and learn from outcome."""
        # Track approach
        closing = self.prev_danger_dist - danger_dist
        if closing > 0.1:
            self.approach_streak += 1
        else:
            self.approach_streak = max(0, self.approach_streak - 1)
        self.prev_danger_dist = danger_dist

        # Compute raw risk
        if danger_dist < self.danger_radius:
            self.risk_raw = (self.danger_radius - danger_dist) / self.danger_radius
        else:
            self.risk_raw = 0.0

        # Apply sensitivity
        self.risk_raw *= self.risk_sensitivity

        # EMA filter
        self.risk_filtered = 0.7 * self.risk_filtered + 0.3 * self.risk_raw

        # Learn from outcome
        if was_hit:
            if not was_in_defense:
                self.missed_dangers += 1
                # Increase sensitivity
                self.risk_sensitivity = min(2.0, self.risk_sensitivity + 0.1)
                self.danger_radius = min(4.0, self.danger_radius + 0.1)
        else:
            if was_in_defense and danger_dist > self.danger_radius * 1.5:
                self.false_alarms += 1
                # Decrease sensitivity (but slowly)
                self.risk_sensitivity = max(0.5, self.risk_sensitivity - 0.02)

    def reset_episode(self):
        """Reset per-episode state."""
        self.risk_raw = 0.0
        self.risk_filtered = 0.0
        self.approach_streak = 0
        self.prev_danger_dist = 999.0


@dataclass
class DefenseCircuit:
    """
    Defense Mode Circuit with Hysteresis.

    Switches between foraging and defense modes.
    Hysteresis prevents flickering.
    """
    # Learnable parameters
    threshold_on: float = 0.4        # Risk level to enter defense
    threshold_off: float = 0.2       # Risk level to exit defense
    min_defense_steps: int = 3       # Minimum steps in defense mode

    # Internal state
    in_defense: bool = False
    defense_step_count: int = 0

    # Learning statistics
    mode_switches: int = 0
    defense_time: int = 0

    def update(self, risk: float, approach_streak: int) -> bool:
        """Update defense mode and return current state."""
        if self.in_defense:
            self.defense_step_count += 1
            self.defense_time += 1

            # Exit defense?
            if (risk < self.threshold_off and
                self.defense_step_count >= self.min_defense_steps):
                self.in_defense = False
                self.defense_step_count = 0
                self.mode_switches += 1
        else:
            # Enter defense?
            if risk > self.threshold_on or approach_streak >= 2:
                self.in_defense = True
                self.defense_step_count = 0
                self.mode_switches += 1

        return self.in_defense

    def learn_from_hit(self):
        """Adjust thresholds after being hit."""
        # Lower threshold = more cautious
        self.threshold_on = max(0.2, self.threshold_on - 0.03)
        self.threshold_off = max(0.1, self.threshold_off - 0.02)

    def learn_from_success(self):
        """Slightly relax thresholds after successful foraging."""
        self.threshold_on = min(0.6, self.threshold_on + 0.005)
        self.threshold_off = min(0.3, self.threshold_off + 0.003)

    def reset_episode(self):
        """Reset per-episode state."""
        self.in_defense = False
        self.defense_step_count = 0


@dataclass
class TransitionCircuit:
    """
    Transition Model Learning.

    Learns Q(s, a) from experience using position-based states.
    State = (agent_pos, goal_dir, agent_direction)
    """
    grid_size: int = 10
    n_actions: int = 7

    # Q-table: [x, y, direction, goal_dir, action] -> value
    Q: np.ndarray = None

    # Visit counts for uncertainty
    visits: np.ndarray = None

    # Learning parameters
    lr: float = 0.2
    gamma: float = 0.95

    def __post_init__(self):
        # State: (x, y, direction, goal_dir) x action
        # direction: 0-3, goal_dir: 0-8
        self.Q = np.zeros((self.grid_size, self.grid_size, 4, 9, self.n_actions))
        self.visits = np.ones((self.grid_size, self.grid_size, 4, 9, self.n_actions)) * 0.1

    def get_state(self, agent_pos: np.ndarray, direction: int, goal_dir: int) -> Tuple:
        """Get state tuple."""
        x = int(np.clip(agent_pos[0], 0, self.grid_size - 1))
        y = int(np.clip(agent_pos[1], 0, self.grid_size - 1))
        d = int(direction) % 4
        gd = int(goal_dir) % 9
        return (x, y, d, gd)

    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple, done: bool):
        """Update Q-value using TD learning."""
        x, y, d, gd = state
        nx, ny, nd, ngd = next_state

        # Best next Q
        if done:
            best_next = 0
        else:
            best_next = np.max(self.Q[nx, ny, nd, ngd])

        # TD update
        target = reward + self.gamma * best_next
        error = target - self.Q[x, y, d, gd, action]
        self.Q[x, y, d, gd, action] += self.lr * error

        # Update visit count
        self.visits[x, y, d, gd, action] += 1

    def get_Q_values(self, state: Tuple) -> np.ndarray:
        """Get Q(s, a) for all actions."""
        x, y, d, gd = state
        return self.Q[x, y, d, gd].copy()

    def get_uncertainty(self, state: Tuple, action: int) -> float:
        """Get uncertainty based on visit count."""
        x, y, d, gd = state
        visits = self.visits[x, y, d, gd, action]
        return 1.0 / np.sqrt(visits + 1)


@dataclass
class MemoryCircuit:
    """
    Long-term Memory.

    Stores experiences across episodes.
    Used to bias G(a) based on past outcomes.
    """
    # Spatial memory
    danger_map: np.ndarray = None   # Where we got hit
    reward_map: np.ndarray = None   # Where we got rewards
    visit_map: np.ndarray = None    # How often we visited

    # Episode memory
    episode_outcomes: List[Dict] = field(default_factory=list)

    grid_size: int = 20  # Max grid size

    def __post_init__(self):
        self.danger_map = np.zeros((self.grid_size, self.grid_size))
        self.reward_map = np.zeros((self.grid_size, self.grid_size))
        self.visit_map = np.ones((self.grid_size, self.grid_size)) * 0.1

    def record_visit(self, x: int, y: int):
        """Record a visit to location."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.visit_map[x, y] += 1

    def record_danger(self, x: int, y: int):
        """Record danger at location."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.danger_map[x, y] += 1

    def record_reward(self, x: int, y: int, reward: float):
        """Record reward at location."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.reward_map[x, y] += reward

    def get_location_value(self, x: int, y: int) -> float:
        """Get learned value of a location."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            visits = self.visit_map[x, y]
            danger = self.danger_map[x, y] / visits
            reward = self.reward_map[x, y] / visits
            return reward - danger * 0.5
        return 0.0

    def record_episode(self, steps: int, reward: float, success: bool):
        """Record episode outcome."""
        self.episode_outcomes.append({
            'steps': steps,
            'reward': reward,
            'success': success
        })

        # Decay old memories slowly
        self.danger_map *= 0.99
        self.reward_map *= 0.99

    def get_success_rate(self, last_n: int = 20) -> float:
        """Get recent success rate."""
        if not self.episode_outcomes:
            return 0.0
        recent = self.episode_outcomes[-last_n:]
        return sum(e['success'] for e in recent) / len(recent)


@dataclass
class CuriosityCircuit:
    """
    Curiosity Circuit - Dopaminergic Novelty-Seeking.

    Mimics the dopamine system's response to novelty and prediction errors.
    Provides intrinsic motivation to explore unfamiliar states.

    Key mechanisms:
    1. State prediction model - predicts next state given action
    2. Prediction error → Curiosity signal (dopamine burst)
    3. Novelty bonus → G(a) reduction for unexplored areas
    4. Habituation → Familiar states become less interesting

    Philosophy: "The brain seeks information, not just reward."
    """
    grid_size: int = 10
    n_actions: int = 7

    # State visitation counts (novelty detection)
    state_counts: np.ndarray = None

    # Transition prediction model (simple: just count-based)
    # transition_counts[x, y, dir, action, next_x, next_y] = count
    # Simplified: just track if we know the outcome
    known_transitions: np.ndarray = None

    # Curiosity parameters
    novelty_weight: float = 0.5      # How much to value novelty
    habituation_rate: float = 0.1   # How fast novelty decreases
    prediction_lr: float = 0.3      # Learning rate for predictions

    # Tracking
    curiosity_signal: float = 0.0   # Current curiosity level
    total_surprises: int = 0        # Count of surprising events

    def __post_init__(self):
        # State visit counts: [x, y, direction]
        self.state_counts = np.zeros((self.grid_size, self.grid_size, 4))

        # Known transitions: [x, y, direction, action] -> bool (do we know outcome?)
        self.known_transitions = np.zeros((self.grid_size, self.grid_size, 4, self.n_actions))

        # Prediction errors for each state-action
        self.prediction_errors = np.zeros((self.grid_size, self.grid_size, 4, self.n_actions))

    def reset_episode(self):
        """Reset per-episode state (but keep learned knowledge)."""
        self.curiosity_signal = 0.0

    def get_novelty_bonus(self, x: int, y: int, direction: int) -> float:
        """
        Get novelty bonus for a state.
        Novel (unvisited) states have high bonus.
        Familiar states have low bonus (habituation).
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0.0

        visits = self.state_counts[x, y, direction]

        # Novelty = 1 / (1 + visits)
        # First visit: novelty = 1.0, after 10 visits: novelty = 0.09
        novelty = 1.0 / (1.0 + visits * self.habituation_rate)

        return novelty * self.novelty_weight

    def get_action_curiosity(self, x: int, y: int, direction: int, action: int) -> float:
        """
        Get curiosity bonus for taking an action from current state.
        Unknown outcomes are more curious.
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0.0

        # How many times have we done this action from this state?
        known = self.known_transitions[x, y, direction, action]

        if known < 1:
            return self.novelty_weight  # Maximum curiosity for unknown
        elif known < 5:
            return self.novelty_weight * 0.5  # Still somewhat curious
        else:
            return 0.0  # Well-known, no curiosity bonus

    def record_transition(self, x: int, y: int, direction: int, action: int,
                          next_x: int, next_y: int, next_dir: int):
        """
        Record a state transition and update prediction model.
        Compute prediction error (surprise).
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return

        # Update visit count
        self.state_counts[x, y, direction] += 1

        # Update known transitions
        prev_known = self.known_transitions[x, y, direction, action]
        self.known_transitions[x, y, direction, action] += 1

        # Compute surprise (prediction error)
        if prev_known < 1:
            # First time seeing this transition - maximum surprise
            surprise = 1.0
            self.total_surprises += 1
        else:
            # We've seen this before - less surprising
            surprise = 1.0 / (1.0 + prev_known)

        # Update prediction error EMA
        self.prediction_errors[x, y, direction, action] = (
            0.7 * self.prediction_errors[x, y, direction, action] +
            0.3 * surprise
        )

        # Update curiosity signal (global)
        self.curiosity_signal = 0.8 * self.curiosity_signal + 0.2 * surprise

    def get_stats(self) -> Dict:
        """Get curiosity circuit statistics."""
        total_states = np.sum(self.state_counts > 0)
        total_transitions = np.sum(self.known_transitions > 0)
        avg_pred_error = np.mean(self.prediction_errors[self.known_transitions > 0]) if total_transitions > 0 else 0

        return {
            'explored_states': int(total_states),
            'known_transitions': int(total_transitions),
            'avg_prediction_error': avg_pred_error,
            'total_surprises': self.total_surprises,
            'curiosity_signal': self.curiosity_signal,
        }


@dataclass
class MetaCognitionCircuit:
    """
    MetaCognition Circuit - Prefrontal Cortex Abstraction.

    KEY DESIGN PRINCIPLE:
    Learn from STATE TRANSITION EVENTS, not object detection.

    Events tracked:
    - KEY_ACQUIRED: carrying_key: False → True
    - DOOR_UNLOCKED: door_is_open: False → True (with key)
    - DOOR_BLOCKED: tried toggle without key (implied by failure pattern)

    Rules learned:
    - "Key enables Door" = P(DOOR_UNLOCKED | KEY_ACQUIRED before)
    - "Door blocks Path" = P(success | DOOR_UNLOCKED)

    Philosophy: "World laws emerge from causal state transitions"
    """
    # Abstract rule confidence (learned from state transitions)
    rule_key_enables_door: float = 0.0   # "Having key → can open door"
    rule_door_blocks_path: float = 0.0   # "Door open → can reach goal"

    # Event sequence tracking (per episode)
    event_log: List[Tuple[int, str]] = field(default_factory=list)  # [(step, event_name), ...]

    # Previous state (for transition detection)
    prev_carrying_key: bool = False
    prev_door_open: bool = False

    # Episode pattern memory
    episode_patterns: List[Dict] = field(default_factory=list)

    # Learning parameters
    rule_lr: float = 0.15  # Slightly higher for faster learning

    # Statistics
    key_door_sequences: int = 0  # Times KEY_ACQUIRED → DOOR_UNLOCKED observed
    door_without_key_attempts: int = 0  # Times tried door without key

    def reset_episode(self):
        """Reset per-episode tracking."""
        self.event_log = []
        self.prev_carrying_key = False
        self.prev_door_open = False

    def record_state_transition(self, step: int,
                                 carrying_key: bool, door_is_open: bool,
                                 action: int):
        """
        Record state transitions and detect events.

        This is the KEY method - we track CHANGES in state, not static observations.
        """
        events = []

        # Detect KEY_ACQUIRED event
        if carrying_key and not self.prev_carrying_key:
            events.append('KEY_ACQUIRED')

        # Detect DOOR_UNLOCKED event
        if door_is_open and not self.prev_door_open:
            if self.prev_carrying_key or carrying_key:
                events.append('DOOR_UNLOCKED')
            else:
                # Door opened without key? (shouldn't happen in DoorKey)
                events.append('DOOR_UNLOCKED_UNEXPECTED')

        # Detect DOOR_BLOCKED (action=5 Toggle, but door didn't open, and no key)
        if action == 5 and not door_is_open and not self.prev_door_open and not carrying_key:
            events.append('DOOR_BLOCKED')
            self.door_without_key_attempts += 1

        # Log events
        for event in events:
            self.event_log.append((step, event))

        # Update previous state
        self.prev_carrying_key = carrying_key
        self.prev_door_open = door_is_open

        return events

    def record_episode_end(self, success: bool, total_steps: int):
        """
        Learn causal rules from the event sequence.

        KEY INSIGHT: Rules are learned from temporal ordering of events.
        """
        # Extract event names in order
        event_names = [e[1] for e in self.event_log]

        # Check for KEY_ACQUIRED → DOOR_UNLOCKED sequence
        key_idx = None
        door_idx = None
        for i, name in enumerate(event_names):
            if name == 'KEY_ACQUIRED' and key_idx is None:
                key_idx = i
            if name == 'DOOR_UNLOCKED' and door_idx is None:
                door_idx = i

        # Pattern analysis
        had_key_event = key_idx is not None
        had_door_event = door_idx is not None
        key_before_door = (key_idx is not None and door_idx is not None and key_idx < door_idx)
        had_door_blocked = 'DOOR_BLOCKED' in event_names

        # Store pattern
        pattern = {
            'success': success,
            'steps': total_steps,
            'events': event_names,
            'key_before_door': key_before_door,
            'had_door_blocked': had_door_blocked,
        }
        self.episode_patterns.append(pattern)

        # ============================================================
        # RULE LEARNING from state transition patterns
        # ============================================================

        # Rule: "Key enables Door"
        # Strengthen when: KEY_ACQUIRED before DOOR_UNLOCKED in success
        # Also strengthen when: DOOR_BLOCKED happened (evidence that key is needed)
        if success and key_before_door:
            # Strong evidence: succeeded with key→door sequence
            self.rule_key_enables_door += self.rule_lr * (1.0 - self.rule_key_enables_door)
            self.key_door_sequences += 1

        if had_door_blocked:
            # Evidence from failure: tried door without key
            self.rule_key_enables_door += self.rule_lr * 0.5 * (1.0 - self.rule_key_enables_door)

        # Rule: "Door blocks Path"
        # Strengthen when: DOOR_UNLOCKED happened in success
        if success and had_door_event:
            self.rule_door_blocks_path += self.rule_lr * (1.0 - self.rule_door_blocks_path)

        # Keep only last 100 patterns
        if len(self.episode_patterns) > 100:
            self.episode_patterns = self.episode_patterns[-100:]

    def get_strategy_bias(self, has_key: bool, door_open: bool) -> Dict[str, float]:
        """
        Get action biases based on learned causal rules.

        Returns dict of action_name -> bias (negative = encourage)
        """
        biases = {}

        # If we've learned "key enables door" rule
        if self.rule_key_enables_door > 0.3:
            if not has_key:
                # Prioritize getting key first
                biases['seek_key'] = -self.rule_key_enables_door * 0.5
                # Deprioritize door interaction until we have key
                biases['avoid_door'] = self.rule_key_enables_door * 0.3

        # If we've learned "door blocks path" rule
        if self.rule_door_blocks_path > 0.3:
            if has_key and not door_open:
                # Prioritize opening door
                biases['open_door'] = -self.rule_door_blocks_path * 0.5

        return biases

    def get_stats(self) -> Dict:
        """Get metacognition statistics."""
        n_patterns = len(self.episode_patterns)
        if n_patterns > 0:
            recent = self.episode_patterns[-20:]
            success_rate = sum(1 for p in recent if p['success']) / len(recent)
            key_door_rate = sum(1 for p in recent if p.get('key_before_door', False)) / len(recent)
        else:
            success_rate = 0
            key_door_rate = 0

        return {
            'rule_key_enables_door': self.rule_key_enables_door,
            'rule_door_blocks_path': self.rule_door_blocks_path,
            'key_door_sequences': self.key_door_sequences,
            'door_blocked_attempts': self.door_without_key_attempts,
            'pattern_count': n_patterns,
            'recent_success_rate': success_rate,
            'recent_key_door_rate': key_door_rate,
        }


@dataclass
class InteroceptionCircuit:
    """
    Interoception Circuit - Internal State Awareness.

    Tracks the agent's internal physiological states:
    - energy: Depletes over time, restored by reaching goals
    - pain: Increases from wall hits and danger proximity
    - time_pressure: Urgency based on remaining time

    Homeostasis via P(o) = Beta distributions:
    - energy: Beta(3,2) → setpoint ~0.6 (want moderate-high energy)
    - pain: Beta(1,5) → setpoint ~0.0 (want minimal pain)

    Philosophy: "The body's needs shape the mind's priorities."
    Internal states → G(a) modulation, not direct action selection.
    """
    # Internal states (0-1 range)
    energy: float = 1.0              # Starts full, depletes over time
    pain: float = 0.0                # Starts zero, increases from damage
    time_pressure: float = 0.0       # Increases as episode progresses

    # Homeostatic setpoints (mode of Beta distributions)
    energy_setpoint: float = 0.6     # Beta(3,2) mode
    pain_setpoint: float = 0.0       # Beta(1,5) mode

    # Depletion/recovery rates
    energy_decay: float = 0.003      # Per-step energy loss
    pain_decay: float = 0.05         # Natural pain recovery
    pain_from_wall: float = 0.15     # Pain from hitting walls
    pain_from_danger: float = 0.3    # Pain from danger proximity

    # Homeostatic drive (deviation from setpoint)
    energy_drive: float = 0.0        # Urgency to restore energy
    pain_drive: float = 0.0          # Urgency to reduce pain

    # Episode tracking
    max_steps: int = 100             # For time pressure calculation
    step_count: int = 0

    # Learning statistics
    energy_crises: int = 0           # Times energy dropped critically low
    pain_events: int = 0             # Times pain spiked

    def reset_episode(self, max_steps: int = 100):
        """Reset internal states for new episode."""
        self.energy = 1.0
        self.pain = 0.0
        self.time_pressure = 0.0
        self.energy_drive = 0.0
        self.pain_drive = 0.0
        self.max_steps = max_steps
        self.step_count = 0

    def update(self, wall_hit: bool, danger_dist: float, got_reward: bool):
        """
        Update internal states based on events.

        Args:
            wall_hit: Whether agent hit a wall this step
            danger_dist: Distance to nearest danger
            got_reward: Whether agent got positive reward (reaching goal)
        """
        self.step_count += 1

        # 1. Energy dynamics
        # Natural depletion
        self.energy = max(0.0, self.energy - self.energy_decay)

        # Reward restores energy (reaching goal = feeding)
        if got_reward:
            self.energy = min(1.0, self.energy + 0.5)

        # Track crisis
        if self.energy < 0.2:
            self.energy_crises += 1

        # 2. Pain dynamics
        # Natural recovery
        self.pain = max(0.0, self.pain - self.pain_decay)

        # Pain from wall hits
        if wall_hit:
            self.pain = min(1.0, self.pain + self.pain_from_wall)
            self.pain_events += 1

        # Pain from danger proximity
        if danger_dist < 2.0:
            danger_pain = self.pain_from_danger * (2.0 - danger_dist) / 2.0
            self.pain = min(1.0, self.pain + danger_pain)
            self.pain_events += 1

        # 3. Time pressure
        self.time_pressure = self.step_count / max(1, self.max_steps)

        # 4. Compute homeostatic drives (KL-like deviation from setpoint)
        # Energy drive: higher when energy is below setpoint
        self.energy_drive = max(0.0, self.energy_setpoint - self.energy)

        # Pain drive: higher when pain is above setpoint
        self.pain_drive = max(0.0, self.pain - self.pain_setpoint)

    def get_G_modulation(self) -> Dict[str, float]:
        """
        Get modulation values for G(a) computation.

        Returns action biases based on internal states:
        - Low energy → increase goal urgency
        - High pain → increase caution
        - High time pressure → decrease exploration
        """
        modulations = {}

        # 1. Goal urgency from low energy
        # When starving, become more goal-directed
        modulations['goal_urgency'] = self.energy_drive * 1.5

        # 2. Caution from pain
        # When in pain, become more risk-averse
        modulations['risk_sensitivity'] = 1.0 + self.pain_drive * 0.5

        # 3. Time pressure effects
        # Late in episode: more focused, less exploratory
        if self.time_pressure > 0.5:
            modulations['exploration_penalty'] = (self.time_pressure - 0.5) * 0.5
        else:
            modulations['exploration_penalty'] = 0.0

        # 4. Desperation mode: very low energy AND high time pressure
        if self.energy < 0.3 and self.time_pressure > 0.7:
            modulations['desperation'] = 1.0  # Take bigger risks
        else:
            modulations['desperation'] = 0.0

        return modulations

    def get_stats(self) -> Dict:
        """Get interoception statistics."""
        return {
            'energy': self.energy,
            'pain': self.pain,
            'time_pressure': self.time_pressure,
            'energy_drive': self.energy_drive,
            'pain_drive': self.pain_drive,
            'energy_crises': self.energy_crises,
            'pain_events': self.pain_events,
        }


@dataclass
class WorkingMemoryCircuit:
    """
    Working Memory Circuit - Prefrontal Buffer.

    Maintains short-term memory for:
    1. Recent observations (temporal buffer)
    2. Object locations (spatial working memory)
    3. Action history (motor memory)

    Critical for partial observation where agent can't see everything.
    Mimics prefrontal cortex working memory capacity (~4-7 items).

    Philosophy: "Hold the plan in mind while executing."
    """
    # Capacity limits (based on human working memory)
    capacity: int = 7                # Maximum items to hold
    temporal_window: int = 10        # Steps to remember

    # Memory buffers
    observations: List[Dict] = field(default_factory=list)   # Recent observations
    object_memory: Dict[str, Tuple] = field(default_factory=dict)  # Object → (pos, step_seen)
    action_history: List[int] = field(default_factory=list)  # Recent actions

    # Decay parameters
    position_decay: float = 0.9      # How fast position certainty decays
    memory_confidence: Dict[str, float] = field(default_factory=dict)  # Object → confidence

    # Episode tracking
    current_step: int = 0

    def reset_episode(self):
        """Reset working memory for new episode."""
        self.observations = []
        self.object_memory = {}
        self.action_history = []
        self.memory_confidence = {}
        self.current_step = 0

    def update(self, obs_info: Dict, action: int):
        """
        Update working memory with new observation.

        Args:
            obs_info: Extracted observation info (agent_pos, key_pos, door_pos, etc.)
            action: Action just taken
        """
        self.current_step += 1

        # 1. Store observation (with capacity limit)
        self.observations.append({
            'step': self.current_step,
            'agent_pos': tuple(obs_info['agent_pos']),
            'direction': obs_info['direction'],
        })
        if len(self.observations) > self.temporal_window:
            self.observations.pop(0)

        # 2. Update object memory (spatial working memory)
        # If we can see an object, update its position with high confidence
        objects_to_track = ['key_pos', 'door_pos', 'goal_pos', 'danger_pos']
        for obj_name in objects_to_track:
            if obj_name in obs_info and obs_info[obj_name] is not None:
                pos = tuple(obs_info[obj_name])
                self.object_memory[obj_name] = (pos, self.current_step)
                self.memory_confidence[obj_name] = 1.0  # Just seen = full confidence

        # Decay confidence for objects not seen this step
        for obj_name in list(self.memory_confidence.keys()):
            if obj_name not in obs_info or obs_info[obj_name] is None:
                self.memory_confidence[obj_name] *= self.position_decay
                # Remove if confidence too low
                if self.memory_confidence[obj_name] < 0.1:
                    del self.memory_confidence[obj_name]
                    if obj_name in self.object_memory:
                        del self.object_memory[obj_name]

        # 3. Store action history
        self.action_history.append(action)
        if len(self.action_history) > self.temporal_window:
            self.action_history.pop(0)

    def recall_object(self, obj_name: str) -> Optional[Tuple[Tuple, float]]:
        """
        Recall object position from working memory.

        Returns:
            (position, confidence) or None if not remembered
        """
        if obj_name in self.object_memory and obj_name in self.memory_confidence:
            pos, last_seen = self.object_memory[obj_name]
            confidence = self.memory_confidence[obj_name]
            return (pos, confidence)
        return None

    def get_remembered_objects(self) -> Dict[str, Tuple[Tuple, float]]:
        """Get all remembered objects with their positions and confidence."""
        result = {}
        for obj_name in self.object_memory:
            recall = self.recall_object(obj_name)
            if recall is not None:
                result[obj_name] = recall
        return result

    def get_action_pattern(self) -> Dict[str, int]:
        """Analyze recent action patterns (e.g., stuck detection)."""
        if not self.action_history:
            return {'dominant': -1, 'variety': 0}

        from collections import Counter
        counts = Counter(self.action_history)
        most_common = counts.most_common(1)[0]

        return {
            'dominant': most_common[0],      # Most frequent action
            'dominant_ratio': most_common[1] / len(self.action_history),  # How dominant
            'variety': len(counts),          # Number of different actions
        }

    def is_repeating(self) -> bool:
        """Check if agent is in a repetitive loop."""
        if len(self.action_history) < 6:
            return False

        # Check for 2-action loop (e.g., left-right-left-right)
        recent = self.action_history[-6:]
        if recent[0] == recent[2] == recent[4] and recent[1] == recent[3] == recent[5]:
            return True

        return False

    def get_stats(self) -> Dict:
        """Get working memory statistics."""
        return {
            'objects_remembered': len(self.object_memory),
            'avg_confidence': np.mean(list(self.memory_confidence.values())) if self.memory_confidence else 0,
            'is_repeating': self.is_repeating(),
            'action_variety': len(set(self.action_history)) if self.action_history else 0,
        }


@dataclass
class PlannerCircuit:
    """
    Explicit Planner Circuit - Prefrontal Executive Control.

    Generates hierarchical step-by-step plans.
    PC-Z controls when to engage planning (not always active).

    Key insight from user's roadmap:
    "Planner가 있되, 언제 작동할지는 PC-Z가 결정"

    Plan format: List of sub-goals with preconditions
    Example: [(GET_KEY, None), (OPEN_DOOR, HAS_KEY), (REACH_GOAL, DOOR_OPEN)]

    Philosophy: "Know when to think vs when to act reflexively."
    """
    # Plan states
    IDLE = 0           # Not currently planning
    PLANNING = 1       # Generating plan
    EXECUTING = 2      # Following plan

    # Sub-goal types
    GET_KEY = 'GET_KEY'
    OPEN_DOOR = 'OPEN_DOOR'
    REACH_GOAL = 'REACH_GOAL'

    # Current state
    state: int = 0                    # IDLE
    current_plan: List[str] = field(default_factory=list)
    current_step: int = 0             # Index in plan

    # Activation thresholds (PC-Z controlled)
    activation_threshold: float = 0.4  # Stuck score above this → plan
    deactivation_threshold: float = 0.2  # Progress below this → stop planning

    # Plan execution tracking
    plans_generated: int = 0
    plans_completed: int = 0
    plan_steps_executed: int = 0

    # Learning: which plans work?
    plan_success_rate: float = 0.5

    def reset_episode(self):
        """Reset planning state for new episode."""
        self.state = self.IDLE
        self.current_plan = []
        self.current_step = 0

    def should_activate(self, stuck_score: float, uncertainty: float) -> bool:
        """
        Determine if planner should activate.
        Called by PC-Z when agent is struggling.

        Args:
            stuck_score: How stuck the agent is (0-1)
            uncertainty: Current action uncertainty
        """
        if self.state == self.IDLE:
            # Activate if stuck enough
            combined = stuck_score * 0.7 + uncertainty * 0.3
            return combined > self.activation_threshold
        return False

    def generate_plan(self, has_key: bool, door_open: bool) -> List[str]:
        """
        Generate a plan based on current state.

        This is NOT search-based planning. It's template-based:
        The agent knows the structure of DoorKey environments.
        """
        plan = []

        if not has_key:
            plan.append(self.GET_KEY)

        if not door_open:
            plan.append(self.OPEN_DOOR)

        plan.append(self.REACH_GOAL)

        self.current_plan = plan
        self.current_step = 0
        self.state = self.EXECUTING
        self.plans_generated += 1

        return plan

    def get_current_subgoal(self) -> Optional[str]:
        """Get the current sub-goal to pursue."""
        if self.state != self.EXECUTING or not self.current_plan:
            return None

        if self.current_step < len(self.current_plan):
            return self.current_plan[self.current_step]
        return None

    def update_progress(self, has_key: bool, door_open: bool, reached_goal: bool):
        """
        Update plan progress based on state changes.
        Advance to next step when current sub-goal is achieved.
        """
        if self.state != self.EXECUTING:
            return

        current = self.get_current_subgoal()

        if current == self.GET_KEY and has_key:
            self.current_step += 1
            self.plan_steps_executed += 1

        elif current == self.OPEN_DOOR and door_open:
            self.current_step += 1
            self.plan_steps_executed += 1

        elif current == self.REACH_GOAL and reached_goal:
            self.current_step += 1
            self.plan_steps_executed += 1
            self.plans_completed += 1
            self.state = self.IDLE  # Plan complete!

        # Check if plan is complete
        if self.current_step >= len(self.current_plan):
            self.state = self.IDLE

    def get_G_bias(self, has_key: bool, door_open: bool,
                   key_in_front: bool, door_in_front: bool) -> Dict[int, float]:
        """
        Get G(a) biases based on current plan step.

        Returns dict of action_idx -> bias (negative = encourage)
        """
        biases = {}
        current = self.get_current_subgoal()

        if current == self.GET_KEY:
            # Strong bias for pickup if key is in front
            if key_in_front:
                biases[3] = -2.0  # Pickup action

        elif current == self.OPEN_DOOR:
            # Strong bias for toggle if door is in front
            if door_in_front:
                biases[5] = -2.0  # Toggle action

        # Always slightly penalize Done action
        biases[6] = 1.0

        return biases

    def should_deactivate(self, progress_score: float) -> bool:
        """Check if planner should deactivate (things are going well)."""
        if self.state == self.EXECUTING:
            return progress_score < self.deactivation_threshold
        return False

    def get_stats(self) -> Dict:
        """Get planner statistics."""
        return {
            'state': ['IDLE', 'PLANNING', 'EXECUTING'][self.state],
            'current_step': self.current_step,
            'plan_length': len(self.current_plan),
            'plans_generated': self.plans_generated,
            'plans_completed': self.plans_completed,
            'completion_rate': self.plans_completed / max(1, self.plans_generated),
        }


@dataclass
class NarrativeCircuit:
    """
    Narrative Self Circuit - Autobiographical Memory.

    Creates episodic summaries and tracks patterns across episodes.
    This is the agent's "sense of self" - a continuous story of who it is
    and what it has done.

    Key functions:
    1. Episode summarization (what happened, key events)
    2. Pattern extraction (recurring challenges, typical outcomes)
    3. Identity formation (am I good at this? what's my strategy?)
    4. Prospective memory (what should I remember for next time?)

    Philosophy: "I am the story I tell myself about my experiences."
    """
    # Episode narratives
    episode_summaries: List[Dict] = field(default_factory=list)

    # Identity markers (learned self-concept)
    average_success: float = 0.5          # Baseline expectation
    typical_steps: float = 50.0           # How long tasks usually take
    challenge_areas: Dict[str, int] = field(default_factory=dict)  # What causes trouble

    # Recurring patterns (cross-episode learning)
    key_acquisition_rate: float = 0.5     # How often do I get the key first?
    door_first_attempts: int = 0          # Times I tried door before key
    efficient_episodes: int = 0           # Episodes completed under average
    struggling_episodes: int = 0          # Episodes with many retries

    # Current episode narrative building
    current_events: List[str] = field(default_factory=list)
    current_challenges: List[str] = field(default_factory=list)

    # Prospective memory (learned tips for future)
    learned_tips: List[str] = field(default_factory=list)

    def reset_episode(self):
        """Reset per-episode narrative tracking."""
        self.current_events = []
        self.current_challenges = []

    def record_event(self, event: str, step: int):
        """Record significant event during episode."""
        self.current_events.append(f"Step {step}: {event}")

    def record_challenge(self, challenge: str, step: int):
        """Record a challenge or difficulty."""
        self.current_challenges.append(f"Step {step}: {challenge}")

        # Update challenge areas
        if challenge not in self.challenge_areas:
            self.challenge_areas[challenge] = 0
        self.challenge_areas[challenge] += 1

    def summarize_episode(self, success: bool, steps: int,
                          key_acquired: bool, door_opened: bool,
                          wall_hits: int, got_stuck: bool):
        """
        Create episode summary and update identity.

        This is the narrative moment - creating a story from events.
        """
        # Build summary
        summary = {
            'success': success,
            'steps': steps,
            'efficiency': 'efficient' if steps < self.typical_steps else 'slow',
            'key_event': 'got_key' if key_acquired else 'no_key',
            'door_event': 'opened_door' if door_opened else 'door_blocked',
            'challenges': len(self.current_challenges),
            'wall_hits': wall_hits,
            'got_stuck': got_stuck,
        }

        # Create narrative text (internal monologue)
        if success:
            if steps < self.typical_steps * 0.7:
                summary['narrative'] = "Quick success - I'm getting better at this"
                self.efficient_episodes += 1
            elif got_stuck:
                summary['narrative'] = "Struggled but succeeded - need to avoid getting stuck"
            else:
                summary['narrative'] = "Completed the task"
        else:
            if got_stuck:
                summary['narrative'] = "Failed due to getting stuck - need different approach"
                self.struggling_episodes += 1
            elif wall_hits > 5:
                summary['narrative'] = "Failed with many wall hits - need to navigate better"
            else:
                summary['narrative'] = "Failed - unclear why"

        # Store summary
        self.episode_summaries.append(summary)
        if len(self.episode_summaries) > 50:
            self.episode_summaries.pop(0)

        # Update identity (running averages)
        alpha = 0.1  # Learning rate for identity update
        self.average_success = (1 - alpha) * self.average_success + alpha * (1.0 if success else 0.0)
        self.typical_steps = (1 - alpha) * self.typical_steps + alpha * steps

        # Track key acquisition pattern
        if key_acquired and success:
            self.key_acquisition_rate = (1 - alpha) * self.key_acquisition_rate + alpha * 1.0

        # Generate prospective memory (tips for future)
        if success and steps < self.typical_steps * 0.5:
            tip = "Current strategy is working well"
            if tip not in self.learned_tips:
                self.learned_tips.append(tip)

        if got_stuck and len(self.learned_tips) < 5:
            tip = "Break loops early when stuck"
            if tip not in self.learned_tips:
                self.learned_tips.append(tip)

    def get_self_assessment(self) -> Dict[str, float]:
        """
        Get self-assessment for modulating behavior.

        Returns confidence/caution levels based on identity.
        """
        return {
            'confidence': self.average_success,  # Higher = more confident
            'expected_difficulty': 1.0 - self.average_success,  # Higher = expect struggle
            'efficiency_score': self.efficient_episodes / max(1, len(self.episode_summaries)),
        }

    def get_stats(self) -> Dict:
        """Get narrative circuit statistics."""
        recent = self.episode_summaries[-10:] if self.episode_summaries else []
        recent_success = sum(1 for s in recent if s['success']) / max(1, len(recent))

        return {
            'episodes_remembered': len(self.episode_summaries),
            'average_success': self.average_success,
            'typical_steps': self.typical_steps,
            'efficient_episodes': self.efficient_episodes,
            'struggling_episodes': self.struggling_episodes,
            'challenge_count': len(self.challenge_areas),
            'learned_tips': len(self.learned_tips),
            'recent_success': recent_success,
        }


class PCZBridge:
    """
    Predictive Coding - Z-State Bridge for MiniGrid.

    Simplified version of Genesis PC-Z dynamics.
    Computes internal "surprise/uncertainty" signals and uses them
    to modulate exploration vs exploitation.

    Z-States (simplified):
    - z=0: Stable - low surprise, exploit learned knowledge
    - z=1: Exploring - high surprise, increase exploration
    - z=2: Stuck - repeated failures, try different strategies

    PC Signals (adapted for MiniGrid):
    - epsilon_spike: Wall hits, failed actions
    - progress_error: Distance to goal not decreasing
    - action_margin: Difference between best and second-best G(a)
    """

    def __init__(self):
        """Initialize PC-Z bridge with instance variables."""
        # Z-state (0=stable, 1=exploring, 2=stuck)
        self.z_state = 0

        # PC signals
        self.epsilon_spike = 0.0       # Recent surprises (wall hits, failures)
        self.progress_error = 0.0      # Goal not getting closer
        self.action_margin = 1.0       # Decision confidence

        # Internal tracking
        self.prev_goal_dist = 999.0
        self.no_progress_count = 0
        self.wall_hit_count = 0
        self.step_count = 0

        # EMA smoothing
        self.epsilon_ema = 0.0
        self.progress_ema = 0.0

        # Z-state transition thresholds
        self.explore_threshold = 0.4   # epsilon > this → z=1
        self.stuck_threshold = 10      # no_progress > this → z=2
        self.stable_threshold = 0.2    # epsilon < this → z=0

        # Modulation outputs
        self.temperature_mult = 1.0    # Multiply base temperature
        self.exploration_mult = 1.0    # Multiply exploration bonus
        self.spatial_mult = 1.0        # Multiply spatial guidance weight
        self.subgoal_mult = 1.0        # Multiply sub-goal action bonus

        # Tracking for analysis
        self.z_state_history = []
        self.transitions = {'0→1': 0, '0→2': 0, '1→0': 0, '1→2': 0, '2→0': 0, '2→1': 0}

    def reset_episode(self):
        """Reset per-episode state."""
        self.prev_goal_dist = 999.0
        self.no_progress_count = 0
        self.wall_hit_count = 0
        self.step_count = 0
        # Don't reset z_state - it carries across episodes

    def update(self, goal_dist: float, wall_hit: bool, G_values: np.ndarray):
        """
        Update PC signals and Z-state.

        Args:
            goal_dist: Current distance to goal
            wall_hit: Whether agent hit a wall this step
            G_values: G(a) values for all actions
        """
        self.step_count += 1

        # 1. Compute epsilon_spike (surprise signal)
        spike = 0.0
        if wall_hit:
            spike = 1.0
            self.wall_hit_count += 1

        self.epsilon_ema = 0.8 * self.epsilon_ema + 0.2 * spike
        self.epsilon_spike = self.epsilon_ema

        # 2. Compute progress_error
        if goal_dist >= self.prev_goal_dist - 0.1:  # Not making progress
            self.no_progress_count += 1
            progress_err = min(1.0, self.no_progress_count / 20.0)
        else:
            self.no_progress_count = max(0, self.no_progress_count - 2)
            progress_err = 0.0

        self.progress_ema = 0.9 * self.progress_ema + 0.1 * progress_err
        self.progress_error = self.progress_ema
        self.prev_goal_dist = goal_dist

        # 3. Compute action_margin (decision confidence)
        if len(G_values) >= 2:
            sorted_G = np.sort(G_values)
            self.action_margin = abs(sorted_G[1] - sorted_G[0])  # Margin between best two
        else:
            self.action_margin = 1.0

        # 4. Update Z-state
        self._update_z_state()

        # 5. Compute modulation outputs
        self._compute_modulation()

    def _update_z_state(self):
        """Update Z-state based on PC signals."""
        combined_signal = self.epsilon_spike * 0.5 + self.progress_error * 0.5
        prev_z = self.z_state

        if self.z_state == 0:  # Currently stable
            if combined_signal > self.explore_threshold:
                self.z_state = 1  # → exploring
            elif self.no_progress_count > self.stuck_threshold:
                self.z_state = 2  # → stuck

        elif self.z_state == 1:  # Currently exploring
            if combined_signal < self.stable_threshold:
                self.z_state = 0  # → stable
            elif self.no_progress_count > self.stuck_threshold * 1.5:
                self.z_state = 2  # → stuck

        elif self.z_state == 2:  # Currently stuck
            if combined_signal < self.stable_threshold and self.no_progress_count < 5:
                self.z_state = 0  # → stable
            elif combined_signal < self.explore_threshold:
                self.z_state = 1  # → exploring

        # Track transitions
        if prev_z != self.z_state:
            key = f"{prev_z}→{self.z_state}"
            self.transitions[key] = self.transitions.get(key, 0) + 1

        # Record history (for analysis)
        self.z_state_history.append(self.z_state)

    def _compute_modulation(self):
        """
        Compute modulation outputs based on Z-state.

        Key insight: For DoorKey-style puzzles, being "stuck" should mean
        "focus harder on sub-goals", NOT "be more random".
        Random exploration doesn't help solve structured puzzles.
        """
        if self.z_state == 0:  # Stable: exploit
            self.temperature_mult = 0.8    # Lower temperature (more greedy)
            self.exploration_mult = 0.5    # Less exploration
            self.spatial_mult = 1.2        # Trust spatial guidance more
            self.subgoal_mult = 1.0        # Normal sub-goal weight

        elif self.z_state == 1:  # Exploring: try new things
            self.temperature_mult = 1.2    # Slightly higher temperature
            self.exploration_mult = 1.5    # More exploration
            self.spatial_mult = 0.8        # Trust spatial guidance less
            self.subgoal_mult = 1.2        # Slightly more sub-goal focused

        elif self.z_state == 2:  # Stuck: focus on sub-goals, not randomness
            # Key change: DON'T increase randomness
            # Instead, increase commitment to sub-goals
            self.temperature_mult = 0.6    # LOWER temp = more decisive
            self.exploration_mult = 0.3    # LESS random exploration
            self.spatial_mult = 0.3        # Ignore spatial (not helping)
            self.subgoal_mult = 2.0        # DOUBLE sub-goal weight!

    def get_state_name(self) -> str:
        """Get human-readable Z-state name."""
        return ['stable', 'exploring', 'stuck'][self.z_state]


@dataclass
class HippocampalCircuit:
    """
    Hippocampus-inspired Spatial Planning Circuit.

    Mimics key hippocampal functions:
    1. Place Cells - Location-specific activation
    2. Wall Memory - Remember blocked paths
    3. Goal Gradient - Diffuse goal signal through space
    4. Path Replay - Strengthen successful paths

    This is NOT A* or explicit path planning.
    Instead, it learns spatial "preferences" that guide navigation.
    """
    grid_size: int = 10

    # Place cell activations (familiarity with locations)
    place_cells: np.ndarray = None

    # Wall memory: which (pos, direction) pairs are blocked
    # Shape: [x, y, direction] -> blocked count
    wall_memory: np.ndarray = None

    # Goal gradient: learned "pull" toward goal from each position
    # Shape: [x, y] -> goal attractiveness (higher = closer to goal path)
    goal_gradient: np.ndarray = None

    # Successful path memory: positions visited during success
    success_paths: List[List[Tuple]] = field(default_factory=list)

    # Learning parameters
    wall_lr: float = 1.0       # How fast to learn wall locations (instant for random layouts)
    gradient_lr: float = 0.3   # How fast to update goal gradient
    path_decay: float = 0.95   # Path memory decay

    def __post_init__(self):
        self.place_cells = np.zeros((self.grid_size, self.grid_size))
        self.wall_memory = np.zeros((self.grid_size, self.grid_size, 4))
        self.goal_gradient = np.zeros((self.grid_size, self.grid_size))
        self.current_path = []

    def reset_episode(self, random_layout: bool = True):
        """Reset per-episode state.

        Args:
            random_layout: If True, reset all spatial memory (for random layouts)
        """
        self.current_path = []

        if random_layout:
            # For random layouts (like DoorKey), reset ALL spatial memory
            # because walls and goal paths will be in different positions
            self.wall_memory = np.zeros_like(self.wall_memory)
            self.goal_gradient = np.zeros_like(self.goal_gradient)
            # Note: success_paths is still kept for learning general patterns

    def record_position(self, x: int, y: int):
        """Record visit to position (place cell activation)."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.place_cells[x, y] += 1
            self.current_path.append((x, y))

    def record_wall_hit(self, x: int, y: int, direction: int):
        """
        Record that moving forward from (x,y) facing direction was blocked.
        This is hippocampal "wall memory" - learning spatial constraints.
        """
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # Increase wall count for this position-direction pair
            self.wall_memory[x, y, direction] = (
                (1 - self.wall_lr) * self.wall_memory[x, y, direction] +
                self.wall_lr * 1.0
            )

    def record_successful_path(self, goal_pos: Tuple[int, int]):
        """
        After reaching goal, strengthen goal gradient along path.
        This is like hippocampal "replay" during rest.
        """
        if not self.current_path:
            return

        # Store path for potential future use
        self.success_paths.append(self.current_path.copy())
        if len(self.success_paths) > 10:
            self.success_paths.pop(0)

        # Update goal gradient: positions closer to goal get higher values
        path_len = len(self.current_path)
        for i, (x, y) in enumerate(self.current_path):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # Earlier positions in path get lower gradient
                # Later positions (closer to goal) get higher
                progress = (i + 1) / path_len
                self.goal_gradient[x, y] = (
                    (1 - self.gradient_lr) * self.goal_gradient[x, y] +
                    self.gradient_lr * progress
                )

        # Goal position gets maximum gradient
        gx, gy = goal_pos
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            self.goal_gradient[gx, gy] = 1.0

    def get_wall_probability(self, x: int, y: int, direction: int) -> float:
        """Get learned probability that this direction is blocked."""
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.wall_memory[x, y, direction]
        return 0.0

    def get_direction_preference(self, x: int, y: int, direction: int) -> float:
        """
        Get preference for moving in a direction based on:
        1. Wall memory (avoid known walls)
        2. Goal gradient (prefer directions toward goal)
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0.0

        # Penalty for known walls
        wall_penalty = self.wall_memory[x, y, direction] * 2.0

        # Compute next position
        dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = dir_vecs[direction]
        nx, ny = x + dx, y + dy

        # Gradient bonus: prefer directions leading to higher gradient
        gradient_bonus = 0.0
        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            current_gradient = self.goal_gradient[x, y]
            next_gradient = self.goal_gradient[nx, ny]
            # Bonus if moving toward higher gradient (closer to goal)
            gradient_bonus = (next_gradient - current_gradient) * 1.5

        return gradient_bonus - wall_penalty

    def decay_memories(self):
        """Decay old memories to allow adaptation to new layouts."""
        self.wall_memory *= self.path_decay
        self.goal_gradient *= self.path_decay
        self.place_cells *= self.path_decay


# =============================================================================
# GENESIS BRAIN - INTEGRATED CIRCUITS
# =============================================================================

class GenesisBrain:
    """
    Genesis Brain for MiniGrid.

    Integrates all circuits to compute G(a) and select actions.
    """

    def __init__(self, n_actions: int = 7, grid_size: int = 10):
        self.n_actions = n_actions

        # Initialize circuits
        self.risk = RiskCircuit()
        self.defense = DefenseCircuit()
        self.transition = TransitionCircuit(n_actions=n_actions, grid_size=grid_size)
        self.memory = MemoryCircuit()
        self.hippocampus = HippocampalCircuit(grid_size=grid_size)  # Spatial planning
        self.curiosity = CuriosityCircuit(grid_size=grid_size, n_actions=n_actions)  # Novelty seeking
        self.metacog = MetaCognitionCircuit()  # Abstract rule learning
        self.interoception = InteroceptionCircuit()  # Internal state awareness
        self.working_memory = WorkingMemoryCircuit()  # Short-term buffer for partial obs
        self.planner = PlannerCircuit()  # Explicit planning (PC-Z controlled)
        self.narrative = NarrativeCircuit()  # Autobiographical memory / sense of self
        self.pcz = PCZBridge()  # PC-Z dynamics for adaptive exploration

        # Base action selection parameters (modulated by PC-Z and interoception)
        self.base_temperature = 0.25      # Softmax temperature
        self.goal_weight = 2.0            # Weight for goal-seeking
        self.safety_weight = 1.5          # Weight for safety
        self.base_exploration = 0.4       # Exploration bonus
        self.base_spatial_weight = 1.0    # Weight for hippocampal spatial guidance

        # Episode state
        self.prev_state = None
        self.prev_action = None
        self.prev_pos = None  # For wall detection
        self.prev_direction = None
        self.step_count = 0
        self.episode_reward = 0
        self.carrying_key = False  # For DoorKey environments
        self.stuck_count = 0  # Track consecutive failed forward moves
        self.wall_hit_this_step = False  # For PC-Z update

    def reset(self):
        """Reset for new episode."""
        self.risk.reset_episode()
        self.defense.reset_episode()
        # Reset hippocampus with random_layout=True for DoorKey-style envs
        # This means wall memory resets each episode (since walls move)
        # but successful paths are preserved (general learning)
        self.hippocampus.reset_episode(random_layout=True)
        self.curiosity.reset_episode()  # Reset curiosity signal but keep knowledge
        self.metacog.reset_episode()  # Reset episode tracking but keep learned rules
        self.interoception.reset_episode(max_steps=100)  # Reset internal states
        self.working_memory.reset_episode()  # Clear short-term buffers
        self.planner.reset_episode()  # Clear current plan
        self.narrative.reset_episode()  # Reset episode narrative
        self.pcz.reset_episode()  # Reset PC-Z but keep z_state
        self.prev_state = None
        self.prev_action = None
        self.prev_info = None
        self.prev_pos = None
        self.prev_direction = None
        self.step_count = 0
        self.episode_reward = 0
        self.carrying_key = False
        self.stuck_count = 0
        self.wall_hit_this_step = False
        self.wall_hits_episode = 0  # For narrative summary
        self.got_stuck_episode = False  # For narrative summary

    def _extract_info(self, obs: Dict, env=None) -> Dict:
        """Extract relevant information from MiniGrid observation."""
        # FullyObsWrapper gives us {'image': array, 'direction': int, 'mission': str}
        img = obs['image']
        direction = obs['direction']

        # Agent position and carrying status from env
        if env is not None:
            agent_pos = np.array(env.unwrapped.agent_pos)
            # Check if agent is carrying something
            carrying = env.unwrapped.carrying
            self.carrying_key = carrying is not None
        else:
            agent_pos = np.array([1, 1])
            self.carrying_key = False

        # Find goal and other objects in grid
        # MiniGrid image shape is (width, height, 3) where [x, y, :] gives cell at (x, y)
        goal_pos = None
        danger_pos = None
        key_pos = None
        door_pos = None

        width, height = img.shape[0], img.shape[1]
        door_is_open = False
        for x in range(width):
            for y in range(height):
                obj = img[x, y, 0]  # Correct: img[x, y] for cell at position (x, y)
                if obj == OBJ_GOAL:
                    goal_pos = np.array([x, y])
                elif obj == OBJ_LAVA:
                    danger_pos = np.array([x, y])
                elif obj == OBJ_KEY:
                    key_pos = np.array([x, y])
                elif obj == OBJ_DOOR:
                    door_pos = np.array([x, y])
                    # Check door state: 0=open, 1=closed, 2=locked
                    door_state = img[x, y, 2]
                    door_is_open = (door_state == 0)

        # Default goal position if not found
        if goal_pos is None:
            goal_pos = np.array([width-2, height-2])

        # Compute distances and directions
        goal_diff = goal_pos - agent_pos
        goal_dist = np.linalg.norm(goal_diff)

        # Direction encoding: 0-8 for 3x3 grid
        # 0=NW, 1=N, 2=NE, 3=W, 4=Center, 5=E, 6=SW, 7=S, 8=SE
        def diff_to_dir(diff):
            dx = np.sign(diff[0]) + 1  # 0, 1, 2
            dy = np.sign(diff[1]) + 1  # 0, 1, 2
            return dy * 3 + dx

        goal_dir = diff_to_dir(goal_diff)

        # Danger info
        if danger_pos is not None:
            danger_diff = danger_pos - agent_pos
            danger_dist = np.linalg.norm(danger_diff)
            danger_dir = diff_to_dir(danger_diff)
        else:
            danger_dist = 999.0
            danger_dir = 4

        return {
            'agent_pos': agent_pos,
            'goal_pos': goal_pos,
            'danger_pos': danger_pos,
            'key_pos': key_pos,
            'door_pos': door_pos,
            'door_is_open': door_is_open,
            'goal_dir': goal_dir,
            'danger_dir': danger_dir,
            'goal_dist': goal_dist,
            'danger_dist': danger_dist,
            'direction': direction,
        }

    def _compute_G(self, state: Tuple, info: Dict) -> np.ndarray:
        """
        Compute Expected Free Energy G(a) for all actions.

        G(a) = -Q(s,a) + Ambiguity(a) + Goal_bias(a)

        MiniGrid actions:
        0: Turn left, 1: Turn right, 2: Move forward
        3: Pick up, 4: Drop, 5: Toggle (door), 6: Done
        """
        # Get learned Q-values
        Q_values = self.transition.get_Q_values(state)

        # Get interoception modulation (internal states → G bias)
        intero_mod = self.interoception.get_G_modulation()
        goal_urgency = intero_mod['goal_urgency']       # Low energy → more goal-focused
        risk_sensitivity = intero_mod['risk_sensitivity']  # Pain → more cautious
        exploration_penalty = intero_mod['exploration_penalty']  # Time pressure → less explore
        desperation = intero_mod['desperation']         # Very low energy + time → risk-taking

        # Agent's current direction (0=right, 1=down, 2=left, 3=up)
        direction = info.get('direction', 0)
        agent_pos = info['agent_pos']

        # Direction vectors
        dir_vecs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        forward_vec = np.array(dir_vecs[direction])

        # ============================================================
        # Hierarchical sub-goal selection (DoorKey logic)
        # Priority: 1) Get key if needed, 2) Open door if locked, 3) Go to goal
        # ============================================================
        target_pos = info['goal_pos']  # Default target
        action_bonus = np.zeros(self.n_actions)

        key_pos = info.get('key_pos')
        door_pos = info.get('door_pos')
        carrying_key = self.carrying_key

        # Check if object is directly in front of agent
        def is_in_front(obj_pos):
            if obj_pos is None:
                return False
            front_pos = agent_pos + np.array(forward_vec)
            return np.allclose(obj_pos, front_pos)

        door_is_open = info.get('door_is_open', False)

        if key_pos is not None and not carrying_key:
            # Sub-goal 1: Go to key
            target_pos = key_pos
            if is_in_front(key_pos) and self.n_actions > 3:
                # Key is directly in front - pickup!
                action_bonus[3] = -3.0  # Strong bias for pickup
        elif door_pos is not None and not door_is_open:
            # Sub-goal 2: Open door (only if closed/locked)
            if is_in_front(door_pos) and self.n_actions > 5:
                # Door is directly in front - toggle!
                action_bonus[5] = -3.0  # Strong bias for toggle
            else:
                # Navigate to door
                target_pos = door_pos
        # If door is open, or no door, target is goal (default)

        # Penalize rarely useful actions: Drop(4) and Done(6)
        # Only if environment has these actions (n_actions >= 7)
        if self.n_actions > 4:
            action_bonus[4] = 2.0  # Strongly discourage Drop
        if self.n_actions > 6:
            action_bonus[6] = 2.0  # Strongly discourage Done

        # Compute alignment with current target
        target_diff = target_pos - agent_pos
        target_dist = np.linalg.norm(target_diff)

        target_unit = np.array([0.0, 0.0])
        forward_alignment = 0.0
        if target_dist > 0:
            target_unit = target_diff / target_dist
            forward_alignment = np.dot(forward_vec, target_unit)

        G = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            # 1. Learned value (negative because we minimize G)
            learned_value = -Q_values[a]

            # 2. Uncertainty (exploration bonus, modulated by PC-Z and interoception)
            # Time pressure reduces exploration; desperation can override
            uncertainty = self.transition.get_uncertainty(state, a)
            explore_mult = self.pcz.exploration_mult * max(0.2, 1.0 - exploration_penalty)
            if desperation > 0:
                explore_mult *= 0.5  # Less exploration when desperate
            effective_exploration = self.base_exploration * explore_mult
            exploration = -effective_exploration * uncertainty

            # 3. Goal-directed bias (initial guidance before learning)
            # When stuck, increase goal-direction weight
            # Also increase when internal energy is low (homeostatic urgency)
            goal_mult = 1.0 + (self.pcz.subgoal_mult - 1.0) * 0.5 + goal_urgency  # 1.0 → 1.5+ when stuck/hungry

            goal_bias = 0.0
            if a == 2:  # Forward
                goal_bias = -forward_alignment * self.goal_weight * 0.5 * goal_mult
            elif a == 0:  # Turn left
                new_dir = (direction - 1) % 4
                new_forward = np.array(dir_vecs[new_dir])
                new_alignment = np.dot(new_forward, target_unit)
                if new_alignment > forward_alignment + 0.1:
                    goal_bias = -0.3 * goal_mult
            elif a == 1:  # Turn right
                new_dir = (direction + 1) % 4
                new_forward = np.array(dir_vecs[new_dir])
                new_alignment = np.dot(new_forward, target_unit)
                if new_alignment > forward_alignment + 0.1:
                    goal_bias = -0.3 * goal_mult

            # 4. Action bonus for sub-goals (pickup, toggle), modulated by PC-Z
            subgoal_bias = action_bonus[a] * self.pcz.subgoal_mult

            # 5. Hippocampal spatial guidance (wall memory + goal gradient, modulated by PC-Z)
            # Pain increases risk sensitivity (more cautious about walls)
            x, y = int(agent_pos[0]), int(agent_pos[1])
            effective_spatial = self.base_spatial_weight * self.pcz.spatial_mult
            spatial_bias = 0.0
            if a == 2:  # Forward
                # Penalize if wall is remembered in current direction
                # Higher penalty when in pain (learned caution from past mistakes)
                wall_prob = self.hippocampus.get_wall_probability(x, y, direction)
                spatial_bias += wall_prob * 2.0 * risk_sensitivity  # Positive = bad (minimize G)

                # Bonus for direction preference based on goal gradient
                dir_pref = self.hippocampus.get_direction_preference(x, y, direction)
                spatial_bias -= dir_pref * effective_spatial  # Negative = good

            elif a == 0:  # Turn left -> check new direction
                new_dir = (direction - 1) % 4
                dir_pref = self.hippocampus.get_direction_preference(x, y, new_dir)
                # If turning left leads to better direction, small bonus
                if dir_pref > self.hippocampus.get_direction_preference(x, y, direction):
                    spatial_bias -= 0.2

            elif a == 1:  # Turn right -> check new direction
                new_dir = (direction + 1) % 4
                dir_pref = self.hippocampus.get_direction_preference(x, y, new_dir)
                if dir_pref > self.hippocampus.get_direction_preference(x, y, direction):
                    spatial_bias -= 0.2

            # 6. Curiosity bonus (dopaminergic novelty-seeking)
            # Actions that lead to unexplored outcomes get a bonus (lower G)
            curiosity_bonus = -self.curiosity.get_action_curiosity(x, y, direction, a)

            # Combine: G = learned + goal_bias + subgoal + spatial + exploration + curiosity
            visits = self.transition.visits[state[0], state[1], state[2], state[3], a]
            learning_weight = min(0.5, visits / 20.0)

            G[a] = (
                learned_value * learning_weight
                + goal_bias
                + subgoal_bias
                + spatial_bias
                + exploration
                + curiosity_bonus  # Novelty-seeking
            )

        return G

    def act(self, obs: Dict, env=None) -> int:
        """Select action based on G(a)."""
        self.step_count += 1

        # Extract information (pass env to get agent position)
        info = self._extract_info(obs, env)

        # Wall/stuck detection: if last action was Forward but position unchanged
        current_pos = tuple(info['agent_pos'])
        current_dir = info['direction']

        self.wall_hit_this_step = False
        if self.prev_pos is not None and self.prev_action == 2:  # 2 = Forward
            if current_pos == self.prev_pos:
                self.stuck_count += 1
                self.wall_hit_this_step = True  # For PC-Z
                # Record wall hit in hippocampus
                x, y = int(self.prev_pos[0]), int(self.prev_pos[1])
                self.hippocampus.record_wall_hit(x, y, self.prev_direction)
            else:
                self.stuck_count = 0

        # Record current position in hippocampus (place cell activation)
        x, y = int(current_pos[0]), int(current_pos[1])
        self.hippocampus.record_position(x, y)

        self.prev_pos = current_pos
        self.prev_direction = current_dir

        # Update risk circuit
        self.risk.update(
            info['danger_dist'],
            was_hit=False,
            was_in_defense=self.defense.in_defense
        )

        # Update defense mode
        self.defense.update(
            self.risk.risk_filtered,
            self.risk.approach_streak
        )

        # Get current state
        state = self.transition.get_state(
            info['agent_pos'],
            info['direction'],
            info['goal_dir']
        )

        # Compute G(a)
        G = self._compute_G(state, info)

        # Compute distance to CURRENT sub-goal target (not final goal)
        # This is critical for PC-Z to correctly detect progress
        key_pos = info.get('key_pos')
        door_pos = info.get('door_pos')
        door_is_open = info.get('door_is_open', False)
        agent_pos = info['agent_pos']

        if key_pos is not None and not self.carrying_key:
            # Sub-goal: get key
            current_target_dist = np.linalg.norm(agent_pos - key_pos)
        elif door_pos is not None and not door_is_open:
            # Sub-goal: open door
            current_target_dist = np.linalg.norm(agent_pos - door_pos)
        else:
            # Sub-goal: reach goal
            current_target_dist = info['goal_dist']

        # Update PC-Z bridge with CURRENT sub-goal distance
        self.pcz.update(
            goal_dist=current_target_dist,
            wall_hit=self.wall_hit_this_step,
            G_values=G
        )

        # Wall avoidance: if stuck, penalize Forward and favor turns
        if self.stuck_count >= 2:
            G[2] += 2.0  # Penalize Forward
            G[0] -= 0.5  # Favor Turn Left
            G[1] -= 0.5  # Favor Turn Right

        # Working memory loop detection: break repetitive patterns
        if self.working_memory.is_repeating():
            # Agent is in a 2-action loop - add randomness to break it
            G += np.random.randn(self.n_actions) * 0.3

        # PC-Z controlled planner activation
        # When stuck (z_state=2), activate planning if not already executing
        stuck_score = self.pcz.progress_error
        uncertainty = self.pcz.action_margin  # Low margin = high uncertainty
        if self.planner.should_activate(stuck_score, 1.0 - uncertainty):
            # Generate a plan based on current state
            self.planner.generate_plan(self.carrying_key, door_is_open)

        # If planner is executing, add its G biases
        if self.planner.state == PlannerCircuit.EXECUTING:
            # Check if objects are in front for precise action biases
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

        # Apply PC-Z modulation to temperature
        effective_temperature = self.base_temperature * self.pcz.temperature_mult

        # Softmax action selection with modulated temperature
        probs = np.exp(-G / effective_temperature)
        probs = probs / (probs.sum() + 1e-10)

        action = np.random.choice(self.n_actions, p=probs)

        # Store for learning
        self.prev_state = state
        self.prev_action = action
        self.prev_info = info

        return action

    def learn(self, obs: Dict, reward: float, done: bool, env_info: Dict, env=None):
        """Learn from experience."""
        self.episode_reward += reward

        # Extract current state info (pass env to get agent position)
        state_info = self._extract_info(obs, env)

        current_state = self.transition.get_state(
            state_info['agent_pos'],
            state_info['direction'],
            state_info['goal_dir']
        )

        # Update transition model (Q-learning)
        if self.prev_state is not None:
            self.transition.update(
                self.prev_state,
                self.prev_action,
                reward,
                current_state,
                done
            )

            # Update curiosity circuit (record transition and compute surprise)
            prev_x, prev_y, prev_dir, _ = self.prev_state
            curr_x, curr_y, curr_dir, _ = current_state
            self.curiosity.record_transition(
                prev_x, prev_y, prev_dir, self.prev_action,
                curr_x, curr_y, curr_dir
            )

            # Update metacognition with STATE TRANSITIONS (not object detection)
            # This tracks: KEY_ACQUIRED, DOOR_UNLOCKED, DOOR_BLOCKED events
            self.metacog.record_state_transition(
                step=self.step_count,
                carrying_key=self.carrying_key,
                door_is_open=state_info.get('door_is_open', False),
                action=self.prev_action
            )

        # Update interoception (internal states: energy, pain, time pressure)
        got_reward = reward > 0
        self.interoception.update(
            wall_hit=self.wall_hit_this_step,
            danger_dist=state_info['danger_dist'],
            got_reward=got_reward
        )

        # Update working memory (short-term buffer for partial observation)
        if self.prev_action is not None:
            self.working_memory.update(state_info, self.prev_action)

        # Update planner progress (track sub-goal achievements)
        self.planner.update_progress(
            has_key=self.carrying_key,
            door_open=state_info.get('door_is_open', False),
            reached_goal=(reward > 0 and done)  # Positive reward at done = goal reached
        )

        # Track narrative events (for episode summary)
        if self.wall_hit_this_step:
            self.wall_hits_episode += 1
            if self.wall_hits_episode == 1:
                self.narrative.record_challenge("first_wall_hit", self.step_count)

        if self.stuck_count >= 3:
            self.got_stuck_episode = True
            self.narrative.record_challenge("got_stuck", self.step_count)

        # Record key acquisition as narrative event
        if self.prev_info is not None:
            if self.carrying_key and not self.prev_info.get('carrying_key', False):
                self.narrative.record_event("acquired_key", self.step_count)
            if state_info.get('door_is_open', False) and not self.prev_info.get('door_is_open', False):
                self.narrative.record_event("opened_door", self.step_count)

        # Store carrying_key in info for next step comparison
        state_info['carrying_key'] = self.carrying_key

        # Learn from outcomes
        was_hit = reward < -0.5

        if was_hit:
            self.risk.update(
                state_info['danger_dist'],
                was_hit=True,
                was_in_defense=self.defense.in_defense
            )
            self.defense.learn_from_hit()
            pos = state_info['agent_pos']
            self.memory.record_danger(int(pos[0]), int(pos[1]))

        elif reward > 0:
            self.defense.learn_from_success()
            pos = state_info['agent_pos']
            self.memory.record_reward(int(pos[0]), int(pos[1]), reward)

        # Record visit
        pos = state_info['agent_pos']
        self.memory.record_visit(int(pos[0]), int(pos[1]))

        # Episode end
        if done:
            success = reward > 0
            self.memory.record_episode(self.step_count, self.episode_reward, success)

            # Hippocampal path replay: strengthen successful paths
            if success:
                goal_pos = tuple(state_info['goal_pos'])
                self.hippocampus.record_successful_path(goal_pos)

            # MetaCognition: learn abstract rules from event sequence
            self.metacog.record_episode_end(success, self.step_count)

            # Narrative: create episode summary and update identity
            self.narrative.summarize_episode(
                success=success,
                steps=self.step_count,
                key_acquired=self.carrying_key,
                door_opened=state_info.get('door_is_open', False),
                wall_hits=self.wall_hits_episode,
                got_stuck=self.got_stuck_episode
            )

    def get_stats(self) -> Dict:
        """Get current circuit statistics."""
        # Hippocampal stats
        wall_knowledge = np.sum(self.hippocampus.wall_memory > 0.5)  # Known walls
        gradient_coverage = np.sum(self.hippocampus.goal_gradient > 0.1)  # Learned paths

        # Z-state distribution (from history)
        z_hist = self.pcz.z_state_history
        if z_hist:
            z_dist = {
                'stable': sum(1 for z in z_hist if z == 0) / len(z_hist),
                'exploring': sum(1 for z in z_hist if z == 1) / len(z_hist),
                'stuck': sum(1 for z in z_hist if z == 2) / len(z_hist),
            }
        else:
            z_dist = {'stable': 0, 'exploring': 0, 'stuck': 0}

        # Curiosity stats
        curiosity_stats = self.curiosity.get_stats()

        # MetaCognition stats
        metacog_stats = self.metacog.get_stats()

        # Interoception stats
        intero_stats = self.interoception.get_stats()

        # Working memory stats
        wm_stats = self.working_memory.get_stats()

        # Planner stats
        planner_stats = self.planner.get_stats()

        # Narrative stats
        narrative_stats = self.narrative.get_stats()

        return {
            'risk_sensitivity': self.risk.risk_sensitivity,
            'danger_radius': self.risk.danger_radius,
            'defense_threshold_on': self.defense.threshold_on,
            'defense_threshold_off': self.defense.threshold_off,
            'defense_time_ratio': self.defense.defense_time / max(1, self.step_count),
            'false_alarms': self.risk.false_alarms,
            'missed_dangers': self.risk.missed_dangers,
            'success_rate': self.memory.get_success_rate(),
            'total_episodes': len(self.memory.episode_outcomes),
            # Hippocampal stats
            'wall_knowledge': wall_knowledge,
            'gradient_coverage': gradient_coverage,
            'successful_paths': len(self.hippocampus.success_paths),
            # Curiosity stats
            'explored_states': curiosity_stats['explored_states'],
            'known_transitions': curiosity_stats['known_transitions'],
            'curiosity_signal': curiosity_stats['curiosity_signal'],
            # MetaCognition stats
            'rule_key_door': metacog_stats['rule_key_enables_door'],
            'rule_door_path': metacog_stats['rule_door_blocks_path'],
            # Interoception stats
            'energy': intero_stats['energy'],
            'pain': intero_stats['pain'],
            'time_pressure': intero_stats['time_pressure'],
            'energy_drive': intero_stats['energy_drive'],
            'pain_drive': intero_stats['pain_drive'],
            'energy_crises': intero_stats['energy_crises'],
            'pain_events': intero_stats['pain_events'],
            # Working memory stats
            'wm_objects_remembered': wm_stats['objects_remembered'],
            'wm_avg_confidence': wm_stats['avg_confidence'],
            'wm_is_repeating': wm_stats['is_repeating'],
            'wm_action_variety': wm_stats['action_variety'],
            # Planner stats
            'planner_state': planner_stats['state'],
            'planner_plans_generated': planner_stats['plans_generated'],
            'planner_plans_completed': planner_stats['plans_completed'],
            'planner_completion_rate': planner_stats['completion_rate'],
            # Narrative stats
            'narrative_episodes': narrative_stats['episodes_remembered'],
            'narrative_avg_success': narrative_stats['average_success'],
            'narrative_typical_steps': narrative_stats['typical_steps'],
            'narrative_efficient': narrative_stats['efficient_episodes'],
            'narrative_struggling': narrative_stats['struggling_episodes'],
            'narrative_tips': narrative_stats['learned_tips'],
            # PC-Z stats
            'z_state': self.pcz.z_state,
            'z_state_name': self.pcz.get_state_name(),
            'z_distribution': z_dist,
            'z_transitions': self.pcz.transitions.copy(),
            'epsilon_spike': self.pcz.epsilon_spike,
            'progress_error': self.pcz.progress_error,
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_genesis_on_minigrid(
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
    n_episodes: int = 500,
    verbose: bool = True,
    debug: bool = False
) -> Tuple[GenesisBrain, List[Dict]]:
    """Train Genesis Brain on MiniGrid environment."""

    # Create environment with full observability
    env = gym.make(env_id)
    env = FullyObsWrapper(env)

    # Determine grid size from env name (e.g., "5x5", "8x8")
    match = re.search(r'(\d+)x\d+', env_id)
    grid_size = int(match.group(1)) + 2 if match else 10  # +2 for walls

    # Create brain with appropriate grid size
    brain = GenesisBrain(n_actions=env.action_space.n, grid_size=grid_size)

    history = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        brain.reset()

        done = False
        truncated = False
        ep_reward = 0
        step = 0

        while not (done or truncated):
            action = brain.act(obs, env)

            # Debug output for first episode
            if debug and ep == 0 and step < 30:
                info = brain._extract_info(obs, env)
                dir_names = ['R', 'D', 'L', 'U']  # Right, Down, Left, Up
                action_names = ['TurnL', 'TurnR', 'Fwd', 'Pick', 'Drop', 'Toggle', 'Done']
                dir_name = dir_names[info['direction']]
                act_name = action_names[action]
                front = info['agent_pos'] + np.array([(1,0), (0,1), (-1,0), (0,-1)][info['direction']])
                door_state = "open" if info.get('door_is_open') else "closed"
                print(f"  Step {step}: pos={info['agent_pos']}, dir={dir_name}, "
                      f"key={info.get('key_pos')}, door={info.get('door_pos')}({door_state}), "
                      f"carry={brain.carrying_key}, act={act_name}")

            next_obs, reward, done, truncated, env_info = env.step(action)
            brain.learn(next_obs, reward, done or truncated, env_info, env)

            obs = next_obs
            ep_reward += reward
            step += 1

        # Record history
        stats = brain.get_stats()
        stats['episode'] = ep
        stats['reward'] = ep_reward
        stats['steps'] = brain.step_count
        stats['success'] = ep_reward > 0
        history.append(stats)

        if verbose and (ep + 1) % 50 == 0:
            recent_success = np.mean([h['success'] for h in history[-50:]])
            stats = brain.get_stats()
            z_dist = stats['z_distribution']
            # Show MetaCognition rule learning for DoorKey envs
            rules_str = ""
            if stats['rule_key_door'] > 0 or stats['rule_door_path'] > 0:
                rules_str = f", rules=[K:{stats['rule_key_door']:.0%}/D:{stats['rule_door_path']:.0%}]"
            print(f"Episode {ep+1}: "
                  f"success={recent_success:.0%}, "
                  f"explored={stats['explored_states']}, "
                  f"z=[S:{z_dist['stable']:.0%}/E:{z_dist['exploring']:.0%}/K:{z_dist['stuck']:.0%}]"
                  f"{rules_str}")

    env.close()
    return brain, history


def main():
    print("=" * 60)
    print("GENESIS BRAIN on MiniGrid")
    print("Circuit-based learning, NO LSTM")
    print("=" * 60)
    print()

    # Curriculum: Easy -> Hard (DoorKey focus for MetaCognition learning)
    curriculum = [
        ("MiniGrid-Empty-5x5-v0", 200),
        ("MiniGrid-DoorKey-5x5-v0", 500),
        ("MiniGrid-DoorKey-6x6-v0", 500),  # Harder: larger grid
        ("MiniGrid-DoorKey-8x8-v0", 500),  # Even harder
    ]

    results = []

    for env_id, n_episodes in curriculum:
        print(f"\n{'='*60}")
        print(f"Training on {env_id}...")
        print(f"{'='*60}\n")

        brain, history = train_genesis_on_minigrid(
            env_id=env_id,
            n_episodes=n_episodes
        )

        early_success = np.mean([h['success'] for h in history[:50]])
        late_success = np.mean([h['success'] for h in history[-50:]])

        results.append({
            'env': env_id,
            'early': early_success,
            'late': late_success,
            'brain': brain
        })

        print(f"\n  Result: {early_success:.0%} -> {late_success:.0%}")

    # Final summary
    print("\n" + "=" * 60)
    print("CURRICULUM RESULTS")
    print("=" * 60)
    for r in results:
        status = "PASS" if r['late'] >= 0.5 else "FAIL"
        growth = r['late'] - r['early']
        print(f"  [{status}] {r['env']}: {r['early']:.0%} -> {r['late']:.0%} (growth: {growth:+.0%})")

    # Brain Analysis (for DoorKey-6x6)
    if len(results) >= 3:
        doorkey_brain = results[2]['brain']  # DoorKey-6x6
        stats = doorkey_brain.get_stats()

        print("\n" + "-" * 60)
        print("BRAIN CIRCUIT ANALYSIS (DoorKey-6x6)")
        print("-" * 60)

        print("\n  [Curiosity Circuit - Dopaminergic]")
        print(f"    Explored States:    {stats['explored_states']}")
        print(f"    Known Transitions:  {stats['known_transitions']}")
        print(f"    Curiosity Signal:   {stats['curiosity_signal']:.3f}")

        print("\n  [MetaCognition Circuit - Prefrontal]")
        print(f"    Rule 'Key→Door':    {stats['rule_key_door']:.1%}")
        print(f"    Rule 'Door→Path':   {stats['rule_door_path']:.1%}")

        print("\n  [PC-Z Dynamics - State Regulation]")
        z_dist = stats['z_distribution']
        print(f"    Stable:    {z_dist['stable']:.1%}")
        print(f"    Exploring: {z_dist['exploring']:.1%}")
        print(f"    Stuck:     {z_dist['stuck']:.1%}")

        print("\n  [Interoception Circuit - Internal States]")
        print(f"    Energy:       {stats['energy']:.2f} (crises: {stats['energy_crises']})")
        print(f"    Pain:         {stats['pain']:.2f} (events: {stats['pain_events']})")
        print(f"    Energy Drive: {stats['energy_drive']:.2f}")
        print(f"    Pain Drive:   {stats['pain_drive']:.2f}")

        print("\n  [Working Memory Circuit - Prefrontal Buffer]")
        print(f"    Objects Remembered: {stats['wm_objects_remembered']}")
        print(f"    Avg Confidence:     {stats['wm_avg_confidence']:.2f}")
        print(f"    Action Variety:     {stats['wm_action_variety']}")
        print(f"    In Repetitive Loop: {stats['wm_is_repeating']}")

        print("\n  [Planner Circuit - Executive Control]")
        print(f"    State:           {stats['planner_state']}")
        print(f"    Plans Generated: {stats['planner_plans_generated']}")
        print(f"    Plans Completed: {stats['planner_plans_completed']}")
        print(f"    Completion Rate: {stats['planner_completion_rate']:.1%}")

        print("\n  [Narrative Circuit - Autobiographical Self]")
        print(f"    Episodes Remembered: {stats['narrative_episodes']}")
        print(f"    Self-Assessed Success: {stats['narrative_avg_success']:.1%}")
        print(f"    Typical Steps: {stats['narrative_typical_steps']:.0f}")
        print(f"    Efficient Episodes: {stats['narrative_efficient']}")
        print(f"    Struggling Episodes: {stats['narrative_struggling']}")
        print(f"    Learned Tips: {stats['narrative_tips']}")


if __name__ == "__main__":
    main()
