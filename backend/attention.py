"""
Attention Module for Consciousness Simulation

Core Concept: Attention as a spotlight that amplifies relevant information
and suppresses irrelevant information.

Key Properties:
1. Focus: What direction/stimulus is being attended to
2. Width: Narrow (goal-directed) vs Broad (exploratory)
3. Gain: How much attended stimuli are amplified
4. Shift: When and how attention moves

Integration:
- Agency HIGH → Narrow focus, strong gain (confident pursuit)
- Agency LOW → Broad focus, weak gain (uncertain exploration)
- Prediction Error → Attention shift (something unexpected)
- Memory → Biases attention toward remembered directions

Biological Basis:
- Thalamic gating of sensory input
- Prefrontal modulation of attention
- Salience detection (novelty, reward prediction error)
"""

from typing import Dict, Optional, List, Tuple
from collections import deque
import math


class AttentionSystem:
    """
    Attention system that filters and amplifies sensory input.

    Two modes:
    1. FOCUSED: High gain on one direction, suppression of others
    2. DIFFUSE: Equal attention to all directions (exploratory)

    Attention is influenced by:
    - Current focus direction
    - Agency level (high agency = more focused)
    - Memory (attend to remembered locations)
    - Salience (unexpected events capture attention)
    """

    def __init__(self,
                 base_gain: float = 1.5,      # Amplification for attended stimuli
                 suppression: float = 0.3,    # Suppression for unattended stimuli
                 shift_threshold: float = 0.4, # Prediction error threshold for shift
                 focus_decay: float = 0.95):   # How quickly focus fades without reinforcement

        self.directions = ['up', 'down', 'left', 'right']

        # Attention parameters
        self.base_gain = base_gain
        self.suppression = suppression
        self.shift_threshold = shift_threshold
        self.focus_decay = focus_decay

        # Current attention state
        self.focus_direction: Optional[str] = None  # Where attention is focused
        self.focus_strength: float = 0.0            # How strongly focused (0-1)
        self.attention_width: float = 1.0           # 0=narrow, 1=broad
        self.attention_mode: str = "DIFFUSE"        # "FOCUSED" or "DIFFUSE"

        # Attention weights for each direction (how much each is attended)
        self.attention_weights: Dict[str, float] = {d: 1.0 for d in self.directions}

        # Salience tracking (what's "interesting")
        self.salience: Dict[str, float] = {d: 0.0 for d in self.directions}
        self.last_sensory: Dict[str, float] = {d: 0.0 for d in self.directions}

        # History for visualization
        self.focus_history: deque = deque(maxlen=50)
        self.width_history: deque = deque(maxlen=50)

        # Statistics
        self.attention_shifts = 0
        self.total_updates = 0

        # v2 Improvements: Anti-jitter and tunnel vision prevention
        self.min_dwell_steps: int = 10        # Minimum steps to hold focus before shift allowed
        self.steps_since_shift: int = 0       # Counter since last shift
        self.consecutive_failures: int = 0    # Failed actions while focused
        self.tunnel_vision_relaxed: bool = False  # Whether gain is currently relaxed

    def update(self,
               sensory_state: Dict[str, float],
               agency_level: float,
               memory_direction: Optional[str] = None,
               prediction_error: float = 0.0) -> Dict[str, float]:
        """
        Update attention state and return attention-modulated sensory input.

        Args:
            sensory_state: Raw sensory input {'up': 0.8, 'down': 0.0, ...}
            agency_level: Current agency level (0-1)
            memory_direction: Direction from working memory (if any)
            prediction_error: Recent prediction error from agency detector

        Returns:
            Attention-filtered sensory input (same format as input)
        """
        self.total_updates += 1
        self.steps_since_shift += 1  # Track time since last shift

        # 1. Update salience (what's interesting/novel)
        self._update_salience(sensory_state)

        # 2. Determine attention width based on agency
        self._update_attention_width(agency_level)

        # 3. Check if attention should shift
        self._check_attention_shift(sensory_state, prediction_error, memory_direction)

        # 4. Update attention weights
        self._update_attention_weights()

        # 5. Apply attention to sensory input
        filtered_sensory = self._apply_attention(sensory_state)

        # Record history
        self.focus_history.append(self.focus_direction)
        self.width_history.append(self.attention_width)

        return filtered_sensory

    def _update_salience(self, sensory_state: Dict[str, float]):
        """
        Update salience map based on sensory changes.
        Novel or changing stimuli are more salient.
        """
        for direction in self.directions:
            current = sensory_state.get(direction, 0.0)
            last = self.last_sensory.get(direction, 0.0)

            # Salience = change + absolute value
            change = abs(current - last)
            self.salience[direction] = 0.7 * self.salience[direction] + 0.3 * (change + current * 0.5)

            self.last_sensory[direction] = current

    def _update_attention_width(self, agency_level: float):
        """
        Update attention width based on agency level.

        High agency → Narrow focus (confident, goal-directed)
        Low agency → Broad focus (uncertain, exploratory)
        """
        # Width inversely related to agency
        # agency=1.0 → width=0.2 (narrow)
        # agency=0.0 → width=1.0 (broad)
        target_width = 0.2 + 0.8 * (1 - agency_level)

        # Smooth transition
        self.attention_width = 0.8 * self.attention_width + 0.2 * target_width

        # Update mode
        if self.attention_width < 0.5:
            self.attention_mode = "FOCUSED"
        else:
            self.attention_mode = "DIFFUSE"

    def _check_attention_shift(self,
                               sensory_state: Dict[str, float],
                               prediction_error: float,
                               memory_direction: Optional[str]):
        """
        Check if attention should shift to a new focus.

        Shift triggers (with priority):
        1. Collision (handled externally via on_collision)
        2. High prediction error (something unexpected) - can bypass dwell
        3. Strong sensory input in unattended direction
        4. Memory suggesting different direction
        5. No current focus

        v2: Minimum dwell time prevents jittery shifts
        """
        should_shift = False
        new_focus = None
        bypass_dwell = False  # Some triggers can bypass minimum dwell

        # Check if we're still in dwell period
        in_dwell_period = self.steps_since_shift < self.min_dwell_steps

        # Trigger 1: High prediction error - CAN bypass dwell (urgent)
        if prediction_error > self.shift_threshold:
            should_shift = True
            bypass_dwell = True  # Prediction error is urgent
            new_focus = max(self.salience, key=self.salience.get)

        # Trigger 2: Strong sensory input in unattended direction
        # Only if not in dwell period
        if not should_shift and not in_dwell_period:
            for direction in self.directions:
                if direction != self.focus_direction:
                    if sensory_state.get(direction, 0) > 0.7:
                        should_shift = True
                        new_focus = direction
                        break

        # Trigger 3: Memory suggests different direction (gentle bias)
        # Only if not in dwell period and focus is weak
        if not should_shift and not in_dwell_period:
            if memory_direction and memory_direction != self.focus_direction:
                if self.focus_strength < 0.3:
                    should_shift = True
                    new_focus = memory_direction

        # Trigger 4: No current focus - attend to strongest sensory
        if not should_shift and self.focus_direction is None:
            max_sensory = max(sensory_state.values()) if sensory_state else 0
            if max_sensory > 0.1:
                should_shift = True
                new_focus = max(sensory_state, key=sensory_state.get)

        # Apply shift (respecting dwell unless bypassed)
        can_shift = should_shift and new_focus and (bypass_dwell or not in_dwell_period)

        if can_shift:
            if new_focus != self.focus_direction:
                self.attention_shifts += 1
                self.steps_since_shift = 0  # Reset dwell counter
                self.consecutive_failures = 0  # Reset failure counter on new focus
                print(f"[ATTENTION] Shift: {self.focus_direction or 'NONE'} → {new_focus.upper()}")
            self.focus_direction = new_focus
            self.focus_strength = 0.8
        else:
            # Decay focus strength over time
            self.focus_strength *= self.focus_decay

            # If focus is too weak, clear it (no dwell restriction for fading)
            if self.focus_strength < 0.1:
                if self.focus_direction:
                    print(f"[ATTENTION] Focus faded: {self.focus_direction.upper()} → DIFFUSE")
                self.focus_direction = None
                self.focus_strength = 0.0
                self.steps_since_shift = 0  # Reset for next focus

    def _update_attention_weights(self):
        """
        Update attention weights based on focus and width.

        v2 Improvements:
        - Gain scheduling: narrower width → softer contrast (prevent tunnel vision)
        - Consecutive failures → relax gain temporarily
        """
        if self.attention_mode == "FOCUSED" and self.focus_direction:
            # v2: Gain scheduling based on width
            # Narrower width = stronger focus BUT softer suppression
            # This prevents "tunnel vision" where we can't see alternatives
            #
            # width=0.2 (very narrow): gain=1.3, suppression=0.5
            # width=0.5 (medium):      gain=1.5, suppression=0.3
            # Formula: As width decreases, suppress less harshly
            effective_gain = self.base_gain - 0.4 * (0.5 - self.attention_width)  # 1.3~1.5
            effective_suppression = self.suppression + 0.4 * (0.5 - self.attention_width)  # 0.3~0.5

            # Clamp values
            effective_gain = max(1.2, min(1.5, effective_gain))
            effective_suppression = max(0.3, min(0.6, effective_suppression))

            # v2: Relax gain if tunnel vision detected (consecutive failures)
            if self.tunnel_vision_relaxed:
                effective_gain = 1.2  # Reduced amplification
                effective_suppression = 0.6  # Much less suppression
                # Note: tunnel_vision_relaxed is set by on_failure() method

            focus_weight = effective_gain * self.focus_strength
            other_weight = effective_suppression + (1 - effective_suppression) * self.attention_width

            for direction in self.directions:
                if direction == self.focus_direction:
                    self.attention_weights[direction] = 1.0 + focus_weight
                else:
                    self.attention_weights[direction] = other_weight
        else:
            # Diffuse mode: equal attention
            for direction in self.directions:
                self.attention_weights[direction] = 1.0

    def _apply_attention(self, sensory_state: Dict[str, float]) -> Dict[str, float]:
        """
        Apply attention weights to sensory input.
        """
        filtered = {}
        for direction in self.directions:
            raw = sensory_state.get(direction, 0.0)
            weight = self.attention_weights[direction]
            filtered[direction] = min(1.0, raw * weight)  # Cap at 1.0
        return filtered

    # =========================================
    # EXTERNAL TRIGGERS
    # =========================================

    def on_reward(self, reward: float, action_direction: str):
        """
        Reward received - reinforce attention to successful direction.

        v2: Also resets failure counter and tunnel vision relaxation.
        """
        if reward > 0.3 and action_direction:
            # Success! Reset failure tracking
            self.consecutive_failures = 0
            if self.tunnel_vision_relaxed:
                self.tunnel_vision_relaxed = False
                print(f"[ATTENTION] Tunnel vision recovered (success)")

            # Positive reward reinforces current focus
            if action_direction == self.focus_direction:
                self.focus_strength = min(1.0, self.focus_strength + 0.2)
            else:
                # Only shift to rewarded direction if dwell period passed
                if self.steps_since_shift >= self.min_dwell_steps:
                    self.focus_direction = action_direction
                    self.focus_strength = 0.7
                    self.steps_since_shift = 0

        elif reward < -0.1 and action_direction == self.focus_direction:
            # Failed while focused on this direction
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3 and not self.tunnel_vision_relaxed:
                self.tunnel_vision_relaxed = True
                print(f"[ATTENTION] Tunnel vision detected! Relaxing gain...")

    def on_collision(self, direction: str):
        """
        Collision detected - strongly suppress attention to blocked direction.

        Wall collision is SELF-CAUSED, so we need aggressive suppression
        to prevent repeated wall-banging behavior.
        """
        # Strongly reduce salience of blocked direction
        self.salience[direction] *= 0.1  # More aggressive than before

        # Directly reduce attention weight for this direction
        self.attention_weights[direction] = max(0.2, self.attention_weights[direction] * 0.5)

        # If focused on blocked direction, force shift away
        if self.focus_direction == direction:
            self.focus_strength *= 0.3  # Much weaker
            self.consecutive_failures += 1
            self.steps_since_shift = self.min_dwell_steps  # Allow immediate shift

            # Find alternative direction with highest salience
            alt_directions = [d for d in self.directions if d != direction]
            if alt_directions:
                best_alt = max(alt_directions, key=lambda d: self.salience.get(d, 0))
                if self.salience.get(best_alt, 0) > 0.05:
                    self.focus_direction = best_alt
                    self.focus_strength = 0.6
                    print(f"[ATTENTION] Wall hit! Shifting: {direction.upper()} → {best_alt.upper()}")
                else:
                    print(f"[ATTENTION] Wall hit! Suppressing: {direction.upper()}")

            # Check for tunnel vision (repeated wall hits)
            if self.consecutive_failures >= 2 and not self.tunnel_vision_relaxed:
                self.tunnel_vision_relaxed = True
                self.attention_width = max(0.7, self.attention_width)  # Force broader attention
                print(f"[ATTENTION] Repeated wall hits! Broadening attention...")

    def on_wm_fast_decay(self):
        """
        Called when Working Memory triggers fast decay (panic mode).
        Attention should widen to explore alternatives.

        v2: WM-Attention coordination to prevent both being stuck.
        """
        # Immediately widen attention for exploration
        self.attention_width = max(0.8, self.attention_width)  # Force broad
        self.attention_mode = "DIFFUSE"
        self.focus_strength *= 0.5  # Weaken current focus

        print(f"[ATTENTION] WM panic → Widening for exploration")

    def force_focus(self, direction: str):
        """
        Externally force attention to a direction (for testing).
        """
        self.focus_direction = direction
        self.focus_strength = 1.0
        self.attention_mode = "FOCUSED"
        print(f"[ATTENTION] Forced focus: {direction.upper()}")

    # =========================================
    # STATE ACCESS
    # =========================================

    def get_focus(self) -> Tuple[Optional[str], float]:
        """Get current focus direction and strength."""
        return self.focus_direction, self.focus_strength

    def get_mode(self) -> str:
        """Get current attention mode."""
        return self.attention_mode

    def to_dict(self) -> Dict:
        """Get full state for API response."""
        return {
            "focus_direction": self.focus_direction,
            "focus_strength": round(self.focus_strength, 3),
            "attention_width": round(self.attention_width, 3),
            "attention_mode": self.attention_mode,
            "attention_weights": {k: round(v, 3) for k, v in self.attention_weights.items()},
            "salience": {k: round(v, 3) for k, v in self.salience.items()},
            "attention_shifts": self.attention_shifts
        }

    def get_visualization_data(self) -> Dict:
        """Get data optimized for frontend visualization."""
        return {
            "focus": self.focus_direction,
            "strength": round(self.focus_strength, 3),
            "width": round(self.attention_width, 3),
            "mode": self.attention_mode,
            "weights": {k: round(v, 3) for k, v in self.attention_weights.items()},
            "salience": {k: round(v, 3) for k, v in self.salience.items()}
        }

    def clear(self):
        """Reset attention state."""
        self.focus_direction = None
        self.focus_strength = 0.0
        self.attention_width = 1.0
        self.attention_mode = "DIFFUSE"
        self.attention_weights = {d: 1.0 for d in self.directions}
        self.salience = {d: 0.0 for d in self.directions}
        # v2 resets
        self.steps_since_shift = 0
        self.consecutive_failures = 0
        self.tunnel_vision_relaxed = False
