"""
Self-Model Module for Consciousness Simulation

Core Concept: "What kind of being am I right now?"

This is NOT a new intelligence - it's a META-LAYER that summarizes
existing signals into a coherent self-representation.

Inputs (all already exist):
- Agency level & prediction errors
- Attention width & focus duration
- Working memory activity
- Recent reward trend
- External event frequency

Outputs (self_state):
- confidence: "I am in control"
- uncertainty: "I am confused"
- effort: "Focus fatigue"
- exploration_need: "I should explore"
- stability: "State is steady"

Why This Matters:
1. Attention now responds to SELF-STATE, not just sensory input
2. Agency becomes sustained self-sense, not just per-step detection
3. Safe foundation for Value Conflict later

No learning required - rule-based summarization is more transparent
and actually more convincing for consciousness demonstration.
"""

from typing import Dict, Optional, List
from collections import deque
import math


class SelfModel:
    """
    Self-Model: The agent's representation of its own state.

    Creates a unified "I" from distributed cognitive signals.
    This is the layer where "subjective experience" emerges -
    not as magic, but as integrated self-monitoring.
    """

    def __init__(self,
                 history_window: int = 20,      # Steps to consider for trends
                 confidence_decay: float = 0.9,  # How quickly confidence fades
                 effort_recovery: float = 0.05): # How quickly effort recovers

        self.history_window = history_window
        self.confidence_decay = confidence_decay
        self.effort_recovery = effort_recovery

        # === Core Self-State ===
        self.confidence: float = 0.5      # "I am in control" (0-1)
        self.uncertainty: float = 0.3     # "I am confused" (0-1)
        self.effort: float = 0.0          # "Focus fatigue" (0-1)
        self.exploration_need: float = 0.3 # "I should explore" (0-1)
        self.stability: float = 0.5       # "My state is steady" (0-1)

        # === Attribution: "Whose fault is this?" ===
        # Key insight: When things go wrong, is it MY fault or the WORLD's fault?
        # This affects response strategy:
        # - externality ↑ → "wait it out" (world is chaotic, not my problem)
        # - internal_fault ↑ → "change strategy" (my approach is wrong)
        self.externality: float = 0.0     # "The world is causing problems" (0-1)
        self.internal_fault: float = 0.0  # "My policy/model is wrong" (0-1)

        # === Tracking History ===
        self.agency_history: deque = deque(maxlen=history_window)
        self.pred_error_history: deque = deque(maxlen=history_window)
        self.reward_history: deque = deque(maxlen=history_window)
        self.attention_width_history: deque = deque(maxlen=history_window)
        self.external_event_history: deque = deque(maxlen=history_window)

        # === Derived Metrics ===
        self.avg_agency: float = 0.5
        self.avg_pred_error: float = 0.3
        self.reward_trend: float = 0.0  # positive = improving, negative = degrading
        self.external_pressure: float = 0.0  # frequency of external events

        # === State Change Detection ===
        self.last_self_state: Dict = {}
        self.state_change_magnitude: float = 0.0

        # === Focus Tracking ===
        self.focus_start_step: int = 0
        self.current_focus_duration: int = 0
        self.total_focus_time: int = 0

        # === Statistics ===
        self.total_updates: int = 0
        self.high_confidence_steps: int = 0
        self.high_uncertainty_steps: int = 0

        # === Behavioral State Hysteresis ===
        # Emotions/states don't switch instantly - they "linger"
        self.current_behavioral_state: str = "STABLE"
        self.state_duration: int = 0          # How long in current state
        self.pending_state: Optional[str] = None  # State waiting to transition
        self.pending_duration: int = 0        # How long pending state has been candidate
        self.min_state_duration: int = 15     # Minimum steps before state can change
        self.transition_threshold: int = 8    # Steps a new state must persist to trigger change

        # Entry thresholds (stricter - need strong signal to enter)
        self.state_entry_thresholds = {
            "CONFIDENT": {"confidence": 0.70, "uncertainty": 0.30},
            "EXPLORING": {"exploration_need": 0.60},
            "STRUGGLING": {"uncertainty": 0.60, "effort": 0.50},
            "FATIGUED": {"effort": 0.70},
            "REACTIVE": {"external_pressure": 0.30},
            "STABLE": {"stability": 0.60},
        }

        # Exit thresholds (looser - state "holds on" longer)
        self.state_exit_thresholds = {
            "CONFIDENT": {"confidence": 0.50, "uncertainty": 0.45},
            "EXPLORING": {"exploration_need": 0.40},
            "STRUGGLING": {"uncertainty": 0.45, "effort": 0.35},
            "FATIGUED": {"effort": 0.50},
            "REACTIVE": {"external_pressure": 0.15},
            "STABLE": {"stability": 0.40},
        }

    def update(self,
               agency_level: float,
               prediction_error: float,
               attention_width: float,
               attention_focus: Optional[str],
               memory_active: bool,
               reward: float,
               was_external_event: bool,
               focus_changed: bool = False) -> Dict[str, float]:
        """
        Update self-model based on current cognitive state.

        Args:
            agency_level: Current agency detection level (0-1)
            prediction_error: Current prediction error (0-1)
            attention_width: Current attention width (0=narrow, 1=broad)
            attention_focus: Current focus direction or None
            memory_active: Whether working memory is being used
            reward: Current reward signal
            was_external_event: Whether an external event just occurred
            focus_changed: Whether attention focus just shifted

        Returns:
            Updated self_state dictionary
        """
        self.total_updates += 1

        # === Update History ===
        self.agency_history.append(agency_level)
        self.pred_error_history.append(prediction_error)
        self.reward_history.append(reward)
        self.attention_width_history.append(attention_width)
        self.external_event_history.append(1.0 if was_external_event else 0.0)

        # === Compute Derived Metrics ===
        self._compute_derived_metrics()

        # === Update Focus Tracking ===
        if focus_changed:
            self.focus_start_step = self.total_updates
        self.current_focus_duration = self.total_updates - self.focus_start_step
        if attention_focus:
            self.total_focus_time += 1

        # === Compute Self-State ===
        self._compute_confidence()
        self._compute_uncertainty(prediction_error, attention_width)
        self._compute_effort(attention_width, memory_active)
        self._compute_exploration_need()
        self._compute_stability()
        self._compute_attribution(prediction_error, was_external_event)

        # === Update Behavioral State with Hysteresis ===
        self._update_behavioral_state()

        # === Detect State Changes ===
        current_state = self.get_state()
        self._detect_state_change(current_state)
        self.last_self_state = current_state

        # === Update Statistics ===
        if self.confidence > 0.7:
            self.high_confidence_steps += 1
        if self.uncertainty > 0.7:
            self.high_uncertainty_steps += 1

        return current_state

    def _compute_derived_metrics(self):
        """Compute running averages and trends from history."""
        # Average agency over window
        if self.agency_history:
            self.avg_agency = sum(self.agency_history) / len(self.agency_history)

        # Average prediction error
        if self.pred_error_history:
            self.avg_pred_error = sum(self.pred_error_history) / len(self.pred_error_history)

        # Reward trend (recent vs older)
        if len(self.reward_history) >= 4:
            recent = list(self.reward_history)[-len(self.reward_history)//2:]
            older = list(self.reward_history)[:-len(self.reward_history)//2]
            recent_avg = sum(recent) / len(recent) if recent else 0
            older_avg = sum(older) / len(older) if older else 0
            self.reward_trend = recent_avg - older_avg

        # External pressure (frequency of external events)
        if self.external_event_history:
            self.external_pressure = sum(self.external_event_history) / len(self.external_event_history)

    def _compute_confidence(self):
        """
        Confidence = "I am in control of what's happening"

        High when:
        - Agency is consistently high
        - Prediction errors are low
        - Few external perturbations
        - Positive reward trend
        """
        # Base: sustained agency
        agency_factor = self.avg_agency

        # Reduce for prediction errors
        error_penalty = self.avg_pred_error * 0.5

        # Reduce for external pressure
        external_penalty = self.external_pressure * 0.3

        # Boost for positive rewards
        reward_boost = max(0, self.reward_trend) * 0.2

        # Compute confidence
        raw_confidence = agency_factor - error_penalty - external_penalty + reward_boost

        # Smooth transition (don't jump suddenly)
        self.confidence = 0.7 * self.confidence + 0.3 * max(0, min(1, raw_confidence))

    def _compute_uncertainty(self, current_pred_error: float, attention_width: float):
        """
        Uncertainty = "I am confused about what's happening"

        High when:
        - Prediction errors are high
        - Attention is broad (not focused)
        - Agency is fluctuating
        """
        # Base: current prediction error
        error_factor = current_pred_error

        # Add from broad attention (searching = uncertain)
        attention_factor = attention_width * 0.3

        # Add from agency variance
        agency_variance = 0.0
        if len(self.agency_history) >= 3:
            recent = list(self.agency_history)[-5:]
            mean = sum(recent) / len(recent)
            agency_variance = sum((x - mean) ** 2 for x in recent) / len(recent)
            agency_variance = min(1.0, agency_variance * 4)  # Scale up

        # Compute uncertainty
        raw_uncertainty = error_factor * 0.5 + attention_factor + agency_variance * 0.2

        # Smooth transition
        self.uncertainty = 0.8 * self.uncertainty + 0.2 * max(0, min(1, raw_uncertainty))

    def _compute_effort(self, attention_width: float, memory_active: bool):
        """
        Effort = "I am mentally fatigued"

        High when:
        - Sustained narrow focus (concentration depletes)
        - Active working memory use
        - Low confidence but high uncertainty (struggling)

        Recovers naturally over time.
        """
        # Natural recovery
        self.effort = max(0, self.effort - self.effort_recovery)

        # Increase from sustained focus
        if attention_width < 0.4:  # Narrow focus
            focus_cost = 0.02 * (self.current_focus_duration / 10)
            self.effort += min(0.05, focus_cost)

        # Increase from memory use
        if memory_active:
            self.effort += 0.01

        # Increase from struggling (low confidence + high uncertainty)
        if self.confidence < 0.4 and self.uncertainty > 0.6:
            self.effort += 0.03

        # Cap at 1.0
        self.effort = min(1.0, self.effort)

    def _compute_exploration_need(self):
        """
        Exploration Need = "I should try something different"

        High when:
        - Reward trend is negative (current strategy failing)
        - Stuck in same place too long
        - Low confidence
        - High effort (current approach is hard)
        """
        # Base: inverse of reward trend
        reward_factor = max(0, -self.reward_trend) * 0.5

        # Add from low confidence
        confidence_factor = (1 - self.confidence) * 0.3

        # Add from high effort
        effort_factor = self.effort * 0.2

        # Reduce if things are going well
        if self.reward_trend > 0.1 and self.confidence > 0.6:
            reduction = 0.3
        else:
            reduction = 0.0

        # Compute exploration need
        raw_exploration = reward_factor + confidence_factor + effort_factor - reduction

        # Smooth transition
        self.exploration_need = 0.8 * self.exploration_need + 0.2 * max(0, min(1, raw_exploration))

    def _compute_stability(self):
        """
        Stability = "My state is steady, not fluctuating"

        High when:
        - Self-state values are not changing much
        - Low uncertainty
        - Moderate confidence (not overconfident)
        """
        # Inverse of state change magnitude
        change_factor = 1 - min(1, self.state_change_magnitude * 2)

        # Inverse of uncertainty
        uncertainty_factor = 1 - self.uncertainty

        # Penalty for extreme confidence (overconfidence is unstable)
        confidence_penalty = 0.2 if self.confidence > 0.9 else 0.0

        # Compute stability
        raw_stability = change_factor * 0.5 + uncertainty_factor * 0.5 - confidence_penalty

        # Smooth transition
        self.stability = 0.9 * self.stability + 0.1 * max(0, min(1, raw_stability))

    def _compute_attribution(self, prediction_error: float, was_external_event: bool):
        """
        Attribution = "Whose fault is this?"

        This is crucial for sense of agency and appropriate responses:
        - externality ↑: "The world is chaotic, not my fault"
          → Response: wait, stabilize, don't change strategy
        - internal_fault ↑: "My approach is wrong"
          → Response: explore, try different strategies

        Key insight: High prediction error can be external OR internal:
        - If external events are frequent → externality
        - If no external events but still failing → internal fault
        """
        # === Externality: "The world is causing my problems" ===
        # Based on frequency of external events (wind, pushes, etc.)
        # When external_pressure is high, errors are "not my fault"
        raw_externality = self.external_pressure * 0.7

        # Boost externality if current event was external
        if was_external_event:
            raw_externality += 0.3

        # High externality when agency is low AND external events are happening
        if self.avg_agency < 0.5 and self.external_pressure > 0.2:
            raw_externality += 0.2

        # Smooth transition
        self.externality = 0.8 * self.externality + 0.2 * max(0, min(1, raw_externality))

        # === Internal Fault: "My policy/model is wrong" ===
        # High when: prediction errors are high BUT no external explanation
        # "I'm failing and it's not because of the environment"

        # Base: prediction error when no external events
        internal_signal = 0.0
        if self.external_pressure < 0.15:  # Few external events
            # High prediction error without external cause = my fault
            internal_signal = self.avg_pred_error * 0.6

            # Add from negative reward trend (strategy is failing)
            if self.reward_trend < 0:
                internal_signal += abs(self.reward_trend) * 0.3

        # Reduce internal_fault if externality is high (can't blame both)
        internal_signal *= (1 - self.externality * 0.5)

        # Add from high uncertainty without external cause
        if self.uncertainty > 0.5 and self.external_pressure < 0.1:
            internal_signal += 0.2

        # Smooth transition
        self.internal_fault = 0.8 * self.internal_fault + 0.2 * max(0, min(1, internal_signal))

    def _detect_state_change(self, current_state: Dict):
        """Detect how much self-state has changed since last update."""
        if not self.last_self_state:
            self.state_change_magnitude = 0.0
            return

        changes = []
        for key in ['confidence', 'uncertainty', 'effort', 'exploration_need']:
            if key in self.last_self_state and key in current_state:
                changes.append(abs(current_state[key] - self.last_self_state[key]))

        self.state_change_magnitude = sum(changes) / len(changes) if changes else 0.0

    # === Modulation Outputs ===

    def get_attention_modulation(self) -> Dict[str, float]:
        """
        Get how self-state should modulate attention.

        Returns:
            width_bias: Add to attention width (-0.3 to +0.3)
            shift_threshold: Modify shift threshold (-0.2 to +0.2)
            gain_modifier: Modify attention gain (0.8 to 1.2)
            dwell_modifier: Modify min dwell time (0.5 to 1.5)
        """
        # High uncertainty → broader attention
        width_bias = self.uncertainty * 0.3 - self.confidence * 0.1

        # High exploration need → lower shift threshold (shift more easily)
        shift_threshold = -self.exploration_need * 0.2

        # High effort → reduced gain (fatigued attention)
        gain_modifier = 1.0 - self.effort * 0.2

        # High effort → shorter dwell time (attention shifts more easily when tired)
        # Low effort + high confidence → longer dwell time (focused)
        dwell_modifier = 1.0 - self.effort * 0.5 + self.confidence * 0.3
        dwell_modifier = max(0.5, min(1.5, dwell_modifier))

        return {
            'width_bias': max(-0.3, min(0.3, width_bias)),
            'shift_threshold': max(-0.2, min(0.2, shift_threshold)),
            'gain_modifier': max(0.8, min(1.2, gain_modifier)),
            'dwell_modifier': dwell_modifier
        }

    def get_agency_modulation(self) -> Dict[str, float]:
        """
        Get how self-state should modulate agency detection.

        Returns:
            sensitivity: Agency detection sensitivity modifier
            baseline: Baseline agency level adjustment
        """
        # Low stability → more sensitive to agency changes
        sensitivity = 1.0 + (1 - self.stability) * 0.3

        # Sustained high confidence → slight baseline boost
        baseline = 0.0
        if self.high_confidence_steps > 10:
            baseline = 0.05

        return {
            'sensitivity': sensitivity,
            'baseline': baseline
        }

    def get_memory_modulation(self) -> Dict[str, float]:
        """
        Get how self-state should modulate working memory.

        Returns:
            decay_modifier: Multiply base decay (0.85 to 1.0)
                           - stability ↑ → decay ↓ (memories persist longer)
                           - stability ↓ → decay ↑ (faster forgetting)
            fast_decay_threshold: When to trigger fast decay
                           - stability ↓ + uncertainty ↑ → easier fast decay
        """
        # High stability → slower decay (memories last longer)
        # Low stability → faster decay (strategy switching)
        decay_modifier = 0.85 + self.stability * 0.15  # 0.85 to 1.0

        # Low stability + high uncertainty → lower threshold for fast decay
        # This means unstable states trigger memory clearing more easily
        fast_decay_sensitivity = (1 - self.stability) * 0.5 + self.uncertainty * 0.3

        return {
            'decay_modifier': max(0.85, min(1.0, decay_modifier)),
            'fast_decay_sensitivity': max(0, min(1, fast_decay_sensitivity))
        }

    def _compute_raw_behavioral_state(self) -> str:
        """
        Compute behavioral state based on current values only (no hysteresis).
        This is the "instantaneous" reading of what state we'd be in.
        """
        if self.external_pressure > 0.3:
            return "REACTIVE"
        elif self.confidence > 0.7 and self.uncertainty < 0.3:
            return "CONFIDENT"
        elif self.exploration_need > 0.6:
            return "EXPLORING"
        elif self.uncertainty > 0.6 and self.effort > 0.5:
            return "STRUGGLING"
        elif self.effort > 0.7:
            return "FATIGUED"
        elif self.stability > 0.6:
            return "STABLE"
        else:
            return "TRANSITIONING"

    def _should_exit_current_state(self) -> bool:
        """
        Check if we should exit the current behavioral state.
        Uses looser exit thresholds (states "hold on" longer).
        """
        state = self.current_behavioral_state
        if state == "TRANSITIONING":
            return True  # Always ready to exit transitioning

        thresholds = self.state_exit_thresholds.get(state, {})

        if state == "CONFIDENT":
            # Stay confident as long as confidence isn't too low
            return self.confidence < thresholds.get("confidence", 0.50) or \
                   self.uncertainty > thresholds.get("uncertainty", 0.45)
        elif state == "EXPLORING":
            return self.exploration_need < thresholds.get("exploration_need", 0.40)
        elif state == "STRUGGLING":
            return self.uncertainty < thresholds.get("uncertainty", 0.45) or \
                   self.effort < thresholds.get("effort", 0.35)
        elif state == "FATIGUED":
            return self.effort < thresholds.get("effort", 0.50)
        elif state == "REACTIVE":
            return self.external_pressure < thresholds.get("external_pressure", 0.15)
        elif state == "STABLE":
            return self.stability < thresholds.get("stability", 0.40)

        return True

    def _can_enter_state(self, state: str) -> bool:
        """
        Check if we can enter a new state.
        Uses stricter entry thresholds (need strong signal to change).
        """
        thresholds = self.state_entry_thresholds.get(state, {})

        if state == "CONFIDENT":
            return self.confidence >= thresholds.get("confidence", 0.70) and \
                   self.uncertainty <= thresholds.get("uncertainty", 0.30)
        elif state == "EXPLORING":
            return self.exploration_need >= thresholds.get("exploration_need", 0.60)
        elif state == "STRUGGLING":
            return self.uncertainty >= thresholds.get("uncertainty", 0.60) and \
                   self.effort >= thresholds.get("effort", 0.50)
        elif state == "FATIGUED":
            return self.effort >= thresholds.get("effort", 0.70)
        elif state == "REACTIVE":
            return self.external_pressure >= thresholds.get("external_pressure", 0.30)
        elif state == "STABLE":
            return self.stability >= thresholds.get("stability", 0.60)
        elif state == "TRANSITIONING":
            return True  # Can always enter transitioning

        return False

    def _update_behavioral_state(self):
        """
        Update behavioral state with hysteresis.

        Key principles:
        1. States have "inertia" - need sustained signal to change
        2. Entry thresholds are stricter than exit thresholds
        3. Minimum duration before a state can change
        4. Pending states must persist to trigger transition
        """
        self.state_duration += 1

        raw_state = self._compute_raw_behavioral_state()

        # If raw state matches current, we're stable - reset pending
        if raw_state == self.current_behavioral_state:
            self.pending_state = None
            self.pending_duration = 0
            return

        # Check if we've been in current state long enough to consider changing
        if self.state_duration < self.min_state_duration:
            # Not ready to change yet - but still track pending
            if raw_state == self.pending_state:
                self.pending_duration += 1
            else:
                self.pending_state = raw_state
                self.pending_duration = 1
            return

        # Check if we should exit current state (using loose thresholds)
        if not self._should_exit_current_state():
            # Current state still holds - but track the challenger
            if raw_state == self.pending_state:
                self.pending_duration += 1
            else:
                self.pending_state = raw_state
                self.pending_duration = 1
            return

        # Current state wants to exit - check if new state can enter
        if not self._can_enter_state(raw_state):
            # New state doesn't meet entry criteria
            # Check TRANSITIONING as fallback
            if raw_state == self.pending_state:
                self.pending_duration += 1
            else:
                self.pending_state = raw_state
                self.pending_duration = 1

            # If stuck too long, go to TRANSITIONING
            if self.pending_duration > self.transition_threshold * 2:
                self._transition_to("TRANSITIONING")
            return

        # New state meets entry criteria - check if it's been pending long enough
        if raw_state == self.pending_state:
            self.pending_duration += 1
        else:
            self.pending_state = raw_state
            self.pending_duration = 1

        # Transition if pending state has persisted long enough
        if self.pending_duration >= self.transition_threshold:
            self._transition_to(raw_state)

    def _transition_to(self, new_state: str):
        """Execute state transition."""
        old_state = self.current_behavioral_state
        self.current_behavioral_state = new_state
        self.state_duration = 0
        self.pending_state = None
        self.pending_duration = 0

        # Could add logging/callbacks here for state transition events
        # print(f"[SELF-MODEL] State transition: {old_state} → {new_state}")

    def get_behavioral_state(self) -> str:
        """
        Get current behavioral state with hysteresis applied.

        Returns one of:
        - "CONFIDENT": High control, low uncertainty (stable, not easily shaken)
        - "EXPLORING": High exploration need (curious, seeking novelty)
        - "STRUGGLING": High uncertainty, high effort (confused, working hard)
        - "FATIGUED": High effort (tired, needs rest)
        - "STABLE": Balanced, steady state (calm baseline)
        - "REACTIVE": Responding to external events (alert, vigilant)
        - "TRANSITIONING": Between states (uncertain which way to go)
        """
        return self.current_behavioral_state

    def get_raw_behavioral_state(self) -> str:
        """Get instantaneous behavioral state without hysteresis (for debugging)."""
        return self._compute_raw_behavioral_state()

    # === State Access ===

    def get_state(self) -> Dict[str, float]:
        """Get current self-state."""
        return {
            'confidence': round(self.confidence, 3),
            'uncertainty': round(self.uncertainty, 3),
            'effort': round(self.effort, 3),
            'exploration_need': round(self.exploration_need, 3),
            'stability': round(self.stability, 3),
            # Attribution: whose fault is this?
            'externality': round(self.externality, 3),
            'internal_fault': round(self.internal_fault, 3)
        }

    def get_visualization_data(self) -> Dict:
        """Get data optimized for frontend visualization."""
        return {
            'state': self.get_state(),
            'behavioral_label': self.get_behavioral_state(),
            'derived': {
                'avg_agency': round(self.avg_agency, 3),
                'avg_pred_error': round(self.avg_pred_error, 3),
                'reward_trend': round(self.reward_trend, 3),
                'external_pressure': round(self.external_pressure, 3)
            },
            'focus_duration': self.current_focus_duration,
            'state_change': round(self.state_change_magnitude, 3)
        }

    def to_dict(self) -> Dict:
        """Full state for API response."""
        return {
            **self.get_state(),
            'behavioral_state': self.get_behavioral_state(),
            'raw_behavioral_state': self.get_raw_behavioral_state(),  # For comparison
            'derived_metrics': {
                'avg_agency': round(self.avg_agency, 3),
                'avg_pred_error': round(self.avg_pred_error, 3),
                'reward_trend': round(self.reward_trend, 3),
                'external_pressure': round(self.external_pressure, 3)
            },
            'modulation': {
                'attention': self.get_attention_modulation(),
                'agency': self.get_agency_modulation()
            },
            'hysteresis': {
                'state_duration': self.state_duration,
                'pending_state': self.pending_state,
                'pending_duration': self.pending_duration,
                'min_duration': self.min_state_duration,
                'transition_threshold': self.transition_threshold
            },
            'statistics': {
                'total_updates': self.total_updates,
                'high_confidence_steps': self.high_confidence_steps,
                'high_uncertainty_steps': self.high_uncertainty_steps,
                'focus_time_ratio': round(
                    self.total_focus_time / max(1, self.total_updates), 3
                )
            }
        }

    def clear(self):
        """Reset self-model state."""
        self.confidence = 0.5
        self.uncertainty = 0.3
        self.effort = 0.0
        self.exploration_need = 0.3
        self.stability = 0.5
        self.externality = 0.0
        self.internal_fault = 0.0

        self.agency_history.clear()
        self.pred_error_history.clear()
        self.reward_history.clear()
        self.attention_width_history.clear()
        self.external_event_history.clear()

        self.avg_agency = 0.5
        self.avg_pred_error = 0.3
        self.reward_trend = 0.0
        self.external_pressure = 0.0

        self.focus_start_step = 0
        self.current_focus_duration = 0
        self.total_focus_time = 0

        self.total_updates = 0
        self.high_confidence_steps = 0
        self.high_uncertainty_steps = 0

        # Reset behavioral state hysteresis
        self.current_behavioral_state = "STABLE"
        self.state_duration = 0
        self.pending_state = None
        self.pending_duration = 0
