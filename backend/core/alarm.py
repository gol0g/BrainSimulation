"""
Alarm System - Fast Pathway (Low Road)

Core Concept:
Brains have two response pathways:
1. FAST (this file): Rough but immediate - bypasses deliberation
2. SLOW (policy.py): Accurate but takes time - uses full evaluation

The alarm system:
- Detects hazard patterns IMMEDIATELY
- Changes policy parameters WITHOUT reasoning
- The "reason" comes later from the slow pathway

Why two pathways?
- Evolution: waiting to "think" about a predator = death
- The fast path evolved as a survival mechanism
- It's not "irrational" - it's optimized for speed over accuracy

What the alarm does when triggered:
1. Attention: Narrows to threat
2. Policy: Strong avoidance bias
3. Learning: Boost memory encoding
4. Computation: Increase rollout depth ("think harder")

The alarm is NOT "fear as emotion":
- It's a control signal that PRODUCES fear-like behavior
- The label "fear" is assigned by observers
- Inside, it's just: "p_absorb high → change parameters"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class AlarmState:
    """Current state of the alarm system."""
    # Alarm level (0-1)
    level: float = 0.0

    # Source of alarm
    source: Optional[str] = None  # 'threat', 'damage', 'viability'

    # How long alarm has been active
    duration: int = 0

    # Is this a false alarm being resolved?
    resolving: bool = False


@dataclass
class PolicyModulation:
    """
    How the alarm modifies policy parameters.

    These are the "symptoms" of fear - not fear itself.
    """
    # Attention parameters
    attention_width: float = 1.0     # 0.2 = narrow, 1.0 = normal, 2.0 = wide
    attention_bias: Dict[str, float] = None  # Bias toward threat direction

    # Action selection parameters
    avoidance_strength: float = 0.0  # 0-5, added to avoidance scores
    freeze_probability: float = 0.0  # 0-1, chance of freezing

    # Learning parameters
    memory_boost: float = 1.0       # 1.0 = normal, 3.0 = strong encoding
    learning_rate_mod: float = 1.0  # Multiplier on learning rate

    # Computation parameters
    deliberation_depth: int = 2     # How many steps to think ahead
    exploration_suppression: float = 0.0  # How much to reduce exploration


class AlarmSystem:
    """
    Fast-pathway hazard detection and response.

    This is the "amygdala" of the system:
    - Pattern matches against known threats
    - Triggers BEFORE conscious deliberation
    - Biases policy toward survival

    The alarm is learned through experience:
    - Predator → pain → alarm learns predator pattern
    - Next time, alarm triggers BEFORE pain
    """

    def __init__(self):
        self.state = AlarmState()

        # === LEARNED THREAT PATTERNS ===
        # What patterns trigger the alarm?
        self.threat_patterns = {
            'predator_near': 0.0,      # Learned: predators are dangerous
            'low_viability': 0.0,      # Innate: low viability is bad
            'rapid_decline': 0.0,      # Learned: fast drops are dangerous
            'unknown_stimulus': 0.0,   # Innate: unknown could be threat
        }

        # === ALARM PARAMETERS ===
        self.trigger_threshold = 0.3   # Hazard level to trigger alarm
        self.resolve_threshold = 0.1   # Below this, alarm resolves
        self.decay_rate = 0.1          # How fast alarm decreases
        self.learning_rate = 0.2       # How fast patterns are learned

        # === ALARM HISTORY ===
        self._alarm_history = []
        self._false_alarm_count = 0
        self._true_alarm_count = 0

    def assess_hazard(self,
                      sensory: Dict[str, float],
                      viability_metrics: Dict,
                      took_damage: bool = False,
                      predator_distance: Optional[float] = None) -> float:
        """
        Rapidly assess current hazard level.

        This is FAST - no deliberation, just pattern matching.
        Returns hazard level (0-1).
        """
        hazard = 0.0

        # === PATTERN: PREDATOR NEAR ===
        if predator_distance is not None:
            if predator_distance < 3:
                proximity_hazard = 1.0 - (predator_distance / 3.0)
                # Amplified by learned danger
                hazard = max(hazard, proximity_hazard * (0.5 + self.threat_patterns['predator_near']))

        # === PATTERN: LOW VIABILITY ===
        p_absorb = viability_metrics.get('p_absorb', 0)
        if p_absorb > 0.2:
            hazard = max(hazard, p_absorb * (0.8 + self.threat_patterns['low_viability']))

        # === PATTERN: RAPID DECLINE ===
        d_energy = viability_metrics.get('rates', {}).get('d_energy', 0)
        d_integrity = viability_metrics.get('rates', {}).get('d_integrity', 0)
        if d_energy < -0.02 or d_integrity < -0.05:
            decline_hazard = abs(min(d_energy, d_integrity)) * 10
            hazard = max(hazard, decline_hazard)

        # === IMMEDIATE DAMAGE ===
        # Damage triggers alarm instantly (no learning needed)
        if took_damage:
            hazard = max(hazard, 0.9)
            # Also learn that current context is dangerous
            self._learn_threat_pattern('predator_near', 0.8)

        return min(1.0, hazard)

    def update(self,
               hazard_level: float,
               was_false_alarm: bool = False) -> PolicyModulation:
        """
        Update alarm state and compute policy modulation.

        Returns how policy parameters should be changed.
        """
        # === UPDATE ALARM LEVEL ===
        if hazard_level > self.trigger_threshold:
            # Trigger or maintain alarm
            if self.state.level < hazard_level:
                self.state.level = hazard_level
                self.state.source = 'hazard'
                self.state.duration = 0
            else:
                self.state.duration += 1
            self.state.resolving = False
        else:
            # Alarm decaying
            if self.state.level > self.resolve_threshold:
                self.state.level *= (1 - self.decay_rate)
                self.state.resolving = True
            else:
                self.state.level = 0
                self.state.source = None
                self.state.resolving = False

        # === TRACK FALSE ALARMS ===
        if was_false_alarm:
            self._false_alarm_count += 1
            # Reduce sensitivity if too many false alarms
            self.trigger_threshold = min(0.5, self.trigger_threshold + 0.02)
        elif self.state.level > 0.5:
            self._true_alarm_count += 1

        # === COMPUTE POLICY MODULATION ===
        return self._compute_modulation()

    def _compute_modulation(self) -> PolicyModulation:
        """
        Compute how alarm modifies policy parameters.

        These are the observable "fear symptoms":
        - Narrow attention
        - Avoidance behavior
        - Enhanced memory
        - More deliberation
        """
        level = self.state.level
        mod = PolicyModulation()

        if level < 0.1:
            # No alarm - normal operation
            return mod

        # === ATTENTION MODULATION ===
        # Narrow attention to threat
        mod.attention_width = max(0.2, 1.0 - level * 0.8)

        # === AVOIDANCE MODULATION ===
        # Strong bias toward avoiding threat direction
        mod.avoidance_strength = level * 5.0

        # === FREEZE RESPONSE ===
        # Very high alarm → freeze
        if level > 0.8:
            mod.freeze_probability = (level - 0.8) / 0.2

        # === MEMORY MODULATION ===
        # High alarm → strong memory encoding
        mod.memory_boost = 1.0 + level * 2.0

        # === LEARNING MODULATION ===
        # Moderate alarm → enhanced learning
        # Very high alarm → learning impaired (panic)
        if level < 0.7:
            mod.learning_rate_mod = 1.0 + level * 0.5
        else:
            mod.learning_rate_mod = 1.5 - (level - 0.7) * 2

        # === DELIBERATION MODULATION ===
        # Alarm → think more carefully about next steps
        mod.deliberation_depth = 2 + int(level * 3)

        # === EXPLORATION SUPPRESSION ===
        # High alarm → no exploration, only escape
        mod.exploration_suppression = level * 0.9

        return mod

    def _learn_threat_pattern(self, pattern: str, strength: float):
        """Learn that a pattern is associated with danger."""
        if pattern in self.threat_patterns:
            current = self.threat_patterns[pattern]
            self.threat_patterns[pattern] = min(1.0,
                current + self.learning_rate * (strength - current))

    def on_damage(self, source: str = 'predator'):
        """Called when agent takes damage - rapid learning."""
        if source == 'predator':
            self._learn_threat_pattern('predator_near', 1.0)
        self._learn_threat_pattern('rapid_decline', 0.5)

    def on_survival(self, was_close_call: bool = False):
        """Called when agent survives - potentially reduce alarm."""
        if was_close_call:
            self._true_alarm_count += 1
        else:
            # If alarm was high but nothing bad happened
            if self.state.level > 0.3:
                self._false_alarm_count += 1

    def get_threat_direction(self,
                             predator_pos: Tuple[int, int],
                             agent_pos: Tuple[int, int]) -> str:
        """Determine which direction the threat is in."""
        dx = predator_pos[0] - agent_pos[0]
        dy = predator_pos[1] - agent_pos[1]

        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'

    def is_active(self) -> bool:
        """Is alarm currently active?"""
        return self.state.level > 0.1

    def reset(self):
        """Reset alarm state."""
        self.state = AlarmState()
        self._alarm_history = []

    def clear_learning(self):
        """Clear learned threat patterns."""
        self.threat_patterns = {k: 0.0 for k in self.threat_patterns}
        self._false_alarm_count = 0
        self._true_alarm_count = 0

    def get_visualization_data(self) -> Dict:
        """Data for frontend."""
        mod = self._compute_modulation()
        return {
            'level': self.state.level,
            'source': self.state.source,
            'duration': self.state.duration,
            'is_active': self.is_active(),
            'is_resolving': self.state.resolving,
            'threat_patterns': dict(self.threat_patterns),
            'modulation': {
                'attention_width': mod.attention_width,
                'avoidance_strength': mod.avoidance_strength,
                'freeze_probability': mod.freeze_probability,
                'memory_boost': mod.memory_boost,
                'exploration_suppression': mod.exploration_suppression,
                'deliberation_depth': mod.deliberation_depth
            },
            'alarm_stats': {
                'true_alarms': self._true_alarm_count,
                'false_alarms': self._false_alarm_count
            }
        }
