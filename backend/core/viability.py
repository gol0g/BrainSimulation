"""
Viability System - The Foundation of Existence

Core Concept:
- Living systems must stay within a "viability kernel" - a set of states
  where continued existence is possible
- Death is not a game rule (HP=0), but an ABSORBING STATE where:
  - Future agency becomes 0
  - No more learning, action, or experience possible
  - The process of "self" terminates

Why fear death?
- Not because it's programmed, but because:
  - Any system that can predict its own termination
  - And has any preference over future states
  - Will necessarily avoid paths leading to termination
- Fear EMERGES from p_absorb (probability of entering absorbing state)

Viability Channels (not HP):
- energy: Metabolic resources (0 = starvation death)
- integrity: Tissue damage accumulation (0 = damage death)
- threat_exposure: Cumulative danger exposure (affects integrity)
- Each has a "recovery zone" and a "critical zone"

Key insight:
- Viability is not a goal, it's a PRECONDITION for having goals
- The system doesn't "want" to live - living is what allows wanting
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import math


@dataclass
class ViabilityState:
    """Current state of viability channels."""
    energy: float = 0.8        # 0-1, metabolic resources
    integrity: float = 1.0     # 0-1, tissue/structural health
    threat_exposure: float = 0.0  # 0-1, accumulated danger

    # Rates of change (for prediction)
    d_energy: float = 0.0
    d_integrity: float = 0.0
    d_threat: float = 0.0


@dataclass
class ViabilityParams:
    """Parameters defining the viability kernel."""
    # Critical thresholds (below = absorbing state risk)
    critical_energy: float = 0.1
    critical_integrity: float = 0.15
    critical_threat: float = 0.9

    # Recovery thresholds (above = stable)
    stable_energy: float = 0.4
    stable_integrity: float = 0.5
    stable_threat: float = 0.3

    # Decay/recovery rates
    energy_decay: float = 0.002       # Per step energy loss
    integrity_recovery: float = 0.01  # Natural healing
    threat_decay: float = 0.05        # Threat fades over time

    # Damage parameters
    predator_damage: float = 0.25     # Integrity loss per hit
    food_energy: float = 0.3          # Energy gain from food

    # Absorbing state parameters
    absorb_threshold: float = 0.05    # Below this = death


class ViabilitySystem:
    """
    Tracks the agent's distance from the absorbing state (death).

    Key outputs:
    - p_absorb: Probability of entering absorbing state soon
    - viability_margin: How far from the kernel boundary
    - channel_urgencies: Which channels need attention

    This replaces homeostasis but with deeper meaning:
    - Not "trying to maintain HP"
    - But "the system's existence depends on staying viable"
    """

    def __init__(self, params: ViabilityParams = None):
        self.params = params or ViabilityParams()
        self.state = ViabilityState()

        # Track history for rate estimation
        self._history_len = 10
        self._energy_history = [0.8] * self._history_len
        self._integrity_history = [1.0] * self._history_len

        # Absorbing state flag
        self.is_absorbed = False
        self._steps_in_critical = 0

    def update(self,
               ate_food: bool = False,
               took_damage: float = 0.0,
               near_threat: bool = False,
               resting: bool = False) -> Dict:
        """
        Update viability state based on events.

        Returns dict with:
        - p_absorb: Probability of death
        - viability_margin: Distance from kernel boundary
        - urgencies: Per-channel urgency scores
        - is_critical: Whether any channel is critical
        """
        if self.is_absorbed:
            return self._absorbed_state()

        # === ENERGY DYNAMICS ===
        # Energy always decays (metabolism)
        energy_change = -self.params.energy_decay
        if ate_food:
            energy_change += self.params.food_energy
        if resting:
            energy_change *= 0.5  # Resting slows metabolism

        self.state.energy = max(0.0, min(1.0,
            self.state.energy + energy_change))

        # === INTEGRITY DYNAMICS ===
        # Damage reduces integrity, natural healing restores
        integrity_change = -took_damage
        if self.state.integrity < 1.0 and not near_threat:
            integrity_change += self.params.integrity_recovery

        self.state.integrity = max(0.0, min(1.0,
            self.state.integrity + integrity_change))

        # === THREAT EXPOSURE ===
        # Being near threats accumulates exposure
        if near_threat:
            self.state.threat_exposure = min(1.0,
                self.state.threat_exposure + 0.1)
        else:
            self.state.threat_exposure = max(0.0,
                self.state.threat_exposure - self.params.threat_decay)

        # === RATE ESTIMATION ===
        self._energy_history.append(self.state.energy)
        self._energy_history.pop(0)
        self._integrity_history.append(self.state.integrity)
        self._integrity_history.pop(0)

        self.state.d_energy = (self._energy_history[-1] -
                               self._energy_history[0]) / self._history_len
        self.state.d_integrity = (self._integrity_history[-1] -
                                  self._integrity_history[0]) / self._history_len

        # === CHECK FOR ABSORBING STATE ===
        if (self.state.energy < self.params.absorb_threshold or
            self.state.integrity < self.params.absorb_threshold):
            self._steps_in_critical += 1
            if self._steps_in_critical > 3:  # Grace period
                self.is_absorbed = True
                return self._absorbed_state()
        else:
            self._steps_in_critical = max(0, self._steps_in_critical - 1)

        return self._compute_viability_metrics()

    def _compute_viability_metrics(self) -> Dict:
        """Compute key viability metrics."""
        p = self.params
        s = self.state

        # === CHANNEL URGENCIES ===
        # How close each channel is to its critical threshold
        # Uses sigmoid for smooth transition

        def urgency(value: float, critical: float, stable: float) -> float:
            """0 at stable, 1 at critical, smooth transition."""
            if value >= stable:
                return 0.0
            if value <= critical:
                return 1.0
            # Linear interpolation in the danger zone
            return 1.0 - (value - critical) / (stable - critical)

        energy_urgency = urgency(s.energy, p.critical_energy, p.stable_energy)
        integrity_urgency = urgency(s.integrity, p.critical_integrity, p.stable_integrity)
        threat_urgency = urgency(1.0 - s.threat_exposure,
                                 1.0 - p.critical_threat,
                                 1.0 - p.stable_threat)

        # === P_ABSORB: Probability of entering absorbing state ===
        # Based on current urgencies and rates of change

        # Base probability from current state
        p_base = max(energy_urgency, integrity_urgency) ** 2

        # Trajectory adjustment: if getting worse, probability higher
        trajectory_factor = 1.0
        if s.d_energy < -0.01:  # Energy dropping fast
            trajectory_factor += abs(s.d_energy) * 10
        if s.d_integrity < -0.02:  # Taking continuous damage
            trajectory_factor += abs(s.d_integrity) * 20

        # Threat exposure increases base probability
        threat_factor = 1.0 + s.threat_exposure * 0.5

        p_absorb = min(0.99, p_base * trajectory_factor * threat_factor)

        # === VIABILITY MARGIN ===
        # Minimum distance from any critical threshold
        margins = [
            (s.energy - p.critical_energy) / (1.0 - p.critical_energy),
            (s.integrity - p.critical_integrity) / (1.0 - p.critical_integrity),
            (p.critical_threat - s.threat_exposure) / p.critical_threat
        ]
        viability_margin = max(0.0, min(margins))

        # === IS CRITICAL ===
        is_critical = (energy_urgency > 0.7 or
                       integrity_urgency > 0.7 or
                       threat_urgency > 0.7)

        return {
            'p_absorb': p_absorb,
            'viability_margin': viability_margin,
            'is_critical': is_critical,
            'urgencies': {
                'energy': energy_urgency,
                'integrity': integrity_urgency,
                'threat': threat_urgency
            },
            'state': {
                'energy': s.energy,
                'integrity': s.integrity,
                'threat_exposure': s.threat_exposure
            },
            'rates': {
                'd_energy': s.d_energy,
                'd_integrity': s.d_integrity
            },
            'is_absorbed': False
        }

    def _absorbed_state(self) -> Dict:
        """Return metrics for absorbed (dead) state."""
        return {
            'p_absorb': 1.0,
            'viability_margin': 0.0,
            'is_critical': True,
            'urgencies': {'energy': 1.0, 'integrity': 1.0, 'threat': 1.0},
            'state': {'energy': 0.0, 'integrity': 0.0, 'threat_exposure': 1.0},
            'rates': {'d_energy': 0.0, 'd_integrity': 0.0},
            'is_absorbed': True
        }

    def predict_future(self, steps: int,
                       actions: list = None) -> list:
        """
        Predict viability trajectory into the future.

        This is crucial for the agent to "fear" death:
        - It can see that certain paths lead to p_absorb → 1
        - And therefore avoid them

        Returns list of predicted p_absorb values.
        """
        # Simple linear extrapolation based on current rates
        predictions = []

        pred_energy = self.state.energy
        pred_integrity = self.state.integrity

        for i in range(steps):
            # Project forward
            pred_energy += self.state.d_energy
            pred_integrity += self.state.d_integrity

            # Clamp
            pred_energy = max(0.0, min(1.0, pred_energy))
            pred_integrity = max(0.0, min(1.0, pred_integrity))

            # Compute p_absorb for this future state
            e_urg = max(0.0, 1.0 - (pred_energy - self.params.critical_energy) /
                        (self.params.stable_energy - self.params.critical_energy))
            i_urg = max(0.0, 1.0 - (pred_integrity - self.params.critical_integrity) /
                        (self.params.stable_integrity - self.params.critical_integrity))

            p_absorb = min(0.99, max(e_urg, i_urg) ** 2)
            predictions.append(p_absorb)

        return predictions

    def on_damage(self, amount: float):
        """Record damage event (from predator, etc.)"""
        self.state.integrity = max(0.0, self.state.integrity - amount)
        self.state.threat_exposure = min(1.0, self.state.threat_exposure + 0.3)

    def on_food(self, amount: float = None):
        """Record food consumption."""
        amount = amount or self.params.food_energy
        self.state.energy = min(1.0, self.state.energy + amount)

    def reset(self):
        """Reset to initial viable state."""
        self.state = ViabilityState()
        self.is_absorbed = False
        self._steps_in_critical = 0
        self._energy_history = [0.8] * self._history_len
        self._integrity_history = [1.0] * self._history_len

    def get_visualization_data(self) -> Dict:
        """Data for frontend."""
        metrics = self._compute_viability_metrics()
        return {
            'energy': self.state.energy,
            'integrity': self.state.integrity,
            'threat_exposure': self.state.threat_exposure,
            'p_absorb': metrics['p_absorb'],
            'viability_margin': metrics['viability_margin'],
            'is_critical': metrics['is_critical'],
            'urgencies': metrics['urgencies'],
            'd_energy': self.state.d_energy,
            'd_integrity': self.state.d_integrity,
            'is_absorbed': self.is_absorbed
        }
