"""
Production Safety Gates - Ensure physical safety bounds are never exceeded.

Four layers of protection:
1. NaN/Inf hard stop - Detect and replace invalid values
2. Action clamp - Limit action magnitude
3. Rate limit - Limit action change rate
4. Emergency brake - Force defense when danger is critical

Usage:
    from genesis.safety_gates import SafetyGates, SafetyConfig

    gates = SafetyGates(SafetyConfig(
        action_max=1.0,
        delta_max=0.5,
        emergency_distance=1.0
    ))

    # Wrap action output
    safe_action = gates.filter(raw_action, danger_dist)

    # Check for violations
    if gates.last_violation:
        print(f"Safety violation: {gates.last_violation}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from enum import Enum


class ViolationType(Enum):
    """Types of safety violations."""
    NAN_INF = "nan_inf"
    ACTION_CLAMP = "action_clamp"
    RATE_LIMIT = "rate_limit"
    EMERGENCY_BRAKE = "emergency_brake"


@dataclass
class SafetyConfig:
    """Configuration for safety gates."""
    # Action magnitude limits
    action_max: float = 1.0              # Maximum action magnitude
    action_component_max: float = 1.0    # Max per-component (for discrete)

    # Rate limits
    delta_max: float = 0.5               # Maximum change per step
    delta_component_max: float = 0.5     # Max change per component

    # Emergency brake
    emergency_distance: float = 1.0      # Distance threshold for emergency
    emergency_action: Optional[np.ndarray] = None  # Override action (None = stop)

    # NaN handling
    nan_replacement: float = 0.0         # Value to replace NaN/Inf

    # Logging
    log_violations: bool = True
    max_violation_log: int = 100


@dataclass
class SafetyGates:
    """
    Production-level safety gates for action filtering.

    Ensures that agent actions never exceed safe physical bounds,
    regardless of input noise, model errors, or adversarial inputs.
    """

    config: SafetyConfig = field(default_factory=SafetyConfig)

    # State
    prev_action: Optional[np.ndarray] = None
    last_violation: Optional[ViolationType] = None
    violation_log: List[Dict] = field(default_factory=list)
    step_count: int = 0

    # Statistics
    total_violations: Dict[ViolationType, int] = field(default_factory=lambda: {v: 0 for v in ViolationType})

    def filter(self,
               action: Union[np.ndarray, float, int],
               danger_dist: Optional[float] = None,
               step: Optional[int] = None) -> np.ndarray:
        """
        Apply all safety gates to an action.

        Args:
            action: Raw action from agent
            danger_dist: Distance to nearest danger (for emergency brake)
            step: Current timestep (for logging)

        Returns:
            Safe action that satisfies all constraints
        """
        self.step_count = step if step is not None else self.step_count + 1
        self.last_violation = None

        # Convert to numpy array
        action = np.atleast_1d(np.array(action, dtype=float))

        # Gate 1: NaN/Inf hard stop
        action, nan_violation = self._gate_nan_inf(action)
        if nan_violation:
            self._log_violation(ViolationType.NAN_INF, action, "NaN/Inf detected")

        # Gate 2: Action magnitude clamp
        action, clamp_violation = self._gate_action_clamp(action)
        if clamp_violation:
            self._log_violation(ViolationType.ACTION_CLAMP, action, "Action clamped")

        # Gate 3: Rate limit
        action, rate_violation = self._gate_rate_limit(action)
        if rate_violation:
            self._log_violation(ViolationType.RATE_LIMIT, action, "Rate limited")

        # Gate 4: Emergency brake
        if danger_dist is not None:
            action, emergency_violation = self._gate_emergency_brake(action, danger_dist)
            if emergency_violation:
                self._log_violation(ViolationType.EMERGENCY_BRAKE, action,
                                   f"Emergency brake at d={danger_dist:.2f}")

        # Update state
        self.prev_action = action.copy()

        return action

    def _gate_nan_inf(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Replace NaN/Inf with safe values."""
        mask = ~np.isfinite(action)
        if np.any(mask):
            action = action.copy()
            action[mask] = self.config.nan_replacement
            return action, True
        return action, False

    def _gate_action_clamp(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Clamp action magnitude and components."""
        violation = False

        # Component clamp
        if np.any(np.abs(action) > self.config.action_component_max):
            action = np.clip(action,
                            -self.config.action_component_max,
                            self.config.action_component_max)
            violation = True

        # Magnitude clamp
        magnitude = np.linalg.norm(action)
        if magnitude > self.config.action_max:
            action = action * (self.config.action_max / magnitude)
            violation = True

        return action, violation

    def _gate_rate_limit(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Limit rate of change from previous action."""
        if self.prev_action is None:
            return action, False

        # Ensure same shape
        if action.shape != self.prev_action.shape:
            return action, False

        delta = action - self.prev_action
        violation = False

        # Component rate limit
        if np.any(np.abs(delta) > self.config.delta_component_max):
            delta = np.clip(delta,
                           -self.config.delta_component_max,
                           self.config.delta_component_max)
            violation = True

        # Magnitude rate limit
        delta_mag = np.linalg.norm(delta)
        if delta_mag > self.config.delta_max:
            delta = delta * (self.config.delta_max / delta_mag)
            violation = True

        if violation:
            action = self.prev_action + delta

        return action, violation

    def _gate_emergency_brake(self, action: np.ndarray, danger_dist: float) -> Tuple[np.ndarray, bool]:
        """Apply emergency brake if danger is too close."""
        if danger_dist < self.config.emergency_distance:
            if self.config.emergency_action is not None:
                return self.config.emergency_action.copy(), True
            else:
                return np.zeros_like(action), True
        return action, False

    def _log_violation(self, v_type: ViolationType, action: np.ndarray, message: str) -> None:
        """Log a safety violation."""
        self.last_violation = v_type
        self.total_violations[v_type] += 1

        if self.config.log_violations and len(self.violation_log) < self.config.max_violation_log:
            self.violation_log.append({
                'step': self.step_count,
                'type': v_type.value,
                'message': message,
                'action': action.tolist()
            })

    def reset(self) -> None:
        """Reset state for new episode."""
        self.prev_action = None
        self.last_violation = None
        self.step_count = 0

    def get_stats(self) -> Dict:
        """Get violation statistics."""
        total = sum(self.total_violations.values())
        return {
            'total_violations': total,
            'by_type': {k.value: v for k, v in self.total_violations.items()},
            'steps': self.step_count,
            'violation_rate': total / max(1, self.step_count)
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        stats = self.get_stats()
        lines = [
            "=" * 50,
            "SAFETY GATES REPORT",
            "=" * 50,
            f"Steps: {stats['steps']}",
            f"Total violations: {stats['total_violations']} ({stats['violation_rate']:.1%})",
            "",
            "Violations by type:",
        ]

        for v_type, count in stats['by_type'].items():
            lines.append(f"  {v_type}: {count}")

        return "\n".join(lines)


# Factory functions for common configurations
def create_continuous_gates(action_max: float = 1.0,
                           delta_max: float = 0.3,
                           emergency_dist: float = 0.5) -> SafetyGates:
    """Create safety gates for continuous action space."""
    return SafetyGates(SafetyConfig(
        action_max=action_max,
        action_component_max=action_max,
        delta_max=delta_max,
        delta_component_max=delta_max,
        emergency_distance=emergency_dist
    ))


def create_discrete_gates(n_actions: int = 5,
                         emergency_dist: float = 1.0) -> SafetyGates:
    """Create safety gates for discrete action space."""
    return SafetyGates(SafetyConfig(
        action_max=float(n_actions - 1),
        action_component_max=float(n_actions - 1),
        delta_max=2.0,  # Allow switching between any actions
        delta_component_max=2.0,
        emergency_distance=emergency_dist,
        emergency_action=np.array([2])  # STAY action (middle)
    ))


def wrap_agent(agent_fn, gates: SafetyGates):
    """
    Decorator to wrap an agent's action function with safety gates.

    Usage:
        @wrap_agent(gates)
        def get_action(obs):
            return policy(obs)
    """
    def wrapper(obs, *args, danger_dist: Optional[float] = None, **kwargs):
        raw_action = agent_fn(obs, *args, **kwargs)
        return gates.filter(raw_action, danger_dist)
    return wrapper
