"""
Working Memory Module for Consciousness Simulation (v2)

Improvements over v1:
1. Dynamic alpha: Memory augments but NEVER overrides strong sensory input
2. Soft competition: Gentle inhibition instead of winner-take-all
3. Reset triggers: Fast forget on collision, reward drop, prediction failure
4. Sensory-gated: α scales inversely with sensory strength

Core Principle: "Memory is advisory, not authoritative"
- Strong sensory → memory has minimal influence
- Weak sensory → memory fills the gap
- Conflict detected → memory rapidly clears
"""

from typing import Dict, Optional, List
from collections import deque
import math


class MemoryNeuron:
    """
    A neuron with sustained activity capability.

    Uses a simple recurrent model:
    activity(t+1) = decay * activity(t) + input(t) * gain

    When activity > threshold, the neuron is considered "active"
    and contributes to downstream processing.
    """

    def __init__(self,
                 neuron_id: str,
                 decay: float = 0.92,
                 gain: float = 0.4,
                 threshold: float = 0.15):
        self.id = neuron_id
        self.base_decay = decay
        self.current_decay = decay  # Can be temporarily modified
        self.gain = gain
        self.threshold = threshold

        # Current activity level (0.0 to 1.0)
        self.activity: float = 0.0

        # History for visualization
        self.history: deque = deque(maxlen=100)

    def update(self, sensory_input: float, decay_override: Optional[float] = None) -> float:
        """
        Update memory activity based on new sensory input.

        Args:
            sensory_input: Current sensory intensity (0.0 to 1.0)
            decay_override: Temporary decay rate (for fast forgetting)

        Returns:
            Current activity level after update
        """
        # Use override decay if provided (for reset triggers)
        decay = decay_override if decay_override is not None else self.base_decay

        # Decay existing activity
        self.activity *= decay

        # Add new input (stronger input = stronger memory formation)
        if sensory_input > 0.1:
            self.activity += sensory_input * self.gain

        # Clamp to [0, 1]
        self.activity = max(0.0, min(1.0, self.activity))

        # Record history
        self.history.append(self.activity)

        return self.activity

    def suppress(self, amount: float):
        """Directly reduce activity (for reset triggers)."""
        self.activity = max(0.0, self.activity - amount)

    def is_active(self) -> bool:
        """Check if memory is currently active (above threshold)."""
        return self.activity > self.threshold

    def get_output(self) -> float:
        """Get memory output (0 if below threshold, scaled activity if above)."""
        if self.activity < self.threshold:
            return 0.0
        return (self.activity - self.threshold) / (1.0 - self.threshold)

    def clear(self):
        """Clear the memory."""
        self.activity = 0.0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "activity": round(self.activity, 3),
            "is_active": self.is_active(),
            "output": round(self.get_output(), 3)
        }


class WorkingMemorySystem:
    """
    Working Memory system v2 with improved behavior.

    Key improvements:
    1. Dynamic alpha modulation (sensory strength gates memory influence)
    2. Soft competition (gentle mutual inhibition)
    3. Reset triggers (fast forget on conflict/failure)
    """

    def __init__(self,
                 decay: float = 0.92,
                 gain: float = 0.4,
                 threshold: float = 0.15,
                 soft_inhibition: float = 0.05):  # Gentler than before

        self.directions = ['up', 'down', 'left', 'right']
        self.base_decay = decay

        # Create memory neurons
        self.memories: Dict[str, MemoryNeuron] = {}
        for direction in self.directions:
            self.memories[f"m_{direction}"] = MemoryNeuron(
                neuron_id=f"m_{direction}",
                decay=decay,
                gain=gain,
                threshold=threshold
            )

        self.soft_inhibition = soft_inhibition

        # Statistics
        self.total_updates = 0
        self.memory_used_count = 0

        # Reset trigger state
        self.fast_decay_steps = 0  # Counter for temporary fast decay
        self.fast_decay_rate = 0.5  # Very fast decay when triggered

        # Tracking for reset triggers
        self.last_reward = 0.0
        self.consecutive_negative_rewards = 0
        self.last_action_direction: Optional[str] = None

    def update(self, sensory_state: Dict[str, float]) -> Dict[str, float]:
        """
        Update all memory neurons based on sensory input.
        """
        self.total_updates += 1

        # Determine decay rate (normal or fast)
        if self.fast_decay_steps > 0:
            decay = self.fast_decay_rate
            self.fast_decay_steps -= 1
        else:
            decay = self.base_decay

        # Update each memory neuron
        for direction in self.directions:
            sensory_input = sensory_state.get(direction, 0.0)
            self.memories[f"m_{direction}"].update(sensory_input, decay_override=decay)

        # Apply SOFT competition (not winner-take-all)
        self._apply_soft_competition()

        return self.get_outputs()

    def _apply_soft_competition(self):
        """
        Soft competition: All memories gently inhibit each other.
        This prevents one memory from completely dominating,
        while still allowing a clear winner to emerge.

        Formula: m_i -= β * Σ(m_j for j≠i) / 3
        """
        if self.soft_inhibition <= 0:
            return

        activities = {d: self.memories[f"m_{d}"].activity for d in self.directions}
        total_activity = sum(activities.values())

        if total_activity < 0.1:
            return  # No significant activity

        for direction in self.directions:
            mem = self.memories[f"m_{direction}"]
            # Sum of other memories
            others_sum = total_activity - mem.activity
            # Soft inhibition proportional to others (averaged)
            inhibition = self.soft_inhibition * (others_sum / 3)
            mem.activity = max(0, mem.activity - inhibition)

    # =========================================
    # RESET TRIGGERS
    # =========================================

    def on_collision(self, direction: str):
        """
        Trigger: Agent hit a wall or obstacle.
        Effect: Completely clear memory for that direction.

        Wall collision = this direction is BLOCKED.
        Agent should forget this direction and explore alternatives.
        """
        mem_id = f"m_{direction}"
        if mem_id in self.memories:
            # Complete suppression - this direction is blocked
            self.memories[mem_id].activity = 0.0
            print(f"[MEMORY] Wall hit: {direction.upper()} memory cleared")

    def on_reward_degradation(self, current_reward: float):
        """
        Trigger: Reward is consistently negative.
        Effect: Speed up forgetting to encourage exploration.

        Recovery: Positive reward immediately exits fast decay ("panic over").
        """
        if current_reward < -0.1:
            self.consecutive_negative_rewards += 1
        elif current_reward > 0.1:
            # Positive reward = situation improved, exit panic mode
            self.consecutive_negative_rewards = 0
            if self.fast_decay_steps > 0:
                self.fast_decay_steps = 0
                print(f"[MEMORY] Positive reward: Fast decay cancelled")
        else:
            # Neutral reward - just reset counter, don't cancel fast decay
            self.consecutive_negative_rewards = 0

        # After 3 consecutive negative rewards, trigger fast decay
        if self.consecutive_negative_rewards >= 3:
            self.fast_decay_steps = 5  # Fast decay for next 5 steps
            self.consecutive_negative_rewards = 0
            print(f"[MEMORY] Reward degradation: Fast decay triggered")

    def on_prediction_failure(self, prediction_error: float):
        """
        Trigger: Agency detector reports high prediction error.
        Effect: Current memory is likely outdated, speed up decay.
        """
        if prediction_error > 0.5:  # High error
            self.fast_decay_steps = 3
            print(f"[MEMORY] Prediction failure (err={prediction_error:.2f}): Fast decay")

    def on_direction_change(self, new_direction: str, old_direction: Optional[str]):
        """
        Trigger: Agent successfully changed direction.
        Effect: Slightly boost new direction memory, reduce old.
        """
        if old_direction and old_direction != new_direction:
            old_mem = self.memories.get(f"m_{old_direction}")
            new_mem = self.memories.get(f"m_{new_direction}")

            if old_mem:
                old_mem.activity *= 0.8  # Reduce old direction
            if new_mem:
                new_mem.activity = min(1.0, new_mem.activity + 0.1)  # Boost new

    # =========================================
    # DYNAMIC ALPHA (Sensory-Gated Memory Influence)
    # =========================================

    def get_memory_guidance(self, sensory_state: Dict[str, float]) -> Dict[str, float]:
        """
        Get memory-based directional guidance with DYNAMIC alpha.

        Key principle: α = f(sensory_strength)
        - Strong sensory → α ≈ α_min (tiny memory influence)
        - Weak sensory → α ≈ 0.5 (memory fills the gap)

        Formula: α(s) = α_min + α_max * (1 - s)^2
        Where: α_min = 0.01 (safety floor for noisy sensors)

        This ensures memory NEVER overrides clear sensory signals,
        but retains minimal influence for sensor noise robustness.
        """
        guidance = {}
        alpha_min = 0.01  # Floor value - memory never fully ignored
        alpha_max = 0.49  # Max additional influence (total max = 0.5)

        for direction in self.directions:
            sensory = sensory_state.get(direction, 0.0)
            memory_output = self.memories[f"m_{direction}"].get_output()

            # Dynamic alpha with floor: α = α_min + α_max * (1-s)²
            # When sensory=0: alpha=0.5, When sensory=1: alpha=0.01
            alpha = alpha_min + alpha_max * ((1 - sensory) ** 2)

            # Memory guidance (to be added to sensory current)
            guidance[direction] = memory_output * alpha

        return guidance

    def should_use_memory(self, sensory_state: Dict[str, float]) -> bool:
        """
        Determine if memory should guide behavior.
        Now more conservative: only when sensory is really weak.
        """
        max_sensory = max(sensory_state.values()) if sensory_state else 0
        has_memory = any(m.is_active() for m in self.memories.values())

        # More conservative threshold
        if max_sensory < 0.2 and has_memory:
            self.memory_used_count += 1
            return True
        return False

    # =========================================
    # STANDARD METHODS
    # =========================================

    def get_outputs(self) -> Dict[str, float]:
        """Get all memory outputs."""
        return {
            f"m_{d}": self.memories[f"m_{d}"].get_output()
            for d in self.directions
        }

    def get_active_memories(self) -> List[str]:
        """Get list of currently active memory directions."""
        return [d for d in self.directions if self.memories[f"m_{d}"].is_active()]

    def get_strongest_memory(self) -> Optional[str]:
        """Get the direction with strongest memory, or None if all weak."""
        outputs = self.get_outputs()
        if all(v < 0.01 for v in outputs.values()):
            return None
        return max(outputs, key=outputs.get).replace("m_", "")

    def clear_all(self):
        """Clear all memories (e.g., on reset)."""
        for mem in self.memories.values():
            mem.clear()
        self.consecutive_negative_rewards = 0
        self.fast_decay_steps = 0

    def to_dict(self) -> Dict:
        """Get full state for API response."""
        outputs = self.get_outputs()
        active = self.get_active_memories()
        strongest = self.get_strongest_memory()

        return {
            "memories": {
                mid: mem.to_dict() for mid, mem in self.memories.items()
            },
            "outputs": {k: round(v, 3) for k, v in outputs.items()},
            "active_directions": active,
            "strongest": strongest,
            "memory_usage_rate": round(
                self.memory_used_count / max(1, self.total_updates), 3
            ),
            "fast_decay_active": self.fast_decay_steps > 0
        }

    def get_visualization_data(self) -> Dict:
        """Get data optimized for frontend visualization."""
        return {
            "m_up": {
                "activity": round(self.memories["m_up"].activity, 3),
                "output": round(self.memories["m_up"].get_output(), 3),
                "history": list(self.memories["m_up"].history)[-50:]
            },
            "m_down": {
                "activity": round(self.memories["m_down"].activity, 3),
                "output": round(self.memories["m_down"].get_output(), 3),
                "history": list(self.memories["m_down"].history)[-50:]
            },
            "m_left": {
                "activity": round(self.memories["m_left"].activity, 3),
                "output": round(self.memories["m_left"].get_output(), 3),
                "history": list(self.memories["m_left"].history)[-50:]
            },
            "m_right": {
                "activity": round(self.memories["m_right"].activity, 3),
                "output": round(self.memories["m_right"].get_output(), 3),
                "history": list(self.memories["m_right"].history)[-50:]
            },
            "strongest": self.get_strongest_memory(),
            "using_memory": any(m.is_active() for m in self.memories.values()),
            "fast_decay_active": self.fast_decay_steps > 0
        }
