from typing import List, Optional
from collections import deque
import math

class Synapse:
    """
    Represents a synaptic connection with STDP learning.
    
    STDP (Spike-Timing-Dependent Plasticity):
    - If pre fires before post: LTP (Long-Term Potentiation) → strengthen
    - If pre fires after post: LTD (Long-Term Depression) → weaken
    """
    
    # STDP parameters (in simulation time steps)
    A_PLUS = 3.0       # Strong LTP for reinforcement learning
    A_MINUS = 0.3      # Weaker LTD - only punish truly bad timing
    TAU_PLUS = 25.0    # ms - wider window for LTP
    TAU_MINUS = 15.0   # ms - narrower window for LTD
    TAU_E = 1000.0     # Slow eligibility decay (ms)
    ACTIVITY_WINDOW = 500  # Steps to consider synapse "recently active"

    # Weight bounds (reduced max to prevent over-dominance)
    W_MIN = 5.0   # Increased from 2.0 to ensure meaningful synaptic strength
    W_MAX = 80.0  # Reduced from 150.0 to prevent extreme weights
    
    def __init__(
        self,
        pre_neuron_id: str,
        post_neuron_id: str,
        weight: float = 5.0,
        delay: int = 3,
        enable_stdp: bool = True,
        is_inhibitory: bool = False
    ):
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        # Weight is always stored as absolute magnitude
        self.weight = abs(weight)
        self.initial_weight = abs(weight)  # Store for reset
        self.delay = delay
        self.enable_stdp = enable_stdp
        self.is_inhibitory = is_inhibitory  # Sign of the connection
        
        # Spike queue for delay
        self.spike_queue: deque = deque()
        
        # STDP timing tracking
        self.last_pre_spike_time: Optional[int] = None
        self.last_post_spike_time: Optional[int] = None
        
        # 3-Factor STDP: Eligibility Trace
        self.eligibility = 0.0

        # Synaptic current with decay
        self.synaptic_current = 0.0

        # Activity tracking for smart homeostasis
        self.last_activity_time = 0
        self.current_time = 0
        
        # Learning history for visualization
        self.weight_history: List[float] = [weight]
        self.stdp_events: List[dict] = []  # {time, delta_w, type}
        
    def receive_spike(self, current_time: int):
        """Called when pre-synaptic neuron fires."""
        self.spike_queue.append(self.delay)
        self.last_pre_spike_time = current_time
        self.last_activity_time = current_time  # Track activity

        # LTD: pre fires after post
        if self.enable_stdp and self.last_post_spike_time is not None:
            delta_t = current_time - self.last_post_spike_time
            if delta_t > 0:
                # Update eligibility (trace) instead of weight
                delta_e = -self.A_MINUS * math.exp(-delta_t / self.TAU_MINUS)
                self.eligibility += delta_e
        
    def notify_post_spike(self, current_time: int):
        """Called when post-synaptic neuron fires."""
        self.last_post_spike_time = current_time
        self.last_activity_time = current_time  # Track activity

        # LTP: post fires after pre
        if self.enable_stdp and self.last_pre_spike_time is not None:
            delta_t = current_time - self.last_pre_spike_time
            if delta_t > 0:
                # Update eligibility (trace) instead of weight
                delta_e = self.A_PLUS * math.exp(-delta_t / self.TAU_PLUS)
                self.eligibility += delta_e
                
    def apply_reward(self, reward: float, current_time: int):
        """
        3rd factor: Apply global reward signal to update weight.
        Weight change is proportional to (eligibility * reward).
        """
        if not self.enable_stdp:
            return
            
        # Only update if there is significant eligibility or reward
        delta_w = self.eligibility * reward
        if abs(delta_w) > 0.001:
            self._apply_weight_change(delta_w, current_time, "LTP" if delta_w > 0 else "LTD")
        
    def _apply_weight_change(self, delta_w: float, time: int, event_type: str):
        """Apply weight change with bounds checking."""
        self.weight = max(self.W_MIN, min(self.W_MAX, self.weight + delta_w))
        
        # Record the event
        self.stdp_events.append({
            "time": time,
            "delta_w": delta_w,
            "type": event_type,
            "new_weight": self.weight
        })
        
        if len(self.stdp_events) > 100:
            self.stdp_events = self.stdp_events[-100:]
            
    def step(self) -> float:
        """Advance time. Returns synaptic current. Decays eligibility."""
        self.current_time += 1

        # Decays eligibility trace (Slow decay to maintain learning signal)
        # 0.999 means eligibility halves in ~693 steps
        self.eligibility *= 0.999

        # Activity-based homeostasis: Only decay INACTIVE synapses
        # Active synapses (recently used) are protected from decay
        # This prevents useful pathways from losing their strength
        if self.enable_stdp and self.weight > self.W_MIN:
            time_since_activity = self.current_time - self.last_activity_time
            if time_since_activity > self.ACTIVITY_WINDOW:
                # Inactive synapse: slow decay toward baseline
                decay_rate = 0.00005  # Very slow decay for unused synapses
                self.weight -= decay_rate * (self.weight - self.W_MIN)

        # Decays synaptic current (Exponential decay)
        self.synaptic_current *= 0.7

        # Decrement all delays
        for i in range(len(self.spike_queue)):
            self.spike_queue[i] -= 1

        while self.spike_queue and self.spike_queue[0] <= 0:
            self.spike_queue.popleft()
            self.synaptic_current += self.weight

        self.weight_history.append(self.weight)
        if len(self.weight_history) > 1000:
            self.weight_history = self.weight_history[-1000:]

        # Apply inhibitory sign when returning current
        return -self.synaptic_current if self.is_inhibitory else self.synaptic_current
    
    def to_dict(self):
        return {
            "pre": self.pre_neuron_id,
            "post": self.post_neuron_id,
            "weight": round(self.weight, 2),
            "is_inhibitory": self.is_inhibitory,
            "delay": self.delay,
            "pending_spikes": len(self.spike_queue),
            "stdp_enabled": self.enable_stdp,
            "recent_stdp": self.stdp_events[-5:] if self.stdp_events else []
        }
