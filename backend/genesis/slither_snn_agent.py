"""
Slither.io SNN Agent - Biological Brain for Snake Survival

Architecture: 500K+ neurons with multi-channel sensory processing

Sensory Channels:
- Food Eye: Detects food (attract)
- Enemy Eye: Detects enemy bodies (avoid)
- Body Eye: Detects own body (self-collision avoidance)

Motor Output:
- Heading neurons: Direction control (left/right)
- Boost neurons: Emergency escape

Learning: DA-STDP with reward from eating, penalty from death
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from snn_scalable import ScalableSNNConfig, SparseSynapses, SparseLIFLayer, AdaptiveLIFLayer, DEVICE
from slither_gym import SlitherGym, SlitherConfig

# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "slither"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SlitherBrainConfig:
    """Configuration for Slither SNN Brain"""
    # Sensory neurons (per channel, per ray direction)
    n_rays: int = 32  # Number of sensor rays

    # Sensory layers
    n_food_eye: int = 8000       # Food detection (8K)
    n_enemy_eye: int = 8000      # Enemy detection (8K)
    n_body_eye: int = 4000       # Self-body detection (4K)

    # Processing layers (Mushroom Body analog)
    n_integration_1: int = 50000   # First integration layer (50K)
    n_integration_2: int = 50000   # Second integration layer (50K)

    # Motor neurons
    n_motor_left: int = 5000     # Turn left
    n_motor_right: int = 5000    # Turn right
    n_motor_boost: int = 3000    # Emergency boost

    # Specialized circuits
    n_fear_circuit: int = 10000   # Enemy avoidance (10K)
    n_hunger_circuit: int = 10000 # Food seeking (10K)

    # SNN parameters
    sparsity: float = 0.005  # 0.5% connectivity (more sparse for larger network)
    lif_beta: float = 0.9
    lif_threshold: float = 1.0

    # Learning
    stdp_tau: float = 500.0
    a_plus: float = 0.005   # Smaller for larger network
    a_minus: float = 0.006

    # Inhibition
    fear_inhibition: float = 0.9  # Strong inhibition when enemy near

    @property
    def total_neurons(self) -> int:
        return (self.n_food_eye + self.n_enemy_eye + self.n_body_eye +
                self.n_integration_1 + self.n_integration_2 +
                self.n_motor_left + self.n_motor_right + self.n_motor_boost +
                self.n_fear_circuit + self.n_hunger_circuit)


class SlitherBrain:
    """
    500K+ Neuron SNN Brain for Slither.io

    Architecture:
    ```
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Food Eye   │  │  Enemy Eye   │  │   Body Eye   │
    │    (8K)      │  │    (8K)      │  │    (4K)      │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐
    │Hunger Circuit│  │ Fear Circuit │
    │    (10K)     │  │    (10K)     │
    └──────┬───────┘  └──────┬───────┘
           │                 │
           └────────┬────────┘
                    ▼
         ┌────────────────────┐
         │   Integration 1    │
         │       (50K)        │
         └─────────┬──────────┘
                   ▼
         ┌────────────────────┐
         │   Integration 2    │
         │       (50K)        │
         └─────────┬──────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    ┌───────┐  ┌───────┐  ┌───────┐
    │ Left  │  │ Right │  │ Boost │
    │ (5K)  │  │ (5K)  │  │ (3K)  │
    └───────┘  └───────┘  └───────┘
    ```
    """

    def __init__(self, config: Optional[SlitherBrainConfig] = None):
        self.config = config or SlitherBrainConfig()
        lif = ScalableSNNConfig(beta=self.config.lif_beta, threshold=self.config.lif_threshold)

        print(f"Initializing SlitherBrain with {self.config.total_neurons:,} neurons...")

        # === SENSORY LAYERS ===
        self.food_eye = SparseLIFLayer(self.config.n_food_eye, lif)
        self.enemy_eye = SparseLIFLayer(self.config.n_enemy_eye, lif)
        self.body_eye = SparseLIFLayer(self.config.n_body_eye, lif)

        # === SPECIALIZED CIRCUITS ===
        self.hunger_circuit = SparseLIFLayer(self.config.n_hunger_circuit, lif)
        self.fear_circuit = SparseLIFLayer(self.config.n_fear_circuit, lif)

        # === INTEGRATION LAYERS (Mushroom Body) ===
        self.integration_1 = SparseLIFLayer(self.config.n_integration_1, lif)
        self.integration_2 = SparseLIFLayer(self.config.n_integration_2, lif)

        # === MOTOR LAYERS (Adaptive - Neural Fatigue) ===
        # 운동 뉴런에 적응(피로) 적용: 같은 방향 반복 시 피로 → 자연스러운 방향 전환
        # 핵심: 피로가 빨리 쌓이고 느리게 회복 → 방향 전환 유도
        self.motor_left = AdaptiveLIFLayer(
            self.config.n_motor_left, lif,
            tau_adapt=200.0,   # 200ms 회복 (느린 회복)
            adapt_beta=1.0     # 강한 피로 축적
        )
        self.motor_right = AdaptiveLIFLayer(
            self.config.n_motor_right, lif,
            tau_adapt=200.0,
            adapt_beta=1.0
        )
        self.motor_boost = SparseLIFLayer(self.config.n_motor_boost, lif)  # 부스트는 적응 없음

        # === SYNAPSES ===
        sp = self.config.sparsity

        # Sensory → Specialized circuits
        self.syn_food_hunger = SparseSynapses(self.config.n_food_eye, self.config.n_hunger_circuit, sp)
        self.syn_enemy_fear = SparseSynapses(self.config.n_enemy_eye, self.config.n_fear_circuit, sp)
        self.syn_body_fear = SparseSynapses(self.config.n_body_eye, self.config.n_fear_circuit, sp * 0.5)

        # Specialized → Integration 1
        self.syn_hunger_int1 = SparseSynapses(self.config.n_hunger_circuit, self.config.n_integration_1, sp)
        self.syn_fear_int1 = SparseSynapses(self.config.n_fear_circuit, self.config.n_integration_1, sp)

        # Integration 1 → Integration 2
        self.syn_int1_int2 = SparseSynapses(self.config.n_integration_1, self.config.n_integration_2, sp)

        # Integration 2 → Motor
        self.syn_int2_left = SparseSynapses(self.config.n_integration_2, self.config.n_motor_left, sp)
        self.syn_int2_right = SparseSynapses(self.config.n_integration_2, self.config.n_motor_right, sp)
        self.syn_int2_boost = SparseSynapses(self.config.n_integration_2, self.config.n_motor_boost, sp)

        # === CROSS-INHIBITION: Fear --| Hunger (don't eat when scared) ===
        self.syn_fear_hunger_inhib = SparseSynapses(self.config.n_fear_circuit, self.config.n_hunger_circuit, sp * 2)

        # === DIRECT REFLEX: Fear → Boost (emergency escape) ===
        self.syn_fear_boost = SparseSynapses(self.config.n_fear_circuit, self.config.n_motor_boost, sp * 3)

        # === DIRECT REFLEX: Food → Motor (food seeking) ===
        # Biologically valid: even simple organisms have direct sensorimotor reflexes
        # Food on left → turn left, Food on right → turn right
        self.syn_food_motor_left = SparseSynapses(self.config.n_food_eye, self.config.n_motor_left, sp * 2)
        self.syn_food_motor_right = SparseSynapses(self.config.n_food_eye, self.config.n_motor_right, sp * 2)

        # === INNATE REFLEX: Enemy → Motor (avoidance - EVOLVED!) ===
        # This is NOT a hack - it's "evolutionary pre-wiring"
        # Enemy on LEFT → activate RIGHT motor (turn AWAY from danger)
        # Enemy on RIGHT → activate LEFT motor (turn AWAY from danger)
        # Initial weights are HIGH (0.5x boost) = survival instinct from birth
        self.syn_enemy_motor_left = SparseSynapses(self.config.n_enemy_eye, self.config.n_motor_left, sp * 2)
        self.syn_enemy_motor_right = SparseSynapses(self.config.n_enemy_eye, self.config.n_motor_right, sp * 2)

        # Scale up innate avoidance weights (진화된 본능 = 강한 초기 가중치)
        # These are STILL LEARNABLE via DA-STDP - can be modified by experience
        innate_boost = 3.0  # 3x stronger than random initialization
        self.syn_enemy_motor_left.weights = self.syn_enemy_motor_left.weights * innate_boost
        self.syn_enemy_motor_right.weights = self.syn_enemy_motor_right.weights * innate_boost
        self.syn_enemy_motor_left._rebuild_sparse()
        self.syn_enemy_motor_right._rebuild_sparse()

        # State
        self.dopamine = 0.5
        self.fear_level = 0.0
        self.hunger_level = 0.5
        self.current_heading = 0.0  # Current heading angle
        self.steps = 0

        # Statistics
        self.stats = {
            'food_eaten': 0,
            'boosts': 0,
            'fear_triggers': 0,
            'left_turns': 0,
            'right_turns': 0,
            'fatigue_switches': 0  # 피로로 인한 방향 전환 횟수
        }
        self._last_turn_dir = None  # 이전 회전 방향 추적

        print(f"  Food Eye: {self.config.n_food_eye:,}")
        print(f"  Enemy Eye: {self.config.n_enemy_eye:,}")
        print(f"  Body Eye: {self.config.n_body_eye:,}")
        print(f"  Hunger Circuit: {self.config.n_hunger_circuit:,}")
        print(f"  Fear Circuit: {self.config.n_fear_circuit:,}")
        print(f"  Integration: {self.config.n_integration_1 + self.config.n_integration_2:,}")
        print(f"  Motor: {self.config.n_motor_left + self.config.n_motor_right + self.config.n_motor_boost:,} (Adaptive)")
        print(f"  Total: {self.config.total_neurons:,} neurons")
        print(f"  Sparsity: {sp*100:.1f}%")
        print(f"  Neural Adaptation: tau=200ms, beta=1.0")

    def process(self, sensor_input: np.ndarray, reward: float = 0.0) -> Tuple[float, float, bool]:
        """
        Process sensor input and return action

        Args:
            sensor_input: (3, n_rays) array from SlitherGym.get_sensor_input()
                - [0, :] = food signals (0-1, closer = higher)
                - [1, :] = enemy signals
                - [2, :] = body signals (wall proximity)
            reward: Learning signal

        Returns:
            (target_x, target_y, boost): Action tuple (normalized 0-1)
        """
        # Unpack sensor input
        food_signal = sensor_input[0]  # (n_rays,)
        enemy_signal = sensor_input[1]
        body_signal = sensor_input[2]

        # === ENCODE TO POPULATION ===
        food_input = self._encode_rays(food_signal, self.config.n_food_eye)
        enemy_input = self._encode_rays(enemy_signal, self.config.n_enemy_eye)
        body_input = self._encode_rays(body_signal, self.config.n_body_eye)

        # === SENSORY PROCESSING ===
        food_spikes = self.food_eye.forward(food_input * 15.0)
        enemy_spikes = self.enemy_eye.forward(enemy_input * 20.0)  # Higher gain for danger
        body_spikes = self.body_eye.forward(body_input * 10.0)

        # === SPECIALIZED CIRCUITS ===
        # Hunger (food seeking)
        hunger_in = self.syn_food_hunger.forward(food_spikes)
        hunger_spikes = self.hunger_circuit.forward(hunger_in * 30.0)
        self.hunger_level = hunger_spikes.sum().item() / self.config.n_hunger_circuit

        # Fear (danger avoidance)
        fear_enemy = self.syn_enemy_fear.forward(enemy_spikes)
        fear_body = self.syn_body_fear.forward(body_spikes)
        fear_in = fear_enemy + fear_body
        fear_spikes = self.fear_circuit.forward(fear_in * 40.0)
        self.fear_level = fear_spikes.sum().item() / self.config.n_fear_circuit

        if self.fear_level > 0.1:
            self.stats['fear_triggers'] += 1

        # === CROSS-INHIBITION: Fear suppresses Hunger ===
        fear_inhibition = self.syn_fear_hunger_inhib.forward(fear_spikes)
        # Suppress hunger when scared
        hunger_spikes = hunger_spikes * (1.0 - self.config.fear_inhibition * fear_inhibition.mean())

        # === INTEGRATION ===
        int1_in = (self.syn_hunger_int1.forward(hunger_spikes) +
                   self.syn_fear_int1.forward(fear_spikes))
        int1_spikes = self.integration_1.forward(int1_in * 20.0)

        int2_in = self.syn_int1_int2.forward(int1_spikes)
        int2_spikes = self.integration_2.forward(int2_in * 20.0)

        # === MOTOR OUTPUT ===
        left_in = self.syn_int2_left.forward(int2_spikes)
        right_in = self.syn_int2_right.forward(int2_spikes)
        boost_in = self.syn_int2_boost.forward(int2_spikes)

        # Direct fear → boost reflex
        fear_boost = self.syn_fear_boost.forward(fear_spikes)
        boost_in = boost_in + fear_boost * 2.0

        # === DIRECT FOOD REFLEX ===
        # Encode food with directional bias (left half → left motor, right half → right motor)
        food_left_input, food_right_input = self._encode_food_directional(food_signal)
        food_reflex_left = self.syn_food_motor_left.forward(food_left_input)
        food_reflex_right = self.syn_food_motor_right.forward(food_right_input)

        # Add direct food reflex (strong weight for Phase 1)
        left_in = left_in + food_reflex_left * 3.0
        right_in = right_in + food_reflex_right * 3.0

        # === INNATE ENEMY AVOIDANCE REFLEX (진화된 본능!) ===
        # Enemy on LEFT → activate RIGHT motor (turn AWAY)
        # Enemy on RIGHT → activate LEFT motor (turn AWAY)
        # Note: OPPOSITE of food reflex - we turn AWAY from danger
        enemy_left_input, enemy_right_input = self._encode_enemy_directional(enemy_signal)
        # Cross-wiring: left enemy → right motor, right enemy → left motor
        enemy_reflex_to_right = self.syn_enemy_motor_right.forward(enemy_left_input)
        enemy_reflex_to_left = self.syn_enemy_motor_left.forward(enemy_right_input)

        # Add innate avoidance (weights already boosted in __init__)
        left_in = left_in + enemy_reflex_to_left * 2.0
        right_in = right_in + enemy_reflex_to_right * 2.0

        left_spikes = self.motor_left.forward(left_in * 30.0)
        right_spikes = self.motor_right.forward(right_in * 30.0)
        boost_spikes = self.motor_boost.forward(boost_in * 30.0)

        # === COMPUTE ACTIONS ===
        left_rate = left_spikes.sum().item() / self.config.n_motor_left
        right_rate = right_spikes.sum().item() / self.config.n_motor_right
        boost_rate = boost_spikes.sum().item() / self.config.n_motor_boost

        # Direction: Difference between left and right
        # Also bias toward food direction
        food_direction = self._compute_direction_bias(food_signal)
        enemy_direction = self._compute_direction_bias(enemy_signal)

        # Compute target direction (relative angle)
        # Motor rate difference now has higher weight so fatigue affects steering
        angle_delta = (right_rate - left_rate) * 0.8  # Increased from 0.3 to 0.8
        # Food seeking (reduced weight when no food nearby to allow exploration)
        food_weight = 0.4 if food_signal.max() > 0.2 else 0.2  # Less heuristic when no food
        angle_delta += food_direction * food_weight * (0.5 + self.hunger_level)
        angle_delta -= enemy_direction * 0.25 * self.fear_level   # Avoid enemy

        # Convert angle_delta to target position (normalized 0-1)
        # Target is ahead of current heading + angle_delta
        target_angle = self.current_heading + angle_delta

        # Use relative forward direction - the gym will calculate angle from head to target
        # Output a target that's 20% of map ahead in the desired direction
        # This works because gym.step() calculates angle from head to target
        target_x = 0.5 + 0.2 * np.cos(target_angle)
        target_y = 0.5 + 0.2 * np.sin(target_angle)

        # Bias toward food when detected (simple heuristic to help initial learning)
        if food_signal.max() > 0.1:  # Food detected
            # Find direction of strongest food signal
            best_ray = np.argmax(food_signal)
            n_rays = len(food_signal)
            food_angle = self.current_heading + (2 * np.pi * best_ray / n_rays) - np.pi
            # Blend toward food
            blend = min(0.5, food_signal.max())
            target_x = target_x * (1 - blend) + (0.5 + 0.2 * np.cos(food_angle)) * blend
            target_y = target_y * (1 - blend) + (0.5 + 0.2 * np.sin(food_angle)) * blend

        target_x = np.clip(target_x, 0.05, 0.95)
        target_y = np.clip(target_y, 0.05, 0.95)

        # Boost decision: Only when ACTUAL enemy detected in sensor
        # Check raw sensor input, not just Fear circuit (which can have noise)
        enemy_detected = enemy_signal.max() > 0.1  # Actually see enemy in view
        boost = boost_rate > 0.15 and enemy_detected

        # Track turns and fatigue-induced switches
        current_turn = 'left' if left_rate > right_rate else 'right'
        if current_turn == 'left':
            self.stats['left_turns'] += 1
        else:
            self.stats['right_turns'] += 1

        # 피로로 인한 방향 전환 감지
        if self._last_turn_dir is not None and self._last_turn_dir != current_turn:
            self.stats['fatigue_switches'] += 1
        self._last_turn_dir = current_turn

        if boost:
            self.stats['boosts'] += 1

        # === LEARNING ===
        if reward != 0:
            if reward > 0:
                self.stats['food_eaten'] += 1
            self.dopamine = np.clip(self.dopamine + reward * 0.1, 0.0, 1.0)
            self._learn()
        else:
            # Baseline dopamine based on state
            self.dopamine = 0.5 + 0.1 * self.hunger_level - 0.1 * self.fear_level

        self.steps += 1
        return target_x, target_y, boost

    def _encode_rays(self, ray_signal: np.ndarray, n_neurons: int) -> torch.Tensor:
        """Encode ray signals to population activity - VECTORIZED"""
        # Convert to tensor once
        signal_tensor = torch.from_numpy(ray_signal).float().to(DEVICE)
        n_rays = len(ray_signal)
        neurons_per_ray = n_neurons // n_rays
        actual_size = neurons_per_ray * n_rays  # May be less than n_neurons

        # Create output tensor
        input_tensor = torch.zeros(n_neurons, device=DEVICE)

        # Expand signals to neuron populations (vectorized)
        expanded = signal_tensor.repeat_interleave(neurons_per_ray)

        # Add noise for stochastic activation (match expanded size)
        noise = torch.rand(actual_size, device=DEVICE)
        # Activate neurons where signal > threshold and random < signal
        mask = (expanded > 0.05) & (noise < expanded * 0.5)
        input_tensor[:actual_size][mask] = expanded[mask]

        return input_tensor

    def _compute_direction_bias(self, ray_signal: np.ndarray) -> float:
        """Compute direction bias from ray signals (left vs right)"""
        n_rays = len(ray_signal)
        half = n_rays // 2

        # Front-left vs front-right
        left_signal = ray_signal[:half].sum()
        right_signal = ray_signal[half:].sum()

        return (right_signal - left_signal) / (left_signal + right_signal + 0.01)

    def _encode_food_directional(self, ray_signal: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode food signals with left/right directional bias - VECTORIZED"""
        n_rays = len(ray_signal)
        half = n_rays // 2
        neurons_per_half = self.config.n_food_eye // 2
        n_per_ray = neurons_per_half // half
        actual_size = n_per_ray * half  # May be less than neurons_per_half

        # Convert to tensors
        left_signal = torch.from_numpy(ray_signal[:half]).float().to(DEVICE)
        right_signal = torch.from_numpy(ray_signal[half:]).float().to(DEVICE)

        # Expand signals to neuron populations
        left_expanded = left_signal.repeat_interleave(n_per_ray)
        right_expanded = right_signal.repeat_interleave(n_per_ray)

        # Create output tensors
        left_input = torch.zeros(self.config.n_food_eye, device=DEVICE)
        right_input = torch.zeros(self.config.n_food_eye, device=DEVICE)

        # Stochastic activation (match expanded size)
        noise_left = torch.rand(actual_size, device=DEVICE)
        noise_right = torch.rand(actual_size, device=DEVICE)

        # Left food → left motor
        mask_left = (left_expanded > 0.05) & (noise_left < left_expanded * 0.8)
        left_input[:actual_size][mask_left] = left_expanded[mask_left] * 2.0

        # Right food → right motor
        mask_right = (right_expanded > 0.05) & (noise_right < right_expanded * 0.8)
        right_input[neurons_per_half:neurons_per_half + actual_size][mask_right] = right_expanded[mask_right] * 2.0

        return left_input, right_input

    def _encode_enemy_directional(self, ray_signal: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode enemy signals with left/right directional bias - VECTORIZED"""
        n_rays = len(ray_signal)
        half = n_rays // 2
        neurons_per_half = self.config.n_enemy_eye // 2
        n_per_ray = neurons_per_half // half
        actual_size = n_per_ray * half  # May be less than neurons_per_half

        # Convert to tensors
        left_signal = torch.from_numpy(ray_signal[:half]).float().to(DEVICE)
        right_signal = torch.from_numpy(ray_signal[half:]).float().to(DEVICE)

        # Expand signals to neuron populations
        left_expanded = left_signal.repeat_interleave(n_per_ray)
        right_expanded = right_signal.repeat_interleave(n_per_ray)

        # Create output tensors
        left_input = torch.zeros(self.config.n_enemy_eye, device=DEVICE)
        right_input = torch.zeros(self.config.n_enemy_eye, device=DEVICE)

        # Stochastic activation (match expanded size)
        noise_left = torch.rand(actual_size, device=DEVICE)
        noise_right = torch.rand(actual_size, device=DEVICE)

        # Enemy left → (will be sent to RIGHT motor for avoidance)
        mask_left = (left_expanded > 0.05) & (noise_left < left_expanded * 0.8)
        left_input[:actual_size][mask_left] = left_expanded[mask_left] * 3.0  # Strong danger signal

        # Enemy right → (will be sent to LEFT motor for avoidance)
        mask_right = (right_expanded > 0.05) & (noise_right < right_expanded * 0.8)
        right_input[neurons_per_half:neurons_per_half + actual_size][mask_right] = right_expanded[mask_right] * 3.0

        return left_input, right_input

    def _learn(self):
        """DA-STDP learning for all synapses"""
        tau = self.config.stdp_tau
        a_plus = self.config.a_plus
        a_minus = self.config.a_minus
        dt = 1.0

        # Sensory → Specialized
        self.syn_food_hunger.update_eligibility(self.food_eye.spikes, self.hunger_circuit.spikes, tau, dt)
        self.syn_food_hunger.apply_dopamine(self.dopamine, a_plus, a_minus)

        self.syn_enemy_fear.update_eligibility(self.enemy_eye.spikes, self.fear_circuit.spikes, tau, dt)
        self.syn_enemy_fear.apply_dopamine(self.dopamine, a_plus, a_minus)

        # Specialized → Integration
        self.syn_hunger_int1.update_eligibility(self.hunger_circuit.spikes, self.integration_1.spikes, tau, dt)
        self.syn_hunger_int1.apply_dopamine(self.dopamine, a_plus, a_minus)

        self.syn_fear_int1.update_eligibility(self.fear_circuit.spikes, self.integration_1.spikes, tau, dt)
        self.syn_fear_int1.apply_dopamine(self.dopamine, a_plus, a_minus)

        # Integration → Motor
        self.syn_int2_left.update_eligibility(self.integration_2.spikes, self.motor_left.spikes, tau, dt)
        self.syn_int2_left.apply_dopamine(self.dopamine, a_plus, a_minus)

        self.syn_int2_right.update_eligibility(self.integration_2.spikes, self.motor_right.spikes, tau, dt)
        self.syn_int2_right.apply_dopamine(self.dopamine, a_plus, a_minus)

        self.syn_int2_boost.update_eligibility(self.integration_2.spikes, self.motor_boost.spikes, tau, dt)
        self.syn_int2_boost.apply_dopamine(self.dopamine, a_plus, a_minus)

        # Direct food reflex (modifiable even though it's a reflex)
        self.syn_food_motor_left.update_eligibility(self.food_eye.spikes, self.motor_left.spikes, tau, dt)
        self.syn_food_motor_left.apply_dopamine(self.dopamine, a_plus, a_minus)

        self.syn_food_motor_right.update_eligibility(self.food_eye.spikes, self.motor_right.spikes, tau, dt)
        self.syn_food_motor_right.apply_dopamine(self.dopamine, a_plus, a_minus)

        # Innate enemy avoidance reflex (STILL LEARNABLE - can be modified by experience!)
        # If avoiding enemies leads to survival → reinforce
        # If avoiding prevents eating → might weaken (learning "courage")
        self.syn_enemy_motor_left.update_eligibility(self.enemy_eye.spikes, self.motor_left.spikes, tau, dt)
        self.syn_enemy_motor_left.apply_dopamine(self.dopamine, a_plus, a_minus)

        self.syn_enemy_motor_right.update_eligibility(self.enemy_eye.spikes, self.motor_right.spikes, tau, dt)
        self.syn_enemy_motor_right.apply_dopamine(self.dopamine, a_plus, a_minus)

    def reset(self):
        """Reset state for new episode"""
        self.food_eye.reset()
        self.enemy_eye.reset()
        self.body_eye.reset()
        self.hunger_circuit.reset()
        self.fear_circuit.reset()
        self.integration_1.reset()
        self.integration_2.reset()
        self.motor_left.reset()
        self.motor_right.reset()
        self.motor_boost.reset()
        self.dopamine = 0.5
        self.fear_level = 0.0
        self.hunger_level = 0.5
        self.current_heading = 0.0
        self._last_turn_dir = None  # Reset turn tracking
        # Reset stats (ensures all keys exist)
        self.stats = {
            'food_eaten': 0, 'boosts': 0, 'fear_triggers': 0,
            'left_turns': 0, 'right_turns': 0, 'fatigue_switches': 0
        }

    def save(self, path: Path):
        """Save all synapse weights"""
        state = {
            'syn_food_hunger': self.syn_food_hunger.weights.cpu(),
            'syn_enemy_fear': self.syn_enemy_fear.weights.cpu(),
            'syn_body_fear': self.syn_body_fear.weights.cpu(),
            'syn_hunger_int1': self.syn_hunger_int1.weights.cpu(),
            'syn_fear_int1': self.syn_fear_int1.weights.cpu(),
            'syn_int1_int2': self.syn_int1_int2.weights.cpu(),
            'syn_int2_left': self.syn_int2_left.weights.cpu(),
            'syn_int2_right': self.syn_int2_right.weights.cpu(),
            'syn_int2_boost': self.syn_int2_boost.weights.cpu(),
            'syn_fear_hunger_inhib': self.syn_fear_hunger_inhib.weights.cpu(),
            'syn_fear_boost': self.syn_fear_boost.weights.cpu(),
            'syn_food_motor_left': self.syn_food_motor_left.weights.cpu(),
            'syn_food_motor_right': self.syn_food_motor_right.weights.cpu(),
            # Innate enemy avoidance (진화된 본능 - learnable)
            'syn_enemy_motor_left': self.syn_enemy_motor_left.weights.cpu(),
            'syn_enemy_motor_right': self.syn_enemy_motor_right.weights.cpu(),
            'stats': self.stats.copy(),
        }
        torch.save(state, path)
        print(f"  Model saved: {path}")

    def load(self, path: Path) -> bool:
        """Load synapse weights"""
        if not path.exists():
            print(f"  No checkpoint: {path}")
            return False
        state = torch.load(path, map_location=DEVICE)
        self.syn_food_hunger.weights = state['syn_food_hunger'].to(DEVICE)
        self.syn_enemy_fear.weights = state['syn_enemy_fear'].to(DEVICE)
        self.syn_body_fear.weights = state['syn_body_fear'].to(DEVICE)
        self.syn_hunger_int1.weights = state['syn_hunger_int1'].to(DEVICE)
        self.syn_fear_int1.weights = state['syn_fear_int1'].to(DEVICE)
        self.syn_int1_int2.weights = state['syn_int1_int2'].to(DEVICE)
        self.syn_int2_left.weights = state['syn_int2_left'].to(DEVICE)
        self.syn_int2_right.weights = state['syn_int2_right'].to(DEVICE)
        self.syn_int2_boost.weights = state['syn_int2_boost'].to(DEVICE)
        self.syn_fear_hunger_inhib.weights = state['syn_fear_hunger_inhib'].to(DEVICE)
        self.syn_fear_boost.weights = state['syn_fear_boost'].to(DEVICE)
        # Load new reflex synapses (optional for backward compatibility)
        if 'syn_food_motor_left' in state:
            self.syn_food_motor_left.weights = state['syn_food_motor_left'].to(DEVICE)
        if 'syn_food_motor_right' in state:
            self.syn_food_motor_right.weights = state['syn_food_motor_right'].to(DEVICE)
        # Load innate enemy avoidance (if saved)
        if 'syn_enemy_motor_left' in state:
            self.syn_enemy_motor_left.weights = state['syn_enemy_motor_left'].to(DEVICE)
            self.syn_enemy_motor_left._rebuild_sparse()
        if 'syn_enemy_motor_right' in state:
            self.syn_enemy_motor_right.weights = state['syn_enemy_motor_right'].to(DEVICE)
            self.syn_enemy_motor_right._rebuild_sparse()
        if 'stats' in state:
            self.stats = state['stats']
        print(f"  Model loaded: {path}")
        return True


class SlitherAgent:
    """Complete Slither.io Agent with SNN Brain"""

    def __init__(self, brain_config: Optional[SlitherBrainConfig] = None,
                 env_config: Optional[SlitherConfig] = None,
                 render_mode: str = "none"):
        self.brain = SlitherBrain(brain_config)
        self.env = SlitherGym(env_config, render_mode)

        self.scores = []
        self.best_score = 0

    def run_episode(self, max_steps: int = 1000) -> dict:
        """Run one episode"""
        obs = self.env.reset()
        self.brain.reset()

        total_reward = 0
        step = 0

        while step < max_steps:
            # Update brain's heading from environment
            self.brain.current_heading = obs.get('heading', 0.0)

            # Get sensor input
            sensor = self.env.get_sensor_input(self.brain.config.n_rays)

            # Process through brain
            target_x, target_y, boost = self.brain.process(sensor)

            # Step environment
            obs, reward, done, info = self.env.step((target_x, target_y, boost))
            total_reward += reward

            # Learn from reward
            if reward != 0:
                self.brain.process(sensor, reward)

            # Render
            self.env.render()

            step += 1

            if done:
                break

        return {
            'length': info['length'],
            'steps': info['steps'],
            'reward': total_reward,
            'food_eaten': info.get('foods_eaten', 0)
        }

    def train(self, n_episodes: int = 100, resume: bool = False):
        """Train the agent"""
        print("\n" + "="*60)
        print(f"Slither.io SNN Training ({self.brain.config.total_neurons:,} neurons)")
        print("="*60)

        if resume:
            self.load_model("best")

        for ep in range(n_episodes):
            result = self.run_episode()
            self.scores.append(result['length'])

            # Save best
            if result['length'] > self.best_score:
                self.best_score = result['length']
                self.save_model(f"best_{result['length']}")
                self.save_model("best")
                print(f"  ★ NEW BEST! Length={result['length']}")

            high = max(self.scores)
            avg = sum(self.scores[-10:]) / min(len(self.scores), 10)

            print(f"[Ep {ep+1:3d}] Length: {result['length']:3d} | "
                  f"High: {high} | Avg(10): {avg:.0f} | "
                  f"Food: {result['food_eaten']} | Steps: {result['steps']}")

        # Final save
        self.save_model("final")

        print("\n" + "="*60)
        print(f"Training Complete!")
        print(f"  Best Length: {max(self.scores)}")
        print(f"  Final Avg: {sum(self.scores)/len(self.scores):.1f}")
        print(f"  Brain Stats: {self.brain.stats}")
        print(f"  Saved to: {CHECKPOINT_DIR}")
        print("="*60)

    def save_model(self, name: str):
        """Save model"""
        self.brain.save(CHECKPOINT_DIR / f"{name}.pt")

    def load_model(self, name: str) -> bool:
        """Load model"""
        return self.brain.load(CHECKPOINT_DIR / f"{name}.pt")

    def close(self):
        """Clean up"""
        self.env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--render', choices=['none', 'pygame', 'ascii'], default='pygame')
    parser.add_argument('--resume', action='store_true', help='Resume from best')
    parser.add_argument('--enemies', type=int, default=0, help='Number of enemy bots')
    parser.add_argument('--small', action='store_true', help='Use smaller brain (15K neurons) for testing')
    args = parser.parse_args()

    print("Slither.io SNN Agent")
    print(f"Render mode: {args.render}")
    print(f"Enemies: {args.enemies}")
    print(f"Small mode: {args.small}")
    print()

    # Configure environment
    env_config = SlitherConfig(n_enemies=args.enemies)

    # Configure brain (small for testing, full for real training)
    if args.small:
        brain_config = SlitherBrainConfig(
            n_food_eye=1000,
            n_enemy_eye=1000,
            n_body_eye=500,
            n_integration_1=5000,
            n_integration_2=5000,
            n_motor_left=500,
            n_motor_right=500,
            n_motor_boost=300,
            n_fear_circuit=1000,
            n_hunger_circuit=1000,
            sparsity=0.02  # Higher sparsity for smaller network
        )
    else:
        brain_config = None  # Use defaults (153K)

    # Create agent
    agent = SlitherAgent(
        brain_config=brain_config,
        env_config=env_config,
        render_mode=args.render
    )

    try:
        agent.train(n_episodes=args.episodes, resume=args.resume)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving...")
        agent.save_model("interrupted")
    finally:
        agent.close()
