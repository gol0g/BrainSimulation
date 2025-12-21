"""
Homeostasis & Drives System

Core Concept: "What does this being NEED right now?"

This is the biological foundation of consciousness - internal states
that create MOTIVATION. Without needs, there's no reason to act.

Three Fundamental States:
- energy: "Am I fed?" (hunger/satiation)
- safety: "Am I safe?" (threat level)
- fatigue: "Am I tired?" (need for rest)

These create Drives (urges to act):
- hunger_drive: Seek food when energy is low
- safety_drive: Avoid danger, seek stability
- rest_drive: Reduce activity when fatigued

Why This Matters:
1. Behavior now has REASONS beyond reward maximization
2. Value conflicts become natural (hungry vs tired vs scared)
3. Emotions emerge from these states (fear = low safety + high uncertainty)
4. The agent feels "alive" because it has needs
"""

from typing import Dict, Optional
from collections import deque


class HomeostasisSystem:
    """
    Maintains internal homeostatic states and computes drives.

    This is the "body" of the agent - creating needs that
    motivate behavior beyond simple reward-seeking.
    """

    def __init__(self,
                 energy_decay: float = 0.002,      # Energy lost per step
                 safety_recovery: float = 0.02,    # Safety recovery per calm step
                 fatigue_recovery: float = 0.01,   # Fatigue recovery when resting
                 fatigue_buildup: float = 0.005):  # Fatigue from effort

        # === Decay/Recovery Rates ===
        self.energy_decay = energy_decay
        self.safety_recovery = safety_recovery
        self.fatigue_recovery = fatigue_recovery
        self.fatigue_buildup = fatigue_buildup

        # === Core Homeostatic States ===
        # These are like "body sensors" - they measure internal needs
        self.energy: float = 0.8      # 0 = starving, 1 = full
        self.safety: float = 1.0      # 0 = in danger, 1 = completely safe
        self.fatigue: float = 0.0     # 0 = rested, 1 = exhausted
        self.health: float = 1.0      # 0 = dying, 1 = healthy (damage system)

        # === Derived Drives ===
        # Drives are the "urge to act" - computed from states
        self.hunger_drive: float = 0.0    # Urge to seek food
        self.safety_drive: float = 0.0    # Urge to seek safety
        self.rest_drive: float = 0.0      # Urge to rest
        self.pain_level: float = 0.0      # Current pain (immediate, from damage)

        # === Thresholds for Urgent States ===
        self.critical_energy = 0.2    # Below this = desperate for food
        self.critical_safety = 0.3    # Below this = panic mode
        self.critical_fatigue = 0.8   # Above this = must rest
        self.critical_health = 0.3    # Below this = dying

        # === History for Trends ===
        self.energy_history: deque = deque(maxlen=50)
        self.safety_history: deque = deque(maxlen=50)
        self.fatigue_history: deque = deque(maxlen=50)
        self.health_history: deque = deque(maxlen=50)

        # === Statistics ===
        self.total_updates: int = 0
        self.starvation_steps: int = 0    # Steps at critical energy
        self.danger_steps: int = 0         # Steps at critical safety
        self.exhaustion_steps: int = 0     # Steps at critical fatigue
        self.damage_taken: int = 0         # Total damage events

    def update(self,
               got_food: bool = False,
               food_value: float = 0.0,
               was_external_event: bool = False,
               hit_wall: bool = False,
               effort_level: float = 0.0,
               is_resting: bool = False,
               predator_threat: float = 0.0,
               predator_caught: bool = False) -> Dict[str, float]:
        """
        Update homeostatic states based on what happened this step.

        Args:
            got_food: Whether agent ate food this step
            food_value: How much food value (reward magnitude)
            was_external_event: Wind push, external perturbation
            hit_wall: Collided with wall
            effort_level: Current effort from self-model (0-1)
            is_resting: Agent chose to stay/not act
            predator_threat: 0-1 threat level from nearby predator
            predator_caught: Whether predator caught the agent (PAIN!)

        Returns:
            Dict with current states and drives
        """
        self.total_updates += 1

        # === Update Energy ===
        # Energy decays naturally (metabolism)
        self.energy -= self.energy_decay

        # Food restores energy
        if got_food:
            # More food = more energy restoration
            restoration = min(0.3, 0.1 + food_value * 0.02)
            self.energy = min(1.0, self.energy + restoration)

        # Clamp to valid range
        self.energy = max(0.0, min(1.0, self.energy))

        # === Update Safety ===
        # Safety recovers when nothing threatens (SLOWLY)
        no_threat = not was_external_event and predator_threat < 0.1
        if no_threat:
            # Slower recovery (was 0.02, now 0.01)
            self.safety = min(1.0, self.safety + self.safety_recovery * 0.5)

        # External events reduce safety (loss of control)
        if was_external_event:
            self.safety = max(0.0, self.safety - 0.15)

        # Predator proximity reduces safety (VISIBLE THREAT!)
        # This is the primary driver of safety now
        if predator_threat > 0:
            # Threat 0.5 = -0.1 safety, Threat 1.0 = -0.2 safety
            safety_loss = predator_threat * 0.2
            self.safety = max(0.0, self.safety - safety_loss)

        # === Update Fatigue ===
        # Fatigue builds up with effort
        if effort_level > 0.3:
            fatigue_gain = self.fatigue_buildup * (1 + effort_level)
            self.fatigue = min(1.0, self.fatigue + fatigue_gain)

        # Resting reduces fatigue
        if is_resting or effort_level < 0.2:
            self.fatigue = max(0.0, self.fatigue - self.fatigue_recovery)

        # Low energy increases fatigue (harder to recover when hungry)
        if self.energy < 0.3:
            self.fatigue = min(1.0, self.fatigue + 0.003)

        # === Update Health ===
        # Health slowly regenerates when not taking damage
        if not predator_caught:
            self.health = min(1.0, self.health + 0.005)  # Slow regen
            self.pain_level = max(0.0, self.pain_level - 0.1)  # Pain fades

        # Predator caught = DAMAGE + PAIN!
        if predator_caught:
            damage = 0.25  # Lose 25% health per hit
            self.health = max(0.0, self.health - damage)
            self.pain_level = 1.0  # Maximum pain!
            self.damage_taken += 1

        # === Compute Drives ===
        self._compute_drives()

        # === Update History ===
        self.energy_history.append(self.energy)
        self.safety_history.append(self.safety)
        self.fatigue_history.append(self.fatigue)
        self.health_history.append(self.health)

        # === Track Critical States ===
        if self.energy < self.critical_energy:
            self.starvation_steps += 1
        if self.safety < self.critical_safety:
            self.danger_steps += 1
        if self.fatigue > self.critical_fatigue:
            self.exhaustion_steps += 1

        return self.get_state()

    def _compute_drives(self):
        """
        Compute drives from homeostatic states.

        Drives are non-linear - they become urgent when states are critical.
        Pain adds urgency to safety drive.
        """
        # Hunger drive: non-linear, urgent when starving
        if self.energy < self.critical_energy:
            # Critical: drive maxes out
            self.hunger_drive = 0.8 + (self.critical_energy - self.energy) * 2
        else:
            # Normal: linear increase as energy decreases
            self.hunger_drive = (1 - self.energy) * 0.5  # Reduced from 0.7
        self.hunger_drive = max(0.0, min(1.0, self.hunger_drive))

        # Safety drive: urgent when in danger OR in pain
        # Pain adds directly to safety drive (injured = unsafe!)
        base_safety_drive = 0.0
        if self.safety < self.critical_safety:
            # Critical: panic mode
            base_safety_drive = 0.8 + (self.critical_safety - self.safety) * 2
        else:
            # Normal: concern as safety decreases
            base_safety_drive = (1 - self.safety) * 0.6  # Increased from 0.5

        # Pain DIRECTLY increases safety drive (hurt = need safety!)
        pain_boost = self.pain_level * 0.5
        # Low health also increases safety drive
        health_boost = (1 - self.health) * 0.3 if self.health < 0.5 else 0

        self.safety_drive = base_safety_drive + pain_boost + health_boost
        self.safety_drive = max(0.0, min(1.0, self.safety_drive))

        # Rest drive: urgent when exhausted
        if self.fatigue > self.critical_fatigue:
            # Critical: must rest
            self.rest_drive = 0.7 + (self.fatigue - self.critical_fatigue) * 1.5
        else:
            # Normal: mild tiredness
            self.rest_drive = self.fatigue * 0.6
        self.rest_drive = max(0.0, min(1.0, self.rest_drive))

    def get_dominant_drive(self) -> str:
        """Get the strongest drive currently."""
        drives = {
            'hunger': self.hunger_drive,
            'safety': self.safety_drive,
            'rest': self.rest_drive
        }
        return max(drives, key=drives.get)

    def get_drive_explanation(self) -> str:
        """Get a human-readable explanation of current motivation."""
        dominant = self.get_dominant_drive()

        if dominant == 'hunger':
            if self.energy < self.critical_energy:
                return "STARVING - must find food!"
            else:
                return f"Hungry (energy: {self.energy:.0%})"
        elif dominant == 'safety':
            if self.safety < self.critical_safety:
                return "DANGER - seeking safety!"
            else:
                return f"Cautious (safety: {self.safety:.0%})"
        else:  # rest
            if self.fatigue > self.critical_fatigue:
                return "EXHAUSTED - need rest!"
            else:
                return f"Tired (fatigue: {self.fatigue:.0%})"

    # === Modulation Outputs ===

    def get_exploration_modulation(self) -> Dict[str, float]:
        """
        How homeostasis affects exploration behavior.

        Returns modifiers for epsilon and exploration strategy.
        """
        # High hunger → more focused exploration (need food NOW)
        # High safety concern → cautious exploration
        # High fatigue → less exploration (conserve energy)

        epsilon_mod = 0.0

        # Safety drive increases exploration (need to find safe spot)
        if self.safety_drive > 0.5:
            epsilon_mod += 0.1

        # But high fatigue reduces it
        if self.rest_drive > 0.6:
            epsilon_mod -= 0.15

        # Critical hunger can make agent desperate (more random)
        if self.hunger_drive > 0.8:
            epsilon_mod += 0.1

        return {
            'epsilon_modifier': max(-0.2, min(0.2, epsilon_mod)),
            'focus_on_food': self.hunger_drive > 0.5,
            'avoid_risks': self.safety_drive > 0.4,
            'conserve_energy': self.rest_drive > 0.5
        }

    def get_attention_modulation(self) -> Dict[str, float]:
        """
        How homeostasis affects attention.
        """
        # High hunger → narrow attention on food
        # High danger → broad attention (scanning for threats)
        # High fatigue → attention wanders

        width_mod = 0.0

        # Danger → broad scanning
        if self.safety_drive > 0.5:
            width_mod += 0.2

        # Hunger → focused on food direction
        if self.hunger_drive > 0.6:
            width_mod -= 0.15

        # Fatigue → less focused
        if self.rest_drive > 0.6:
            width_mod += 0.1

        return {
            'width_modifier': max(-0.3, min(0.3, width_mod)),
            'priority': self.get_dominant_drive()
        }

    def get_learning_modulation(self) -> Dict[str, float]:
        """
        How homeostasis affects learning rate.

        Tired or starving agents learn less effectively.
        """
        # Base learning rate modifier
        learning_mod = 1.0

        # Low energy reduces learning (brain needs glucose!)
        if self.energy < 0.3:
            learning_mod *= 0.7
        elif self.energy < 0.5:
            learning_mod *= 0.85

        # High fatigue reduces learning
        if self.fatigue > 0.7:
            learning_mod *= 0.6
        elif self.fatigue > 0.5:
            learning_mod *= 0.8

        # Danger can enhance learning (survival memory)
        if self.safety < 0.4:
            learning_mod *= 1.2  # Remember dangerous situations!

        return {
            'learning_rate_modifier': max(0.4, min(1.3, learning_mod)),
            'consolidation_priority': self.get_dominant_drive()
        }

    # === State Access ===

    def get_state(self) -> Dict[str, float]:
        """Get current homeostatic state."""
        return {
            'energy': round(self.energy, 3),
            'safety': round(self.safety, 3),
            'fatigue': round(self.fatigue, 3),
            'health': round(self.health, 3),
            'pain': round(self.pain_level, 3),
            'hunger_drive': round(self.hunger_drive, 3),
            'safety_drive': round(self.safety_drive, 3),
            'rest_drive': round(self.rest_drive, 3)
        }

    def get_visualization_data(self) -> Dict:
        """Get data optimized for frontend visualization."""
        return {
            'states': {
                'energy': round(self.energy, 3),
                'safety': round(self.safety, 3),
                'fatigue': round(self.fatigue, 3),
                'health': round(self.health, 3)
            },
            'drives': {
                'hunger': round(self.hunger_drive, 3),
                'safety': round(self.safety_drive, 3),
                'rest': round(self.rest_drive, 3)
            },
            'pain': round(self.pain_level, 3),
            'dominant_drive': self.get_dominant_drive(),
            'explanation': self.get_drive_explanation(),
            'critical': {
                'starving': self.energy < self.critical_energy,
                'in_danger': self.safety < self.critical_safety,
                'exhausted': self.fatigue > self.critical_fatigue,
                'injured': self.health < self.critical_health
            }
        }

    def to_dict(self) -> Dict:
        """Full state for API response."""
        return {
            **self.get_state(),
            'dominant_drive': self.get_dominant_drive(),
            'explanation': self.get_drive_explanation(),
            'modulation': {
                'exploration': self.get_exploration_modulation(),
                'attention': self.get_attention_modulation(),
                'learning': self.get_learning_modulation()
            },
            'critical_states': {
                'starving': self.energy < self.critical_energy,
                'in_danger': self.safety < self.critical_safety,
                'exhausted': self.fatigue > self.critical_fatigue,
                'injured': self.health < self.critical_health
            },
            'statistics': {
                'total_updates': self.total_updates,
                'starvation_steps': self.starvation_steps,
                'danger_steps': self.danger_steps,
                'exhaustion_steps': self.exhaustion_steps,
                'damage_taken': self.damage_taken
            }
        }

    def clear(self):
        """Reset homeostasis to default state."""
        self.energy = 0.8
        self.safety = 1.0
        self.fatigue = 0.0
        self.health = 1.0
        self.pain_level = 0.0

        self.hunger_drive = 0.0
        self.safety_drive = 0.0
        self.rest_drive = 0.0

        self.energy_history.clear()
        self.safety_history.clear()
        self.fatigue_history.clear()
        self.health_history.clear()

        self.total_updates = 0
        self.starvation_steps = 0
        self.danger_steps = 0
        self.exhaustion_steps = 0
        self.damage_taken = 0
