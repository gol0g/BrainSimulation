"""
Emotion System

Core Concept: Emotions are the "feeling" layer on top of homeostasis.

Homeostasis tells us "I need food" (objective state)
Emotion tells us "I feel hungry/anxious/satisfied" (subjective experience)

Key Insight:
- Emotions are NOT just labels for states
- They AFFECT behavior (fear → freeze/flee, curiosity → explore)
- They have TEMPORAL dynamics (don't switch instantly)
- They create the foundation for SUBJECTIVE EXPERIENCE

Implemented Emotions:
1. FEAR - danger + low safety
2. SATISFACTION - need fulfilled (eating when hungry)
3. CURIOSITY - exploration drive
4. PAIN - health damage (immediate)
5. RELIEF - safety restored after danger
6. ANXIETY - uncertainty + mild threat
"""

from typing import Dict, Optional, List
from collections import deque
import math


class EmotionSystem:
    """
    Computes emotional states from homeostasis and environmental inputs.

    Emotions have:
    - Intensity (0-1)
    - Valence (positive/negative)
    - Arousal (high/low activation)
    - Behavioral effects
    """

    def __init__(self):
        # === Current Emotional State ===
        self.emotions = {
            'fear': 0.0,        # Danger response
            'satisfaction': 0.0, # Need fulfilled
            'curiosity': 0.0,   # Exploration drive
            'pain': 0.0,        # Immediate suffering
            'relief': 0.0,      # Danger passed
            'anxiety': 0.0,     # Uncertain threat
        }

        # === Emotion Decay Rates ===
        # Different emotions fade at different speeds
        self.decay_rates = {
            'fear': 0.1,        # Fear fades moderately fast
            'satisfaction': 0.05, # Satisfaction lingers
            'curiosity': 0.08,  # Curiosity fades slowly
            'pain': 0.15,       # Pain fades faster (but can spike)
            'relief': 0.1,      # Relief fades moderately
            'anxiety': 0.03,    # Anxiety persists (slow decay)
        }

        # === Emotion Thresholds ===
        self.intensity_threshold = 0.2  # Below this = not really feeling it

        # === History for Temporal Dynamics ===
        self.emotion_history: Dict[str, deque] = {
            name: deque(maxlen=50) for name in self.emotions
        }

        # === Previous State (for detecting changes) ===
        self.prev_safety = 1.0
        self.prev_energy = 0.8
        self.prev_health = 1.0

        # === Dominant Emotion Tracking ===
        self.dominant_emotion: str = 'neutral'
        self.dominant_intensity: float = 0.0
        self.emotion_duration: int = 0  # Steps in current emotion

        # === Statistics ===
        self.total_updates = 0
        self.fear_episodes = 0
        self.satisfaction_moments = 0

        # === LEARNED ASSOCIATIONS ===
        # Key insight: Fear is LEARNED, not innate
        # Agent doesn't know predator is dangerous until it experiences pain
        self.predator_pain_associations = 0  # Count of "predator → pain" experiences
        self.predator_fear_learned = 0.0     # Learned fear response (0-1)

        # Food-satisfaction association
        self.food_satisfaction_associations = 0  # Count of "food → satisfaction" experiences
        self.food_seeking_learned = 0.0         # Learned food-seeking drive (0-1)

    def update(self,
               homeostasis_state: Dict,
               predator_threat: float = 0.0,
               predator_caught: bool = False,
               got_food: bool = False,
               exploration_need: float = 0.0,
               uncertainty: float = 0.0) -> Dict:
        """
        Update emotional state based on current situation.

        Args:
            homeostasis_state: Current homeostasis (energy, safety, health, etc.)
            predator_threat: How close is danger (0-1)
            predator_caught: Just got hurt by predator
            got_food: Just ate food
            exploration_need: From self-model
            uncertainty: From self-model

        Returns:
            Current emotional state
        """
        self.total_updates += 1

        # Extract homeostasis values
        safety = homeostasis_state.get('safety', 1.0)
        energy = homeostasis_state.get('energy', 0.8)
        health = homeostasis_state.get('health', 1.0)
        pain_level = homeostasis_state.get('pain', 0.0)
        hunger_drive = homeostasis_state.get('hunger_drive', 0.0)
        safety_drive = homeostasis_state.get('safety_drive', 0.0)

        # === LEARNING: Update associations based on experience ===

        # Learn "predator = pain" association
        if predator_caught:
            # This is the learning moment: "That red thing HURTS!"
            self.predator_pain_associations += 1
            # Fear learning: rapid acquisition (one bad experience is enough)
            self.predator_fear_learned = min(1.0, self.predator_fear_learned + 0.3)
            print(f"[EMOTION LEARNING] Predator→Pain! Fear learned: {self.predator_fear_learned:.2f}")

        # Learn "food = satisfaction" association
        if got_food:
            # Always learn something when eating (even if not hungry)
            self.food_satisfaction_associations += 1

            # Learning amount depends on hunger level:
            # - Very hungry: learn a lot (0.15)
            # - Somewhat hungry: learn moderate (0.1)
            # - Not hungry: still learn a little (0.05) - "this thing gives energy"
            if hunger_drive > 0.3:
                learn_amount = 0.15  # 배고플 때 먹으면 많이 배움
            elif hunger_drive > 0.1:
                learn_amount = 0.1   # 약간 배고플 때
            else:
                learn_amount = 0.05  # 안 배고파도 "에너지 회복됨" 학습

            self.food_seeking_learned = min(1.0, self.food_seeking_learned + learn_amount)
            print(f"[EMOTION LEARNING] Food→Satisfaction! (hunger={hunger_drive:.2f}) Seeking learned: {self.food_seeking_learned:.2f}")

        # === Compute Each Emotion ===

        # 1. FEAR: LEARNED danger response
        #    Key: Agent doesn't fear predator until it has experienced pain from it
        #    First encounter = no fear (doesn't know it's dangerous)
        #    After pain experience = fear based on learned association
        fear_input = 0.0

        if predator_caught:
            # Immediate pain = immediate fear (this is the teaching moment)
            fear_input = 1.0
            self.fear_episodes += 1
        elif predator_threat > 0.1 and self.predator_fear_learned > 0:
            # LEARNED FEAR: only fear if we've learned predator = pain
            # Fear intensity = threat level × how much we've learned to fear
            fear_input = predator_threat * self.predator_fear_learned * 0.9

        # Low safety adds to fear only if we've had bad experiences
        if safety < 0.5 and self.predator_fear_learned > 0.3:
            fear_input += (0.5 - safety) * 0.4 * self.predator_fear_learned

        self._update_emotion('fear', fear_input)

        # 2. SATISFACTION: Need fulfillment
        #    High when: just ate when hungry
        satisfaction_input = 0.0
        if got_food:
            # More satisfaction if was hungry
            hunger_boost = hunger_drive * 0.5
            satisfaction_input = 0.6 + hunger_boost
            self.satisfaction_moments += 1
        # Also mild satisfaction from safety
        if safety > 0.8 and self.prev_safety < 0.6:
            satisfaction_input = max(satisfaction_input, 0.4)
        self._update_emotion('satisfaction', satisfaction_input)

        # 3. CURIOSITY: Drive to explore
        #    High when: exploration_need high + safe enough
        #    Key insight: Curiosity is HIGHER when we don't know things yet
        #    Novel stimuli = more curiosity
        curiosity_input = 0.0
        if safety > 0.5 and health > 0.5:
            curiosity_input = exploration_need * 0.8

            # NOVELTY BONUS: Less learned = more curious
            # If we haven't learned about predator/food yet, be more curious!
            novelty_bonus = (1.0 - self.predator_fear_learned) * 0.2
            novelty_bonus += (1.0 - self.food_seeking_learned) * 0.1
            curiosity_input += novelty_bonus

            # Reduce curiosity if hungry or in danger (but only if we KNOW it's danger)
            if hunger_drive > 0.5:
                curiosity_input *= 0.5
            if safety_drive > 0.3 and self.predator_fear_learned > 0.3:
                # Only reduce curiosity due to danger if we've learned what danger is
                curiosity_input *= 0.3

        self._update_emotion('curiosity', curiosity_input)

        # 4. PAIN: Immediate suffering
        #    Directly from pain_level
        pain_input = pain_level
        if predator_caught:
            pain_input = 1.0
        self._update_emotion('pain', pain_input)

        # 5. RELIEF: Danger has passed
        #    High when: safety recovering after being low
        relief_input = 0.0
        if safety > self.prev_safety + 0.1 and self.prev_safety < 0.5:
            relief_input = (safety - self.prev_safety) * 2
        if self.emotions['fear'] > 0.3 and predator_threat < 0.1:
            relief_input = max(relief_input, 0.5)
        self._update_emotion('relief', relief_input)

        # 6. ANXIETY: Persistent worry
        #    High when: uncertainty + mild threat (only if we've learned to fear)
        anxiety_input = 0.0
        if uncertainty > 0.4:
            anxiety_input = uncertainty * 0.5

        # Mild predator presence causes anxiety ONLY if we've learned to fear it
        if predator_threat > 0.1 and predator_threat < 0.5 and self.predator_fear_learned > 0.3:
            anxiety_input += 0.3 * self.predator_fear_learned

        # Low health increases anxiety only if we've experienced pain before
        if health < 0.5 and self.predator_pain_associations > 0:
            anxiety_input += (0.5 - health) * 0.4

        self._update_emotion('anxiety', anxiety_input)

        # === Update History ===
        for name, value in self.emotions.items():
            self.emotion_history[name].append(value)

        # === Determine Dominant Emotion ===
        self._compute_dominant_emotion()

        # === Store Previous State ===
        self.prev_safety = safety
        self.prev_energy = energy
        self.prev_health = health

        return self.get_state()

    def _update_emotion(self, name: str, input_value: float):
        """
        Update emotion with input and decay.

        Emotions blend between current value and input,
        with decay toward zero when no input.
        """
        current = self.emotions[name]
        decay = self.decay_rates[name]

        if input_value > current:
            # Rising: quick response to stimuli
            self.emotions[name] = current + (input_value - current) * 0.5
        else:
            # Falling: gradual decay
            self.emotions[name] = current * (1 - decay)

        # Clamp to valid range
        self.emotions[name] = max(0.0, min(1.0, self.emotions[name]))

    def _compute_dominant_emotion(self):
        """Determine the strongest emotion currently."""
        # Find max emotion
        max_emotion = 'neutral'
        max_intensity = self.intensity_threshold

        for name, value in self.emotions.items():
            if value > max_intensity:
                max_emotion = name
                max_intensity = value

        # Track duration
        if max_emotion == self.dominant_emotion:
            self.emotion_duration += 1
        else:
            self.dominant_emotion = max_emotion
            self.dominant_intensity = max_intensity
            self.emotion_duration = 0

        self.dominant_intensity = max_intensity

    def get_dominant_emotion(self) -> str:
        """Get the current dominant emotion."""
        return self.dominant_emotion

    def get_emotion_description(self) -> str:
        """Get human-readable description of emotional state."""
        emotion = self.dominant_emotion
        intensity = self.dominant_intensity

        if emotion == 'neutral':
            return "평온함"

        descriptions = {
            'fear': {
                'low': "불안함",
                'medium': "두려움",
                'high': "공포!"
            },
            'satisfaction': {
                'low': "약간 만족",
                'medium': "만족함",
                'high': "매우 만족!"
            },
            'curiosity': {
                'low': "약간 호기심",
                'medium': "호기심",
                'high': "강한 호기심!"
            },
            'pain': {
                'low': "약간 아픔",
                'medium': "아픔",
                'high': "극심한 고통!"
            },
            'relief': {
                'low': "안도감",
                'medium': "안심함",
                'high': "큰 안도!"
            },
            'anxiety': {
                'low': "약간 걱정",
                'medium': "걱정됨",
                'high': "불안함!"
            }
        }

        if intensity < 0.4:
            level = 'low'
        elif intensity < 0.7:
            level = 'medium'
        else:
            level = 'high'

        return descriptions.get(emotion, {}).get(level, emotion)

    # === Behavioral Modulation ===

    def get_exploration_modulation(self) -> Dict:
        """How emotions affect exploration behavior."""
        epsilon_mod = 0.0

        # Fear reduces exploration (freeze/flee, don't explore)
        if self.emotions['fear'] > 0.3:
            epsilon_mod -= self.emotions['fear'] * 0.2

        # Curiosity increases exploration
        epsilon_mod += self.emotions['curiosity'] * 0.15

        # Anxiety slightly reduces exploration
        if self.emotions['anxiety'] > 0.3:
            epsilon_mod -= self.emotions['anxiety'] * 0.1

        # Satisfaction slightly reduces exploration (content)
        if self.emotions['satisfaction'] > 0.5:
            epsilon_mod -= 0.05

        return {
            'epsilon_modifier': max(-0.3, min(0.2, epsilon_mod)),
            'freeze_response': self.emotions['fear'] > 0.7,
            'explore_drive': self.emotions['curiosity'] > 0.5
        }

    def get_attention_modulation(self) -> Dict:
        """How emotions affect attention."""
        width_mod = 0.0

        # Fear → narrow, focused attention (on threat)
        if self.emotions['fear'] > 0.3:
            width_mod -= self.emotions['fear'] * 0.3

        # Curiosity → broader attention (scanning)
        if self.emotions['curiosity'] > 0.3:
            width_mod += self.emotions['curiosity'] * 0.2

        # Anxiety → broader but unfocused
        if self.emotions['anxiety'] > 0.3:
            width_mod += self.emotions['anxiety'] * 0.1

        return {
            'width_modifier': max(-0.4, min(0.3, width_mod)),
            'threat_focus': self.emotions['fear'] > 0.5
        }

    def get_learning_modulation(self) -> Dict:
        """How emotions affect learning."""
        learning_mod = 1.0

        # Fear enhances learning (survival memories!)
        if self.emotions['fear'] > 0.3:
            learning_mod *= 1.0 + self.emotions['fear'] * 0.3

        # Satisfaction enhances learning (reward memory)
        if self.emotions['satisfaction'] > 0.3:
            learning_mod *= 1.0 + self.emotions['satisfaction'] * 0.2

        # Pain creates strong memories
        if self.emotions['pain'] > 0.3:
            learning_mod *= 1.0 + self.emotions['pain'] * 0.4

        # Anxiety slightly impairs learning (distracted)
        if self.emotions['anxiety'] > 0.5:
            learning_mod *= 0.9

        return {
            'learning_rate_modifier': max(0.5, min(1.5, learning_mod)),
            'memory_priority': self.dominant_emotion
        }

    def get_valence_arousal(self) -> Dict:
        """
        Get emotional state in valence-arousal space.

        Valence: negative (-1) to positive (+1)
        Arousal: calm (0) to excited (1)
        """
        # Positive emotions
        positive = self.emotions['satisfaction'] + self.emotions['relief'] + self.emotions['curiosity'] * 0.5
        # Negative emotions
        negative = self.emotions['fear'] + self.emotions['pain'] + self.emotions['anxiety']

        valence = (positive - negative) / max(1, positive + negative + 0.1)
        valence = max(-1, min(1, valence))

        # Arousal based on intensity of any emotion
        arousal = max(self.emotions.values())

        return {
            'valence': round(valence, 3),
            'arousal': round(arousal, 3)
        }

    # === State Access ===

    def get_state(self) -> Dict:
        """Get current emotional state."""
        return {
            'emotions': {k: round(v, 3) for k, v in self.emotions.items()},
            'dominant': self.dominant_emotion,
            'intensity': round(self.dominant_intensity, 3),
            'description': self.get_emotion_description(),
            'duration': self.emotion_duration
        }

    def get_visualization_data(self) -> Dict:
        """Get data for frontend visualization."""
        va = self.get_valence_arousal()
        return {
            'emotions': {k: round(v, 3) for k, v in self.emotions.items()},
            'dominant': self.dominant_emotion,
            'intensity': round(self.dominant_intensity, 3),
            'description': self.get_emotion_description(),
            'valence': va['valence'],
            'arousal': va['arousal'],
            'duration': self.emotion_duration,
            # Learned associations
            'learned': {
                'predator_fear': round(self.predator_fear_learned, 3),
                'food_seeking': round(self.food_seeking_learned, 3),
                'pain_experiences': self.predator_pain_associations,
                'food_experiences': self.food_satisfaction_associations
            }
        }

    def clear(self):
        """Reset emotional state."""
        for name in self.emotions:
            self.emotions[name] = 0.0
        for name in self.emotion_history:
            self.emotion_history[name].clear()

        self.prev_safety = 1.0
        self.prev_energy = 0.8
        self.prev_health = 1.0

        self.dominant_emotion = 'neutral'
        self.dominant_intensity = 0.0
        self.emotion_duration = 0

        self.total_updates = 0
        self.fear_episodes = 0
        self.satisfaction_moments = 0

        # Reset learned associations
        self.predator_pain_associations = 0
        self.predator_fear_learned = 0.0
        self.food_satisfaction_associations = 0
        self.food_seeking_learned = 0.0
