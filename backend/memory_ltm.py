"""
Long-term Memory System (Episodic Memory)

Core Concept: "I remember what happened here"
- Stores episodes: (where, what state, what action, what outcome, how I felt)
- Emotion-tagged importance: Strong emotions = stronger memories
- Similarity-based recall: Similar situation → recall relevant memories
- Memory-informed decisions: "Last time here, I got hurt"

This completes the conscious loop:
(내 상태 인식) → (미래를 가정해봄) → (선택) → (결과를 회상/갱신)
                                              ↑ THIS

Key Insight:
Imagination predicts "what might happen"
Memory recalls "what actually happened"
Together they create informed, experience-based decision making.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class Episode:
    """A single memory episode - one moment in time."""

    # Where it happened
    position: Tuple[int, int]

    # Internal state at the time
    energy: float
    safety: float

    # What action was taken
    action: str  # 'up', 'down', 'left', 'right', 'stay'

    # What happened as a result
    outcome: str  # 'food', 'pain', 'nothing', 'near_danger', 'escape'
    reward: float

    # Emotional state when it happened
    dominant_emotion: str
    emotion_intensity: float  # 0-1, how strongly felt

    # === NEW: Actual outcome deltas (경험 기반 점수의 핵심) ===
    delta_energy: float = 0.0   # 에너지 변화량 (-1 ~ +1)
    delta_pain: float = 0.0     # 통증 변화량 (0 = 없음, 1 = 최대 고통)
    delta_safety: float = 0.0   # 안전 변화량 (-1 ~ +1)

    # === NEW: Context (상황 요약 - 위치 미신 방지) ===
    context_predator_near: bool = False   # 포식자가 근처에 있었나
    context_energy_low: bool = False      # 에너지가 낮았나
    context_was_fleeing: bool = False     # 도망 중이었나

    # Memory metadata
    importance: float = 0.0  # Calculated from emotion intensity
    recall_count: int = 0  # How many times this was recalled
    age: int = 0  # Steps since this memory was formed

    def __post_init__(self):
        # Calculate importance based on emotion and outcome
        self.importance = self._calculate_importance()

    def _calculate_importance(self) -> float:
        """
        Emotional memories are more important.
        Pain/fear memories are especially salient (survival value).
        """
        base_importance = self.emotion_intensity

        # Outcome-based importance boost
        if self.outcome == 'pain':
            base_importance *= 2.0  # Pain is very memorable
        elif self.outcome == 'food':
            base_importance *= 1.5  # Rewards are memorable
        elif self.outcome == 'escape':
            base_importance *= 1.3  # Successful escapes
        elif self.outcome == 'near_danger':
            base_importance *= 1.2  # Close calls

        # Emotion-based boost
        if self.dominant_emotion in ['fear', 'pain']:
            base_importance *= 1.5
        elif self.dominant_emotion == 'satisfaction':
            base_importance *= 1.2

        return min(1.0, base_importance)


class LongTermMemory:
    """
    Episodic memory system with emotion-based importance.

    Design principles:
    1. Not everything is remembered - only significant events
    2. Emotional events are remembered more strongly
    3. Old, unimportant memories fade
    4. Similar situations trigger recall
    """

    def __init__(self, max_episodes: int = 100):
        self.max_episodes = max_episodes
        self.episodes: deque = deque(maxlen=max_episodes * 2)  # Buffer before pruning

        # Thresholds for what's worth remembering
        self.min_importance_to_store = 0.2
        self.min_emotion_intensity = 0.3

        # Recall settings
        self.position_similarity_radius = 3  # Grid cells
        self.max_recall_count = 5  # Max memories to recall at once

        # Memory decay
        self.decay_rate = 0.001  # Importance decay per step
        self.recall_boost = 0.1  # Importance boost when recalled

        # Statistics
        self.total_stored = 0
        self.total_recalled = 0
        self.last_recall: List[Episode] = []
        self.last_recall_details: Dict[str, Optional[Dict]] = {}  # NEW: Details for UI

    def store(self,
              position: Tuple[int, int],
              energy: float,
              safety: float,
              action: str,
              outcome: str,
              reward: float,
              dominant_emotion: str,
              emotion_intensity: float,
              # NEW: Actual deltas (경험 기반 점수)
              delta_energy: float = 0.0,
              delta_pain: float = 0.0,
              delta_safety: float = 0.0,
              # NEW: Context (상황 요약)
              context_predator_near: bool = False,
              context_energy_low: bool = False,
              context_was_fleeing: bool = False) -> bool:
        """
        Attempt to store a new episode.

        Returns True if the episode was significant enough to store.
        """
        # Create episode
        episode = Episode(
            position=position,
            energy=energy,
            safety=safety,
            action=action,
            outcome=outcome,
            reward=reward,
            dominant_emotion=dominant_emotion,
            emotion_intensity=emotion_intensity,
            delta_energy=delta_energy,
            delta_pain=delta_pain,
            delta_safety=delta_safety,
            context_predator_near=context_predator_near,
            context_energy_low=context_energy_low,
            context_was_fleeing=context_was_fleeing
        )

        # Check if worth storing
        if episode.importance < self.min_importance_to_store:
            return False

        if emotion_intensity < self.min_emotion_intensity and outcome == 'nothing':
            return False

        # Store it
        self.episodes.append(episode)
        self.total_stored += 1

        # Prune if needed
        self._prune_old_memories()

        return True

    def recall(self,
               current_position: Tuple[int, int],
               current_action: Optional[str] = None,
               # NEW: Current context for similarity matching
               current_predator_near: bool = False,
               current_energy_low: bool = False,
               current_fleeing: bool = False) -> List[Episode]:
        """
        Recall memories relevant to current situation.

        Similar position + similar context + similar action = relevant memory.
        """
        relevant_memories = []

        for episode in self.episodes:
            # Check position similarity (Manhattan distance)
            dist = abs(episode.position[0] - current_position[0]) + \
                   abs(episode.position[1] - current_position[1])

            if dist <= self.position_similarity_radius:
                # Position is similar - this memory is relevant
                relevance = 1.0 - (dist / (self.position_similarity_radius + 1))

                # Boost relevance if same action
                if current_action and episode.action == current_action:
                    relevance *= 1.5

                # NEW: Context similarity bonus (위치 미신 방지)
                context_match = 0
                if episode.context_predator_near == current_predator_near:
                    context_match += 1
                if episode.context_energy_low == current_energy_low:
                    context_match += 1
                if episode.context_was_fleeing == current_fleeing:
                    context_match += 1
                # Context가 많이 일치할수록 더 관련 있는 기억
                context_bonus = 1.0 + (context_match * 0.15)  # 최대 1.45배
                relevance *= context_bonus

                # Combine with importance
                score = relevance * episode.importance

                relevant_memories.append((score, episode))

        # Sort by relevance score and take top N
        relevant_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [ep for _, ep in relevant_memories[:self.max_recall_count]]

        # Boost importance of recalled memories (they're useful!)
        for episode in top_memories:
            episode.recall_count += 1
            episode.importance = min(1.0, episode.importance + self.recall_boost)

        self.last_recall = top_memories
        self.total_recalled += len(top_memories)

        return top_memories

    def recall_for_direction(self,
                             current_position: Tuple[int, int],
                             direction: str,
                             # NEW: Context for similarity matching
                             current_predator_near: bool = False,
                             current_energy_low: bool = False,
                             current_fleeing: bool = False) -> Dict:
        """
        Recall memories relevant to moving in a specific direction.

        Returns aggregated info about what happened when moving that way.
        Now uses actual deltas instead of counts!
        """
        # Calculate target position
        dx, dy = {'up': (0, -1), 'down': (0, 1),
                  'left': (-1, 0), 'right': (1, 0)}.get(direction, (0, 0))
        target_pos = (current_position[0] + dx, current_position[1] + dy)

        # Find memories near target position (with context)
        memories = self.recall(target_pos, direction,
                               current_predator_near, current_energy_low, current_fleeing)

        if not memories:
            return {
                'has_memory': False,
                'pain_count': 0,
                'food_count': 0,
                'avg_reward': 0,
                'danger_level': 0,
                # NEW: Delta aggregates
                'avg_delta_energy': 0.0,
                'avg_delta_pain': 0.0,
                'avg_delta_safety': 0.0,
                'memories': []
            }

        # Aggregate memory info (legacy)
        pain_count = sum(1 for m in memories if m.outcome == 'pain')
        food_count = sum(1 for m in memories if m.outcome == 'food')
        near_danger = sum(1 for m in memories if m.outcome == 'near_danger')
        avg_reward = sum(m.reward for m in memories) / len(memories)

        # Calculate danger level from memories
        danger_level = 0.0
        for m in memories:
            if m.outcome == 'pain':
                danger_level += 0.4 * m.importance
            elif m.outcome == 'near_danger':
                danger_level += 0.2 * m.importance
        danger_level = min(1.0, danger_level)

        # NEW: Aggregate actual deltas (importance-weighted average)
        total_importance = sum(m.importance for m in memories)
        if total_importance > 0:
            avg_delta_energy = sum(m.delta_energy * m.importance for m in memories) / total_importance
            avg_delta_pain = sum(m.delta_pain * m.importance for m in memories) / total_importance
            avg_delta_safety = sum(m.delta_safety * m.importance for m in memories) / total_importance
        else:
            avg_delta_energy = 0.0
            avg_delta_pain = 0.0
            avg_delta_safety = 0.0

        return {
            'has_memory': True,
            'pain_count': pain_count,
            'food_count': food_count,
            'near_danger_count': near_danger,
            'avg_reward': avg_reward,
            'danger_level': danger_level,
            'memory_count': len(memories),
            # NEW: Delta aggregates
            'avg_delta_energy': avg_delta_energy,
            'avg_delta_pain': avg_delta_pain,
            'avg_delta_safety': avg_delta_safety,
            'memories': memories
        }

    def get_memory_influence(self,
                             current_position: Tuple[int, int],
                             # NEW: Context for similarity matching
                             current_predator_near: bool = False,
                             current_energy_low: bool = False,
                             current_fleeing: bool = False) -> Dict[str, float]:
        """
        Get memory-based score adjustment for each direction.

        NEW: Uses actual experience deltas instead of fixed counts!
        U = wE * delta_energy - wP * delta_pain + wS * delta_safety

        This makes the memory influence feel like "experience" not "rules".
        """
        # Weight coefficients (튜닝 가능)
        W_ENERGY = 1.0    # 에너지 얻으면 좋음
        W_PAIN = 3.0      # 통증은 강력히 회피 (생존 본능)
        W_SAFETY = 1.5    # 안전도 중요

        influences = {}
        self.last_recall_details = {}  # NEW: Store details for UI

        for direction in ['up', 'down', 'left', 'right']:
            mem_info = self.recall_for_direction(
                current_position, direction,
                current_predator_near, current_energy_low, current_fleeing
            )

            if not mem_info['has_memory']:
                influences[direction] = 0.0
                self.last_recall_details[direction] = None
                continue

            # NEW: Delta-based influence (경험 기반!)
            # U = wE * Δenergy - wP * Δpain + wS * Δsafety
            influence = 0.0
            influence += W_ENERGY * mem_info['avg_delta_energy']
            influence -= W_PAIN * mem_info['avg_delta_pain']  # Pain is negative
            influence += W_SAFETY * mem_info['avg_delta_safety']

            # Also add legacy reward influence (smaller weight)
            influence += mem_info['avg_reward'] * 0.1

            influences[direction] = influence

            # Store details for UI
            self.last_recall_details[direction] = {
                'delta_energy': round(mem_info['avg_delta_energy'], 3),
                'delta_pain': round(mem_info['avg_delta_pain'], 3),
                'delta_safety': round(mem_info['avg_delta_safety'], 3),
                'memory_count': mem_info['memory_count'],
                'computed_score': round(influence, 3)
            }

        return influences

    def step(self):
        """Called each simulation step - handles memory aging and decay."""
        for episode in self.episodes:
            episode.age += 1

            # Decay importance over time (but not below a minimum for traumatic memories)
            if episode.outcome != 'pain':  # Pain memories don't decay as fast
                episode.importance = max(0.1, episode.importance - self.decay_rate)
            else:
                episode.importance = max(0.3, episode.importance - self.decay_rate * 0.5)

    def _prune_old_memories(self):
        """Remove least important memories if over capacity."""
        if len(self.episodes) <= self.max_episodes:
            return

        # Convert to list, sort by importance, keep top N
        memories = list(self.episodes)
        memories.sort(key=lambda e: e.importance, reverse=True)

        # Keep the most important
        self.episodes = deque(memories[:self.max_episodes], maxlen=self.max_episodes * 2)

    def get_visualization_data(self) -> Dict:
        """Get data for frontend visualization."""
        # Summarize recent recalls with MORE detail
        recent_recalls = []
        for ep in self.last_recall[:3]:  # Top 3
            recent_recalls.append({
                'position': ep.position,
                'outcome': ep.outcome,
                'emotion': ep.dominant_emotion,
                'importance': round(ep.importance, 2),
                # NEW: Show actual deltas
                'delta_energy': round(ep.delta_energy, 3),
                'delta_pain': round(ep.delta_pain, 3),
                'delta_safety': round(ep.delta_safety, 3),
                'recall_count': ep.recall_count
            })

        # Memory statistics
        outcome_counts = {'pain': 0, 'food': 0, 'escape': 0, 'near_danger': 0, 'nothing': 0}
        for ep in self.episodes:
            outcome_counts[ep.outcome] = outcome_counts.get(ep.outcome, 0) + 1

        return {
            'total_memories': len(self.episodes),
            'total_stored': self.total_stored,
            'total_recalled': self.total_recalled,
            'recent_recalls': recent_recalls,
            'has_recall': len(self.last_recall) > 0,
            'recall_count': len(self.last_recall),
            'outcome_distribution': outcome_counts,
            'avg_importance': round(
                sum(e.importance for e in self.episodes) / max(1, len(self.episodes)), 2
            ),
            # NEW: Direction-wise recall details
            'recall_details': self.last_recall_details
        }

    def get_recall_reason(self) -> str:
        """Generate human-readable reason from recent recalls."""
        if not self.last_recall:
            return ""

        # Find the most impactful memory
        top_memory = max(self.last_recall, key=lambda e: e.importance)

        if top_memory.outcome == 'pain':
            return f"여기서 아팠어! ({top_memory.recall_count}회 기억)"
        elif top_memory.outcome == 'food':
            return f"여기서 먹었어! ({top_memory.recall_count}회 기억)"
        elif top_memory.outcome == 'near_danger':
            return f"여기 위험했어..."
        elif top_memory.outcome == 'escape':
            return f"여기서 도망쳤어"
        else:
            return f"기억이 있어 ({len(self.last_recall)}개)"
