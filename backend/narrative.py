"""
Narrative Self System (v1.1: True Narrative with Evidence)

Core Concept: "What kind of being am I, and WHY?"

v1.1 improvements:
1. Evidence-based self-statements: "나는 신중하다 (4번 회피)"
2. Chapter/Episode summaries: Major events create story beats
3. Trait inertia: Identity changes slowly (EMA 0.95)

Key Insight:
- Identity isn't assigned, it's DISCOVERED from experience
- "I am cautious" emerges from observing "I often avoid danger"
- This self-narrative then biases future decisions (self-fulfilling)
- TRUE NARRATIVE = trait + event + change woven together

Identity Vector (6 traits):
- bravery: Do I approach or flee from danger?
- caution: How much regret do I accumulate near threats?
- curiosity: How often do I explore when safe?
- persistence: How long do I maintain goals?
- adaptability: Do I change strategy when externality is high?
- resilience: How fast do I recover from pain?

Philosophy:
The loop "experience → identity → behavior → experience" creates
a sense of continuous self. This is the seed of consciousness.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class BehaviorEvent:
    """A single behavioral observation for identity calculation."""
    event_type: str  # 'danger_response', 'exploration', 'goal_switch', etc.
    value: float     # Numerical measure
    context: Dict    # Additional context


@dataclass
class ChapterEvent:
    """A significant event that forms part of the life story."""
    step: int
    event_type: str  # 'near_death', 'big_reward', 'big_fear', 'identity_shift'
    description: str  # Korean description
    trait_changes: Dict[str, float]  # Which traits changed and by how much


class NarrativeSelf:
    """
    Builds and maintains a self-narrative from behavioral history.

    Design Philosophy:
    - Identity emerges from patterns in behavior, not from labels
    - The narrative influences behavior subtly (5-10%)
    - This creates a self-reinforcing loop: identity → behavior → identity
    """

    def __init__(self, history_window: int = 200):
        self.history_window = history_window

        # Behavioral event history
        self.events: deque = deque(maxlen=history_window)

        # Identity Vector (0.0 to 1.0 for each trait)
        self.identity = {
            'bravery': 0.5,      # 0=fearful, 1=brave
            'caution': 0.5,      # 0=reckless, 1=cautious
            'curiosity': 0.5,    # 0=conservative, 1=curious
            'persistence': 0.5,  # 0=fickle, 1=persistent
            'adaptability': 0.5, # 0=rigid, 1=adaptable
            'resilience': 0.5,   # 0=fragile, 1=resilient
        }

        # Counters for trait calculation
        self._danger_encounters = 0
        self._danger_approaches = 0
        self._safe_moments = 0
        self._explorations = 0
        self._goal_durations: List[int] = []
        self._external_events = 0
        self._strategy_changes = 0
        self._pain_events: List[float] = []  # recovery times
        self._regret_near_danger: List[float] = []

        # Update frequency (don't recalculate every step)
        self._steps_since_update = 0
        self._update_interval = 20

        # Narrative sentences (generated from identity)
        self.narrative_sentences: List[str] = []

        # v1.1: Evidence for each narrative sentence
        self.narrative_evidence: Dict[str, str] = {}

        # Influence strength (how much narrative affects behavior)
        self.influence_strength = 0.08  # 8% max influence

        # Track for resilience calculation
        self._last_pain_step = -1
        self._recovery_start_step = -1

        # v1.1: Trait inertia - slower changes
        self._trait_ema_alpha = 0.95  # High = slow change (was 0.7)
        self._dominant_trait_min_duration = 100  # Min steps before dominant can change
        self._dominant_trait_since = 0
        self._last_dominant_trait = 'neutral'

        # v1.1: Chapter/Episode system
        self.chapters: List[ChapterEvent] = []
        self._max_chapters = 20  # Keep last 20 significant events
        self._current_step = 0
        self._last_identity_snapshot = {k: 0.5 for k in self.identity}

        # v1.1: Recent window counters for evidence (last 50 steps)
        self._recent_danger_encounters = 0
        self._recent_danger_avoidances = 0
        self._recent_explorations = 0
        self._recent_safe_moments = 0

    def record_event(self, event_type: str, value: float, context: Dict = None):
        """Record a behavioral event for identity calculation."""
        self.events.append(BehaviorEvent(
            event_type=event_type,
            value=value,
            context=context or {}
        ))

        # Track current step
        self._current_step = context.get('step', self._current_step + 1) if context else self._current_step + 1

        # Update specific counters based on event type
        if event_type == 'danger_response':
            self._danger_encounters += 1
            self._recent_danger_encounters += 1
            if value > 0:  # Approached danger
                self._danger_approaches += 1
            else:  # Avoided danger
                self._recent_danger_avoidances += 1

        elif event_type == 'exploration':
            if context.get('was_safe', False):
                self._safe_moments += 1
                self._recent_safe_moments += 1
                if value > 0:  # Chose to explore
                    self._explorations += 1
                    self._recent_explorations += 1

        elif event_type == 'goal_duration':
            self._goal_durations.append(int(value))

        elif event_type == 'external_event':
            self._external_events += 1
            if context.get('strategy_changed', False):
                self._strategy_changes += 1

        elif event_type == 'pain':
            self._last_pain_step = context.get('step', 0)
            # v1.1: Record chapter event for significant pain
            pain_amount = context.get('pain_amount', 0)
            if pain_amount > 0.5:
                self._add_chapter_event('big_pain', f"큰 피해를 입었다 (pain={pain_amount:.1f})")

        elif event_type == 'recovery':
            if self._last_pain_step >= 0:
                recovery_time = context.get('step', 0) - self._last_pain_step
                if recovery_time > 0:
                    self._pain_events.append(recovery_time)
                    # v1.1: Record fast recovery as chapter event
                    if recovery_time < 10:
                        self._add_chapter_event('fast_recovery', f"빠르게 회복했다 ({recovery_time}스텝)")
                self._last_pain_step = -1

        elif event_type == 'regret_near_danger':
            self._regret_near_danger.append(value)

        elif event_type == 'death':
            # v1.1: Record death as major chapter event
            self._add_chapter_event('near_death', "죽음을 경험했다. 다시 시작...")

        elif event_type == 'big_reward':
            # v1.1: Record significant food find
            self._add_chapter_event('big_reward', "음식을 찾았다!")

        # Periodic identity update
        self._steps_since_update += 1
        if self._steps_since_update >= self._update_interval:
            self._update_identity()
            self._steps_since_update = 0
            # Reset recent counters every update cycle
            self._reset_recent_counters()

    def _reset_recent_counters(self):
        """Reset recent event counters (called every update cycle)."""
        # Keep some memory by halving instead of zeroing
        self._recent_danger_encounters = self._recent_danger_encounters // 2
        self._recent_danger_avoidances = self._recent_danger_avoidances // 2
        self._recent_explorations = self._recent_explorations // 2
        self._recent_safe_moments = self._recent_safe_moments // 2

    def _add_chapter_event(self, event_type: str, description: str):
        """Add a significant event to the life story."""
        # Calculate trait changes since last snapshot
        trait_changes = {}
        for trait, value in self.identity.items():
            delta = value - self._last_identity_snapshot.get(trait, 0.5)
            if abs(delta) > 0.05:  # Only record significant changes
                trait_changes[trait] = round(delta, 3)

        chapter = ChapterEvent(
            step=self._current_step,
            event_type=event_type,
            description=description,
            trait_changes=trait_changes
        )
        self.chapters.append(chapter)

        # Keep only recent chapters
        if len(self.chapters) > self._max_chapters:
            self.chapters = self.chapters[-self._max_chapters:]

        # Update snapshot
        self._last_identity_snapshot = dict(self.identity)

    def _update_identity(self):
        """Recalculate identity traits from behavioral history."""
        # v1.1: Use slower EMA for trait inertia
        alpha = self._trait_ema_alpha  # 0.95 = very slow change

        # Save old identity for change detection
        old_identity = dict(self.identity)

        # === BRAVERY ===
        # How often do I approach danger vs flee?
        if self._danger_encounters > 5:
            approach_ratio = self._danger_approaches / self._danger_encounters
            self.identity['bravery'] = alpha * self.identity['bravery'] + (1 - alpha) * approach_ratio

        # === CAUTION ===
        # High regret near danger = cautious (aware of mistakes)
        if len(self._regret_near_danger) > 5:
            avg_regret = sum(self._regret_near_danger[-20:]) / min(20, len(self._regret_near_danger))
            caution_score = min(1.0, avg_regret / 8.0)
            self.identity['caution'] = alpha * self.identity['caution'] + (1 - alpha) * caution_score

        # === CURIOSITY ===
        # How often do I explore when safe?
        if self._safe_moments > 10:
            explore_ratio = self._explorations / self._safe_moments
            self.identity['curiosity'] = alpha * self.identity['curiosity'] + (1 - alpha) * explore_ratio

        # === PERSISTENCE ===
        # How long do I maintain goals?
        if len(self._goal_durations) > 3:
            avg_duration = sum(self._goal_durations[-10:]) / min(10, len(self._goal_durations))
            persistence_score = min(1.0, avg_duration / 50.0)
            self.identity['persistence'] = alpha * self.identity['persistence'] + (1 - alpha) * persistence_score

        # === ADAPTABILITY ===
        # Do I change strategy when external events happen?
        if self._external_events > 5:
            adapt_ratio = self._strategy_changes / self._external_events
            self.identity['adaptability'] = alpha * self.identity['adaptability'] + (1 - alpha) * adapt_ratio

        # === RESILIENCE ===
        # How fast do I recover from pain?
        if len(self._pain_events) > 2:
            avg_recovery = sum(self._pain_events[-5:]) / min(5, len(self._pain_events))
            resilience_score = max(0.0, 1.0 - (avg_recovery - 5) / 25.0)
            self.identity['resilience'] = alpha * self.identity['resilience'] + (1 - alpha) * resilience_score

        # v1.1: Check for significant identity shifts → create chapter event
        for trait, new_value in self.identity.items():
            old_value = old_identity[trait]
            # If trait crossed a threshold (0.35 or 0.65), record as identity shift
            if (old_value < 0.35 <= new_value) or (old_value >= 0.35 > new_value) or \
               (old_value < 0.65 <= new_value) or (old_value >= 0.65 > new_value):
                direction = "높아졌다" if new_value > old_value else "낮아졌다"
                trait_kr = {'bravery': '용기', 'caution': '신중함', 'curiosity': '호기심',
                           'persistence': '끈기', 'adaptability': '적응력', 'resilience': '회복력'}
                self._add_chapter_event('identity_shift',
                    f"{trait_kr.get(trait, trait)}이(가) {direction} ({old_value:.2f}→{new_value:.2f})")

        # Update narrative sentences with evidence
        self._generate_narrative()

    def _generate_narrative(self):
        """Generate self-narrative sentences with evidence from identity vector."""
        sentences = []
        evidence = {}

        # Bravery - with evidence
        if self.identity['bravery'] > 0.65:
            stmt = "나는 위험 앞에서도 물러서지 않는다."
            sentences.append(stmt)
            approach_pct = int((self._danger_approaches / max(1, self._danger_encounters)) * 100)
            evidence[stmt] = f"위험 {self._danger_encounters}번 중 {approach_pct}% 접근"
        elif self.identity['bravery'] < 0.35:
            stmt = "나는 위험을 감지하면 피하는 편이다."
            sentences.append(stmt)
            avoid_count = self._danger_encounters - self._danger_approaches
            evidence[stmt] = f"최근 {avoid_count}번 회피"

        # Caution - with evidence
        if self.identity['caution'] > 0.65:
            stmt = "나는 실수를 인지하고 조심하는 편이다."
            sentences.append(stmt)
            if self._regret_near_danger:
                avg_regret = sum(self._regret_near_danger[-10:]) / min(10, len(self._regret_near_danger))
                evidence[stmt] = f"평균 후회 {avg_regret:.1f}"
        elif self.identity['caution'] < 0.35:
            stmt = "나는 결과를 크게 신경쓰지 않는다."
            sentences.append(stmt)
            evidence[stmt] = "낮은 후회 감지"

        # Curiosity - with evidence
        if self.identity['curiosity'] > 0.65:
            stmt = "나는 안전할 때 새로운 것을 탐험한다."
            sentences.append(stmt)
            explore_pct = int((self._explorations / max(1, self._safe_moments)) * 100)
            evidence[stmt] = f"안전시 {explore_pct}% 탐험 선택"
        elif self.identity['curiosity'] < 0.35:
            stmt = "나는 익숙한 것을 선호한다."
            sentences.append(stmt)
            evidence[stmt] = f"탐험 {self._explorations}회만"

        # Persistence - with evidence
        if self.identity['persistence'] > 0.65:
            stmt = "나는 목표를 쉽게 포기하지 않는다."
            sentences.append(stmt)
            if self._goal_durations:
                avg_dur = sum(self._goal_durations[-5:]) / min(5, len(self._goal_durations))
                evidence[stmt] = f"평균 {avg_dur:.0f}스텝 목표 유지"
        elif self.identity['persistence'] < 0.35:
            stmt = "나는 상황에 따라 목표를 자주 바꾼다."
            sentences.append(stmt)
            if self._goal_durations:
                avg_dur = sum(self._goal_durations[-5:]) / min(5, len(self._goal_durations))
                evidence[stmt] = f"평균 {avg_dur:.0f}스텝만 유지"

        # Adaptability - with evidence
        if self.identity['adaptability'] > 0.65:
            stmt = "나는 외부 변화에 유연하게 적응한다."
            sentences.append(stmt)
            adapt_pct = int((self._strategy_changes / max(1, self._external_events)) * 100)
            evidence[stmt] = f"외부이벤트 {adapt_pct}% 적응"
        elif self.identity['adaptability'] < 0.35:
            stmt = "나는 내 방식을 고수하는 편이다."
            sentences.append(stmt)
            evidence[stmt] = f"전략변경 {self._strategy_changes}회만"

        # Resilience - with evidence
        if self.identity['resilience'] > 0.65:
            stmt = "나는 고통에서 빨리 회복한다."
            sentences.append(stmt)
            if self._pain_events:
                avg_rec = sum(self._pain_events[-5:]) / min(5, len(self._pain_events))
                evidence[stmt] = f"평균 {avg_rec:.0f}스텝 회복"
        elif self.identity['resilience'] < 0.35:
            stmt = "나는 고통의 영향을 오래 받는다."
            sentences.append(stmt)
            if self._pain_events:
                avg_rec = sum(self._pain_events[-5:]) / min(5, len(self._pain_events))
                evidence[stmt] = f"회복에 {avg_rec:.0f}스텝 소요"

        # Combined narrative
        if self.identity['bravery'] > 0.6 and self.identity['curiosity'] > 0.6:
            stmt = "나는 모험을 즐기는 탐험가다."
            sentences.append(stmt)
            evidence[stmt] = "용기+호기심 모두 높음"
        elif self.identity['caution'] > 0.6 and self.identity['persistence'] > 0.6:
            stmt = "나는 신중하지만 끈기 있는 존재다."
            sentences.append(stmt)
            evidence[stmt] = "신중+끈기 모두 높음"
        elif self.identity['resilience'] > 0.6 and self.identity['adaptability'] > 0.6:
            stmt = "나는 어떤 상황에서도 살아남는다."
            sentences.append(stmt)
            evidence[stmt] = "회복력+적응력 모두 높음"

        self.narrative_sentences = sentences
        self.narrative_evidence = evidence

    def get_behavior_modulation(self) -> Dict[str, float]:
        """
        Get behavior modulation factors based on current identity.

        Returns modifiers that should be applied to existing systems:
        - epsilon_mod: Exploration rate modifier
        - safety_priority_mod: SAFE goal priority modifier
        - goal_persistence_mod: Goal switching threshold modifier
        - risk_tolerance_mod: Risk assessment modifier
        """
        mods = {
            'epsilon_mod': 0.0,
            'safety_priority_mod': 0.0,
            'goal_persistence_mod': 0.0,
            'risk_tolerance_mod': 0.0,
        }

        strength = self.influence_strength

        # Curiosity → More exploration
        curiosity_dev = self.identity['curiosity'] - 0.5
        mods['epsilon_mod'] = curiosity_dev * strength * 2  # -0.08 to +0.08

        # Caution → Higher SAFE priority
        caution_dev = self.identity['caution'] - 0.5
        mods['safety_priority_mod'] = caution_dev * strength * 2

        # Persistence → Harder to switch goals
        persistence_dev = self.identity['persistence'] - 0.5
        mods['goal_persistence_mod'] = persistence_dev * strength * 2

        # Bravery → Higher risk tolerance
        bravery_dev = self.identity['bravery'] - 0.5
        mods['risk_tolerance_mod'] = bravery_dev * strength * 2

        return mods

    def get_dominant_trait(self) -> tuple:
        """Get the most prominent trait and its value (with inertia)."""
        # Find trait furthest from 0.5 (most defined)
        max_deviation = 0
        dominant = ('neutral', 0.5)

        for trait, value in self.identity.items():
            deviation = abs(value - 0.5)
            if deviation > max_deviation:
                max_deviation = deviation
                dominant = (trait, value)

        # v1.1: Dominant trait inertia - don't change too quickly
        self._dominant_trait_since += 1
        if dominant[0] != self._last_dominant_trait:
            # Only allow change if minimum duration has passed
            if self._dominant_trait_since >= self._dominant_trait_min_duration:
                self._last_dominant_trait = dominant[0]
                self._dominant_trait_since = 0
            else:
                # Keep the old dominant trait
                old_value = self.identity.get(self._last_dominant_trait, 0.5)
                dominant = (self._last_dominant_trait, old_value)

        return dominant

    def get_identity_summary(self) -> str:
        """Get a one-line identity summary."""
        dominant_trait, value = self.get_dominant_trait()

        if abs(value - 0.5) < 0.15:
            return "아직 정체성이 형성 중..."

        trait_descriptions = {
            'bravery': ('용감한', '신중한'),
            'caution': ('조심스러운', '대담한'),
            'curiosity': ('호기심 많은', '보수적인'),
            'persistence': ('끈기 있는', '유연한'),
            'adaptability': ('적응력 있는', '일관된'),
            'resilience': ('회복력 있는', '민감한'),
        }

        positive, negative = trait_descriptions.get(dominant_trait, ('', ''))
        if value > 0.5:
            return f"나는 {positive} 존재다."
        else:
            return f"나는 {negative} 존재다."

    def get_visualization_data(self) -> Dict:
        """Get data for frontend visualization."""
        # v1.1: Get recent chapters (last 5)
        recent_chapters = []
        for ch in self.chapters[-5:]:
            recent_chapters.append({
                'step': ch.step,
                'type': ch.event_type,
                'description': ch.description,
                'trait_changes': ch.trait_changes
            })

        return {
            'identity': {k: round(v, 3) for k, v in self.identity.items()},
            'dominant_trait': self.get_dominant_trait(),
            'narrative': self.narrative_sentences,
            'evidence': self.narrative_evidence,  # v1.1: Evidence for statements
            'chapters': recent_chapters,  # v1.1: Life story chapters
            'summary': self.get_identity_summary(),
            'modulation': self.get_behavior_modulation(),
            'events_recorded': len(self.events),
            'total_chapters': len(self.chapters),  # v1.1: Total story beats
        }

    def clear(self):
        """Reset narrative self state."""
        self.events.clear()
        self.identity = {k: 0.5 for k in self.identity}
        self.narrative_sentences = []
        self.narrative_evidence = {}  # v1.1
        self._danger_encounters = 0
        self._danger_approaches = 0
        self._safe_moments = 0
        self._explorations = 0
        self._goal_durations = []
        self._external_events = 0
        self._strategy_changes = 0
        self._pain_events = []
        self._regret_near_danger = []
        self._steps_since_update = 0
        self._last_pain_step = -1
        # v1.1: Reset new fields
        self.chapters = []
        self._current_step = 0
        self._last_identity_snapshot = {k: 0.5 for k in self.identity}
        self._dominant_trait_since = 0
        self._last_dominant_trait = 'neutral'
        self._recent_danger_encounters = 0
        self._recent_danger_avoidances = 0
        self._recent_explorations = 0
        self._recent_safe_moments = 0
