"""
v4.1 Memory Consolidation (Sleep/Integration)

핵심 철학:
- Sleep = "졸림 게이지"가 아니라 내부 신호 기반 트리거
- 통합 = LTM을 "조언자"에서 "뇌 구조(prior)"로 전환
- 목표 = transition_std 감소, uncertainty 감소, G 감소

Sleep 트리거 조건:
- low_surprise: 최근 예측 오차가 낮음
- high_redundancy: 유사 에피소드가 자주 저장됨
- stable_context: context가 안정적

Sleep 중 수행:
1. Episode replay → transition model 재학습 (transition_std 감소)
2. 유사 에피소드 클러스터링 → 프로토타입(압축 기억) 생성
3. Context model (expected/transition) 업데이트
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque

from .memory import Episode, LTMStore


@dataclass
class SleepTriggerState:
    """Sleep 트리거를 위한 내부 신호 상태"""
    # Recent surprise (prediction error) history
    surprise_history: deque = field(default_factory=lambda: deque(maxlen=50))

    # Redundancy tracking (merge rate)
    merge_rate_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # Context stability
    context_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Thresholds
    surprise_threshold: float = 0.3  # Low surprise = below this
    redundancy_threshold: float = 0.4  # High redundancy = above this
    context_stability_threshold: float = 0.7  # Stable = same context ratio above this

    # Minimum steps between sleeps
    min_awake_steps: int = 100
    steps_since_sleep: int = 0

    def update(self, surprise: float, merged: bool, context_id: int):
        """Update trigger state with new observations"""
        self.surprise_history.append(surprise)
        self.merge_rate_history.append(1.0 if merged else 0.0)
        self.context_history.append(context_id)
        self.steps_since_sleep += 1

    def should_sleep(self) -> Tuple[bool, Dict[str, float]]:
        """Check if sleep conditions are met"""
        if self.steps_since_sleep < self.min_awake_steps:
            return False, {'reason': 'too_soon', 'steps': self.steps_since_sleep}

        if len(self.surprise_history) < 30:
            return False, {'reason': 'not_enough_data'}

        # Compute signals
        avg_surprise = np.mean(list(self.surprise_history)[-30:])
        low_surprise = avg_surprise < self.surprise_threshold

        if len(self.merge_rate_history) >= 10:
            merge_rate = np.mean(list(self.merge_rate_history)[-10:])
            high_redundancy = merge_rate > self.redundancy_threshold
        else:
            high_redundancy = False
            merge_rate = 0.0

        if len(self.context_history) >= 20:
            recent_contexts = list(self.context_history)[-20:]
            most_common = max(set(recent_contexts), key=recent_contexts.count)
            stability = recent_contexts.count(most_common) / len(recent_contexts)
            stable_context = stability > self.context_stability_threshold
        else:
            stable_context = False
            stability = 0.0

        signals = {
            'avg_surprise': avg_surprise,
            'low_surprise': low_surprise,
            'merge_rate': merge_rate if len(self.merge_rate_history) >= 10 else 0.0,
            'high_redundancy': high_redundancy,
            'context_stability': stability if len(self.context_history) >= 20 else 0.0,
            'stable_context': stable_context,
            'steps_since_sleep': self.steps_since_sleep
        }

        # All conditions must be met
        should = low_surprise and high_redundancy and stable_context

        return should, signals

    def mark_sleep_done(self):
        """Reset counter after sleep"""
        self.steps_since_sleep = 0


@dataclass
class ConsolidationResult:
    """Sleep consolidation 결과"""
    episodes_replayed: int
    transition_updates: int
    prototypes_created: int
    context_updates: int

    # Before/after metrics
    transition_std_before: float
    transition_std_after: float

    # Quality metrics
    compression_ratio: float  # episodes / prototypes
    avg_cluster_size: float


@dataclass
class Prototype:
    """압축된 기억 프로토타입"""
    context_id: int
    obs_centroid: np.ndarray
    action: int
    avg_delta_energy: float
    avg_delta_pain: float
    avg_outcome_score: float
    episode_count: int  # 몇 개의 에피소드가 압축됐는지
    confidence: float  # sqrt(n)/(sqrt(n)+k) 기반


class MemoryConsolidator:
    """
    v4.1 Memory Consolidation System

    FEP 원칙:
    - Sleep은 "게이지"가 아닌 내부 신호로 트리거
    - 통합은 prior를 업데이트하여 미래 F/G를 줄임
    - 심즈식 "휴식 필요" 없이, 정보론적 조건으로 결정
    """

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        min_cluster_size: int = 2,
        replay_batch_size: int = 20,
        transition_learning_rate: float = 0.1,
        context_learning_rate: float = 0.05
    ):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.replay_batch_size = replay_batch_size
        self.transition_lr = transition_learning_rate
        self.context_lr = context_learning_rate

        self.trigger_state = SleepTriggerState()
        self.prototypes: List[Prototype] = []

        # Statistics
        self.total_sleeps = 0
        self.total_episodes_consolidated = 0
        self.last_result: Optional[ConsolidationResult] = None

    def update_trigger(self, surprise: float, merged: bool, context_id: int):
        """Update sleep trigger with new step data"""
        self.trigger_state.update(surprise, merged, context_id)

    def check_sleep_trigger(self) -> Tuple[bool, Dict]:
        """Check if sleep should be triggered"""
        return self.trigger_state.should_sleep()

    def consolidate(
        self,
        ltm_store: LTMStore,
        transition_model: 'TransitionModel',
        hierarchy_controller: Optional['HierarchicalController'] = None
    ) -> ConsolidationResult:
        """
        Execute sleep consolidation

        1. Episode replay → transition model update
        2. Episode clustering → prototype generation
        3. Context model update (if hierarchy enabled)
        """
        episodes = ltm_store.episodes

        if len(episodes) < 3:
            return ConsolidationResult(
                episodes_replayed=0,
                transition_updates=0,
                prototypes_created=0,
                context_updates=0,
                transition_std_before=0.0,
                transition_std_after=0.0,
                compression_ratio=0.0,
                avg_cluster_size=0.0
            )

        # Record before state
        transition_std_before = self._get_avg_transition_std(transition_model)

        # 1. Episode Replay → Transition Model Update
        replay_count, transition_updates = self._replay_episodes(
            episodes, transition_model
        )

        # 2. Episode Clustering → Prototype Generation
        new_prototypes, cluster_sizes = self._cluster_and_compress(episodes)
        self.prototypes.extend(new_prototypes)

        # 3. Context Model Update (if hierarchy enabled)
        context_updates = 0
        if hierarchy_controller is not None:
            context_updates = self._update_context_model(
                episodes, hierarchy_controller
            )

        # Record after state
        transition_std_after = self._get_avg_transition_std(transition_model)

        # Mark sleep done
        self.trigger_state.mark_sleep_done()
        self.total_sleeps += 1
        self.total_episodes_consolidated += len(episodes)

        # Compute result
        compression_ratio = len(episodes) / max(1, len(new_prototypes)) if new_prototypes else 0.0
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0

        result = ConsolidationResult(
            episodes_replayed=replay_count,
            transition_updates=transition_updates,
            prototypes_created=len(new_prototypes),
            context_updates=context_updates,
            transition_std_before=transition_std_before,
            transition_std_after=transition_std_after,
            compression_ratio=compression_ratio,
            avg_cluster_size=avg_cluster_size
        )

        self.last_result = result
        return result

    def _replay_episodes(
        self,
        episodes: List[Episode],
        transition_model: 'TransitionModel'
    ) -> Tuple[int, int]:
        """
        Episode replay: 저장된 경험으로 transition model 재학습

        핵심: transition_std를 줄여서 ambiguity 감소
        """
        # Select episodes to replay (prioritize high outcome_score)
        sorted_episodes = sorted(
            episodes,
            key=lambda e: abs(e.outcome_score),
            reverse=True
        )
        replay_batch = sorted_episodes[:self.replay_batch_size]

        updates = 0
        for ep in replay_batch:
            action = ep.action

            # Compute observed delta from episode
            obs_delta = np.array([
                ep.delta_energy,
                ep.delta_pain,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Pad to match obs dimensions
            ])

            # Update transition model with lower learning rate (consolidation)
            # This should reduce transition_std for this action
            if hasattr(transition_model, 'update_from_replay'):
                # Get obs_dim from transition_model dict
                obs_dim = 8  # Default
                if hasattr(transition_model, 'transition_model'):
                    obs_dim = transition_model.transition_model['delta_mean'].shape[1]

                transition_model.update_from_replay(
                    action=action,
                    obs_delta=obs_delta[:obs_dim],
                    learning_rate=self.transition_lr
                )
                updates += 1

        return len(replay_batch), updates

    def _cluster_and_compress(
        self,
        episodes: List[Episode]
    ) -> Tuple[List[Prototype], List[int]]:
        """
        유사 에피소드 클러스터링 → 프로토타입 생성

        핵심: 반복되는 패턴을 압축하여 일반화
        """
        if len(episodes) < self.min_cluster_size:
            return [], []

        # Group by (context_id, action)
        groups: Dict[Tuple[int, int], List[Episode]] = {}
        for ep in episodes:
            key = (ep.context_id, ep.action)
            if key not in groups:
                groups[key] = []
            groups[key].append(ep)

        prototypes = []
        cluster_sizes = []

        for (context_id, action), group in groups.items():
            if len(group) < self.min_cluster_size:
                continue

            # Compute centroid
            obs_arrays = [ep.obs_summary for ep in group]
            centroid = np.mean(obs_arrays, axis=0)

            # Compute averages
            avg_delta_energy = np.mean([ep.delta_energy for ep in group])
            avg_delta_pain = np.mean([ep.delta_pain for ep in group])
            avg_outcome = np.mean([ep.outcome_score for ep in group])

            # Confidence based on count
            n = len(group)
            k = 2.0
            confidence = np.sqrt(n) / (np.sqrt(n) + k)

            proto = Prototype(
                context_id=context_id,
                obs_centroid=centroid,
                action=action,
                avg_delta_energy=avg_delta_energy,
                avg_delta_pain=avg_delta_pain,
                avg_outcome_score=avg_outcome,
                episode_count=n,
                confidence=confidence
            )

            prototypes.append(proto)
            cluster_sizes.append(n)

        return prototypes, cluster_sizes

    def _update_context_model(
        self,
        episodes: List[Episode],
        hierarchy_controller: 'HierarchicalController'
    ) -> int:
        """
        Context model 업데이트: slow layer가 "세상 규칙"을 더 잘 배움

        핵심: context별 transition 예측을 개선
        """
        if not hasattr(hierarchy_controller, 'slow_layer'):
            return 0

        slow_layer = hierarchy_controller.slow_layer
        updates = 0

        # Group episodes by context
        context_episodes: Dict[int, List[Episode]] = {}
        for ep in episodes:
            if ep.context_id not in context_episodes:
                context_episodes[ep.context_id] = []
            context_episodes[ep.context_id].append(ep)

        # Update context expectations
        for context_id, eps in context_episodes.items():
            if context_id >= slow_layer.K:
                continue

            # Compute statistics for this context
            avg_energy = np.mean([ep.obs_summary[6] if len(ep.obs_summary) > 6 else 0.5
                                  for ep in eps])
            avg_outcome = np.mean([ep.outcome_score for ep in eps])

            # Update context expected values
            if hasattr(slow_layer, 'expected') and context_id < len(slow_layer.expected):
                # EMA update with consolidation learning rate
                current = slow_layer.expected[context_id]
                # Update energy expectation (index 6 in obs_stats)
                if len(current) > 3:  # Assuming obs_stats format
                    current[3] = (1 - self.context_lr) * current[3] + self.context_lr * avg_energy

                updates += 1

        return updates

    def _get_avg_transition_std(self, transition_model) -> float:
        """Get average transition standard deviation across actions"""
        # Handle ActionSelector which has transition_model as a dict
        if hasattr(transition_model, 'transition_model'):
            return float(np.mean(transition_model.transition_model['delta_std']))
        elif hasattr(transition_model, 'delta_std'):
            return float(np.mean(transition_model.delta_std))
        return 0.0

    def get_prototype_bias(self, current_obs: np.ndarray, context_id: int) -> np.ndarray:
        """
        프로토타입 기반 행동 bias 계산

        현재 상황과 유사한 프로토타입의 outcome을 기반으로 G 조정
        """
        bias = np.zeros(6)  # 6 actions (including THINK)

        if not self.prototypes:
            return bias

        # Find matching prototypes
        for proto in self.prototypes:
            if proto.context_id != context_id:
                continue

            # Compute similarity
            if len(current_obs) >= len(proto.obs_centroid):
                obs_subset = current_obs[:len(proto.obs_centroid)]
            else:
                obs_subset = current_obs

            similarity = self._cosine_similarity(obs_subset, proto.obs_centroid)

            if similarity > self.similarity_threshold:
                # Good outcome → negative bias (lower G = prefer)
                # Bad outcome → positive bias (higher G = avoid)
                action = proto.action
                if 0 <= action < 6:
                    # Weight by similarity and confidence
                    weight = similarity * proto.confidence
                    bias[action] -= proto.avg_outcome_score * weight * 0.1

        return bias

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_status(self) -> Dict:
        """Get consolidation system status"""
        should_sleep, signals = self.check_sleep_trigger()

        # Convert numpy types to Python native types for JSON serialization
        def to_python(obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return bool(obj) if isinstance(obj, np.bool_) else int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            return obj

        return {
            'enabled': True,
            'should_sleep': bool(should_sleep),
            'trigger_signals': to_python(signals),
            'stats': {
                'total_sleeps': int(self.total_sleeps),
                'total_episodes_consolidated': int(self.total_episodes_consolidated),
                'prototype_count': len(self.prototypes),
                'steps_since_sleep': int(self.trigger_state.steps_since_sleep)
            },
            'last_result': {
                'episodes_replayed': int(self.last_result.episodes_replayed) if self.last_result else 0,
                'transition_updates': int(self.last_result.transition_updates) if self.last_result else 0,
                'prototypes_created': int(self.last_result.prototypes_created) if self.last_result else 0,
                'transition_std_before': float(self.last_result.transition_std_before) if self.last_result else 0.0,
                'transition_std_after': float(self.last_result.transition_std_after) if self.last_result else 0.0,
                'compression_ratio': float(self.last_result.compression_ratio) if self.last_result else 0.0
            } if self.last_result else None
        }

    def reset(self):
        """Reset consolidation system"""
        self.trigger_state = SleepTriggerState()
        self.prototypes = []
        self.total_sleeps = 0
        self.total_episodes_consolidated = 0
        self.last_result = None
