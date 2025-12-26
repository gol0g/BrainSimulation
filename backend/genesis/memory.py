"""
Genesis Brain v4.0 - Long-Term Memory (LTM)

핵심 철학:
"기억 = 미래 F/G를 줄이는 압축 모델"

불확실성과 서프라이즈를 줄이는 데 도움이 되는 경험을,
압축된 형태로 보존하고 재사용한다.

주요 원칙:
1. 기억은 "행동을 지시"하지 않고 "G를 조정"
2. 불확실할 때 기억에 더 의존 (uncertainty ↑ → memory weight ↑)
3. 저장은 memory_gate로 조절 (surprise/uncertainty 높을 때 저장)
4. 중복 억제로 과적합 방지
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict


@dataclass
class Episode:
    """
    단일 경험 에피소드.

    저장 단위를 하나로 통일 (semantic/skill 구분 없음).
    """
    # 시간/맥락
    t: int                              # 타임스텝
    context_id: int                     # dominant context (0~K-1)
    context_confidence: float           # Q(c) 최대값 (0~1)

    # 관측 요약 (8차원)
    obs_summary: np.ndarray             # 관측 상태 요약

    # 행동과 결과
    action: int                         # 선택한 행동 (0~5)

    # 내부 상태 변화
    delta_energy: float                 # Δenergy
    delta_pain: float                   # Δpain

    # 불확실성 변화
    delta_uncertainty: float            # Δglobal_uncertainty (전→후)
    delta_surprise: float               # Δprediction_error (전→후)

    # 결과 점수 (내부 기준)
    outcome_score: float                # G 감소량 또는 보상 신호

    # 메타 정보
    store_count: int = 1                # 유사 에피소드 병합 횟수
    last_recall_t: int = 0              # 마지막 회상 시점
    recall_count: int = 0               # 총 회상 횟수

    def to_dict(self) -> Dict:
        return {
            't': self.t,
            'context_id': self.context_id,
            'context_confidence': self.context_confidence,
            'obs_summary': self.obs_summary.tolist() if isinstance(self.obs_summary, np.ndarray) else self.obs_summary,
            'action': self.action,
            'delta_energy': self.delta_energy,
            'delta_pain': self.delta_pain,
            'delta_uncertainty': self.delta_uncertainty,
            'delta_surprise': self.delta_surprise,
            'outcome_score': self.outcome_score,
            'store_count': self.store_count,
            'last_recall_t': self.last_recall_t,
            'recall_count': self.recall_count,
        }

    @staticmethod
    def from_dict(d: Dict) -> 'Episode':
        return Episode(
            t=d['t'],
            context_id=d['context_id'],
            context_confidence=d['context_confidence'],
            obs_summary=np.array(d['obs_summary']),
            action=d['action'],
            delta_energy=d['delta_energy'],
            delta_pain=d['delta_pain'],
            delta_uncertainty=d['delta_uncertainty'],
            delta_surprise=d['delta_surprise'],
            outcome_score=d['outcome_score'],
            store_count=d.get('store_count', 1),
            last_recall_t=d.get('last_recall_t', 0),
            recall_count=d.get('recall_count', 0),
        )


@dataclass
class RecallResult:
    """Recall 결과: 행동별 memory bias"""
    memory_bias: np.ndarray      # shape (n_actions,): G에 더할 bias
    recalled_episodes: List[int]  # 회상된 에피소드 인덱스들
    similarity_scores: List[float]  # 유사도 점수들
    recall_weight: float         # 최종 recall 가중치 (uncertainty 기반)

    def to_dict(self) -> Dict:
        return {
            'memory_bias': self.memory_bias.tolist(),
            'recalled_count': len(self.recalled_episodes),
            'avg_similarity': float(np.mean(self.similarity_scores)) if self.similarity_scores else 0.0,
            'recall_weight': self.recall_weight,
        }


class LTMStore:
    """
    Long-Term Memory 저장소.

    핵심 원리:
    - 저장: memory_gate > threshold일 때 확률적으로 저장
    - 회상: 현재 상태와 유사한 에피소드 검색 → G bias 계산
    - 압축: 유사 에피소드 병합으로 중복 억제
    """

    def __init__(self,
                 max_episodes: int = 1000,
                 store_threshold: float = 0.5,
                 store_sharpness: float = 5.0,
                 similarity_threshold: float = 0.95,
                 recall_top_k: int = 5,
                 n_actions: int = 6):
        """
        Args:
            max_episodes: 최대 저장 에피소드 수
            store_threshold: memory_gate 저장 임계값 (θ)
            store_sharpness: sigmoid 기울기 (k)
            similarity_threshold: 중복 억제 유사도 임계값
            recall_top_k: 회상 시 상위 k개 에피소드 사용
            n_actions: 행동 수 (기본 6: 0-4 물리 + 5 THINK)
        """
        self.max_episodes = max_episodes
        self.store_threshold = store_threshold
        self.store_sharpness = store_sharpness
        self.similarity_threshold = similarity_threshold
        self.recall_top_k = recall_top_k
        self.n_actions = n_actions

        # 저장소
        self.episodes: List[Episode] = []

        # 통계
        self.total_store_attempts = 0
        self.total_stored = 0
        self.total_merged = 0
        self.total_recalls = 0

    def compute_store_probability(self, memory_gate: float) -> float:
        """
        저장 확률 계산.

        store_prob = sigmoid((memory_gate - θ) * k)

        memory_gate가 threshold 이상이면 저장 확률 높음.
        """
        x = (memory_gate - self.store_threshold) * self.store_sharpness
        return 1.0 / (1.0 + np.exp(-x))

    def should_store(self, memory_gate: float) -> bool:
        """저장 여부 결정 (확률적)"""
        self.total_store_attempts += 1
        prob = self.compute_store_probability(memory_gate)
        return np.random.random() < prob

    def compute_similarity(self, obs1: np.ndarray, obs2: np.ndarray) -> float:
        """
        관측 유사도 계산 (cosine similarity).

        범위: -1 ~ 1 (1이면 동일)
        """
        norm1 = np.linalg.norm(obs1)
        norm2 = np.linalg.norm(obs2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(obs1, obs2) / (norm1 * norm2))

    def find_similar_episode(self,
                            obs_summary: np.ndarray,
                            action: int,
                            context_id: int) -> Optional[int]:
        """
        유사 에피소드 검색 (중복 억제용).

        같은 context, 같은 action에서 obs 유사도 > threshold면 중복.
        """
        for i, ep in enumerate(self.episodes):
            if ep.action != action:
                continue
            if ep.context_id != context_id:
                continue
            sim = self.compute_similarity(obs_summary, ep.obs_summary)
            if sim > self.similarity_threshold:
                return i
        return None

    def store(self, episode: Episode, memory_gate: float) -> Dict:
        """
        에피소드 저장 시도.

        1. memory_gate로 저장 확률 계산
        2. 확률적으로 저장 결정
        3. 유사 에피소드 있으면 병합
        4. 없으면 새로 추가

        Returns:
            결과 정보 dict
        """
        result = {
            'attempted': True,
            'stored': False,
            'merged': False,
            'store_prob': self.compute_store_probability(memory_gate),
            'memory_gate': memory_gate,
        }

        # 저장 결정
        if not self.should_store(memory_gate):
            return result

        # 유사 에피소드 검색
        similar_idx = self.find_similar_episode(
            episode.obs_summary,
            episode.action,
            episode.context_id
        )

        if similar_idx is not None:
            # 병합: 기존 에피소드 업데이트
            existing = self.episodes[similar_idx]

            # EMA로 outcome_score 업데이트
            n = existing.store_count
            alpha = 1.0 / (n + 1)
            existing.outcome_score = (1 - alpha) * existing.outcome_score + alpha * episode.outcome_score

            # obs_summary도 EMA
            existing.obs_summary = (1 - alpha) * existing.obs_summary + alpha * episode.obs_summary

            # delta 값들도 EMA
            existing.delta_energy = (1 - alpha) * existing.delta_energy + alpha * episode.delta_energy
            existing.delta_pain = (1 - alpha) * existing.delta_pain + alpha * episode.delta_pain
            existing.delta_uncertainty = (1 - alpha) * existing.delta_uncertainty + alpha * episode.delta_uncertainty
            existing.delta_surprise = (1 - alpha) * existing.delta_surprise + alpha * episode.delta_surprise

            existing.store_count += 1
            existing.t = episode.t  # 최신 시간으로 업데이트

            self.total_merged += 1
            result['merged'] = True
            result['merged_idx'] = similar_idx
            result['new_store_count'] = existing.store_count
        else:
            # 새 에피소드 추가
            if len(self.episodes) >= self.max_episodes:
                # 오래된/덜 사용된 에피소드 제거
                self._evict_one()

            self.episodes.append(episode)
            self.total_stored += 1
            result['stored'] = True
            result['episode_idx'] = len(self.episodes) - 1

        return result

    def _evict_one(self):
        """
        에피소드 하나 제거 (용량 초과 시).

        기준: recall_count가 가장 낮고 store_count도 낮은 것
        """
        if not self.episodes:
            return

        # 점수 계산: recall_count + 0.5 * store_count
        scores = [ep.recall_count + 0.5 * ep.store_count for ep in self.episodes]
        min_idx = int(np.argmin(scores))
        self.episodes.pop(min_idx)

    def recall(self,
              current_obs: np.ndarray,
              current_context_id: int,
              current_uncertainty: float) -> RecallResult:
        """
        현재 상태에서 관련 기억 회상 → G bias 계산.

        핵심 원리:
        - "행동을 지시"하지 않고 "G를 조정"
        - 유사 상황에서 좋았던 행동 → G ↓ (더 유리)
        - 유사 상황에서 나빴던 행동 → G ↑ (덜 유리)
        - uncertainty 높을수록 기억에 더 의존

        Args:
            current_obs: 현재 관측 (8차원)
            current_context_id: 현재 context
            current_uncertainty: 현재 global uncertainty (0~1)

        Returns:
            RecallResult with memory_bias for each action
        """
        self.total_recalls += 1

        # 기본 bias (0: 영향 없음)
        memory_bias = np.zeros(self.n_actions)
        recalled_episodes = []
        similarity_scores = []

        if not self.episodes:
            return RecallResult(
                memory_bias=memory_bias,
                recalled_episodes=[],
                similarity_scores=[],
                recall_weight=0.0
            )

        # 유사도 계산 (context 가중)
        similarities = []
        for i, ep in enumerate(self.episodes):
            # 기본 유사도: 관측 cosine
            obs_sim = self.compute_similarity(current_obs, ep.obs_summary)

            # context 일치 보너스
            context_bonus = 0.2 if ep.context_id == current_context_id else 0.0

            # 최종 유사도
            total_sim = obs_sim + context_bonus
            similarities.append((i, total_sim, ep))

        # 상위 k개 선택
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.recall_top_k]

        # 회상된 에피소드에서 action별 bias 계산
        # outcome_score > 0 → 좋은 결과 → G ↓ (bias 음수)
        # outcome_score < 0 → 나쁜 결과 → G ↑ (bias 양수)
        action_outcomes = defaultdict(list)

        for idx, sim, ep in top_k:
            if sim < 0.3:  # 유사도 너무 낮으면 무시
                continue

            recalled_episodes.append(idx)
            similarity_scores.append(sim)

            # 해당 에피소드의 행동에 outcome 기록
            # 유사도로 가중
            weighted_outcome = ep.outcome_score * sim
            action_outcomes[ep.action].append(weighted_outcome)

            # 회상 기록 업데이트
            ep.recall_count += 1
            ep.last_recall_t = ep.t  # 현재 시간으로 업데이트할 수 있지만, t를 모르므로 생략

        # 각 action에 대해 bias 계산
        for action, outcomes in action_outcomes.items():
            if outcomes:
                avg_outcome = np.mean(outcomes)
                # outcome이 양수면 좋은 결과 → bias 음수 (G 감소 → 더 선택됨)
                # outcome이 음수면 나쁜 결과 → bias 양수 (G 증가 → 덜 선택됨)
                memory_bias[action] = -avg_outcome * 0.5  # 스케일 조절

        # Recall weight: uncertainty가 높을수록 기억에 더 의존
        # u=0: weight=0.2 (기본 의존)
        # u=1: weight=1.0 (강한 의존)
        recall_weight = 0.2 + 0.8 * current_uncertainty

        # 최종 bias에 recall_weight 적용
        memory_bias = memory_bias * recall_weight

        return RecallResult(
            memory_bias=memory_bias,
            recalled_episodes=recalled_episodes,
            similarity_scores=similarity_scores,
            recall_weight=recall_weight
        )

    def get_stats(self) -> Dict:
        """통계 정보 반환"""
        return {
            'total_episodes': len(self.episodes),
            'max_episodes': self.max_episodes,
            'total_store_attempts': self.total_store_attempts,
            'total_stored': self.total_stored,
            'total_merged': self.total_merged,
            'total_recalls': self.total_recalls,
            'store_rate': self.total_stored / max(1, self.total_store_attempts),
            'merge_rate': self.total_merged / max(1, self.total_stored + self.total_merged),
        }

    def get_episodes_summary(self, limit: int = 10) -> List[Dict]:
        """최근 에피소드 요약"""
        recent = self.episodes[-limit:] if len(self.episodes) > limit else self.episodes
        return [ep.to_dict() for ep in recent]

    def reset(self):
        """저장소 초기화"""
        self.episodes = []
        self.total_store_attempts = 0
        self.total_stored = 0
        self.total_merged = 0
        self.total_recalls = 0


def compute_outcome_score(
    G_before: float,
    G_after: float,
    delta_energy: float,
    delta_pain: float
) -> float:
    """
    Outcome score 계산.

    내부 기준으로 "이 행동이 좋았는가" 판단.

    요소:
    1. G 감소: G_before - G_after (클수록 좋음)
    2. 에너지 증가: delta_energy (양수면 좋음)
    3. 고통 감소: -delta_pain (고통 줄면 좋음)

    Returns:
        양수: 좋은 결과, 음수: 나쁜 결과
    """
    # G 감소량 (정규화)
    g_improvement = (G_before - G_after) * 2.0  # 스케일 조절

    # 내부 상태 변화
    internal_improvement = delta_energy - delta_pain

    # 가중 합
    score = 0.6 * g_improvement + 0.4 * internal_improvement

    # 클램프
    return float(np.clip(score, -1.0, 1.0))
