"""
v5.3-1: Self-State Memory - (regime_id × z) 기반 메모리 재조직

핵심 변경:
1. 저장 우선순위: z가 "불확실/후회" → memory_gate ↑
2. 회상 억제: uncertainty_spike OR regime_switch → recall_weight ↓
3. cold-start 가속: 새 레짐에서 자동으로 탐색/저장 강화

기존 RegimeLTMStore를 래핑하여 z 기반 조절 추가.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from genesis.memory import RegimeLTMStore, Episode, RecallResult


@dataclass
class SelfStateMemoryConfig:
    """Self-State Memory 설정"""
    # z별 memory_gate modifier
    z_gate_modifiers: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # stable: 보통
        1: 1.5,   # uncertain: 강화 (정보 가치 높음)
        2: 1.4,   # regret: 강화 (실수 기록 중요)
        3: 0.6,   # fatigue: 약화 (노이즈 저장 방지)
    })

    # z별 recall_weight modifier
    z_recall_modifiers: Dict[int, float] = field(default_factory=lambda: {
        0: 1.2,   # stable: 기억 활용 강화
        1: 0.5,   # uncertain: 기억 억제 (새 학습에 집중)
        2: 0.7,   # regret: 약간 억제 (과거 오판 의존 줄임)
        3: 1.0,   # fatigue: 보통 (습관 활용)
    })

    # Suppression 조건
    uncertainty_spike_threshold: float = 0.6  # 이 이상이면 spike
    regime_grace_steps: int = 10  # 레짐 전환 후 억제 유지 스텝

    # Suppression 강도
    spike_suppression: float = 0.3  # uncertainty spike 시 recall 배율
    grace_suppression: float = 0.4  # grace period 시 recall 배율


@dataclass
class SelfStateMemoryState:
    """현재 상태 추적"""
    last_z: int = 0
    last_regime: int = 0
    steps_since_regime_switch: int = 100  # 시작 시 충분히 큼

    # Spike 감지
    uncertainty_history: List[float] = field(default_factory=list)
    uncertainty_spike_active: bool = False

    # Cold-start 추적
    regime_episode_counts: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})


class SelfStateMemory:
    """
    v5.3-1 Self-State Memory

    RegimeLTMStore를 래핑하여:
    - z 기반 저장 우선순위 조절
    - z 기반 회상 억제
    - cold-start 자동 가속

    Usage:
        memory = SelfStateMemory()

        # 저장 시
        result = memory.store(episode, base_memory_gate, regime_id, z)

        # 회상 시
        recall = memory.recall(obs, context, uncertainty, regime, z)
    """

    def __init__(
        self,
        config: Optional[SelfStateMemoryConfig] = None,
        n_regimes: int = 2,
        max_episodes_per_regime: int = 500,
        **ltm_kwargs
    ):
        self.config = config or SelfStateMemoryConfig()
        self.state = SelfStateMemoryState()

        # 기존 RegimeLTMStore 래핑
        self.ltm = RegimeLTMStore(
            n_regimes=n_regimes,
            max_episodes_per_regime=max_episodes_per_regime,
            **ltm_kwargs
        )

        # 통계
        self._store_count = 0
        self._recall_count = 0
        self._suppression_events = 0
        self._cold_start_boosts = 0

    def compute_storage_priority(
        self,
        base_memory_gate: float,
        z: int,
        regime_id: int
    ) -> Tuple[float, Dict]:
        """
        z 기반 저장 우선순위 계산

        Returns:
            (adjusted_memory_gate, info_dict)
        """
        cfg = self.config
        state = self.state

        # 1. z modifier
        z_mod = cfg.z_gate_modifiers.get(z, 1.0)

        # 2. Cold-start boost
        # 새 레짐에서 에피소드 수가 적으면 저장 강화
        regime_count = state.regime_episode_counts.get(regime_id, 0)
        cold_start_boost = 1.0
        if regime_count < 20:  # 20개 미만이면 cold-start
            cold_start_boost = 1.0 + (20 - regime_count) * 0.05  # 최대 2.0배
            self._cold_start_boosts += 1

        # 3. 최종 gate
        adjusted_gate = base_memory_gate * z_mod * cold_start_boost
        adjusted_gate = min(1.0, adjusted_gate)  # cap

        info = {
            'base_gate': base_memory_gate,
            'z_modifier': z_mod,
            'cold_start_boost': cold_start_boost,
            'adjusted_gate': adjusted_gate,
            'regime_episode_count': regime_count,
        }

        return adjusted_gate, info

    def compute_recall_suppression(
        self,
        z: int,
        current_uncertainty: float,
        regime_id: int
    ) -> Tuple[float, Dict]:
        """
        z 기반 회상 억제 계산

        Returns:
            (recall_weight_modifier, info_dict)
        """
        cfg = self.config
        state = self.state

        # 1. z modifier
        z_mod = cfg.z_recall_modifiers.get(z, 1.0)

        # 2. Uncertainty spike check
        state.uncertainty_history.append(current_uncertainty)
        if len(state.uncertainty_history) > 10:
            state.uncertainty_history.pop(0)

        # Spike: 현재가 최근 평균보다 2배 이상 높음
        recent_avg = np.mean(state.uncertainty_history[:-1]) if len(state.uncertainty_history) > 1 else 0.3
        spike_ratio = current_uncertainty / (recent_avg + 0.1)
        is_spike = current_uncertainty > cfg.uncertainty_spike_threshold or spike_ratio > 2.0

        spike_mod = cfg.spike_suppression if is_spike else 1.0
        if is_spike:
            state.uncertainty_spike_active = True
            self._suppression_events += 1
        else:
            state.uncertainty_spike_active = False

        # 3. Regime grace period check
        grace_mod = 1.0
        if state.steps_since_regime_switch < cfg.regime_grace_steps:
            grace_mod = cfg.grace_suppression + (
                (1 - cfg.grace_suppression) *
                (state.steps_since_regime_switch / cfg.regime_grace_steps)
            )
            self._suppression_events += 1

        # 4. 최종 modifier
        final_mod = z_mod * spike_mod * grace_mod
        final_mod = max(0.1, min(1.5, final_mod))  # clamp

        info = {
            'z_modifier': z_mod,
            'uncertainty': current_uncertainty,
            'is_spike': is_spike,
            'spike_modifier': spike_mod,
            'grace_modifier': grace_mod,
            'steps_since_switch': state.steps_since_regime_switch,
            'final_modifier': final_mod,
        }

        return final_mod, info

    def store(
        self,
        episode: Episode,
        base_memory_gate: float,
        regime_id: int,
        z: int
    ) -> Dict:
        """
        z 기반 우선순위로 에피소드 저장

        Args:
            episode: 저장할 에피소드
            base_memory_gate: 기본 memory_gate (surprise/uncertainty 기반)
            regime_id: 현재 레짐
            z: 현재 self-state

        Returns:
            저장 결과 + 우선순위 정보
        """
        self._store_count += 1
        state = self.state

        # 우선순위 계산
        adjusted_gate, priority_info = self.compute_storage_priority(
            base_memory_gate, z, regime_id
        )

        # RegimeLTMStore에 저장
        result = self.ltm.store(episode, adjusted_gate, regime_id)

        # 상태 업데이트
        if result['stored'] or result['merged']:
            state.regime_episode_counts[regime_id] = state.regime_episode_counts.get(regime_id, 0) + 1

        state.last_z = z
        state.last_regime = regime_id

        # 결과에 우선순위 정보 추가
        result['priority_info'] = priority_info
        result['z'] = z

        return result

    def recall(
        self,
        current_obs: np.ndarray,
        current_context_id: int,
        current_uncertainty: float,
        current_regime: int,
        z: int
    ) -> Tuple[RecallResult, Dict]:
        """
        z 기반 억제를 적용한 회상

        Args:
            current_obs: 현재 관측
            current_context_id: 현재 context
            current_uncertainty: 현재 불확실성
            current_regime: 현재 레짐
            z: 현재 self-state

        Returns:
            (RecallResult, suppression_info)
        """
        self._recall_count += 1
        state = self.state

        # 레짐 전환 감지 (억제 계산 전에!)
        if current_regime != state.last_regime:
            state.steps_since_regime_switch = 0
            state.last_regime = current_regime
        else:
            state.steps_since_regime_switch += 1

        # 억제 계산 (레짐 전환 후)
        recall_mod, suppression_info = self.compute_recall_suppression(
            z, current_uncertainty, current_regime
        )

        state.last_z = z

        # RegimeLTMStore에서 회상 (modifier 적용)
        result = self.ltm.recall(
            current_obs,
            current_context_id,
            current_uncertainty,
            current_regime,
            recall_weight_modifier=recall_mod
        )

        return result, suppression_info

    def get_stats(self) -> Dict:
        """통계 정보"""
        ltm_stats = self.ltm.get_stats()

        return {
            **ltm_stats,
            'self_state_memory': {
                'store_count': self._store_count,
                'recall_count': self._recall_count,
                'suppression_events': self._suppression_events,
                'cold_start_boosts': self._cold_start_boosts,
                'regime_episode_counts': dict(self.state.regime_episode_counts),
                'uncertainty_spike_active': self.state.uncertainty_spike_active,
                'steps_since_regime_switch': self.state.steps_since_regime_switch,
            }
        }

    def get_cold_start_status(self, regime_id: int) -> Dict:
        """
        특정 레짐의 cold-start 상태

        Returns:
            {
                'is_cold': bool,
                'episode_count': int,
                'fill_ratio': float,
                'boost_factor': float
            }
        """
        count = self.state.regime_episode_counts.get(regime_id, 0)
        is_cold = count < 20
        fill_ratio = min(1.0, count / 20)
        boost = 1.0 + (20 - count) * 0.05 if is_cold else 1.0

        return {
            'is_cold': is_cold,
            'episode_count': count,
            'fill_ratio': fill_ratio,
            'boost_factor': boost,
        }

    def reset(self):
        """리셋"""
        self.ltm.reset()
        self.state = SelfStateMemoryState()
        self._store_count = 0
        self._recall_count = 0
        self._suppression_events = 0
        self._cold_start_boosts = 0
