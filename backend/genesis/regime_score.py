"""
v5.5: Regime Change Score - 연속 신호 기반 자원 제어

핵심 철학:
- "drift가 왔을 때만 올라가고, 적응되면 자연스럽게 0으로 내려오는" 연속 신호
- action 선택은 건드리지 않고, 자원 배분만 조절 (v5.4 철학 유지)
- learn↑, prior↓, recall↓, store↑를 score에 비례해서 연속 제어

입력:
- volatility (transition_std)
- std_spike_ratio = std / std_baseline
- intended_outcome_error (혹은 prediction_error)
- regret_spike_rate

출력:
- regime_change_score (0~1): 레짐 전환 강도
- ResourceMultipliers: learn, prior, recall, store 배율

성공 조건:
1. G2@v1 PASS 유지
2. multi_seed retention >= 0.95 유지
3. FoodDecomposition primary_source가 A(drift 적응)로 판정
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque


@dataclass
class RegimeScoreConfig:
    """regime_change_score 설정 (v5.5 정본)"""
    # 입력 신호 가중치
    w_error: float = 0.4
    w_volatility: float = 0.3
    w_std: float = 0.2
    w_regret: float = 0.1

    # 정규화 경계
    error_x0: float = 0.15
    error_x1: float = 0.50
    volatility_x0: float = 0.20
    volatility_x1: float = 0.60
    std_x0: float = 1.2
    std_x1: float = 3.0
    regret_x0: float = 0.10
    regret_x1: float = 0.30

    # 합성
    saturation_k: float = 2.5

    # 시간 안정화
    ema_alpha: float = 0.15

    # 연속 제어 계수
    learn_boost_L: float = 0.5
    prior_suppress_P: float = 0.6
    recall_suppress_R: float = 0.8
    store_boost_M: float = 0.5


@dataclass
class RegimeScoreState:
    """regime_change_score 상태"""
    score_ema: float = 0.0
    raw_score: float = 0.0

    # Baseline 추적 (첫 N스텝에서 자동 설정)
    std_baseline: float = 0.15
    baseline_samples: deque = field(default_factory=lambda: deque(maxlen=50))
    baseline_set: bool = False

    # 히스토리 (디버깅용)
    score_history: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class ResourceMultipliers:
    """v5.5 자원 배율 (score 기반)"""
    learn_mult: float = 1.0   # learn_coupling에 곱해짐
    prior_mult: float = 1.0   # prior_strength에 곱해짐
    recall_mult: float = 1.0  # recall_weight에 곱해짐
    store_mult: float = 1.0   # storage_gate에 곱해짐

    def to_dict(self) -> Dict:
        return {
            'learn_mult': round(self.learn_mult, 3),
            'prior_mult': round(self.prior_mult, 3),
            'recall_mult': round(self.recall_mult, 3),
            'store_mult': round(self.store_mult, 3),
        }


class RegimeChangeScore:
    """
    v5.5 Regime Change Score - 연속 신호 기반 자원 제어

    Usage:
        scorer = RegimeChangeScore()

        # 매 스텝
        score, multipliers = scorer.update(
            volatility=transition_std,
            std_ratio=current_std / baseline_std,
            error=prediction_error,
            regret_rate=regret_spike_rate,
        )

        # 자원 배분 적용
        learn_coupling *= multipliers.learn_mult
        prior_strength *= multipliers.prior_mult
        recall_weight *= multipliers.recall_mult
        storage_gate *= multipliers.store_mult
    """

    def __init__(self, config: Optional[RegimeScoreConfig] = None):
        self.config = config or RegimeScoreConfig()
        self.state = RegimeScoreState()
        self._step_count = 0

    def _normalize(self, x: float, x0: float, x1: float) -> float:
        """
        baseline 대비 초과분만 정규화 (0~1)

        x < x0: 0 (정상 범위)
        x0 <= x <= x1: 선형 증가
        x > x1: 1 (포화)
        """
        if x <= x0:
            return 0.0
        if x >= x1:
            return 1.0
        return (x - x0) / (x1 - x0)

    def update(
        self,
        volatility: float,
        std_ratio: float = 1.0,
        error: float = 0.0,
        regret_rate: float = 0.0,
    ) -> tuple:
        """
        regime_change_score 계산 및 자원 배율 반환

        Args:
            volatility: transition_std (0~1)
            std_ratio: current_std / baseline_std (1.0 = 정상)
            error: intended_outcome_error 또는 prediction_error (0~1)
            regret_rate: regret_spike_rate (0~1)

        Returns:
            (score, ResourceMultipliers)
        """
        self._step_count += 1
        cfg = self.config
        state = self.state

        # === 1. Baseline 자동 설정 (처음 50스텝) ===
        if not state.baseline_set:
            state.baseline_samples.append(volatility)
            if len(state.baseline_samples) >= 30:
                state.std_baseline = np.mean(list(state.baseline_samples))
                state.baseline_set = True

        # === 2. 입력 신호 정규화 ===
        v_norm = self._normalize(volatility, cfg.volatility_x0, cfg.volatility_x1)
        s_norm = self._normalize(std_ratio, cfg.std_x0, cfg.std_x1)
        e_norm = self._normalize(error, cfg.error_x0, cfg.error_x1)
        r_norm = self._normalize(regret_rate, cfg.regret_x0, cfg.regret_x1)

        # === 3. 가중합 ===
        raw = (
            cfg.w_error * e_norm +
            cfg.w_volatility * v_norm +
            cfg.w_std * s_norm +
            cfg.w_regret * r_norm
        )

        # === 4. 포화 함수 (1 - exp(-k * raw)) ===
        # raw가 커질수록 1에 빠르게 포화
        score = 1 - np.exp(-cfg.saturation_k * raw)
        state.raw_score = raw

        # === 5. EMA 시간 안정화 (깜빡임 방지) ===
        state.score_ema = (
            cfg.ema_alpha * score +
            (1 - cfg.ema_alpha) * state.score_ema
        )

        # 히스토리 저장
        state.score_history.append(state.score_ema)

        # === 6. 자원 배율 계산 ===
        s = state.score_ema

        multipliers = ResourceMultipliers(
            learn_mult=1 + cfg.learn_boost_L * s,
            prior_mult=1 - cfg.prior_suppress_P * s,
            recall_mult=1 - cfg.recall_suppress_R * s,
            store_mult=1 + cfg.store_boost_M * s,
        )

        return state.score_ema, multipliers

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        state = self.state
        return {
            'enabled': True,
            'version': 'v5.5',
            'score_ema': round(state.score_ema, 4),
            'raw_score': round(state.raw_score, 4),
            'std_baseline': round(state.std_baseline, 4),
            'baseline_set': state.baseline_set,
            'step_count': self._step_count,
            'recent_scores': [round(s, 3) for s in list(state.score_history)[-10:]],
        }

    def get_score_stats(self) -> Dict:
        """스코어 통계 (분석용)"""
        if len(self.state.score_history) < 10:
            return {'avg': 0.0, 'max': 0.0, 'min': 0.0}

        scores = list(self.state.score_history)
        return {
            'avg': round(np.mean(scores), 4),
            'max': round(np.max(scores), 4),
            'min': round(np.min(scores), 4),
            'std': round(np.std(scores), 4),
        }

    def reset(self):
        """리셋"""
        self.state = RegimeScoreState()
        self._step_count = 0


# =============================================================================
# Integration with existing modules
# =============================================================================

def apply_regime_multipliers(
    base_modifiers: Dict,
    multipliers: ResourceMultipliers,
) -> Dict:
    """
    기존 modifier에 v5.5 배율 적용

    Args:
        base_modifiers: SelfModel/InteractionGating의 기본 modifier
        multipliers: RegimeChangeScore의 배율

    Returns:
        조정된 modifier dict
    """
    result = base_modifiers.copy()

    # learn_coupling / learning_rate
    if 'learn_coupling' in result:
        result['learn_coupling'] *= multipliers.learn_mult
    if 'learning_rate' in result:
        result['learning_rate'] *= multipliers.learn_mult
    if 'learning_rate_boost' in result:
        result['learning_rate_boost'] *= multipliers.learn_mult

    # prior_strength
    if 'prior_strength' in result:
        result['prior_strength'] *= multipliers.prior_mult

    # recall_weight
    if 'recall_weight' in result:
        result['recall_weight'] *= multipliers.recall_mult

    # storage/consolidation
    if 'consolidation_boost' in result:
        result['consolidation_boost'] *= multipliers.store_mult

    return result
