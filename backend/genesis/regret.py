"""
Counterfactual + Regret Module - v4.4

핵심 원칙:
    후회(regret)는 "감정 변수"가 아니라,
    선택한 행동이 대안 행동보다 얼마나 더 큰 G를 초래했는지에 대한 '사후 EFE 차이'다.

구성 요소:
    1. Counterfactual 계산: G_pred vs G_post
    2. Regret 신호: regret_pred, regret_real
    3. Regret 연결: memory_gate, lr_boost, THINK 보정

연결 방식 (FEP스럽게):
    - 정책을 직접 바꾸는 보상학습 X
    - 모델/정밀도/기억 게이트 쪽으로 연결 O
    → "후회가 '학습/추론 자원 배분'을 바꾸는 구조"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class CounterfactualResult:
    """단일 step의 counterfactual 분석 결과"""
    step: int
    chosen_action: int

    # 선택 시점 예측 (이미 계산됨)
    G_pred: Dict[int, float]  # {action: predicted G}

    # 사후 재평가 (관측 후)
    G_post: Dict[int, float]  # {action: counterfactual G}

    # Regret 신호
    regret_pred: float  # G_pred(a*) - min_a G_pred(a) - 판단 오류
    regret_real: float  # G_post(a*) - min_a G_post(a) - 실제 결과 기준

    # 진단용
    best_action_was: int  # 사후적으로 최선이었던 행동
    choice_was_optimal: bool  # 선택이 최선이었는가


@dataclass
class RegretState:
    """Regret 추적 상태"""
    # 누적 regret (EMA)
    cumulative_regret: float = 0.0
    regret_ema_alpha: float = 0.1

    # 최근 regret 히스토리
    recent_regret: List[float] = field(default_factory=list)
    history_size: int = 50

    # 통계
    total_counterfactuals: int = 0
    optimal_choices: int = 0
    suboptimal_choices: int = 0

    # Regret spike 감지
    regret_baseline: float = 0.0
    regret_spike_threshold: float = 2.0  # baseline의 2배 이상이면 spike

    def update(self, regret_real: float, was_optimal: bool):
        """Regret 상태 업데이트"""
        # EMA 업데이트
        self.cumulative_regret = (
            (1 - self.regret_ema_alpha) * self.cumulative_regret +
            self.regret_ema_alpha * regret_real
        )

        # 히스토리 업데이트
        self.recent_regret.append(regret_real)
        if len(self.recent_regret) > self.history_size:
            self.recent_regret.pop(0)

        # 통계
        self.total_counterfactuals += 1
        if was_optimal:
            self.optimal_choices += 1
        else:
            self.suboptimal_choices += 1

        # Baseline 업데이트 (느리게)
        if len(self.recent_regret) >= 10:
            self.regret_baseline = np.mean(self.recent_regret[-20:])

    def is_regret_spike(self, regret: float) -> bool:
        """현재 regret가 spike인지"""
        if self.regret_baseline < 0.01:
            return regret > 0.5  # baseline 없으면 절대값 기준
        return regret > self.regret_baseline * self.regret_spike_threshold

    def get_optimality_ratio(self) -> float:
        """최적 선택 비율"""
        if self.total_counterfactuals == 0:
            return 1.0
        return self.optimal_choices / self.total_counterfactuals

    def get_regret_z(self, regret: float) -> float:
        """
        Regret의 z-score 계산 (최근 분포 대비)

        Returns:
            z-score: 0 = 평균, +1 = 1 표준편차 위, -1 = 1 표준편차 아래
        """
        if len(self.recent_regret) < 5:
            return 0.0  # 데이터 부족

        mean = np.mean(self.recent_regret)
        std = np.std(self.recent_regret)

        if std < 1e-6:
            return 0.0  # 분산 없음

        return (regret - mean) / std

    def get_normalized_regret(self, regret: float, G_best: float) -> float:
        """
        Regret 정규화 (G_best 대비 비율)

        Returns:
            normalized: regret / (|G_best| + eps)
        """
        return regret / (abs(G_best) + 0.01)


class CounterfactualEngine:
    """
    Counterfactual 계산 엔진

    매 step에서:
    1. 선택 시점의 G_pred 저장 (이미 action_selection에서 계산됨)
    2. 관측 후 G_post 계산 (전이 모델 기반 반사실 추론)
    3. Regret 신호 계산
    """

    def __init__(self, action_selector, n_actions: int = 5):
        self.action_selector = action_selector
        self.n_actions = n_actions

        # 상태
        self.regret_state = RegretState()
        self.enabled = False

        # 최근 counterfactual 결과 (디버깅용)
        self._last_cf_result: Optional[CounterfactualResult] = None

        # Step 카운터
        self._step_counter = 0

    def enable(self):
        """Counterfactual + Regret 활성화"""
        self.enabled = True
        self.regret_state = RegretState()
        self._step_counter = 0

    def disable(self):
        """비활성화"""
        self.enabled = False

    def compute_counterfactual(
        self,
        chosen_action: int,
        G_pred: Dict[int, float],
        obs_before: np.ndarray,
        obs_after: np.ndarray
    ) -> CounterfactualResult:
        """
        Counterfactual 계산

        Args:
            chosen_action: 실제로 선택한 행동
            G_pred: 선택 시점의 예측 G (action -> G value)
            obs_before: 행동 전 관측
            obs_after: 행동 후 관측 (실제 결과)

        Returns:
            CounterfactualResult with regret signals
        """
        self._step_counter += 1

        # 1. G_post 계산: 각 대안 행동에 대해 "했으면 어땠을까"
        G_post = {}

        for action in range(self.n_actions):
            if action == chosen_action:
                # 실제로 한 행동: 실제 관측 기반 G 계산
                G_post[action] = self._compute_realized_G(action, obs_after)
            else:
                # 대안 행동: 전이 모델 기반 반사실 추론
                G_post[action] = self._compute_counterfactual_G(action, obs_before)

        # 2. Regret 계산
        G_pred_chosen = G_pred.get(chosen_action, 0.0)
        G_post_chosen = G_post.get(chosen_action, 0.0)

        min_G_pred = min(G_pred.values()) if G_pred else 0.0
        min_G_post = min(G_post.values()) if G_post else 0.0

        # regret_pred: 선택 시점 판단 오류 (내가 잘못 판단했나?)
        regret_pred = max(0.0, G_pred_chosen - min_G_pred)

        # regret_real: 실제 결과 기준 후회 (세상이 바뀌었나?)
        regret_real = max(0.0, G_post_chosen - min_G_post)

        # 3. 최적 행동 분석
        best_action = min(G_post.keys(), key=lambda a: G_post[a])
        choice_was_optimal = (chosen_action == best_action)

        # 4. 상태 업데이트
        self.regret_state.update(regret_real, choice_was_optimal)

        result = CounterfactualResult(
            step=self._step_counter,
            chosen_action=chosen_action,
            G_pred=G_pred,
            G_post=G_post,
            regret_pred=regret_pred,
            regret_real=regret_real,
            best_action_was=best_action,
            choice_was_optimal=choice_was_optimal
        )

        self._last_cf_result = result
        return result

    def _compute_realized_G(self, action: int, obs_after: np.ndarray) -> float:
        """
        실제 관측 기반 G 계산

        실제로 한 행동의 결과를 관측했으므로, 진짜 G를 계산
        """
        # 간단한 구현: 현재 상태의 "불편함" = risk proxy
        # energy low = high G, pain high = high G
        energy = obs_after[6] if len(obs_after) > 6 else 0.5
        pain = obs_after[7] if len(obs_after) > 7 else 0.0

        # G는 낮을수록 좋음
        # energy 낮으면 G 높음 (항상성 위반)
        # pain 높으면 G 높음
        energy_penalty = max(0, 0.6 - energy) * 2  # 0.6 이하면 penalty
        pain_penalty = pain * 3

        return energy_penalty + pain_penalty

    def _compute_counterfactual_G(self, action: int, obs_before: np.ndarray) -> float:
        """
        전이 모델 기반 반사실 G 계산

        "이 행동을 했으면 어떤 관측이 나왔을까?" 예측 후 G 계산
        """
        # 전이 모델에서 예측
        delta_mean = self.action_selector.transition_model['delta_mean'][action]
        predicted_obs = obs_before + delta_mean

        # Clip to valid range
        predicted_obs = np.clip(predicted_obs, 0, 1)

        return self._compute_realized_G(action, predicted_obs)

    def get_regret_modulation(self) -> Dict[str, float]:
        """
        Regret 기반 조절 파라미터 반환

        연결 대상:
        1. memory_gate: regret 큰 사건 = 저장 우선순위 ↑
        2. lr_boost: regret + surprise = 모델 재학습 필요
        3. think_benefit: regret 누적 = 메타인지 강화 합리적
        """
        if not self.enabled or self._last_cf_result is None:
            return {
                'memory_gate_boost': 0.0,
                'lr_boost_factor': 1.0,
                'think_benefit_boost': 0.0,
                'regret_real': 0.0,
                'regret_pred': 0.0,
                'is_spike': False,
            }

        regret = self._last_cf_result.regret_real
        regret_pred = self._last_cf_result.regret_pred
        is_spike = self.regret_state.is_regret_spike(regret)
        cumulative = self.regret_state.cumulative_regret

        # 1. Memory gate boost: regret 높으면 저장 가치 ↑
        # sigmoid로 0~0.3 범위
        memory_gate_boost = 0.3 * (1 / (1 + np.exp(-5 * (regret - 0.3))))

        # 2. LR boost: regret spike면 모델 재학습 필요
        # regret_pred가 높고 regret_real도 높으면 = 판단도 틀리고 결과도 나빴음
        if is_spike and regret_pred > 0.2:
            lr_boost_factor = 1.5  # 50% lr 증가
        elif regret > 0.3:
            lr_boost_factor = 1.2
        else:
            lr_boost_factor = 1.0

        # 3. THINK benefit boost: 누적 regret 높으면 메타인지 가치 ↑
        # cumulative regret가 높을수록 THINK의 expected_improvement 추정 ↑
        think_benefit_boost = min(0.2, cumulative * 0.5)

        # v4.4.1: 정규화 지표
        regret_z = self.regret_state.get_regret_z(regret)
        G_best = min(self._last_cf_result.G_post.values()) if self._last_cf_result.G_post else 0.0
        normalized_regret = self.regret_state.get_normalized_regret(regret, G_best)

        # v4.4.1: Spike 원인 분석
        spike_cause = None
        if is_spike:
            # regret_pred 높음 = 판단 오류, regret_real만 높음 = 모델/환경 변화
            if regret_pred > 0.2 and regret > 0.3:
                spike_cause = "judgment_error"  # 판단도 틀리고 결과도 나쁨
            elif regret > regret_pred * 2:
                spike_cause = "model_mismatch"  # 예측과 현실 괴리
            else:
                spike_cause = "environment_change"  # 환경 변화

        return {
            'memory_gate_boost': round(memory_gate_boost, 4),
            'lr_boost_factor': round(lr_boost_factor, 2),
            'think_benefit_boost': round(think_benefit_boost, 4),
            'regret_real': round(regret, 4),
            'regret_pred': round(regret_pred, 4),
            'is_spike': is_spike,
            'spike_cause': spike_cause,  # v4.4.1
            'regret_z': round(regret_z, 2),  # v4.4.1
            'normalized_regret': round(normalized_regret, 3),  # v4.4.1
            'cumulative_regret': round(cumulative, 4),
            'optimality_ratio': round(self.regret_state.get_optimality_ratio(), 3),
            'optimal_basis': 'G_post',  # v4.4.1: 최적 기준 명시
        }

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'enabled': self.enabled,
            'step_counter': self._step_counter,
            'regret_state': {
                'cumulative_regret': round(self.regret_state.cumulative_regret, 4),
                'regret_baseline': round(self.regret_state.regret_baseline, 4),
                'total_counterfactuals': self.regret_state.total_counterfactuals,
                'optimal_choices': self.regret_state.optimal_choices,
                'suboptimal_choices': self.regret_state.suboptimal_choices,
                'optimality_ratio': round(self.regret_state.get_optimality_ratio(), 3),
            },
            'last_result': {
                'step': self._last_cf_result.step if self._last_cf_result else None,
                'chosen_action': self._last_cf_result.chosen_action if self._last_cf_result else None,
                'regret_pred': round(self._last_cf_result.regret_pred, 4) if self._last_cf_result else None,
                'regret_real': round(self._last_cf_result.regret_real, 4) if self._last_cf_result else None,
                'choice_was_optimal': self._last_cf_result.choice_was_optimal if self._last_cf_result else None,
            } if self._last_cf_result else None,
            'modulation': self.get_regret_modulation(),
        }
