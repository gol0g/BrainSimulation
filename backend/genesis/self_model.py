"""
v5.6 Self-model: Internal Latent State for Resource Allocation

핵심 철학:
- 라벨/태그/감정명 금지 → latent state z ∈ {0..M-1}
- action 선택 금지 → 자원 배분 파라미터만 조절
- 게이트로만 생존 → G2 ON/OFF 비교로 검증

v5.6 핵심 변경:
- regime_change_score를 evidence 입력으로 흡수
- "세계 변화 신호가 곧바로 자원 배분을 건드리는 대신,
   먼저 self-state를 바꾸고, 그 self-state가 자원을 바꾸게 만든다"
- score↑ → z=1(탐색/변화감지) evidence 가산
- score↑일 때 → z=3(피로) evidence 감산 (drift를 피로로 오인 방지)

Inputs:
- uncertainty (overall or 4-component)
- regret_spike_rate (최근 후회 빈도)
- transition_std / volatility (환경 변화 감지)
- energy_efficiency (탐색 품질)
- think_rate (최근 THINK 선택률)
- movement_ratio (이동 비율)
- regime_change_score (v5.6): 0~1 연속 변화 신호

Outputs (자원 배분만):
- think_budget_modifier: THINK action 가치 조절
- recall_weight_modifier: 기억 회상 가중치
- sleep_prob_modifier: 수면/통합 확률
- learning_rate_modifier: 학습률 부스트
- prior_strength_modifier: prior λ 강도

완료 조건:
1. 모드 전이 관성: z가 매 스텝 flip하지 않음
2. 자원-불확실성 정합: uncertainty↑ → 자동 자원 재배치
3. 피로/비효율 억제: 헛움직임 구간 감소
4. G2 유지 또는 개선
5. v5.6 추가: Mid-signal activation (score 0.6에서 z=1 발화)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class SelfModelConfig:
    """Self-model 설정"""
    M: int = 4  # number of latent states

    # Transition dynamics
    transition_inertia: float = 0.85  # Q(z) update smoothing
    min_steps_before_switch: int = 5  # hysteresis
    switch_confidence_threshold: float = 0.55  # Q(z) > threshold to switch

    # Signal smoothing
    signal_ema_alpha: float = 0.3  # EMA for input signals

    # Evidence weights (튜닝 가능)
    uncertainty_weight: float = 1.0
    regret_weight: float = 0.8
    efficiency_weight: float = 0.7
    volatility_weight: float = 0.6

    # v5.3-2 B': Evidence scaling (극단 신호에서 더 강한 반응)
    evidence_temperature: float = 2.1  # 2.3 → 2.1 (과민 방지)

    # v5.4-fix: 중간 신호 구간 민감도 개선
    # A) Conflict persistence (누적 메커니즘)
    conflict_persistence_threshold: int = 10  # 8 → 10: 누적 느리게
    conflict_boost: float = 0.15  # 0.3 → 0.15: 가산 강도 줄임

    # B) Nonlinear sensitivity (중간 구간 기울기)
    midrange_boost_center: float = 0.55  # 이 값 근처에서 민감도 증가
    midrange_boost_width: float = 0.15   # 0.20 → 0.15: 범위 좁힘
    midrange_boost_strength: float = 1.2 # 1.5 → 1.2: 강도 줄임

    # v5.6: Regime change score → evidence 연결
    # score↑ → z=1 evidence 가산 (변화 감지 → 탐색)
    # score↑ → z=3 evidence 감산 (drift를 피로로 오인 방지)
    # v5.6-tune2: 더 보수적으로 (83% → 목표 40-60%)
    regime_score_z1_weight: float = 0.15  # z=1에 가산할 비중 (0.25 → 0.15)
    regime_score_z3_suppress: float = 0.10  # z=3에서 감산할 비중 (0.15 → 0.10)
    regime_score_threshold: float = 0.35  # 이 이상이면 효과 적용 (0.25 → 0.35)


@dataclass
class SelfState:
    """현재 self-model 상태"""
    z: int = 0  # current latent state
    Q_z: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    steps_in_current: int = 0
    switch_count: int = 0
    last_switch_step: int = 0

    # Smoothed input signals
    uncertainty_ema: float = 0.3
    regret_ema: float = 0.0
    efficiency_ema: float = 0.5
    volatility_ema: float = 0.1

    # v5.4-fix: Conflict persistence tracking
    conflict_streak: int = 0  # 연속 conflict 스텝 수
    in_conflict: bool = False  # 현재 conflict 상태인지

    # v5.6: Regime change score tracking
    regime_score_ema: float = 0.0  # EMA smoothed score


@dataclass
class ResourceModifiers:
    """자원 배분 modifier (Self-model의 유일한 출력)"""
    think_budget: float = 1.0      # THINK action 가치 배율
    recall_weight: float = 1.0    # 기억 회상 가중치 배율
    sleep_prob: float = 1.0       # 수면/통합 확률 배율
    learning_rate: float = 1.0    # 학습률 배율
    prior_strength: float = 1.0   # prior λ 배율

    def to_dict(self) -> Dict:
        return {
            'think_budget': self.think_budget,
            'recall_weight': self.recall_weight,
            'sleep_prob': self.sleep_prob,
            'learning_rate': self.learning_rate,
            'prior_strength': self.prior_strength,
        }


class SelfModel:
    """
    Self-model: 내부 잠재 상태 기반 자원 배분

    핵심: action을 고르지 않고, gain/weight/budget만 조절

    Usage:
        model = SelfModel()
        signals = {
            'uncertainty': 0.5,
            'regret_spike_rate': 0.1,
            'energy_efficiency': 0.8,
            'volatility': 0.2,
            'think_rate': 0.1,
            'movement_ratio': 0.7,
        }
        modifiers, info = model.update(signals)
        # modifiers.think_budget, modifiers.recall_weight, etc.
    """

    # z별 자원 배분 프로파일 (디버깅용 이름은 주석으로만)
    # z0: 안정/자동 (stable/automatic)
    # z1: 탐색/불확실 (exploration/uncertain)
    # z2: 후회/오판 (regret/misjudgment)
    # z3: 피로/비효율 (fatigue/low-efficiency)

    RESOURCE_PROFILES = {
        0: ResourceModifiers(
            think_budget=0.7,      # THINK 줄임
            recall_weight=1.3,     # 기억 강화
            sleep_prob=1.4,        # 통합 증가
            learning_rate=0.8,     # 학습률 낮춤
            prior_strength=1.2,    # prior 강화
        ),
        1: ResourceModifiers(
            think_budget=1.4,      # deliberation 증가
            recall_weight=0.5,     # 기억 억제 (새 학습에 집중)
            sleep_prob=0.6,        # 통합 줄임
            learning_rate=1.5,     # 학습률 높임
            prior_strength=0.6,    # prior 약화 (열린 마음)
        ),
        2: ResourceModifiers(
            think_budget=1.6,      # 반성적 사고 증가
            recall_weight=0.8,     # 약간 억제
            sleep_prob=1.1,        # 약간 증가
            learning_rate=1.3,     # 학습률 높임
            prior_strength=0.8,    # prior 약간 약화
        ),
        3: ResourceModifiers(
            think_budget=0.6,      # 과도한 deliberation 줄임
            recall_weight=1.1,     # 기존 패턴 활용
            sleep_prob=1.6,        # 통합/휴식 강화
            learning_rate=0.6,     # 학습률 낮춤 (에너지 절약)
            prior_strength=1.4,    # prior 강화 (습관 모드)
        ),
    }

    def __init__(self, config: Optional[SelfModelConfig] = None):
        self.config = config or SelfModelConfig()
        self.state = SelfState(
            Q_z=np.ones(self.config.M) / self.config.M
        )
        self.state.Q_z[0] = 0.7  # 초기 상태는 z0 (안정) 쪽으로 편향
        self.state.Q_z /= self.state.Q_z.sum()

        self._step_count = 0
        self._transition_history: List[Tuple[int, int, int]] = []  # (step, old_z, new_z)

    def update(self, signals: Dict[str, float]) -> Tuple[ResourceModifiers, Dict]:
        """
        신호 기반으로 self-state 업데이트 및 자원 modifier 반환

        Args:
            signals: {
                'uncertainty': float (0-1),
                'regret_spike_rate': float (0-1),
                'energy_efficiency': float (0+),
                'volatility': float (0-1),
                'think_rate': float (0-1),
                'movement_ratio': float (0-1),
            }

        Returns:
            (ResourceModifiers, info_dict)
        """
        self._step_count += 1
        cfg = self.config
        state = self.state

        # === 1. Signal smoothing (EMA) ===
        uncertainty = signals.get('uncertainty', 0.3)
        regret = signals.get('regret_spike_rate', 0.0)
        efficiency = signals.get('energy_efficiency', 0.5)
        volatility = signals.get('volatility', 0.1)

        alpha = cfg.signal_ema_alpha
        state.uncertainty_ema = alpha * uncertainty + (1 - alpha) * state.uncertainty_ema
        state.regret_ema = alpha * regret + (1 - alpha) * state.regret_ema
        state.efficiency_ema = alpha * efficiency + (1 - alpha) * state.efficiency_ema
        state.volatility_ema = alpha * volatility + (1 - alpha) * state.volatility_ema

        # v5.6: regime_change_score EMA
        regime_score = signals.get('regime_change_score', 0.0)
        state.regime_score_ema = alpha * regime_score + (1 - alpha) * state.regime_score_ema

        # === 2. Compute evidence for each state ===
        evidence = self._compute_evidence(signals)

        # === 3. Update Q(z) with inertia ===
        state.Q_z = (
            cfg.transition_inertia * state.Q_z +
            (1 - cfg.transition_inertia) * evidence
        )
        state.Q_z = np.clip(state.Q_z, 0.01, 0.99)
        state.Q_z /= state.Q_z.sum()

        # === 4. State transition with hysteresis ===
        state.steps_in_current += 1
        old_z = state.z
        new_z = int(np.argmax(state.Q_z))

        switched = False
        if new_z != old_z:
            # 전환 조건: 충분한 시간 + 충분한 confidence
            if (state.steps_in_current >= cfg.min_steps_before_switch and
                state.Q_z[new_z] > cfg.switch_confidence_threshold):
                state.z = new_z
                state.steps_in_current = 0
                state.switch_count += 1
                state.last_switch_step = self._step_count
                switched = True
                self._transition_history.append((self._step_count, old_z, new_z))

        # === 5. Get resource modifiers for current state ===
        # Interpolate based on Q(z) for smooth transitions
        modifiers = self._interpolate_modifiers()

        # === 6. Build info dict ===
        info = {
            'z': state.z,
            'Q_z': state.Q_z.tolist(),
            'switched': switched,
            'steps_in_current': state.steps_in_current,
            'switch_count': state.switch_count,
            'evidence': evidence.tolist(),
            'smoothed_signals': {
                'uncertainty': state.uncertainty_ema,
                'regret': state.regret_ema,
                'efficiency': state.efficiency_ema,
                'volatility': state.volatility_ema,
            }
        }

        return modifiers, info

    def _compute_evidence(self, signals: Dict[str, float]) -> np.ndarray:
        """
        신호로부터 각 상태에 대한 evidence 계산

        규칙 (휴리스틱이 아닌 연속 함수로):
        - z0 (안정): 낮은 uncertainty, 낮은 regret, 낮은 volatility
        - z1 (탐색): 높은 uncertainty, 낮은 regret
        - z2 (후회): 높은 regret
        - z3 (피로): 높은 movement + 낮은 efficiency

        v5.4-fix 추가:
        - A) Conflict persistence: 중간 신호가 지속되면 evidence 가산
        - B) Nonlinear sensitivity: 중간 구간(0.55±0.2)에서 민감도 증가
        """
        cfg = self.config
        M = cfg.M
        state = self.state

        # Get smoothed values
        u = state.uncertainty_ema
        r = state.regret_ema
        e = min(1.0, state.efficiency_ema)  # cap at 1.0 for calculation
        v = state.volatility_ema
        m = signals.get('movement_ratio', 0.5)

        # === A) Conflict persistence: 중간 신호 누적 감지 ===
        # Conflict 조건: 불확실성 중간 이상 + 효율 낮음
        is_conflict = (u > 0.45 and e < 0.4) or (v > 0.35 and e < 0.4)

        if is_conflict:
            state.conflict_streak += 1
            state.in_conflict = True
        else:
            state.conflict_streak = 0
            state.in_conflict = False

        # Conflict boost (누적되면 비안정 상태 evidence 증가)
        conflict_factor = 0.0
        if state.conflict_streak >= cfg.conflict_persistence_threshold:
            # 누적 스텝에 비례해서 boost (최대 2배)
            excess = state.conflict_streak - cfg.conflict_persistence_threshold
            conflict_factor = min(2.0, 1.0 + excess * 0.1) * cfg.conflict_boost

        # === B) Nonlinear sensitivity: 중간 구간 민감도 증가 ===
        def midrange_amplify(x: float) -> float:
            """중간 구간(0.55±0.2)에서 기울기 증가"""
            center = cfg.midrange_boost_center
            width = cfg.midrange_boost_width
            strength = cfg.midrange_boost_strength

            # 중간 구간에서만 boost
            dist = abs(x - center)
            if dist < width:
                # 가우시안 형태의 boost (중심에서 최대)
                boost = (1 - dist / width) * (strength - 1)
                return x * (1 + boost)
            return x

        # 중간 구간 민감도 적용
        u_amp = midrange_amplify(u)
        v_amp = midrange_amplify(v)
        inefficiency_amp = midrange_amplify(max(0, 1 - e))

        # === Evidence 계산 ===
        evidence = np.zeros(M)

        # z0: stable/automatic
        # 낮은 불확실성, 낮은 후회, 낮은 변동성
        evidence[0] = (1 - u_amp) * (1 - r) * (1 - v_amp)

        # z1: exploration/uncertain
        # 높은 불확실성, 중간 이하 후회
        evidence[1] = u_amp * (1 - r * 0.5) * (1 - e * 0.3)

        # z2: regret/misjudgment
        # 높은 후회
        evidence[2] = r * (1 + u_amp * 0.3)

        # z3: fatigue/low-efficiency
        # 높은 움직임 + 낮은 효율
        evidence[3] = m * inefficiency_amp * (1 - r * 0.5)

        # === A) Conflict boost 적용 ===
        if conflict_factor > 0:
            # Conflict 상황에서 z=1(탐색)과 z=3(피로)에 가산
            # z=1: 불확실 → 탐색
            # z=3: 비효율 → 피로
            if u > e:
                # 불확실성이 더 높으면 z=1에 더 많이
                evidence[1] += conflict_factor * 0.7
                evidence[3] += conflict_factor * 0.3
            else:
                # 비효율이 더 높으면 z=3에 더 많이
                evidence[1] += conflict_factor * 0.3
                evidence[3] += conflict_factor * 0.7

        # === v5.6: Regime change score → evidence 연결 ===
        # "세계 변화 신호가 곧바로 자원 배분을 건드리는 대신,
        #  먼저 self-state를 바꾸고, 그 self-state가 자원을 바꾸게 만든다"
        score = state.regime_score_ema

        if score > cfg.regime_score_threshold:
            # score가 threshold 이상이면 효과 적용
            # 효과 강도는 (score - threshold) / (1 - threshold)로 정규화
            effect = (score - cfg.regime_score_threshold) / (1 - cfg.regime_score_threshold)

            # z=1 (탐색/변화감지) evidence 가산
            # "세상이 바뀌었다" → "탐색/학습 모드로"
            evidence[1] += cfg.regime_score_z1_weight * effect

            # z=3 (피로) evidence 감산
            # drift shock을 피로로 오인해서 act를 과도하게 닫는 걸 방지
            evidence[3] = max(0.05, evidence[3] - cfg.regime_score_z3_suppress * effect)

        # === Temperature scaling (기존 B') ===
        T = cfg.evidence_temperature
        if T != 1.0:
            evidence = np.clip(evidence, 0.01, None)
            log_evidence = np.log(evidence)
            scaled = np.exp(log_evidence * T)
            evidence = scaled

        # Normalize with floor
        evidence = np.clip(evidence, 0.05, 1.0)
        evidence /= evidence.sum()

        return evidence

    def _interpolate_modifiers(self) -> ResourceModifiers:
        """
        Q(z)를 기반으로 modifier 보간 (부드러운 전환)
        """
        Q = self.state.Q_z
        M = self.config.M

        # Weighted average of all profiles
        think = sum(Q[z] * self.RESOURCE_PROFILES[z].think_budget for z in range(M))
        recall = sum(Q[z] * self.RESOURCE_PROFILES[z].recall_weight for z in range(M))
        sleep = sum(Q[z] * self.RESOURCE_PROFILES[z].sleep_prob for z in range(M))
        lr = sum(Q[z] * self.RESOURCE_PROFILES[z].learning_rate for z in range(M))
        prior = sum(Q[z] * self.RESOURCE_PROFILES[z].prior_strength for z in range(M))

        return ResourceModifiers(
            think_budget=round(think, 3),
            recall_weight=round(recall, 3),
            sleep_prob=round(sleep, 3),
            learning_rate=round(lr, 3),
            prior_strength=round(prior, 3),
        )

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        state = self.state
        return {
            'enabled': True,
            'z': state.z,
            'Q_z': state.Q_z.tolist(),
            'steps_in_current': state.steps_in_current,
            'switch_count': state.switch_count,
            'last_switch_step': state.last_switch_step,
            'total_steps': self._step_count,
            'current_modifiers': self._interpolate_modifiers().to_dict(),
            'smoothed_signals': {
                'uncertainty': round(state.uncertainty_ema, 3),
                'regret': round(state.regret_ema, 3),
                'efficiency': round(state.efficiency_ema, 3),
                'volatility': round(state.volatility_ema, 3),
                'regime_score': round(state.regime_score_ema, 3),  # v5.6
            },
            'transition_history': self._transition_history[-10:],  # last 10
        }

    def get_mode_stability(self) -> Dict:
        """
        모드 전이 관성 분석 (완료 조건 1)

        Returns:
            {
                'avg_steps_per_mode': float,
                'flip_flop_count': int (연속 전환 횟수),
                'stability_score': float (높을수록 안정)
            }
        """
        if self._step_count < 10:
            return {'avg_steps_per_mode': 0, 'flip_flop_count': 0, 'stability_score': 1.0}

        avg_steps = self._step_count / max(1, self.state.switch_count + 1)

        # Flip-flop: 연속으로 빠르게 전환한 횟수
        flip_flops = 0
        history = self._transition_history
        for i in range(1, len(history)):
            step_diff = history[i][0] - history[i-1][0]
            if step_diff < 10:  # 10스텝 미만에 다시 전환
                flip_flops += 1

        # Stability score: 평균 체류 시간 / 목표 (30스텝)
        stability = min(1.0, avg_steps / 30)
        stability *= max(0.5, 1 - flip_flops * 0.1)

        return {
            'avg_steps_per_mode': round(avg_steps, 1),
            'flip_flop_count': flip_flops,
            'stability_score': round(stability, 3),
        }

    def reset(self):
        """리셋"""
        self.state = SelfState(
            Q_z=np.ones(self.config.M) / self.config.M
        )
        self.state.Q_z[0] = 0.7
        self.state.Q_z /= self.state.Q_z.sum()
        self.state.conflict_streak = 0
        self.state.in_conflict = False
        self.state.regime_score_ema = 0.0  # v5.6
        self._step_count = 0
        self._transition_history.clear()


def get_mode_label(z: int) -> str:
    """
    디버깅/UI용 라벨 (모델 내부에서는 사용하지 않음)
    """
    labels = {
        0: "stable",      # 안정/자동
        1: "exploring",   # 탐색/불확실
        2: "reflecting",  # 후회/반성
        3: "fatigued",    # 피로/비효율
    }
    return labels.get(z, f"z{z}")
