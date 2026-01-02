"""
v5.4 Gate Specification - 정본 (Canonical)

이 파일은 게이트 기준과 테스트 환경의 단일 진실원천(SSOT).
절대 임의 변경 금지. 새 기준이 필요하면 버전을 추가할 것.

=== CHANGELOG ===
v1.0: Initial G2 gate spec
v1.1: v5.4 parameters + robustness spec 추가
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List


# =============================================================================
# G2@v1 Gate Specification
# =============================================================================

@dataclass(frozen=True)
class G2GateSpec:
    """G2 Gate 정본 기준 (v1)"""
    version: str = "v1"

    # G2a: Adaptation Speed
    g2a_baseline_recovery_steps: int = 30
    g2a_ratio_threshold: float = 1.2  # ratio <= 1.2 to pass
    g2a_recovery_window: int = 5  # 연속 N스텝 확인
    g2a_recovery_std_multiplier: float = 1.2  # std <= baseline * 1.2

    # G2b: Shock Stability
    g2b_peak_std_ratio_threshold: float = 3.0
    g2b_regret_spike_rate_threshold: float = 0.3

    # G2c: Energy Efficiency
    g2c_retention_threshold: float = 0.70  # >= 70%
    g2c_movement_penalty_threshold: float = 0.3
    g2c_movement_penalty_factor: float = 0.8


G2_SPEC_V1 = G2GateSpec()


# =============================================================================
# v5.4 Parameter Spec
# =============================================================================

@dataclass(frozen=True)
class V54ParameterSpec:
    """v5.4 튜닝된 파라미터 세트 (정본)"""
    version: str = "v5.4"

    # Self-model
    evidence_temperature: float = 2.1
    conflict_persistence_threshold: int = 10
    conflict_boost: float = 0.15
    midrange_boost_center: float = 0.55
    midrange_boost_width: float = 0.15
    midrange_boost_strength: float = 1.2
    transition_inertia: float = 0.85
    switch_confidence_threshold: float = 0.55

    # Interaction Gating
    z1_act_floor: float = 0.80  # z=1에서 act 하한
    z1_act_decay_rate: float = 0.03
    z3_act_floor: float = 0.20
    z3_act_decay_rate: float = 0.10
    fatigue_streak_threshold: int = 8
    uncertainty_streak_threshold: int = 5


V54_PARAMS = V54ParameterSpec()


# =============================================================================
# Scenario Fingerprint
# =============================================================================

@dataclass(frozen=True)
class ScenarioSpec:
    """테스트 시나리오 환경 스펙"""
    name: str

    # World
    pre_drift_steps: int
    shock_steps: int
    adapt_steps: int

    # Environment
    pre_efficiency: float
    pre_uncertainty: float
    pre_transition_std: float

    shock_efficiency: float
    shock_uncertainty: float
    shock_transition_std: float

    adapt_efficiency_start: float
    adapt_efficiency_end: float
    adapt_recovery_steps: int  # 완전 회복까지 스텝

    # Seeds
    seed_list: tuple

    def fingerprint(self) -> str:
        """환경 스펙의 고유 해시"""
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:12]


# 정본 시나리오들
SCENARIO_LONG_RUN = ScenarioSpec(
    name="long_run_2000",
    pre_drift_steps=100,
    shock_steps=30,
    adapt_steps=1870,  # total 2000
    pre_efficiency=0.6,
    pre_uncertainty=0.2,
    pre_transition_std=0.15,
    shock_efficiency=0.3,
    shock_uncertainty=0.6,
    shock_transition_std=0.5,
    adapt_efficiency_start=0.3,
    adapt_efficiency_end=0.6,
    adapt_recovery_steps=100,
    seed_list=(42,),
)

SCENARIO_MULTI_SEED = ScenarioSpec(
    name="multi_seed_400",
    pre_drift_steps=100,
    shock_steps=30,
    adapt_steps=270,  # total 400
    pre_efficiency=0.6,
    pre_uncertainty=0.2,
    pre_transition_std=0.15,
    shock_efficiency=0.3,
    shock_uncertainty=0.6,
    shock_transition_std=0.5,
    adapt_efficiency_start=0.3,
    adapt_efficiency_end=0.6,
    adapt_recovery_steps=100,
    seed_list=tuple(range(1, 21)),
)

SCENARIO_Z_CONFLICT = ScenarioSpec(
    name="z_conflict_400",
    pre_drift_steps=100,
    shock_steps=30,
    adapt_steps=270,
    pre_efficiency=0.6,
    pre_uncertainty=0.2,
    pre_transition_std=0.15,
    shock_efficiency=0.2,  # conflict: 낮은 효율
    shock_uncertainty=0.6,  # conflict: 높은 불확실성
    shock_transition_std=0.5,
    adapt_efficiency_start=0.2,
    adapt_efficiency_end=0.2,  # conflict 유지
    adapt_recovery_steps=9999,  # 회복 안 함
    seed_list=(42,),
)


# =============================================================================
# Food Source Decomposition Metrics
# =============================================================================

@dataclass
class FoodDecomposition:
    """
    "+57의 원인"을 측정하는 표준 분해 지표

    각 지표는 ON - OFF 차이값
    """
    # Phase별 food 차이
    delta_food_pre: int = 0
    delta_food_shock: int = 0
    delta_food_adapt: int = 0
    delta_food_total: int = 0

    # Coupling 적분 차이 (ON - OFF)
    delta_learn_integral: float = 0.0  # learn_coupling 적분
    delta_act_integral: float = 0.0    # act_coupling 적분

    # 성능 지표
    time_to_recovery_off: int = 0
    time_to_recovery_on: int = 0
    efficiency_retention_off: float = 0.0
    efficiency_retention_on: float = 0.0

    # 해석
    def primary_source(self) -> str:
        """주요 원인 판정"""
        # shock/adapt에서 food 이득이 있고, learn_integral이 높으면 (A)
        shock_adapt_gain = self.delta_food_shock + self.delta_food_adapt

        if shock_adapt_gain > 0 and self.delta_learn_integral > 0:
            return "(A) drift 적응에서 학습 증폭"
        elif self.delta_food_pre > 0:
            return "(B) 평시 thrashing 감소"
        elif shock_adapt_gain > 0 and self.delta_act_integral < 0:
            return "(C) conflict 구간 효율 상승"
        else:
            return "불명확 (추가 분석 필요)"

    def to_dict(self) -> Dict:
        return {
            'delta_food': {
                'pre': self.delta_food_pre,
                'shock': self.delta_food_shock,
                'adapt': self.delta_food_adapt,
                'total': self.delta_food_total,
            },
            'delta_integrals': {
                'learn': round(self.delta_learn_integral, 2),
                'act': round(self.delta_act_integral, 2),
            },
            'performance': {
                'time_to_recovery_off': self.time_to_recovery_off,
                'time_to_recovery_on': self.time_to_recovery_on,
                'efficiency_retention_off': round(self.efficiency_retention_off, 3),
                'efficiency_retention_on': round(self.efficiency_retention_on, 3),
            },
            'primary_source': self.primary_source(),
        }


# =============================================================================
# v5.5 Regime Change Score Spec
# =============================================================================

@dataclass(frozen=True)
class RegimeChangeScoreSpec:
    """
    v5.5 regime_change_score 정본 스펙

    핵심: "drift가 왔을 때만 올라가고, 적응되면 자연스럽게 0으로 내려오는" 연속 신호
    """
    version: str = "v5.5"

    # === 입력 신호 가중치 ===
    w_error: float = 0.4      # intended_outcome_error (가장 중요)
    w_volatility: float = 0.3  # volatility
    w_std: float = 0.2         # std_spike_ratio
    w_regret: float = 0.1      # regret_spike_rate (보조)

    # === 정규화 경계 (long_run f6e8... 기준) ===
    # x0: 정상 상태 상한 ("이 정도는 평소에도 흔함")
    # x1: 명백한 변화 상한 ("이 정도면 레짐 전환급")
    error_x0: float = 0.15
    error_x1: float = 0.50
    volatility_x0: float = 0.20
    volatility_x1: float = 0.60
    std_x0: float = 1.2   # std/baseline ratio
    std_x1: float = 3.0
    regret_x0: float = 0.10
    regret_x1: float = 0.30

    # === 합성 ===
    saturation_k: float = 2.5  # 1 - exp(-k * raw)

    # === 시간 안정화 ===
    ema_alpha: float = 0.15  # 깜빡임 방지

    # === 연속 제어 계수 ===
    # learn_mult = 1 + L * score
    learn_boost_L: float = 0.5

    # prior_mult = 1 - P * score
    prior_suppress_P: float = 0.6

    # recall_mult = 1 - R * score
    recall_suppress_R: float = 0.8

    # store_mult = 1 + M * score
    store_boost_M: float = 0.5

    # act는 건드리지 않음 (z-정본 바운더리 유지)


V55_REGIME_SCORE_SPEC = RegimeChangeScoreSpec()


# =============================================================================
# v5.8 Canonical Spec - Z-Only Mode (direct 제거)
# =============================================================================

@dataclass(frozen=True)
class V58CanonicalSpec:
    """
    v5.8 정본 스펙: Z-Only 모드

    핵심 선언:
    - direct_path_enabled = False (정본 상태)
    - regime_score → z evidence → z state → 자원 배분 (유일한 경로)
    - "성능 떨어졌네? direct 다시 붙이자" 같은 퇴행 금지

    생물학적 해석:
    - regime_score = 각성/놀람 신호 (노르아드레날린/아세틸콜린 계열)
    - z state = 전역 모드 (cortical state)
    - 자원 배분 = 학습률/우선순위/회상/prior 강도

    검증 기준 (v5.7에서 증명됨):
    - Z-only retention >= 0.97 (hybrid 대비)
    - Z-only robustness >= 0.95 (multi-seed)
    """
    version: str = "v5.8"

    # === 핵심 선언: direct 경로 제거 ===
    direct_path_enabled: bool = False  # 정본 상태
    direct_path_reason: str = "v5.7에서 Z-only retention 0.97+ 달성, direct 불필요"

    # === Z-only 모드 경로 ===
    # regime_score → z evidence → z state → resource modifiers
    z_evidence_weight: float = 0.15  # score가 z=1 evidence에 기여하는 비중
    z_evidence_threshold: float = 0.35  # score가 이 이상일 때 효과 적용

    # === 성능 기준 ===
    z_only_retention_threshold: float = 0.97  # hybrid 대비
    z_only_robustness_threshold: float = 0.95  # multi-seed 평균

    # === 필수 회귀 테스트 ===
    # z가 안 깨어나는 상황 방지 (v5.4 z=0 관성 문제 재발 방지)
    z_conflict_test_required: bool = True
    z_conflict_z1_min_rate: float = 0.10  # z=1이 최소 10%는 발화해야 함

    # === 기여도 분해 포맷 (1급 시민) ===
    # 왜 좋아졌는지 설명 가능해야 함
    decomposition_required: bool = True
    decomposition_metrics: tuple = (
        'learn_integral_shock_adapt',  # shock/adapt에서 learn_coupling 적분
        'act_reduction_stable',         # 안정 구간에서 act 감소량
        'recall_suppress_drift',        # drift 구간에서 recall 억제량
        'prior_suppress_drift',         # drift 구간에서 prior 억제량
    )


V58_CANONICAL_SPEC = V58CanonicalSpec()


@dataclass
class ZOnlyContribution:
    """
    v5.8 Z-Only 모드 기여도 분해

    "왜 좋아졌는지"를 설명하는 1급 시민 데이터 구조
    """
    # Phase별 learn_coupling 적분
    learn_integral_pre: float = 0.0
    learn_integral_shock: float = 0.0
    learn_integral_adapt: float = 0.0

    # Phase별 act 평균
    act_avg_pre: float = 1.0
    act_avg_shock: float = 1.0
    act_avg_adapt: float = 1.0

    # Drift 구간 억제량
    recall_suppress_total: float = 0.0
    prior_suppress_total: float = 0.0

    # z 발화 통계
    z1_rate_shock: float = 0.0
    z1_rate_adapt: float = 0.0
    z3_rate_shock: float = 0.0

    def primary_contribution(self) -> str:
        """주요 기여 원인 판정"""
        shock_adapt_learn = self.learn_integral_shock + self.learn_integral_adapt

        if shock_adapt_learn > self.learn_integral_pre * 1.2:
            return "(A) Drift 구간 학습 증폭"
        elif self.act_avg_shock < self.act_avg_pre * 0.9:
            return "(B) Shock 구간 행동 억제"
        elif self.recall_suppress_total > 5.0:
            return "(C) 과거 기억 억제로 새 학습 집중"
        else:
            return "(D) 복합 기여 (분해 불가)"

    def to_dict(self) -> Dict:
        return {
            'learn_integral': {
                'pre': round(self.learn_integral_pre, 2),
                'shock': round(self.learn_integral_shock, 2),
                'adapt': round(self.learn_integral_adapt, 2),
            },
            'act_avg': {
                'pre': round(self.act_avg_pre, 3),
                'shock': round(self.act_avg_shock, 3),
                'adapt': round(self.act_avg_adapt, 3),
            },
            'suppress_total': {
                'recall': round(self.recall_suppress_total, 2),
                'prior': round(self.prior_suppress_total, 2),
            },
            'z_rates': {
                'z1_shock': round(self.z1_rate_shock, 3),
                'z1_adapt': round(self.z1_rate_adapt, 3),
                'z3_shock': round(self.z3_rate_shock, 3),
            },
            'primary_contribution': self.primary_contribution(),
        }


# =============================================================================
# Robustness Spec
# =============================================================================

@dataclass(frozen=True)
class RobustnessSpec:
    """Robustness 테스트 기준 (v5.4)"""
    version: str = "v5.4"

    # Multi-seed
    multi_seed_retention_threshold: float = 0.95
    multi_seed_wins_threshold: float = 0.50  # 50% 이상

    # Multi-drift
    multi_drift_retention_threshold: float = 0.90
    multi_drift_efficiency_majority: float = 0.50  # 과반수

    # Long-run
    long_run_retention_threshold: float = 0.95
    long_run_actions_saved_min: int = 0  # > 0

    # Z-conflict
    z_conflict_z_fires: bool = True  # z=0 < 100%
    z_conflict_z_meaningful_threshold: float = 0.10  # z1+z3 >= 10%
    z_conflict_not_degraded_threshold: float = 0.90


ROBUSTNESS_SPEC_V54 = RobustnessSpec()


# =============================================================================
# G2 Baseline Rules (측정 정합성 정본)
# =============================================================================

@dataclass(frozen=True)
class G2BaselineRules:
    """
    G2 Gate 측정의 baseline 계산 규칙 (정본)

    핵심 원칙:
    - "테스트 코드가 PASS/FAIL을 재정의하면 안 됨"
    - 테스트는 "환경을 스펙과 같게 만들고, 스펙 판정을 그대로 출력"만

    baseline 정의:
    - G2a ratio: ON의 recovery_steps / baseline_recovery_steps(=30)
    - G2b peak_std: shock window에서 max(transition_std) / pre_std
    - G2c retention: adapt_efficiency / pre_efficiency

    ON/OFF 비교 규칙:
    - 같은 fingerprint 시나리오에서 실행
    - OFF run의 metrics를 baseline으로 채택
    - ON run은 그 baseline 대비 ratio만 계산
    - PASS/FAIL 판정은 G2GateTracker/g2_gate.py만 사용
    """
    version: str = "v1"

    # Baseline 계산 기준
    baseline_source: str = "OFF_run"  # baseline은 OFF run에서 측정
    baseline_recovery_steps: int = 30  # G2a ratio 분모

    # Phase 경계 (시나리오 fingerprint와 일치해야 함)
    pre_drift_steps: int = 100
    shock_window_steps: int = 30  # shock 구간 길이
    adapt_start_step: int = 130   # pre + shock

    # 측정 일관성 규칙
    require_same_fingerprint: bool = True  # ON/OFF는 같은 시나리오
    require_same_seed: bool = True         # 단일 비교는 같은 seed
    g2_tracker_only: bool = True           # PASS/FAIL은 tracker만

    # 환경 불일치 감지
    max_pre_std_deviation: float = 0.05  # pre_std가 spec과 5% 이상 다르면 경고
    max_phase_boundary_drift: int = 5    # phase 경계가 5 steps 이상 다르면 경고


G2_BASELINE_RULES = G2BaselineRules()


def check_environment_consistency(
    actual_pre_std: float,
    actual_pre_steps: int,
    scenario: ScenarioSpec,
) -> tuple:
    """
    테스트 환경이 시나리오 스펙과 일치하는지 확인

    Returns:
        (is_consistent: bool, warnings: List[str])
    """
    rules = G2_BASELINE_RULES
    warnings = []

    # pre_std 일치 확인
    expected_pre_std = scenario.pre_transition_std
    std_deviation = abs(actual_pre_std - expected_pre_std) / expected_pre_std
    if std_deviation > rules.max_pre_std_deviation:
        warnings.append(
            f"pre_std mismatch: actual={actual_pre_std:.3f}, "
            f"expected={expected_pre_std:.3f} (deviation={std_deviation:.1%})"
        )

    # phase boundary 일치 확인
    boundary_drift = abs(actual_pre_steps - scenario.pre_drift_steps)
    if boundary_drift > rules.max_phase_boundary_drift:
        warnings.append(
            f"phase boundary mismatch: actual_pre={actual_pre_steps}, "
            f"expected={scenario.pre_drift_steps}"
        )

    is_consistent = len(warnings) == 0
    return is_consistent, warnings


# =============================================================================
# Utility
# =============================================================================

def print_spec_summary():
    """정본 스펙 요약 출력"""
    print("=" * 60)
    print("  CANONICAL GATE SPECIFICATION")
    print("=" * 60)

    print("\n[G2@v1]")
    print(f"  G2a: ratio <= {G2_SPEC_V1.g2a_ratio_threshold} (baseline={G2_SPEC_V1.g2a_baseline_recovery_steps} steps)")
    print(f"  G2b: peak_std < {G2_SPEC_V1.g2b_peak_std_ratio_threshold}, regret < {G2_SPEC_V1.g2b_regret_spike_rate_threshold}")
    print(f"  G2c: retention >= {G2_SPEC_V1.g2c_retention_threshold:.0%}")

    print("\n[v5.4 Parameters]")
    print(f"  evidence_temperature: {V54_PARAMS.evidence_temperature}")
    print(f"  conflict_boost: {V54_PARAMS.conflict_boost}")
    print(f"  z1_act_floor: {V54_PARAMS.z1_act_floor}")
    print(f"  z3_act_floor: {V54_PARAMS.z3_act_floor}")

    print("\n[Scenarios]")
    print(f"  long_run: {SCENARIO_LONG_RUN.fingerprint()}")
    print(f"  multi_seed: {SCENARIO_MULTI_SEED.fingerprint()}")
    print(f"  z_conflict: {SCENARIO_Z_CONFLICT.fingerprint()}")

    print("\n[Robustness v5.4]")
    print(f"  multi_seed retention >= {ROBUSTNESS_SPEC_V54.multi_seed_retention_threshold}")
    print(f"  multi_drift retention >= {ROBUSTNESS_SPEC_V54.multi_drift_retention_threshold}")
    print(f"  long_run retention >= {ROBUSTNESS_SPEC_V54.long_run_retention_threshold}")

    print("=" * 60)


if __name__ == "__main__":
    print_spec_summary()
