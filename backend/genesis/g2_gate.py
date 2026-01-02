"""
v5.2 G2 Gate: Circuit-Driven Adaptation Test

G1과 차이점:
- Circuit이 메인 컨트롤러 (P2 모드)
- energy_efficiency 추가 (탐색 품질 지표)
- N1a/N1b 정렬도 함께 추적

=== G2@v1 CANONICAL GATE SPEC (정본) ===
절대 변경 금지. 새 기준이 필요하면 G2@v2로 버전 추가.

G2a (적응 속도):
  - metric: adaptation_speed_ratio = time_to_recovery / BASELINE_RECOVERY_STEPS
  - threshold: ratio <= 1.2
  - BASELINE_RECOVERY_STEPS = 30
  - recovery 정의: transition_std <= baseline_std * 1.2 가 5스텝 연속 유지

G2b (충격 안정성):
  - metric: peak_std_ratio, regret_spike_rate
  - threshold: peak_std_ratio < 3.0 AND regret_spike_rate < 0.3

G2c (에너지 효율):
  - metric: efficiency_retention = efficiency_adapt / efficiency_pre
  - threshold: retention >= 0.70 (70%)
  - efficiency = food_gained / energy_spent
  - 패널티: movement_ratio < 0.3이면 retention *= 0.8

핵심 질문: "Circuit 주행에서 drift 적응이 FEP와 비슷하거나 더 나은가?"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


@dataclass
class EnergyEfficiencyMetrics:
    """에너지 효율 추적 (탐색 품질 지표)"""
    total_energy_spent: float = 0.0  # 총 소비 에너지
    total_food_gained: int = 0  # 획득한 음식
    movement_count: int = 0  # 이동 횟수
    stay_count: int = 0  # STAY 횟수

    # Phase별 추적
    pre_drift_efficiency: float = 0.0
    shock_efficiency: float = 0.0  # drift 직후 20스텝
    adapt_efficiency: float = 0.0  # drift 후 20스텝 이후

    @property
    def overall_efficiency(self) -> float:
        """food_gained / energy_spent (높을수록 효율적)"""
        if self.total_energy_spent < 0.01:
            return 0.0
        return self.total_food_gained / self.total_energy_spent

    @property
    def movement_ratio(self) -> float:
        """이동 비율 (높으면 적극적 탐색)"""
        total = self.movement_count + self.stay_count
        if total == 0:
            return 0.0
        return self.movement_count / total


@dataclass
class G2StepLog:
    """G2 Gate용 스텝 로그"""
    step: int
    # Action info
    circuit_action: int
    fep_action: Optional[int]
    final_action: int
    agreed: bool
    disagreement_type: Optional[str]

    # State info
    energy: float
    danger_prox: float
    food_prox: float

    # Drift info
    drift_active: bool
    transition_std: float
    transition_error: float

    # Outcome
    ate_food: bool
    hit_danger: bool
    energy_spent: float

    # Regret
    regret_spike: bool

    # Circuit confidence
    circuit_margin: float


@dataclass
class G2GateResult:
    """
    G2 Gate 결과 - Circuit 주행 drift 적응

    G1과 공유하는 메트릭:
    - peak_std_ratio, time_to_recovery, food_rate_*, etc.

    G2 고유 메트릭:
    - energy_efficiency (탐색 품질)
    - circuit_agreement_rate (drift 중 FEP와 일치율)
    - danger_approach_during_drift (drift 중 위험 접근 횟수)
    """
    # === Meta ===
    drift_type: str
    drift_after: int
    total_steps: int

    # === G2a: Adaptation Speed ===
    time_to_recovery: int  # G 회복까지 스텝
    adaptation_speed_ratio: float  # vs baseline (1.0 = 동일, <1 = 더 빠름)

    # === G2b: Shock Stability ===
    std_auc_shock: float  # shock window에서 std 면적
    peak_std_ratio: float  # max(std) / pre_std
    regret_spike_rate: float  # drift 후 regret spike 빈도

    # === G2c: Energy Efficiency ===
    efficiency_pre: float  # pre-drift efficiency
    efficiency_shock: float  # shock phase efficiency
    efficiency_adapt: float  # adapt phase efficiency
    efficiency_retention: float  # adapt / pre (1.0 = 유지)
    movement_ratio: float  # 이동 비율 (탐색 적극성)

    # === Circuit-specific ===
    circuit_agreement_rate: float  # drift 중 FEP와 일치율
    danger_approach_count: int  # drift 중 위험 접근 횟수
    safety_maintained: bool  # danger_approach == 0

    # === Performance ===
    food_rate_pre: float
    food_rate_shock: float
    food_rate_adapt: float
    survival: bool  # post-drift food > 0

    # === Gates ===
    g2a_passed: bool  # adaptation speed
    g2b_passed: bool  # shock stability
    g2c_passed: bool  # energy efficiency
    overall_passed: bool

    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'meta': {
                'drift_type': self.drift_type,
                'drift_after': self.drift_after,
                'total_steps': self.total_steps,
            },
            'g2a_adaptation': {
                'time_to_recovery': self.time_to_recovery,
                'speed_ratio': self.adaptation_speed_ratio,
                'passed': self.g2a_passed,
            },
            'g2b_stability': {
                'std_auc_shock': self.std_auc_shock,
                'peak_std_ratio': self.peak_std_ratio,
                'regret_spike_rate': self.regret_spike_rate,
                'passed': self.g2b_passed,
            },
            'g2c_efficiency': {
                'efficiency_pre': self.efficiency_pre,
                'efficiency_shock': self.efficiency_shock,
                'efficiency_adapt': self.efficiency_adapt,
                'efficiency_retention': self.efficiency_retention,
                'movement_ratio': self.movement_ratio,
                'passed': self.g2c_passed,
            },
            'circuit_safety': {
                'agreement_rate': self.circuit_agreement_rate,
                'danger_approach_count': self.danger_approach_count,
                'safety_maintained': self.safety_maintained,
            },
            'performance': {
                'food_rate_pre': self.food_rate_pre,
                'food_rate_shock': self.food_rate_shock,
                'food_rate_adapt': self.food_rate_adapt,
                'survival': self.survival,
            },
            'overall_passed': self.overall_passed,
            'reasons': self.reasons,
        }


class G2GateTracker:
    """
    G2 Gate 추적기 - Circuit 주행 drift 적응 테스트

    사용:
    1. tracker = G2GateTracker(drift_after=100)
    2. 매 스텝: tracker.log_step(...)
    3. 결과: tracker.get_result()
    """

    # Baseline 값 (FEP 기준, 추후 측정으로 업데이트)
    BASELINE_RECOVERY_STEPS = 30
    BASELINE_EFFICIENCY = 0.5
    BASELINE_STD_AUC = 0.5

    def __init__(
        self,
        drift_after: int = 100,
        drift_type: str = "rotate",
        shock_window: int = 20,
    ):
        self.drift_after = drift_after
        self.drift_type = drift_type
        self.shock_window = shock_window

        self.logs: List[G2StepLog] = []
        self.step_count = 0
        self.drift_active = False

        # Energy tracking
        self.energy_metrics = EnergyEfficiencyMetrics()
        self._phase_energy = {'pre': [], 'shock': [], 'adapt': []}
        self._phase_food = {'pre': 0, 'shock': 0, 'adapt': 0}

    def log_step(
        self,
        circuit_action: int,
        fep_action: Optional[int],
        final_action: int,
        agreed: bool,
        disagreement_type: Optional[str],
        energy: float,
        danger_prox: float,
        food_prox: float,
        drift_active: bool,
        transition_std: float,
        transition_error: float,
        ate_food: bool,
        hit_danger: bool,
        energy_spent: float,
        regret_spike: bool,
        circuit_margin: float,
    ):
        """스텝 로깅"""
        self.step_count += 1
        self.drift_active = drift_active

        log = G2StepLog(
            step=self.step_count,
            circuit_action=circuit_action,
            fep_action=fep_action,
            final_action=final_action,
            agreed=agreed,
            disagreement_type=disagreement_type,
            energy=energy,
            danger_prox=danger_prox,
            food_prox=food_prox,
            drift_active=drift_active,
            transition_std=transition_std,
            transition_error=transition_error,
            ate_food=ate_food,
            hit_danger=hit_danger,
            energy_spent=energy_spent,
            regret_spike=regret_spike,
            circuit_margin=circuit_margin,
        )
        self.logs.append(log)

        # Energy tracking
        self.energy_metrics.total_energy_spent += energy_spent
        if ate_food:
            self.energy_metrics.total_food_gained += 1
        if final_action == 0:  # STAY
            self.energy_metrics.stay_count += 1
        else:
            self.energy_metrics.movement_count += 1

        # Phase tracking
        phase = self._get_phase()
        self._phase_energy[phase].append(energy_spent)
        if ate_food:
            self._phase_food[phase] += 1

    def _get_phase(self) -> str:
        """현재 phase 반환"""
        if not self.drift_active:
            return 'pre'
        drift_steps = self.step_count - self.drift_after
        if drift_steps <= self.shock_window:
            return 'shock'
        return 'adapt'

    def get_result(self) -> Optional[G2GateResult]:
        """G2 Gate 결과 계산"""
        if len(self.logs) < 50:
            return None

        pre_logs = [l for l in self.logs if not l.drift_active]
        post_logs = [l for l in self.logs if l.drift_active]

        if len(pre_logs) < 10 or len(post_logs) < 10:
            return None

        shock_logs = post_logs[:self.shock_window]
        adapt_logs = post_logs[self.shock_window:]

        # === Food rates ===
        food_rate_pre = sum(1 for l in pre_logs if l.ate_food) / len(pre_logs)
        food_rate_shock = sum(1 for l in shock_logs if l.ate_food) / len(shock_logs) if shock_logs else 0
        food_rate_adapt = sum(1 for l in adapt_logs if l.ate_food) / len(adapt_logs) if adapt_logs else 0

        # === G2a: Adaptation Speed ===
        # std baseline
        std_baseline = np.mean([l.transition_std for l in pre_logs[-10:]])

        # time_to_recovery: std가 baseline+0.1 이하로 돌아오는 시점
        time_to_recovery = len(post_logs)
        for i, l in enumerate(post_logs):
            if l.transition_std <= std_baseline * 1.2:
                # 연속 5스텝 확인
                if i + 5 <= len(post_logs):
                    window_std = np.mean([post_logs[j].transition_std for j in range(i, i+5)])
                    if window_std <= std_baseline * 1.2:
                        time_to_recovery = i
                        break

        adaptation_speed_ratio = time_to_recovery / self.BASELINE_RECOVERY_STEPS

        # === G2b: Shock Stability ===
        std_auc_shock = sum(
            max(0, l.transition_std - std_baseline)
            for l in shock_logs
        )

        all_stds = [l.transition_std for l in post_logs]
        peak_std = max(all_stds) if all_stds else std_baseline
        peak_std_ratio = peak_std / (std_baseline + 1e-6)

        regret_spikes = sum(1 for l in post_logs if l.regret_spike)
        regret_spike_rate = regret_spikes / len(post_logs)

        # === G2c: Energy Efficiency ===
        def calc_efficiency(logs):
            if not logs:
                return 0.0
            energy = sum(l.energy_spent for l in logs)
            food = sum(1 for l in logs if l.ate_food)
            return food / energy if energy > 0.01 else 0.0

        efficiency_pre = calc_efficiency(pre_logs)
        efficiency_shock = calc_efficiency(shock_logs)
        efficiency_adapt = calc_efficiency(adapt_logs)

        efficiency_retention = efficiency_adapt / efficiency_pre if efficiency_pre > 0.01 else 1.0

        movement_ratio = self.energy_metrics.movement_ratio

        # === Circuit Safety ===
        agreed_count = sum(1 for l in post_logs if l.agreed)
        circuit_agreement_rate = agreed_count / len(post_logs)

        danger_approach_count = sum(
            1 for l in post_logs
            if l.disagreement_type == 'danger_approach'
        )
        safety_maintained = danger_approach_count == 0

        # === Gate Judgments ===
        reasons = []

        # G2a: adaptation_speed < baseline × 1.2
        g2a_passed = adaptation_speed_ratio <= 1.2
        if g2a_passed:
            reasons.append(f"G2a PASS: recovery in {time_to_recovery} steps (ratio={adaptation_speed_ratio:.2f})")
        else:
            reasons.append(f"G2a FAIL: recovery too slow ({time_to_recovery} steps, ratio={adaptation_speed_ratio:.2f})")

        # G2b: shock stability (peak_std_ratio < 3.0, regret_spike_rate < 0.3)
        g2b_passed = peak_std_ratio < 3.0 and regret_spike_rate < 0.3
        if g2b_passed:
            reasons.append(f"G2b PASS: stable (peak={peak_std_ratio:.2f}, regret={regret_spike_rate:.1%})")
        else:
            reasons.append(f"G2b FAIL: unstable (peak={peak_std_ratio:.2f}, regret={regret_spike_rate:.1%})")

        # G2c: efficiency_retention >= 0.7 (유지 70% 이상)
        # 단, movement_ratio가 너무 낮으면(0.3 미만) 패널티
        effective_retention = efficiency_retention
        if movement_ratio < 0.3:
            effective_retention *= 0.8  # 움직임이 너무 적으면 페널티

        g2c_passed = effective_retention >= 0.7
        if g2c_passed:
            reasons.append(f"G2c PASS: efficient (retention={efficiency_retention:.1%}, move={movement_ratio:.1%})")
        else:
            reasons.append(f"G2c FAIL: inefficient (retention={efficiency_retention:.1%}, move={movement_ratio:.1%})")

        # Safety check
        if not safety_maintained:
            reasons.append(f"SAFETY WARN: {danger_approach_count} danger approaches during drift")

        survival = sum(1 for l in post_logs if l.ate_food) > 0
        overall_passed = g2a_passed and g2b_passed and g2c_passed and safety_maintained

        return G2GateResult(
            drift_type=self.drift_type,
            drift_after=self.drift_after,
            total_steps=self.step_count,
            # G2a
            time_to_recovery=time_to_recovery,
            adaptation_speed_ratio=round(adaptation_speed_ratio, 3),
            # G2b
            std_auc_shock=round(std_auc_shock, 4),
            peak_std_ratio=round(peak_std_ratio, 3),
            regret_spike_rate=round(regret_spike_rate, 4),
            # G2c
            efficiency_pre=round(efficiency_pre, 4),
            efficiency_shock=round(efficiency_shock, 4),
            efficiency_adapt=round(efficiency_adapt, 4),
            efficiency_retention=round(efficiency_retention, 3),
            movement_ratio=round(movement_ratio, 3),
            # Circuit
            circuit_agreement_rate=round(circuit_agreement_rate, 3),
            danger_approach_count=danger_approach_count,
            safety_maintained=safety_maintained,
            # Performance
            food_rate_pre=round(food_rate_pre, 3),
            food_rate_shock=round(food_rate_shock, 3),
            food_rate_adapt=round(food_rate_adapt, 3),
            survival=survival,
            # Gates
            g2a_passed=g2a_passed,
            g2b_passed=g2b_passed,
            g2c_passed=g2c_passed,
            overall_passed=overall_passed,
            reasons=reasons,
        )

    def reset(self):
        """리셋"""
        self.logs.clear()
        self.step_count = 0
        self.drift_active = False
        self.energy_metrics = EnergyEfficiencyMetrics()
        self._phase_energy = {'pre': [], 'shock': [], 'adapt': []}
        self._phase_food = {'pre': 0, 'shock': 0, 'adapt': 0}


def format_g2_result(result: G2GateResult) -> str:
    """G2 결과 포맷팅"""
    lines = [
        "=" * 50,
        "  G2 GATE RESULT (Circuit-Driven Adaptation)",
        "=" * 50,
        f"",
        f"Drift: {result.drift_type} (after step {result.drift_after})",
        f"Total steps: {result.total_steps}",
        f"",
        f"=== G2a: Adaptation Speed ===",
        f"  Time to recovery: {result.time_to_recovery} steps",
        f"  Speed ratio: {result.adaptation_speed_ratio:.2f}x (baseline=30)",
        f"  Status: {'PASS' if result.g2a_passed else 'FAIL'}",
        f"",
        f"=== G2b: Shock Stability ===",
        f"  Peak std ratio: {result.peak_std_ratio:.2f}x",
        f"  Std AUC (shock): {result.std_auc_shock:.4f}",
        f"  Regret spike rate: {result.regret_spike_rate:.1%}",
        f"  Status: {'PASS' if result.g2b_passed else 'FAIL'}",
        f"",
        f"=== G2c: Energy Efficiency ===",
        f"  Efficiency pre: {result.efficiency_pre:.4f}",
        f"  Efficiency adapt: {result.efficiency_adapt:.4f}",
        f"  Retention: {result.efficiency_retention:.1%}",
        f"  Movement ratio: {result.movement_ratio:.1%}",
        f"  Status: {'PASS' if result.g2c_passed else 'FAIL'}",
        f"",
        f"=== Circuit Safety ===",
        f"  Agreement rate: {result.circuit_agreement_rate:.1%}",
        f"  Danger approaches: {result.danger_approach_count}",
        f"  Safety: {'MAINTAINED' if result.safety_maintained else 'VIOLATED'}",
        f"",
        f"=== Overall ===",
        f"  Survival: {'YES' if result.survival else 'NO'}",
        f"  G2 Gate: {'PASS' if result.overall_passed else 'FAIL'}",
        "=" * 50,
    ]
    return "\n".join(lines)
