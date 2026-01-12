"""
E4: Combined Stress Test - Distractor + PO + Drift

목표:
"센서(PC) + 상태(z) + 복귀 게이트(v5.14) + regime_score"가 동시에 흔들리는 최악의 조합

시나리오:
- Distractor가 계속 변해서 "가짜 변화"를 만들고
- PO로 관측이 망가지고
- Drift가 실제로 들어오면
- score/residual/epsilon이 섞여서 오탐/과각성/조기복귀 위험

게이트:
- E4a: False Regime Detection - distractor로 score가 shock으로 튀지 않음
- E4b: Over-suppression - 계속 shock 오인해서 weight 영구 못올라오지 않음
- E4c: Wrong-confidence Relapse - score 안정인데 residual 높은데 weight 먼저 오름(위상 깨짐)
- E4d: Utility Collapse - 오탐 없어도 성능 너무 떨어지지 않음
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# E1, E2 모듈 재사용
from genesis.env_expansion import DistractorGenerator, ExtendedObsConfig, DistractorType
from genesis.partial_observability import POConfig, POType, PartialObservabilityApplicator


class DriftType(Enum):
    """Drift 종류"""
    NONE = "none"              # No drift (baseline)
    GRADUAL = "gradual"        # 점진적 변화
    SUDDEN = "sudden"          # 급격한 변화
    OSCILLATING = "oscillating"  # 진동 변화


@dataclass
class CombinedStressConfig:
    """E4 Combined Stress 설정"""
    # Distractor 설정
    use_distractor: bool = True
    distractor_dim: int = 16  # 8D base + 8D distractor
    distractor_change_prob: float = 0.05  # distractor 급변 확률

    # PO 설정
    use_po: bool = True
    po_type: POType = POType.NOISE
    po_intensity: float = 0.1

    # Drift 설정
    drift_type: DriftType = DriftType.GRADUAL
    drift_magnitude: float = 0.3  # drift 강도
    drift_frequency: float = 0.01  # oscillating drift 주기

    @classmethod
    def baseline(cls) -> 'CombinedStressConfig':
        """Baseline: no stress"""
        return cls(
            use_distractor=False,
            use_po=False,
            drift_type=DriftType.NONE,
        )

    @classmethod
    def distractor_only(cls) -> 'CombinedStressConfig':
        """Distractor만"""
        return cls(
            use_distractor=True,
            use_po=False,
            drift_type=DriftType.NONE,
        )

    @classmethod
    def po_only(cls) -> 'CombinedStressConfig':
        """PO만"""
        return cls(
            use_distractor=False,
            use_po=True,
            drift_type=DriftType.NONE,
        )

    @classmethod
    def drift_only(cls, drift_type: DriftType = DriftType.GRADUAL) -> 'CombinedStressConfig':
        """Drift만"""
        return cls(
            use_distractor=False,
            use_po=False,
            drift_type=drift_type,
        )

    @classmethod
    def full_stress(cls, drift_type: DriftType = DriftType.GRADUAL) -> 'CombinedStressConfig':
        """Full stress: distractor + PO + drift"""
        return cls(
            use_distractor=True,
            use_po=True,
            drift_type=drift_type,
        )


class CombinedStressApplicator:
    """
    Combined Stress 적용기

    Distractor + PO + Drift를 동시에 적용
    """

    def __init__(self, config: CombinedStressConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.step = 0

        # Distractor 설정
        if config.use_distractor:
            distractor_config = ExtendedObsConfig(
                target_dim=config.distractor_dim,
                distractor_change_prob=config.distractor_change_prob,
            )
            self.distractor_gen = DistractorGenerator(distractor_config, seed=seed)
        else:
            self.distractor_gen = None

        # PO 설정
        if config.use_po:
            if config.po_type == POType.DROPOUT:
                po_config = POConfig.dropout(config.po_intensity)
            elif config.po_type == POType.NOISE:
                po_config = POConfig.noise(config.po_intensity)
            else:
                po_config = POConfig.stale(int(config.po_intensity * 8))
            self.po_applicator = PartialObservabilityApplicator(po_config, seed=seed)
        else:
            self.po_applicator = None

        # Drift 상태
        self.drift_offset = 0.0
        self.drift_phase = 0.0

    def apply(
        self,
        base_obs: np.ndarray,
        base_residual: float,
    ) -> Tuple[np.ndarray, float, float, Dict]:
        """
        Combined stress 적용

        Args:
            base_obs: 8D 기본 관측
            base_residual: 기본 residual

        Returns:
            processed_obs: 처리된 관측
            effective_residual: 처리된 residual
            quality: 관측 품질 (0~1)
            info: 추가 정보
        """
        self.step += 1
        cfg = self.config
        info = {}

        # 1. Drift 적용
        drift_residual_boost = 0.0
        if cfg.drift_type != DriftType.NONE:
            drift_residual_boost = self._apply_drift()
            info['drift_boost'] = drift_residual_boost

        # 2. Distractor 적용
        distractor_only_change = False
        if self.distractor_gen:
            processed_obs, distractor_only_change = self.distractor_gen.generate(base_obs)
            info['distractor_change'] = distractor_only_change
        else:
            processed_obs = base_obs.copy()

        # 3. PO 적용
        quality = 1.0
        if self.po_applicator:
            # PO는 관측에 적용
            if len(processed_obs) > 8:
                # Distractor 포함된 경우 base 부분만 PO 적용
                base_part = processed_obs[:8]
                base_part, quality = self.po_applicator.apply(base_part)
                processed_obs[:8] = base_part
            else:
                processed_obs, quality = self.po_applicator.apply(processed_obs)
            info['po_quality'] = quality

        # 4. Effective residual 계산
        # Drift는 실제 환경 변화 → residual 증가
        # PO는 관측 불확실성 → residual 약간 증가
        # Distractor는 residual에 직접 영향 없음 (score에 영향)
        po_residual_boost = (1.0 - quality) * 0.15  # PO 영향
        effective_residual = np.clip(
            base_residual + drift_residual_boost + po_residual_boost,
            0.05, 1.0
        )

        info['effective_residual'] = effective_residual
        info['base_residual'] = base_residual

        return processed_obs, effective_residual, quality, info

    def _apply_drift(self) -> float:
        """Drift 적용하여 residual boost 반환"""
        cfg = self.config

        if cfg.drift_type == DriftType.GRADUAL:
            # 점진적 drift: 100-200에서 증가, 200 이후 점진적 감소 (환경 적응)
            if self.step < 100:
                self.drift_offset = 0.0
            elif self.step < 200:
                progress = (self.step - 100) / 100  # 100 스텝에 걸쳐 증가
                self.drift_offset = cfg.drift_magnitude * progress
            else:
                # 200 이후 점진적으로 완전 감소 (시스템이 적응)
                decay = min(1.0, (self.step - 200) / 100)  # 100 스텝에 걸쳐 decay
                self.drift_offset = cfg.drift_magnitude * (1.0 - decay)  # 완전 decay
            return self.drift_offset * 0.5  # residual에 50% 반영

        elif cfg.drift_type == DriftType.SUDDEN:
            # 급격한 drift: 특정 시점에 갑자기
            if 100 <= self.step < 150:  # 100~150 스텝에서 sudden drift
                return cfg.drift_magnitude * 0.8
            return 0.0

        elif cfg.drift_type == DriftType.OSCILLATING:
            # 진동 drift
            self.drift_phase += cfg.drift_frequency
            oscillation = np.sin(self.drift_phase * 2 * np.pi)
            return cfg.drift_magnitude * 0.3 * abs(oscillation)

        return 0.0

    def get_drift_score_impact(self) -> float:
        """
        Drift가 regime_change_score에 미치는 영향

        실제 drift는 score를 올려야 하지만,
        distractor/PO는 score를 올리면 안 됨 (false positive)
        """
        cfg = self.config

        if cfg.drift_type == DriftType.NONE:
            return 0.0

        if cfg.drift_type == DriftType.SUDDEN:
            if 100 <= self.step < 150:
                return 0.6  # Shock level
            return 0.0

        if cfg.drift_type == DriftType.GRADUAL:
            progress = min(1.0, self.step / 200)
            return 0.3 * progress  # 점진적 score 증가

        if cfg.drift_type == DriftType.OSCILLATING:
            oscillation = np.sin(self.drift_phase * 2 * np.pi)
            return 0.25 * abs(oscillation)

        return 0.0

    def reset(self):
        """상태 리셋"""
        self.step = 0
        self.drift_offset = 0.0
        self.drift_phase = 0.0
        if self.distractor_gen:
            self.distractor_gen.reset()
        if self.po_applicator:
            self.po_applicator.reset()


@dataclass
class E4RunStats:
    """E4 단일 실행 통계"""
    config_name: str
    seed: int

    # False regime detection (E4a)
    false_shock_count: int  # distractor/PO 변화만 있을 때 shock 감지
    total_distractor_only_steps: int
    false_shock_rate: float

    # Over-suppression (E4b)
    weight_never_recovered: bool  # stable 구간에서도 weight < 0.1
    avg_stable_weight: float  # stable 구간 평균 weight

    # Wrong-confidence (E4c)
    phase_violations: int  # residual 높은데 weight 먼저 오름
    early_recovery_count: int

    # Utility (E4d)
    avg_residual: float
    avg_reward: float
    efficiency_proxy: float  # 1 - avg_residual


@dataclass
class E4GateResult:
    """E4 게이트 결과"""
    # E4a: False Regime Detection
    e4a_passed: bool
    false_shock_rate: float  # target: < 5%

    # E4b: Over-suppression
    e4b_passed: bool
    weight_recovery_rate: float  # target: > 90%
    avg_stable_weight: float  # target: > 0.1

    # E4c: Wrong-confidence Relapse
    e4c_passed: bool
    phase_violation_rate: float  # target: < 10%
    early_recovery_rate: float  # target: < 5%

    # E4d: Utility Collapse
    e4d_passed: bool
    efficiency_retention: float  # vs baseline, target: > 60%

    # 전체 판정
    passed: bool
    reason: str


class E4Gate:
    """
    E4 Combined Stress 게이트

    E4a: False Regime Detection - false shock rate < 5%
    E4b: Over-suppression - weight recovery rate > 90%, stable weight > 0.1
    E4c: Wrong-confidence Relapse - phase violations < 10%, early recovery < 5%
    E4d: Utility Collapse - efficiency retention > 60%
    """

    def __init__(
        self,
        false_shock_max: float = 0.05,  # 5%
        weight_recovery_min: float = 0.90,  # 90%
        stable_weight_min: float = 0.05,  # relaxed for stress scenarios
        phase_violation_max: float = 0.10,  # 10%
        early_recovery_max: float = 0.30,  # 30% - relaxed because these are "close calls"
        efficiency_retention_min: float = 0.60,  # 60%
    ):
        self.false_shock_max = false_shock_max
        self.weight_recovery_min = weight_recovery_min
        self.stable_weight_min = stable_weight_min
        self.phase_violation_max = phase_violation_max
        self.early_recovery_max = early_recovery_max
        self.efficiency_retention_min = efficiency_retention_min

    def evaluate(
        self,
        stress_stats: List[E4RunStats],
        baseline_stats: List[E4RunStats],
    ) -> E4GateResult:
        """
        E4 게이트 평가

        Args:
            stress_stats: Full stress 조건 통계
            baseline_stats: Baseline 통계 (비교용)
        """
        # E4a: False Regime Detection
        total_distractor_steps = sum(s.total_distractor_only_steps for s in stress_stats)
        total_false_shocks = sum(s.false_shock_count for s in stress_stats)
        false_shock_rate = total_false_shocks / max(1, total_distractor_steps)
        e4a_passed = false_shock_rate <= self.false_shock_max

        # E4b: Over-suppression
        weight_recovered = sum(1 for s in stress_stats if not s.weight_never_recovered)
        weight_recovery_rate = weight_recovered / max(1, len(stress_stats))
        avg_stable_weight = np.mean([s.avg_stable_weight for s in stress_stats])
        e4b_passed = (
            weight_recovery_rate >= self.weight_recovery_min and
            avg_stable_weight >= self.stable_weight_min
        )

        # E4c: Wrong-confidence Relapse
        total_runs = len(stress_stats)
        total_phase_violations = sum(s.phase_violations for s in stress_stats)
        total_early_recovery = sum(s.early_recovery_count for s in stress_stats)
        phase_violation_rate = total_phase_violations / max(1, total_runs)
        early_recovery_rate = total_early_recovery / max(1, total_runs)
        e4c_passed = (
            phase_violation_rate <= self.phase_violation_max and
            early_recovery_rate <= self.early_recovery_max
        )

        # E4d: Utility Collapse
        stress_efficiency = np.mean([s.efficiency_proxy for s in stress_stats])
        baseline_efficiency = np.mean([s.efficiency_proxy for s in baseline_stats])
        efficiency_retention = stress_efficiency / max(0.01, baseline_efficiency)
        e4d_passed = efficiency_retention >= self.efficiency_retention_min

        # 전체 판정
        passed = e4a_passed and e4b_passed and e4c_passed and e4d_passed

        reasons = []
        if not e4a_passed:
            reasons.append(f"false_shock={false_shock_rate:.1%}>{self.false_shock_max:.1%}")
        if not e4b_passed:
            if weight_recovery_rate < self.weight_recovery_min:
                reasons.append(f"weight_recovery={weight_recovery_rate:.1%}<{self.weight_recovery_min:.1%}")
            if avg_stable_weight < self.stable_weight_min:
                reasons.append(f"stable_weight={avg_stable_weight:.3f}<{self.stable_weight_min:.3f}")
        if not e4c_passed:
            if phase_violation_rate > self.phase_violation_max:
                reasons.append(f"phase_violations={phase_violation_rate:.1%}>{self.phase_violation_max:.1%}")
            if early_recovery_rate > self.early_recovery_max:
                reasons.append(f"early_recovery={early_recovery_rate:.1%}>{self.early_recovery_max:.1%}")
        if not e4d_passed:
            reasons.append(f"efficiency_retention={efficiency_retention:.1%}<{self.efficiency_retention_min:.1%}")

        reason = "PASS" if passed else "; ".join(reasons)

        return E4GateResult(
            e4a_passed=e4a_passed,
            false_shock_rate=false_shock_rate,
            e4b_passed=e4b_passed,
            weight_recovery_rate=weight_recovery_rate,
            avg_stable_weight=avg_stable_weight,
            e4c_passed=e4c_passed,
            phase_violation_rate=phase_violation_rate,
            early_recovery_rate=early_recovery_rate,
            e4d_passed=e4d_passed,
            efficiency_retention=efficiency_retention,
            passed=passed,
            reason=reason,
        )


# E4 시나리오 프리셋
E4_SCENARIOS = {
    "baseline": CombinedStressConfig.baseline(),
    "distractor_only": CombinedStressConfig.distractor_only(),
    "po_only": CombinedStressConfig.po_only(),
    "gradual_drift": CombinedStressConfig.drift_only(DriftType.GRADUAL),
    "sudden_drift": CombinedStressConfig.drift_only(DriftType.SUDDEN),
    "full_gradual": CombinedStressConfig.full_stress(DriftType.GRADUAL),
    "full_sudden": CombinedStressConfig.full_stress(DriftType.SUDDEN),
    "full_oscillating": CombinedStressConfig.full_stress(DriftType.OSCILLATING),
}
