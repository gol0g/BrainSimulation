"""
E2: Partial Observability - Uncertainty Calibration & Disambiguation

목표:
- Uncertainty Calibration: 정보량↓ → PC residual/ε↑, z=1↑ (단조 증가)
- No Panic: 정보 부족을 z=3(피로)로 오인하지 않음

3종 Partial Observability:
- PO-1: Dropout (채널 마스킹)
- PO-2: Noise (연속값 교란)
- PO-3: Stale (지연 관측)

게이트:
- E2a: Uncertainty Calibration (단조성)
- E2b: Disambiguation (z=1 vs z=3)
- E2c: Utility Preservation (성과 유지)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class POType(Enum):
    """Partial Observability 종류"""
    DROPOUT = "dropout"      # 채널 마스킹
    NOISE = "noise"          # 가우시안 노이즈
    STALE = "stale"          # 지연 관측


@dataclass
class POConfig:
    """Partial Observability 설정"""
    po_type: POType
    intensity: float  # 0.0 = no PO, higher = more severe

    # PO-1: Dropout
    dropout_prob: float = 0.0  # 각 채널 마스킹 확률

    # PO-2: Noise
    noise_std: float = 0.0  # 가우시안 노이즈 표준편차

    # PO-3: Stale
    stale_steps: int = 0  # 관측 지연 스텝 수

    @classmethod
    def dropout(cls, p: float) -> 'POConfig':
        """Dropout PO 생성"""
        return cls(
            po_type=POType.DROPOUT,
            intensity=p,
            dropout_prob=p,
        )

    @classmethod
    def noise(cls, sigma: float) -> 'POConfig':
        """Noise PO 생성"""
        return cls(
            po_type=POType.NOISE,
            intensity=sigma,
            noise_std=sigma,
        )

    @classmethod
    def stale(cls, k: int) -> 'POConfig':
        """Stale PO 생성"""
        # intensity를 0-1로 정규화 (k=8이면 ~1.0)
        return cls(
            po_type=POType.STALE,
            intensity=k / 8.0,
            stale_steps=k,
        )


class PartialObservabilityApplicator:
    """
    Partial Observability 적용기

    관측에 정보 손실을 주입
    """

    def __init__(self, config: POConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Stale 관측용 버퍼
        self.obs_buffer: List[np.ndarray] = []
        self.step = 0

    def apply(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Partial observability 적용

        Args:
            obs: 원본 관측 벡터

        Returns:
            degraded_obs: 정보 손실된 관측
            quality: 관측 품질 (0~1, 1이 완전한 관측)
        """
        self.step += 1
        cfg = self.config

        if cfg.po_type == POType.DROPOUT:
            return self._apply_dropout(obs)
        elif cfg.po_type == POType.NOISE:
            return self._apply_noise(obs)
        elif cfg.po_type == POType.STALE:
            return self._apply_stale(obs)
        else:
            return obs.copy(), 1.0

    def _apply_dropout(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """채널 마스킹 적용"""
        degraded = obs.copy()
        mask = self.rng.random(len(obs)) < self.config.dropout_prob

        # 마스킹된 채널은 0으로 (또는 평균값으로)
        degraded[mask] = 0.0

        # 품질 = 살아있는 채널 비율
        quality = 1.0 - mask.mean()
        return degraded, quality

    def _apply_noise(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """가우시안 노이즈 적용"""
        noise = self.rng.randn(len(obs)) * self.config.noise_std
        degraded = np.clip(obs + noise, 0, 1)

        # 품질 = 1 / (1 + noise_std)
        quality = 1.0 / (1.0 + self.config.noise_std * 5)  # 스케일 조정
        return degraded, quality

    def _apply_stale(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """지연 관측 적용"""
        self.obs_buffer.append(obs.copy())

        k = self.config.stale_steps
        if k == 0 or len(self.obs_buffer) <= k:
            return obs.copy(), 1.0

        # k 스텝 이전 관측 반환
        stale_obs = self.obs_buffer[-k-1]

        # 버퍼 크기 제한
        if len(self.obs_buffer) > k + 10:
            self.obs_buffer = self.obs_buffer[-(k+5):]

        # 품질 = 지연에 반비례
        quality = 1.0 / (1.0 + k * 0.1)
        return stale_obs, quality

    def reset(self):
        """상태 리셋"""
        self.obs_buffer = []
        self.step = 0


@dataclass
class E2RunStats:
    """E2 단일 실행 통계"""
    po_type: str
    intensity: float
    seed: int

    # PC 신호
    residual_ema_mean: float
    epsilon_spike_rate: float

    # Z-state 점유율
    z0_occupation: float
    z1_occupation: float
    z2_occupation: float
    z3_occupation: float

    # 성과 지표 (간접 측정)
    avg_quality: float  # 평균 관측 품질
    efficiency_proxy: float  # 효율 대리 지표


@dataclass
class E2GateResult:
    """E2 게이트 결과"""
    # E2a: Uncertainty Calibration
    e2a_passed: bool
    monotonicity_score: float  # 단조성 점수 (0~1)
    residual_monotonic: bool
    epsilon_monotonic: bool
    z1_monotonic: bool

    # E2b: Disambiguation
    e2b_passed: bool
    z3_excess_rate: float  # baseline 대비 z3 초과율

    # E2c: Utility Preservation
    e2c_passed: bool
    efficiency_retention: float  # 성과 유지율

    # 전체 판정
    passed: bool
    reason: str


class E2Gate:
    """
    E2 Partial Observability 게이트

    E2a: Uncertainty Calibration (단조성)
    - 강도↑ → residual↑, ε↑, z1↑

    E2b: Disambiguation (z=1 vs z=3)
    - PO에서 z=3 과발화 없음

    E2c: Utility Preservation
    - 중간 강도에서 성과 유지
    """

    def __init__(
        self,
        monotonicity_threshold: float = 0.75,  # 4레벨 중 3개 이상
        z3_excess_max: float = 0.05,  # baseline 대비 +5% 이내
        efficiency_retention_min: float = 0.7,  # 70% 이상 유지
    ):
        self.monotonicity_threshold = monotonicity_threshold
        self.z3_excess_max = z3_excess_max
        self.efficiency_retention_min = efficiency_retention_min

    def evaluate(
        self,
        stats_by_intensity: Dict[float, E2RunStats],
        baseline_stats: E2RunStats,
    ) -> E2GateResult:
        """
        E2 게이트 평가

        Args:
            stats_by_intensity: 강도별 통계 (0.0, 0.2, 0.4, 0.6 등)
            baseline_stats: 강도 0.0의 baseline
        """
        intensities = sorted(stats_by_intensity.keys())

        # E2a: Monotonicity check
        residual_vals = [stats_by_intensity[i].residual_ema_mean for i in intensities]
        epsilon_vals = [stats_by_intensity[i].epsilon_spike_rate for i in intensities]
        z1_vals = [stats_by_intensity[i].z1_occupation for i in intensities]

        residual_monotonic = self._check_monotonicity(residual_vals)
        epsilon_monotonic = self._check_monotonicity(epsilon_vals)
        z1_monotonic = self._check_monotonicity(z1_vals)

        monotonicity_score = (residual_monotonic + epsilon_monotonic + z1_monotonic) / 3.0
        e2a_passed = monotonicity_score >= self.monotonicity_threshold

        # E2b: Disambiguation (z3 excess)
        baseline_z3 = baseline_stats.z3_occupation
        max_z3_excess = 0.0
        for intensity, stats in stats_by_intensity.items():
            if intensity > 0:
                excess = stats.z3_occupation - baseline_z3
                max_z3_excess = max(max_z3_excess, excess)

        e2b_passed = max_z3_excess <= self.z3_excess_max

        # E2c: Utility Preservation (중간 강도에서)
        mid_intensities = [i for i in intensities if 0.1 <= i <= 0.5]
        if mid_intensities:
            baseline_eff = baseline_stats.efficiency_proxy
            mid_effs = [stats_by_intensity[i].efficiency_proxy for i in mid_intensities]
            avg_mid_eff = np.mean(mid_effs)
            efficiency_retention = avg_mid_eff / max(0.01, baseline_eff)
        else:
            efficiency_retention = 1.0

        e2c_passed = efficiency_retention >= self.efficiency_retention_min

        # 전체 판정
        passed = e2a_passed and e2b_passed and e2c_passed

        # 이유 생성
        reasons = []
        if not e2a_passed:
            reasons.append(f"monotonicity={monotonicity_score:.2f}<{self.monotonicity_threshold}")
        if not e2b_passed:
            reasons.append(f"z3_excess={max_z3_excess:.1%}>{self.z3_excess_max:.1%}")
        if not e2c_passed:
            reasons.append(f"efficiency_retention={efficiency_retention:.1%}<{self.efficiency_retention_min:.1%}")

        reason = "PASS" if passed else "; ".join(reasons)

        return E2GateResult(
            e2a_passed=e2a_passed,
            monotonicity_score=monotonicity_score,
            residual_monotonic=residual_monotonic,
            epsilon_monotonic=epsilon_monotonic,
            z1_monotonic=z1_monotonic,
            e2b_passed=e2b_passed,
            z3_excess_rate=max_z3_excess,
            e2c_passed=e2c_passed,
            efficiency_retention=efficiency_retention,
            passed=passed,
            reason=reason,
        )

    def _check_monotonicity(self, values: List[float], tolerance: float = 0.02) -> float:
        """
        단조 증가 체크 (허용 오차 포함)

        Returns:
            단조성 점수 (0~1), 1이면 완전 단조 증가
        """
        if len(values) < 2:
            return 1.0

        increases = 0
        total = len(values) - 1

        for i in range(1, len(values)):
            # 증가했거나, 감소폭이 tolerance 이내면 OK
            if values[i] >= values[i-1] - tolerance:
                increases += 1

        return increases / total


# PO 강도 레벨 정의
PO_INTENSITIES = {
    POType.DROPOUT: [0.0, 0.2, 0.4, 0.6],
    POType.NOISE: [0.0, 0.05, 0.1, 0.2],
    POType.STALE: [0, 2, 4, 8],  # 스텝 단위
}


def create_po_configs() -> List[POConfig]:
    """모든 PO 설정 생성 (3종 × 4강도 = 12개)"""
    configs = []

    for p in PO_INTENSITIES[POType.DROPOUT]:
        configs.append(POConfig.dropout(p))

    for sigma in PO_INTENSITIES[POType.NOISE]:
        configs.append(POConfig.noise(sigma))

    for k in PO_INTENSITIES[POType.STALE]:
        configs.append(POConfig.stale(k))

    return configs
