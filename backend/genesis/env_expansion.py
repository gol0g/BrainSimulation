"""
E1: Environment Expansion - Observation Dimension + Distractors

목표:
- PC residual/epsilon 기반 신호가 "차원 증가"에서 망가지지 않는지 검증
- regime 감지가 "진짜 변화"만 잡는지 검증

관측 공간 확장:
- 8D (baseline) → 16D → 32D
- 의미 있는 채널 + 무의미 채널(distractor) 혼합

게이트:
- E1a Scale Invariance: residual_ema 분포가 차원 증가에서 유지
- E1b Change Specificity: distractor-only 구간에서 score < s_on
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DistractorType(Enum):
    """Distractor 종류"""
    RANDOM_NOISE = "random_noise"           # 매 스텝 랜덤 값
    SLOW_DRIFT = "slow_drift"               # 천천히 변하는 값
    PERIODIC = "periodic"                   # 주기적으로 변하는 값
    BLINK = "blink"                         # 가끔 튀는 값
    CORRELATED = "correlated"               # 실제 관측과 상관있지만 무의미


@dataclass
class ExtendedObsConfig:
    """확장 관측 공간 설정"""
    # 기본 8D + 확장
    base_dim: int = 8
    target_dim: int = 16  # 16 or 32

    # Distractor 설정
    distractor_types: List[DistractorType] = field(default_factory=lambda: [
        DistractorType.RANDOM_NOISE,
        DistractorType.SLOW_DRIFT,
        DistractorType.PERIODIC,
    ])

    # Distractor 파라미터
    noise_std: float = 0.3           # RANDOM_NOISE 표준편차
    drift_speed: float = 0.01        # SLOW_DRIFT 변화 속도
    periodic_freq: float = 0.1       # PERIODIC 주파수
    blink_prob: float = 0.05         # BLINK 발생 확률
    blink_magnitude: float = 0.8     # BLINK 크기

    # Distractor 변화 이벤트 (dynamics 영향 없음)
    distractor_change_prob: float = 0.02  # distractor만 급변하는 확률


class DistractorGenerator:
    """
    Distractor 채널 생성기

    핵심: dynamics에 영향 없이 관측 차원만 늘림
    """

    def __init__(self, config: ExtendedObsConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Distractor 채널 수
        self.n_distractors = config.target_dim - config.base_dim

        # 상태 변수 (slow drift, periodic phase 등)
        self.drift_values = self.rng.randn(self.n_distractors) * 0.5
        self.periodic_phases = self.rng.uniform(0, 2 * np.pi, self.n_distractors)
        self.step = 0

        # Distractor-only change 이벤트 추적
        self.distractor_change_active = False
        self.distractor_change_steps = 0

    def generate(self, base_obs: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        기본 관측에 distractor 채널 추가

        Args:
            base_obs: 8D 기본 관측

        Returns:
            extended_obs: target_dim 크기 관측
            distractor_only_change: 이번 스텝이 distractor-only 변화인지
        """
        self.step += 1
        cfg = self.config

        # Distractor-only change 이벤트 체크
        distractor_only_change = False
        if self.rng.random() < cfg.distractor_change_prob:
            self.distractor_change_active = True
            self.distractor_change_steps = 0
            distractor_only_change = True

        if self.distractor_change_active:
            self.distractor_change_steps += 1
            if self.distractor_change_steps > 10:  # 10 steps 후 종료
                self.distractor_change_active = False

        # 각 타입별 distractor 생성
        distractors = []
        n_per_type = max(1, self.n_distractors // len(cfg.distractor_types))

        for i, dtype in enumerate(cfg.distractor_types):
            start_idx = i * n_per_type
            end_idx = min(start_idx + n_per_type, self.n_distractors)
            n_channels = end_idx - start_idx

            if n_channels <= 0:
                continue

            if dtype == DistractorType.RANDOM_NOISE:
                vals = self.rng.randn(n_channels) * cfg.noise_std
                # Distractor-only change 시 더 큰 변화
                if self.distractor_change_active:
                    vals *= 3.0
                distractors.extend(vals)

            elif dtype == DistractorType.SLOW_DRIFT:
                # 천천히 변화
                drift_delta = self.rng.randn(n_channels) * cfg.drift_speed
                if self.distractor_change_active:
                    drift_delta *= 10.0  # 급격한 drift 변화
                self.drift_values[start_idx:end_idx] += drift_delta
                self.drift_values = np.clip(self.drift_values, -1, 1)
                distractors.extend(self.drift_values[start_idx:end_idx])

            elif dtype == DistractorType.PERIODIC:
                # 주기적 변화
                phases = self.periodic_phases[start_idx:end_idx]
                if self.distractor_change_active:
                    phases += np.pi  # 위상 급변
                vals = np.sin(phases + self.step * cfg.periodic_freq)
                self.periodic_phases[start_idx:end_idx] += cfg.periodic_freq
                distractors.extend(vals * 0.5)

            elif dtype == DistractorType.BLINK:
                # 가끔 튀는 값
                vals = np.zeros(n_channels)
                blink_mask = self.rng.random(n_channels) < cfg.blink_prob
                if self.distractor_change_active:
                    blink_mask = np.ones(n_channels, dtype=bool)  # 전부 blink
                vals[blink_mask] = cfg.blink_magnitude * self.rng.choice([-1, 1], blink_mask.sum())
                distractors.extend(vals)

            elif dtype == DistractorType.CORRELATED:
                # 실제 관측과 상관있지만 무의미
                corr_base = base_obs[:min(n_channels, len(base_obs))]
                if len(corr_base) < n_channels:
                    corr_base = np.pad(corr_base, (0, n_channels - len(corr_base)))
                vals = corr_base * 0.3 + self.rng.randn(n_channels) * 0.2
                if self.distractor_change_active:
                    vals += self.rng.randn(n_channels) * 0.5
                distractors.extend(vals)

        # 부족한 채널 채우기
        while len(distractors) < self.n_distractors:
            distractors.append(self.rng.randn() * cfg.noise_std)

        # 확장 관측 생성
        distractor_array = np.array(distractors[:self.n_distractors])
        extended_obs = np.concatenate([base_obs, distractor_array])

        return extended_obs, distractor_only_change

    def reset(self):
        """상태 리셋"""
        self.drift_values = self.rng.randn(self.n_distractors) * 0.5
        self.periodic_phases = self.rng.uniform(0, 2 * np.pi, self.n_distractors)
        self.step = 0
        self.distractor_change_active = False
        self.distractor_change_steps = 0


@dataclass
class E1GateResult:
    """E1 게이트 결과"""
    # E1a: Scale Invariance
    e1a_passed: bool
    residual_mean_8d: float
    residual_mean_extended: float
    residual_ratio: float  # extended / 8d
    epsilon_spike_mean_8d: float
    epsilon_spike_mean_extended: float
    epsilon_ratio: float

    # E1b: Change Specificity
    e1b_passed: bool
    distractor_only_shock_rate: float  # distractor-only 구간에서 score >= 0.6 비율
    real_change_detection_rate: float  # 실제 변화 시 score >= 0.6 비율

    # 전체 판정
    passed: bool
    reason: str


class E1Gate:
    """
    E1 환경 확장 게이트

    E1a Scale Invariance:
    - residual_ema 분포가 8D→16D→32D에서 유지
    - ratio가 0.5~2.0 범위 내

    E1b Change Specificity:
    - distractor-only 구간에서 score >= s_on 비율 < 5%
    - 실제 변화 시 탐지율 유지
    """

    def __init__(
        self,
        scale_ratio_min: float = 0.5,
        scale_ratio_max: float = 2.0,
        distractor_shock_rate_max: float = 0.05,
        real_change_detection_min: float = 0.7,
    ):
        self.scale_ratio_min = scale_ratio_min
        self.scale_ratio_max = scale_ratio_max
        self.distractor_shock_rate_max = distractor_shock_rate_max
        self.real_change_detection_min = real_change_detection_min

    def evaluate(
        self,
        baseline_8d: Dict,
        extended: Dict,
    ) -> E1GateResult:
        """
        E1 게이트 평가

        Args:
            baseline_8d: 8D 환경에서의 측정값
                - residual_mean, residual_std
                - epsilon_spike_mean, epsilon_spike_std
            extended: 확장 환경에서의 측정값
                - residual_mean, residual_std
                - epsilon_spike_mean, epsilon_spike_std
                - distractor_only_shock_rate
                - real_change_detection_rate
        """
        # E1a: Scale Invariance
        residual_ratio = extended['residual_mean'] / max(0.001, baseline_8d['residual_mean'])
        epsilon_ratio = extended['epsilon_spike_mean'] / max(0.001, baseline_8d['epsilon_spike_mean'])

        e1a_passed = (
            self.scale_ratio_min <= residual_ratio <= self.scale_ratio_max and
            self.scale_ratio_min <= epsilon_ratio <= self.scale_ratio_max
        )

        # E1b: Change Specificity
        distractor_shock_rate = extended.get('distractor_only_shock_rate', 0.0)
        real_detection_rate = extended.get('real_change_detection_rate', 1.0)

        e1b_passed = (
            distractor_shock_rate <= self.distractor_shock_rate_max and
            real_detection_rate >= self.real_change_detection_min
        )

        # 전체 판정
        passed = e1a_passed and e1b_passed

        # 이유 생성
        reasons = []
        if not e1a_passed:
            if not (self.scale_ratio_min <= residual_ratio <= self.scale_ratio_max):
                reasons.append(f"residual_ratio={residual_ratio:.2f} out of [{self.scale_ratio_min}, {self.scale_ratio_max}]")
            if not (self.scale_ratio_min <= epsilon_ratio <= self.scale_ratio_max):
                reasons.append(f"epsilon_ratio={epsilon_ratio:.2f} out of [{self.scale_ratio_min}, {self.scale_ratio_max}]")
        if not e1b_passed:
            if distractor_shock_rate > self.distractor_shock_rate_max:
                reasons.append(f"distractor_shock_rate={distractor_shock_rate:.1%} > {self.distractor_shock_rate_max:.1%}")
            if real_detection_rate < self.real_change_detection_min:
                reasons.append(f"real_detection_rate={real_detection_rate:.1%} < {self.real_change_detection_min:.1%}")

        reason = "PASS" if passed else "; ".join(reasons)

        return E1GateResult(
            e1a_passed=e1a_passed,
            residual_mean_8d=baseline_8d['residual_mean'],
            residual_mean_extended=extended['residual_mean'],
            residual_ratio=residual_ratio,
            epsilon_spike_mean_8d=baseline_8d['epsilon_spike_mean'],
            epsilon_spike_mean_extended=extended['epsilon_spike_mean'],
            epsilon_ratio=epsilon_ratio,
            e1b_passed=e1b_passed,
            distractor_only_shock_rate=distractor_shock_rate,
            real_change_detection_rate=real_detection_rate,
            passed=passed,
            reason=reason,
        )


def normalize_extended_observation(
    obs: np.ndarray,
    base_dim: int = 8,
    method: str = "per_channel"
) -> np.ndarray:
    """
    확장 관측 정규화

    PC residual이 차원 증가로 폭증하지 않도록 정규화

    Args:
        obs: 확장 관측 벡터
        base_dim: 기본 차원 (8)
        method: 정규화 방법
            - "per_channel": 채널별 정규화 (기본)
            - "sqrt_dim": sqrt(dim) 스케일링
            - "none": 정규화 없음

    Returns:
        정규화된 관측
    """
    if method == "none":
        return obs

    if method == "sqrt_dim":
        # 차원 증가에 따른 스케일 조정
        scale = np.sqrt(base_dim / len(obs))
        return obs * scale

    if method == "per_channel":
        # 기본 채널은 그대로, distractor는 스케일 조정
        if len(obs) <= base_dim:
            return obs
        base = obs[:base_dim]
        distractors = obs[base_dim:] * 0.5  # distractor 영향 축소
        return np.concatenate([base, distractors])

    return obs


# E1 시나리오 설정
E1_SCENARIOS = {
    "16d_baseline": ExtendedObsConfig(target_dim=16),
    "32d_baseline": ExtendedObsConfig(target_dim=32),
    "16d_high_distractor": ExtendedObsConfig(
        target_dim=16,
        noise_std=0.5,
        distractor_change_prob=0.05,
    ),
    "32d_all_types": ExtendedObsConfig(
        target_dim=32,
        distractor_types=[
            DistractorType.RANDOM_NOISE,
            DistractorType.SLOW_DRIFT,
            DistractorType.PERIODIC,
            DistractorType.BLINK,
            DistractorType.CORRELATED,
        ],
    ),
}
