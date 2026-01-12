"""
E6-4b: Observation Noise (Gaussian)

E6-3 확장: 관측에 가우시안 노이즈 추가
- 물리 상태는 깨끗하게 유지 (obs만 흔들림)
- σ ∈ {0.01, 0.03, 0.06}

핵심 지표:
1. collision_rate
2. triple_success_rate
3. false_defense_rate: 위험 없는데 방어 모드 켜진 비율
4. near_miss_rate: 장애물 최소거리 < ε 비율

실험군:
- BASE
- +MEM (기존 EMA)
- FULL (t_goal + t_risk + hysteresis)
- FULL+risk_filter (action EMA 없이 risk만 필터)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List

from .e6_multigoal_nav import (
    MultiGoalConfig,
    MultiGoalState,
    MultiGoalNavEnv,
    E6_3Gate,
)
from .e6_obstacle_nav import Obstacle, ActionGate


@dataclass
class NoiseConfig(MultiGoalConfig):
    """E6-4b 환경 설정"""
    # 노이즈 설정
    noise_sigma: float = 0.01  # 관측 노이즈 표준편차

    # 노이즈 적용 채널
    noise_pos: bool = True      # pos_x, pos_y
    noise_vel: bool = True      # vel_x, vel_y
    noise_goal: bool = True     # goal_dx, goal_dy, dist
    noise_obstacle: bool = True  # obstacle info


@dataclass
class NoiseState(MultiGoalState):
    """E6-4b 환경 상태"""
    # False defense 추적
    false_defense_count: int = 0
    total_defense_count: int = 0

    # Near miss 추적
    near_miss_count: int = 0
    min_obstacle_distance: float = float('inf')

    # True risk 추적 (노이즈 없는 실제 값)
    true_risk: float = 0.0


class NoiseNavEnv(MultiGoalNavEnv):
    """
    Observation Noise Navigation Environment

    E6-4b 핵심: 센서 불확실성 테스트
    """

    def __init__(self, config: NoiseConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.config = config

        # 통계
        self.total_false_defense = 0
        self.total_defense = 0
        self.total_near_miss = 0
        self.total_steps = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        obs = super().reset(seed)

        # 상태 초기화
        self.state.false_defense_count = 0
        self.state.total_defense_count = 0
        self.state.near_miss_count = 0
        self.state.min_obstacle_distance = float('inf')
        self.state.true_risk = 0.0

        # 노이즈 적용 (첫 관측)
        return self._apply_noise(obs)

    def _apply_noise(self, obs: np.ndarray) -> np.ndarray:
        """관측에 가우시안 노이즈 적용"""
        noisy_obs = obs.copy()
        sigma = self.config.noise_sigma

        # Position noise (indices 0-1)
        if self.config.noise_pos:
            noisy_obs[0:2] += self.rng.randn(2) * sigma

        # Velocity noise (indices 2-3)
        if self.config.noise_vel:
            noisy_obs[2:4] += self.rng.randn(2) * sigma

        # Goal noise (indices 4-6)
        if self.config.noise_goal:
            noisy_obs[4:7] += self.rng.randn(3) * sigma

        # Obstacle noise (indices 7-16)
        if self.config.noise_obstacle:
            noisy_obs[7:17] += self.rng.randn(10) * sigma

        return noisy_obs

    def _compute_true_risk(self) -> float:
        """실제 상태 기준 risk (노이즈 없음)"""
        min_dist = float('inf')
        for obstacle in self.state.obstacles:
            dist = np.linalg.norm(self.state.pos - obstacle.pos) - obstacle.radius
            min_dist = min(min_dist, dist)

        d_safe = 3.0  # world_size * 0.3
        risk = np.clip((d_safe - min_dist) / d_safe, 0, 1)

        # Near miss 추적 (실제 거리 < 0.5)
        if min_dist < 0.5:
            self.state.near_miss_count += 1
            self.total_near_miss += 1

        self.state.min_obstacle_distance = min(self.state.min_obstacle_distance, min_dist)

        return risk

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
        # 부모 스텝 실행 (깨끗한 물리)
        obs, reward, done, info = super().step(action)

        self.total_steps += 1

        # True risk 계산 (노이즈 없음)
        true_risk = self._compute_true_risk()
        self.state.true_risk = true_risk
        info['true_risk'] = true_risk

        # 노이즈 적용
        noisy_obs = self._apply_noise(obs)

        return noisy_obs, reward, done, info

    def record_defense_event(self, in_defense_mode: bool, perceived_risk: float):
        """
        방어 모드 이벤트 기록 (에이전트가 호출)

        false_defense = 방어 모드인데 실제 risk < 0.2
        """
        if in_defense_mode:
            self.state.total_defense_count += 1
            self.total_defense += 1

            if self.state.true_risk < 0.2:  # 실제로는 안전한데 방어
                self.state.false_defense_count += 1
                self.total_false_defense += 1

    def get_noise_stats(self) -> Dict:
        """노이즈 관련 통계"""
        false_defense_rate = self.total_false_defense / max(1, self.total_defense)
        near_miss_rate = self.total_near_miss / max(1, self.total_steps)

        return {
            'false_defense_count': self.total_false_defense,
            'total_defense_count': self.total_defense,
            'false_defense_rate': false_defense_rate,
            'near_miss_count': self.total_near_miss,
            'near_miss_rate': near_miss_rate,
        }

    def reset_all_stats(self):
        super().reset_all_stats()
        self.total_false_defense = 0
        self.total_defense = 0
        self.total_near_miss = 0
        self.total_steps = 0


# ============================================================================
# E6-4b Gates
# ============================================================================

@dataclass
class E6NoiseGateResult:
    """Noise Gate 결과"""
    passed: bool
    reason: str

    # 각 노이즈 레벨별 결과
    results_by_sigma: Dict

    # FULL이 BASE보다 우위인 레벨 수
    full_advantage_levels: int


@dataclass
class E6FilterComparisonResult:
    """FULL vs FULL+risk_filter 비교"""
    # Collision rates
    full_collision: float
    filter_collision: float

    # False defense rates
    full_false_defense: float
    filter_false_defense: float

    # Filter가 더 좋은지
    filter_better_collision: bool
    filter_better_false_defense: bool


class E6_4bGate(E6_3Gate):
    """E6-4b Gate 평가"""

    # Safety thresholds per noise level
    SAFETY_THRESHOLDS = {
        0.01: 0.15,
        0.03: 0.20,
        0.06: 0.25,
    }

    def get_safety_threshold(self, sigma: float) -> float:
        if sigma <= 0.01:
            return 0.15
        elif sigma <= 0.03:
            return 0.20
        else:
            return 0.25

    def evaluate_noise_robustness(
        self,
        results_by_sigma: Dict,
    ) -> E6NoiseGateResult:
        """Noise robustness 평가"""
        full_advantage_count = 0

        for sigma, results in results_by_sigma.items():
            base_coll = results.get('BASE', {}).get('collision_rate', 1.0)
            full_coll = results.get('FULL', {}).get('collision_rate', 1.0)

            if full_coll < base_coll:
                full_advantage_count += 1

        passed = full_advantage_count >= 2

        if passed:
            reason = f"FULL advantage in {full_advantage_count}/{len(results_by_sigma)} levels"
        else:
            reason = f"FULL advantage only in {full_advantage_count}/{len(results_by_sigma)} levels"

        return E6NoiseGateResult(
            passed=passed,
            reason=reason,
            results_by_sigma=results_by_sigma,
            full_advantage_levels=full_advantage_count,
        )

    def compare_filter_variants(
        self,
        full_results: Dict,
        filter_results: Dict,
    ) -> E6FilterComparisonResult:
        """FULL vs FULL+risk_filter 비교"""
        full_coll = full_results.get('collision_rate', 0)
        filter_coll = filter_results.get('collision_rate', 0)

        full_fd = full_results.get('false_defense_rate', 0)
        filter_fd = filter_results.get('false_defense_rate', 0)

        return E6FilterComparisonResult(
            full_collision=full_coll,
            filter_collision=filter_coll,
            full_false_defense=full_fd,
            filter_false_defense=filter_fd,
            filter_better_collision=filter_coll <= full_coll,
            filter_better_false_defense=filter_fd < full_fd,
        )
