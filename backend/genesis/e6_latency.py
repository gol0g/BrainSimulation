"""
E6-4c: Sensor Latency (Observation Delay)

E6-3 확장: 관측을 k step 지연
- k ∈ {0, 1, 3, 5} step delay
- Ring buffer로 obs[t-k] 제공
- EMA lag + latency = 이중 지연 테스트

핵심 지표:
1. Lag-amplification slope: collision rate vs k
2. Risk reaction time: Δt = avoidance_start - risk_detected

게이트:
- Safety: k=0에서 <15%, k=5에서 <25%
- Robustness: FULL collision 증가 ≤ +10%p (k=0→5)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
from collections import deque

from .e6_multigoal_nav import (
    MultiGoalConfig,
    MultiGoalState,
    MultiGoalNavEnv,
    E6_3Gate,
)
from .e6_obstacle_nav import Obstacle, ActionGate


@dataclass
class LatencyConfig(MultiGoalConfig):
    """E6-4c 환경 설정"""
    # Latency 설정
    latency_k: int = 0  # 지연 step 수

    # Safety thresholds per latency
    # k=0: 15%, k=1: 18%, k=3: 22%, k=5: 25%


@dataclass
class LatencyState(MultiGoalState):
    """E6-4c 환경 상태"""
    # Ring buffer for delayed observations
    obs_buffer: deque = field(default_factory=lambda: deque(maxlen=10))

    # Risk reaction time 추적
    risk_detected_step: int = -1  # risk > threshold 처음 감지된 step
    avoidance_started_step: int = -1  # 회피 시작된 step
    reaction_times: List[int] = field(default_factory=list)  # 모든 reaction time 기록

    # 현재 risk 상태
    in_risk_zone: bool = False


class LatencyNavEnv(MultiGoalNavEnv):
    """
    Sensor Latency Navigation Environment

    E6-4c 핵심: 관측 지연으로 EMA lag 증폭 테스트
    """

    def __init__(self, config: LatencyConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.config = config

        # Reaction time 통계
        self.all_reaction_times = []
        self.risk_events = 0

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        obs = super().reset(seed)

        # Ring buffer 초기화 (k개의 초기 관측으로 채움)
        self.state.obs_buffer = deque(maxlen=max(10, self.config.latency_k + 1))
        for _ in range(self.config.latency_k + 1):
            self.state.obs_buffer.append(obs.copy())

        # Reaction time 초기화
        self.state.risk_detected_step = -1
        self.state.avoidance_started_step = -1
        self.state.reaction_times = []
        self.state.in_risk_zone = False

        return self._get_delayed_observation(obs)

    def _get_delayed_observation(self, current_obs: np.ndarray) -> np.ndarray:
        """지연된 관측 반환"""
        # 현재 관측을 버퍼에 추가
        self.state.obs_buffer.append(current_obs.copy())

        # k step 전 관측 반환
        if len(self.state.obs_buffer) > self.config.latency_k:
            return self.state.obs_buffer[-self.config.latency_k - 1].copy()
        else:
            return self.state.obs_buffer[0].copy()

    def _compute_true_risk(self) -> float:
        """실제 상태 기준 risk 계산 (지연 없이)"""
        min_obs_dist = float('inf')
        for obstacle in self.state.obstacles:
            dist = np.linalg.norm(self.state.pos - obstacle.pos) - obstacle.radius
            min_obs_dist = min(min_obs_dist, dist)

        # risk = (d_safe - dist) / d_safe, clamped to [0, 1]
        d_safe = 3.0  # 실제 거리 기준 (world_size * 0.3)
        risk = np.clip((d_safe - min_obs_dist) / d_safe, 0, 1)
        return risk

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
        # 이전 상태 저장 (reaction time 계산용)
        prev_vel = self.state.vel.copy()

        # 부모 스텝 실행
        obs, reward, done, info = super().step(action)

        # 실제 risk 계산 (지연 없이)
        true_risk = self._compute_true_risk()
        info['true_risk'] = true_risk

        # Risk reaction time 추적
        risk_threshold = 0.4  # risk_on_threshold와 동일

        if true_risk > risk_threshold:
            if not self.state.in_risk_zone:
                # 새로운 위험 구역 진입
                self.state.in_risk_zone = True
                self.state.risk_detected_step = self.state.step
                self.risk_events += 1
        else:
            if self.state.in_risk_zone:
                # 위험 구역 탈출
                self.state.in_risk_zone = False

                # Reaction time 기록 (회피 시작이 감지되었다면)
                if self.state.avoidance_started_step > 0:
                    reaction_time = self.state.avoidance_started_step - self.state.risk_detected_step
                    if reaction_time > 0:
                        self.state.reaction_times.append(reaction_time)
                        self.all_reaction_times.append(reaction_time)

                self.state.risk_detected_step = -1
                self.state.avoidance_started_step = -1

        # 회피 기동 감지 (속도 방향 변화)
        if self.state.in_risk_zone and self.state.avoidance_started_step < 0:
            # 가장 가까운 장애물 방향
            closest_obs = None
            min_dist = float('inf')
            for obstacle in self.state.obstacles:
                dist = np.linalg.norm(self.state.pos - obstacle.pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_obs = obstacle

            if closest_obs is not None:
                # 장애물로부터 멀어지는 방향
                away_dir = self.state.pos - closest_obs.pos
                away_norm = np.linalg.norm(away_dir)
                if away_norm > 1e-6:
                    away_dir = away_dir / away_norm

                    # action이 회피 방향과 일치하는지 (dot product > 0)
                    if np.dot(action, away_dir) > 0.1:
                        self.state.avoidance_started_step = self.state.step

        # 지연된 관측 반환
        delayed_obs = self._get_delayed_observation(obs)

        return delayed_obs, reward, done, info

    def get_reaction_time_stats(self) -> Dict:
        """Reaction time 통계"""
        if len(self.all_reaction_times) > 0:
            return {
                'mean_reaction_time': np.mean(self.all_reaction_times),
                'std_reaction_time': np.std(self.all_reaction_times),
                'max_reaction_time': np.max(self.all_reaction_times),
                'n_events': len(self.all_reaction_times),
                'total_risk_events': self.risk_events,
            }
        else:
            return {
                'mean_reaction_time': 0.0,
                'std_reaction_time': 0.0,
                'max_reaction_time': 0.0,
                'n_events': 0,
                'total_risk_events': self.risk_events,
            }

    def reset_all_stats(self):
        super().reset_all_stats()
        self.all_reaction_times = []
        self.risk_events = 0


# ============================================================================
# E6-4c Gates
# ============================================================================

@dataclass
class E6LatencyGateResult:
    """Latency Gate 결과"""
    passed: bool
    reason: str

    # 각 latency 레벨별 collision rate
    collision_by_k: Dict[int, Dict[str, float]]

    # Lag-amplification slope (collision 증가율)
    base_slope: float
    mem_slope: float
    full_slope: float

    # FULL의 증가폭
    full_increase: float  # k=0 → k=5


@dataclass
class E6ReactionTimeResult:
    """Reaction time 분석 결과"""
    # 각 config별 평균 reaction time
    base_mean_rt: float
    mem_mean_rt: float
    full_mean_rt: float

    # +MEM의 reaction time 증가
    mem_rt_penalty: float


class E6_4cGate(E6_3Gate):
    """E6-4c Gate 평가"""

    # Safety thresholds per latency
    SAFETY_THRESHOLDS = {
        0: 0.15,
        1: 0.18,
        3: 0.22,
        5: 0.25,
    }

    # Robustness: FULL collision increase limit
    MAX_FULL_INCREASE = 0.10  # +10%p

    def get_safety_threshold(self, latency_k: int) -> float:
        """latency에 따른 safety threshold"""
        return self.SAFETY_THRESHOLDS.get(latency_k, 0.25)

    def evaluate_latency_robustness(
        self,
        collision_by_k: Dict[int, Dict[str, float]],
    ) -> E6LatencyGateResult:
        """
        Latency Robustness 평가

        PASS 조건:
        1. FULL의 collision 증가폭이 k=0→5에서 ≤ +10%p
        2. FULL이 모든 k에서 BASE보다 낮거나 같음
        """
        ks = sorted(collision_by_k.keys())

        # Collision rates 추출
        base_colls = [collision_by_k[k]['BASE'] for k in ks]
        mem_colls = [collision_by_k[k]['+MEM'] for k in ks]
        full_colls = [collision_by_k[k]['FULL'] for k in ks]

        # Slope 계산 (선형 회귀 기울기 근사)
        if len(ks) >= 2:
            k_range = ks[-1] - ks[0]
            base_slope = (base_colls[-1] - base_colls[0]) / k_range if k_range > 0 else 0
            mem_slope = (mem_colls[-1] - mem_colls[0]) / k_range if k_range > 0 else 0
            full_slope = (full_colls[-1] - full_colls[0]) / k_range if k_range > 0 else 0
        else:
            base_slope = mem_slope = full_slope = 0

        # FULL 증가폭
        full_increase = full_colls[-1] - full_colls[0] if len(full_colls) >= 2 else 0

        # PASS 조건 체크
        full_robust = full_increase <= self.MAX_FULL_INCREASE
        full_better_than_base = all(full_colls[i] <= base_colls[i] for i in range(len(ks)))

        passed = full_robust and full_better_than_base

        if passed:
            reason = f"FULL increase={full_increase:.1%} (≤{self.MAX_FULL_INCREASE:.0%})"
        else:
            reasons = []
            if not full_robust:
                reasons.append(f"FULL increase={full_increase:.1%}>{self.MAX_FULL_INCREASE:.0%}")
            if not full_better_than_base:
                reasons.append("FULL not always better than BASE")
            reason = ", ".join(reasons)

        return E6LatencyGateResult(
            passed=passed,
            reason=reason,
            collision_by_k=collision_by_k,
            base_slope=base_slope,
            mem_slope=mem_slope,
            full_slope=full_slope,
            full_increase=full_increase,
        )

    def analyze_reaction_time(
        self,
        base_mean_rt: float,
        mem_mean_rt: float,
        full_mean_rt: float,
    ) -> E6ReactionTimeResult:
        """Reaction time 분석"""
        mem_rt_penalty = mem_mean_rt - base_mean_rt

        return E6ReactionTimeResult(
            base_mean_rt=base_mean_rt,
            mem_mean_rt=mem_mean_rt,
            full_mean_rt=full_mean_rt,
            mem_rt_penalty=mem_rt_penalty,
        )
