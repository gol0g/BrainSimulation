"""
v5.14: PC-Z Dynamics - Bidirectional Dynamic Coupling

v5.14 추가 (2026-01-02):
- PC-gated recovery: weight recovery는 score(외부 안정) + residual(내부 안정) 둘 다 확인
- residual_gate_min/max로 internal stability 계산
- internal_lag 시나리오에서 early_recovery_with_cost 100% 감소

v5.13 추가:
- Dynamic past_regime_weight: shock에서 0, 안정에서 서서히 복귀
- Asymmetric EMA: 차단 빠르게, 복귀 느리게
- Anti-flap 히스테리시스

핵심 철학:
- PC ↔ Z가 "게이트"가 아니라 "동역학"으로 주고받음
- PC는 Z의 "감각기관" 중 하나
- Z는 PC의 "정밀도/우선순위/연산량" 조절기

Phase 1: PC → z-evidence (Read-only)
- ε_spike: 예측오차 스파이크 → z=1 trigger
- convergence_cost: 수렴 비용 → z=1 보조 증거
- residual_error_floor: 바닥 오차 → regime 전환 신호
- margin: 행동 경쟁 마진 → 선택 갈등 신호

Phase 2: z → PC (Soft-write)
- Precision/Gain: z=1이면 error gain↑, prior pull↓
- Iteration budget: z=1이면 iteration↑ (숙고)
- Prior blending: z별 prior source 선택

원칙:
1. regime_change_score 유지 + PC 신호를 sensor fusion
2. Z는 "전역 조절기", 정책 선택은 circuit 경쟁에서
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from collections import deque


@dataclass
class PCSignals:
    """PC가 Z에게 주는 신호들 (감각 채널)"""
    # 핵심 4개 신호
    epsilon_spike: float = 0.0      # 예측오차 급등 (0~1)
    convergence_cost: float = 0.0   # 수렴 비용 = iterations / max_iterations
    residual_error: float = 0.0     # 바닥 오차 (수렴 후에도 남는 오차)
    action_margin: float = 1.0      # 행동 경쟁 마진 (낮을수록 갈등)

    # 보조 신호
    error_velocity: float = 0.0     # d(error)/dt (오차 변화 속도)
    prior_pull_ratio: float = 0.5   # prior_force / (data_force + prior_force)

    def to_z_evidence(self, weights: Optional[Dict] = None) -> Dict[int, float]:
        """
        PC 신호를 z-evidence 형태로 변환

        Returns:
            Dict[z_state, evidence_delta] - 각 z-state에 더할 evidence
        """
        w = weights or {
            'epsilon_spike': 0.4,
            'convergence_cost': 0.2,
            'residual_error': 0.2,
            'margin': 0.2,
        }

        # z=0 (stable): 오차 낮고, 수렴 빠르고, 마진 크면 증가
        z0_evidence = (
            (1.0 - self.epsilon_spike) * 0.3 +
            (1.0 - self.convergence_cost) * 0.3 +
            (1.0 - self.residual_error) * 0.2 +
            min(self.action_margin, 1.0) * 0.2
        )

        # z=1 (exploring): 오차 스파이크, 수렴 느림, 마진 작으면 증가
        z1_evidence = (
            self.epsilon_spike * w['epsilon_spike'] +
            self.convergence_cost * w['convergence_cost'] +
            self.residual_error * w['residual_error'] +
            (1.0 - min(self.action_margin, 1.0)) * w['margin']
        )

        # z=2 (reflecting): residual_error 높고 velocity 낮으면 (막힘)
        z2_evidence = (
            self.residual_error * 0.5 +
            (1.0 - abs(self.error_velocity)) * 0.3 +
            self.prior_pull_ratio * 0.2  # prior에 의존 중이면
        )

        # z=3 (fatigued): 이건 PC보다는 외부 신호(에너지, 시간 등)로
        # PC에서는 z=3 evidence를 직접 주지 않음 (오진 방지)
        z3_evidence = 0.0

        return {
            0: z0_evidence,
            1: z1_evidence,
            2: z2_evidence,
            3: z3_evidence,
        }


@dataclass
class DynamicPastRegimeConfig:
    """v5.13/v5.14: 동적 past_regime_weight 설정"""
    # Shock/Stability thresholds
    s_on: float = 0.6   # regime_change_score >= 이 값이면 shock
    s_off: float = 0.2  # regime_change_score <= 이 값이면 안정

    # Weight bounds
    w_max: float = 0.2  # 최대 past_regime_weight (안정시)
    w_min: float = 0.0  # 최소 (shock시)
    z1_ceiling: float = 0.12  # z=1에서의 상한 (v5.12의 0.1보다 약간 여유)

    # Asymmetric EMA rates
    alpha_decay: float = 0.3   # shock시 빠르게 0으로
    alpha_recovery: float = 0.03  # 복귀는 더 느리게 (0.05 → 0.03)

    # v5.14: PC-gated recovery (default since v5.14 production)
    # Note: residual_error is normalized (error_norm/0.5, clipped to [0,1])
    # - stable (error_norm=0.15): normalized=0.3
    # - internal_lag recovery (error_norm=0.4): normalized=0.8
    use_pc_gate: bool = True  # v5.14 production default (2026-01-02)
    residual_gate_min: float = 0.4   # 이 아래면 full recovery (stability=1)
    residual_gate_max: float = 0.8   # 이 위면 no recovery (stability=0)
    residual_ema_alpha: float = 0.1  # residual EMA smoothing


@dataclass
class RecoveryEvent:
    """Too-early recovery 이벤트 샘플"""
    step: int
    score: float
    residual_error: float
    convergence_cost: float
    weight_before: float
    weight_after: float
    weight_delta: float


class RecoveryMonitor:
    """
    v5.13 모니터링: Recovery 동역학 관측

    A) 시계열 로깅:
       - regime_change_score(t)
       - residual_error(t)
       - convergence_cost(t)
       - past_regime_weight(t)

    B) Too-early recovery 샘플링:
       - score ≤ s_off인데 residual이 높은 상태에서 weight 상승 감지
    """

    def __init__(self, buffer_size: int = 500, residual_threshold_pct: float = 0.75):
        """
        Args:
            buffer_size: 시계열 버퍼 크기
            residual_threshold_pct: residual_error percentile 기준 (0.75 = 상위 25%)
        """
        self.buffer_size = buffer_size
        self.residual_threshold_pct = residual_threshold_pct

        # A) Time series buffers
        self.score_history: deque = deque(maxlen=buffer_size)
        self.residual_history: deque = deque(maxlen=buffer_size)
        self.convergence_history: deque = deque(maxlen=buffer_size)
        self.weight_history: deque = deque(maxlen=buffer_size)

        # B) Too-early recovery events
        self.early_recovery_events: List[RecoveryEvent] = []
        self.max_events = 100  # 최대 저장 이벤트 수

        # Statistics
        self.total_steps = 0
        self.steps_in_stable_zone = 0  # score <= s_off
        self.early_recovery_count = 0

        # Running percentile estimation (for residual threshold)
        self.residual_ema = 0.3  # EMA for adaptive threshold

        # v5.14: Zone-wise openness tracking
        self.zone_openness: Dict[str, List[float]] = {
            'shock': [],
            'transition': [],
            'stable': [],
        }
        self.zone_residual_ema: Dict[str, List[float]] = {
            'shock': [],
            'transition': [],
            'stable': [],
        }
        self.openness_history: deque = deque(maxlen=buffer_size)

    def record(
        self,
        score: float,
        residual_error: float,
        convergence_cost: float,
        weight_before: float,
        weight_after: float,
        s_off: float = 0.2,
        openness: float = 1.0,  # v5.14: PC gate openness (0~1)
    ):
        """
        매 스텝 기록

        Args:
            score: regime_change_score
            residual_error: PC residual error
            convergence_cost: PC convergence cost
            weight_before: past_regime_weight before update
            weight_after: past_regime_weight after update
            s_off: stability threshold
            openness: v5.14 PC gate openness (0=closed, 1=open)
        """
        self.total_steps += 1

        # A) Time series logging
        self.score_history.append(score)
        self.residual_history.append(residual_error)
        self.convergence_history.append(convergence_cost)
        self.weight_history.append(weight_after)
        self.openness_history.append(openness)

        # Update residual EMA (for adaptive threshold)
        self.residual_ema = 0.95 * self.residual_ema + 0.05 * residual_error

        # B) Too-early recovery detection
        weight_delta = weight_after - weight_before
        is_stable_zone = score <= s_off
        is_high_residual = residual_error > self.residual_ema * 1.5  # 50% above EMA
        is_weight_rising = weight_delta > 0.005  # Meaningful increase

        if is_stable_zone:
            self.steps_in_stable_zone += 1

            if is_high_residual and is_weight_rising:
                # Too-early recovery detected!
                self.early_recovery_count += 1

                event = RecoveryEvent(
                    step=self.total_steps,
                    score=score,
                    residual_error=residual_error,
                    convergence_cost=convergence_cost,
                    weight_before=weight_before,
                    weight_after=weight_after,
                    weight_delta=weight_delta,
                )

                if len(self.early_recovery_events) < self.max_events:
                    self.early_recovery_events.append(event)

        # C) v5.14: Zone-wise openness tracking
        if score >= 0.6:
            zone = 'shock'
        elif score > 0.2:
            zone = 'transition'
        else:
            zone = 'stable'

        # Keep last 1000 samples per zone
        if len(self.zone_openness[zone]) >= 1000:
            self.zone_openness[zone] = self.zone_openness[zone][-500:]
            self.zone_residual_ema[zone] = self.zone_residual_ema[zone][-500:]
        self.zone_openness[zone].append(openness)
        self.zone_residual_ema[zone].append(self.residual_ema)

    def get_time_series(self, last_n: Optional[int] = None) -> Dict:
        """시계열 데이터 반환"""
        n = last_n or len(self.score_history)
        return {
            'score': list(self.score_history)[-n:],
            'residual_error': list(self.residual_history)[-n:],
            'convergence_cost': list(self.convergence_history)[-n:],
            'past_regime_weight': list(self.weight_history)[-n:],
        }

    def get_early_recovery_summary(self) -> Dict:
        """Too-early recovery 이벤트 요약"""
        if not self.early_recovery_events:
            return {
                'count': 0,
                'rate': 0.0,
                'events': [],
            }

        events = self.early_recovery_events
        return {
            'count': self.early_recovery_count,
            'rate': self.early_recovery_count / max(1, self.steps_in_stable_zone),
            'avg_residual': np.mean([e.residual_error for e in events]),
            'avg_weight_delta': np.mean([e.weight_delta for e in events]),
            'events': [
                {
                    'step': e.step,
                    'score': round(e.score, 3),
                    'residual': round(e.residual_error, 3),
                    'weight_delta': round(e.weight_delta, 4),
                }
                for e in events[-10:]  # 최근 10개만
            ],
        }

    def get_score_zone_distribution(self) -> Dict:
        """Score 구간별 weight 분포"""
        if len(self.score_history) < 10:
            return {}

        scores = np.array(self.score_history)
        weights = np.array(self.weight_history)

        zones = {
            'shock': scores >= 0.6,
            'transition': (scores > 0.2) & (scores < 0.6),
            'stable': scores <= 0.2,
        }

        result = {}
        for zone_name, mask in zones.items():
            if mask.sum() > 0:
                zone_weights = weights[mask]
                result[zone_name] = {
                    'count': int(mask.sum()),
                    'weight_mean': float(np.mean(zone_weights)),
                    'weight_std': float(np.std(zone_weights)),
                    'weight_max': float(np.max(zone_weights)),
                }

        return result

    def get_phase_analysis(self) -> Dict:
        """Weight vs Residual 위상 분석 (lead/lag)"""
        if len(self.weight_history) < 20:
            return {'status': 'insufficient_data'}

        weights = np.array(self.weight_history)
        residuals = np.array(self.residual_history)

        # Cross-correlation to detect lead/lag
        # Positive lag = weight lags behind residual (good)
        # Negative lag = weight leads residual (bad - too early recovery)
        weight_diff = np.diff(weights)
        residual_diff = np.diff(residuals)

        if len(weight_diff) < 10:
            return {'status': 'insufficient_data'}

        # Simple correlation of derivatives
        corr = np.corrcoef(weight_diff, residual_diff)[0, 1]

        # Check if weight rises when residual is still high
        # (bad pattern: residual high + weight rising)
        bad_pattern_count = 0
        for i in range(len(weight_diff)):
            if residuals[i] > self.residual_ema and weight_diff[i] > 0.003:
                bad_pattern_count += 1

        return {
            'derivative_correlation': float(corr) if not np.isnan(corr) else 0.0,
            'bad_pattern_count': bad_pattern_count,
            'bad_pattern_rate': bad_pattern_count / max(1, len(weight_diff)),
            'interpretation': 'healthy' if bad_pattern_count < len(weight_diff) * 0.05 else 'needs_v5.14',
        }

    def get_lag_metric(self) -> Dict:
        """
        Lag Metric: residual 안정 → weight 복귀 사이의 위상 지연 측정

        건강한 시스템: residual이 먼저 내려가고, weight가 뒤따라 올라옴
        문제 시스템: weight가 residual보다 먼저 올라감 (premature recovery)

        Returns:
            {
                'mean_lag': float (positive = healthy, negative = premature)
                'recovery_episodes': int (분석된 에피소드 수)
                'healthy_recoveries': int (정상 순서 복귀)
                'premature_recoveries': int (조기 복귀)
            }
        """
        if len(self.residual_history) < 50:
            return {'status': 'insufficient_data'}

        residuals = np.array(self.residual_history)
        weights = np.array(self.weight_history)

        # Find recovery episodes: residual drops significantly
        residual_threshold = self.residual_ema * 0.7  # 30% below EMA
        weight_rise_threshold = 0.01

        recovery_episodes = 0
        healthy_recoveries = 0
        premature_recoveries = 0
        lag_values = []

        # Scan for recovery patterns
        i = 0
        while i < len(residuals) - 20:
            # Look for residual drop
            if residuals[i] > self.residual_ema and i + 10 < len(residuals):
                # Find when residual stabilizes below threshold
                residual_stable_step = None
                for j in range(i, min(i + 30, len(residuals))):
                    if residuals[j] < residual_threshold:
                        residual_stable_step = j
                        break

                if residual_stable_step is not None:
                    # Find when weight starts rising
                    weight_rise_step = None
                    for j in range(i, min(residual_stable_step + 30, len(weights) - 1)):
                        if weights[j + 1] - weights[j] > weight_rise_threshold:
                            weight_rise_step = j
                            break

                    if weight_rise_step is not None:
                        recovery_episodes += 1
                        lag = weight_rise_step - residual_stable_step

                        if lag >= 0:
                            healthy_recoveries += 1
                        else:
                            premature_recoveries += 1

                        lag_values.append(lag)
                        i = max(i + 1, residual_stable_step + 10)
                        continue

            i += 1

        if not lag_values:
            return {
                'status': 'no_recovery_episodes',
                'recovery_episodes': 0,
            }

        return {
            'mean_lag': float(np.mean(lag_values)),
            'std_lag': float(np.std(lag_values)),
            'recovery_episodes': recovery_episodes,
            'healthy_recoveries': healthy_recoveries,
            'premature_recoveries': premature_recoveries,
            'premature_rate': premature_recoveries / max(1, recovery_episodes),
        }

    def get_premature_recovery_impact(self, window: int = 20) -> Dict:
        """
        Premature Recovery Impact: early recovery 이벤트 이후 성능 영향 측정

        early recovery 이벤트 발생 후 N스텝의:
        - 평균 residual (baseline 대비)
        - residual 상승 여부
        - 효율 영향 (convergence_cost로 대리)

        Returns:
            {
                'events_analyzed': int
                'avg_post_residual': float
                'baseline_residual': float
                'residual_increase_rate': float (이벤트 후 residual 상승 비율)
                'cost_detected': bool (실제 비용 발생 여부)
            }
        """
        if not self.early_recovery_events:
            return {'status': 'no_early_recovery_events', 'cost_detected': False}

        residuals = list(self.residual_history)
        convergences = list(self.convergence_history)

        if len(residuals) < window * 2:
            return {'status': 'insufficient_data', 'cost_detected': False}

        # Baseline: average residual in stable periods
        baseline_residual = self.residual_ema

        events_analyzed = 0
        post_residuals = []
        post_convergences = []
        residual_increases = 0

        for event in self.early_recovery_events:
            event_idx = event.step - 1  # Convert to 0-indexed

            if event_idx < 0 or event_idx + window >= len(residuals):
                continue

            events_analyzed += 1

            # Get post-event window
            post_window_residuals = residuals[event_idx:event_idx + window]
            post_window_convergences = convergences[event_idx:event_idx + window]

            post_residuals.extend(post_window_residuals)
            post_convergences.extend(post_window_convergences)

            # Check if residual increases after event
            if len(post_window_residuals) >= 5:
                if np.mean(post_window_residuals[-5:]) > np.mean(post_window_residuals[:5]):
                    residual_increases += 1

        if events_analyzed == 0:
            return {'status': 'no_valid_events', 'cost_detected': False}

        avg_post_residual = np.mean(post_residuals)
        avg_post_convergence = np.mean(post_convergences)
        residual_increase_rate = residual_increases / events_analyzed

        # Cost detection criteria:
        # 1. Post-event residual is higher than baseline, OR
        # 2. Residual increases after the event in >30% of cases
        cost_detected = (
            avg_post_residual > baseline_residual * 1.2 or
            residual_increase_rate > 0.3
        )

        return {
            'events_analyzed': events_analyzed,
            'avg_post_residual': float(avg_post_residual),
            'avg_post_convergence': float(avg_post_convergence),
            'baseline_residual': float(baseline_residual),
            'residual_ratio': float(avg_post_residual / max(0.01, baseline_residual)),
            'residual_increase_rate': float(residual_increase_rate),
            'cost_detected': cost_detected,
        }

    def get_zone_openness_stats(self) -> Dict:
        """
        v5.14: Zone별 PC gate openness 통계

        Returns:
            {
                'shock': {'mean': float, 'low_rate': float, ...},
                'transition': {'mean': float, 'low_rate': float, 'diagnostic': str, ...},
                'stable': {'mean': float, 'low_rate': float, 'low_reason': str, ...},
            }

        low_rate = openness < 0.2 비율 (gate가 닫혀있는 비율)
        """
        result = {}
        for zone, values in self.zone_openness.items():
            if not values:
                result[zone] = {'mean': 1.0, 'low_rate': 0.0, 'count': 0}
            else:
                arr = np.array(values)
                stats = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'low_rate': float(np.mean(arr < 0.2)),  # openness < 0.2
                    'high_rate': float(np.mean(arr > 0.8)),  # openness > 0.8
                    'count': len(values),
                }

                # Zone-specific diagnostics (with zone-specific residual_ema)
                zone_residual_mean = np.mean(self.zone_residual_ema[zone]) if self.zone_residual_ema[zone] else self.residual_ema
                if zone == 'transition':
                    stats['diagnostic'] = self._diagnose_transition_zone(stats, zone_residual_mean)
                    stats['avg_residual_ema'] = float(zone_residual_mean)
                elif zone == 'stable':
                    stats['low_reason'] = self._explain_stable_low_rate(stats, zone_residual_mean)
                    stats['avg_residual_ema'] = float(zone_residual_mean)

                result[zone] = stats
        return result

    def _diagnose_transition_zone(self, stats: Dict, zone_residual_mean: float) -> str:
        """
        Transition zone 진단: mean=0.0 + low_rate=100%가 문제인지 판단

        문제 케이스: residual은 내려갔는데 openness가 0에 붙어있음
        정상 케이스: residual이 실제로 gate_max 근처라서 openness=0
        """
        if stats['mean'] > 0.1 or stats['low_rate'] < 0.9:
            return "NORMAL - gate responding to conditions"

        # Transition에서 mean=0, low_rate~100%일 때
        # zone의 평균 residual_ema를 체크해서 진단
        if zone_residual_mean > 0.7:
            return f"EXPECTED - avg_residual({zone_residual_mean:.2f}) near gate_max, gate correctly closed"
        elif zone_residual_mean > 0.5:
            return f"BORDERLINE - avg_residual({zone_residual_mean:.2f}) dropping but gate still conservative"
        else:
            return f"INVESTIGATE - avg_residual({zone_residual_mean:.2f}) low but gate not opening (check threshold)"

    def _explain_stable_low_rate(self, stats: Dict, zone_residual_mean: float) -> str:
        """
        Stable zone에서 low_rate가 높을 때 이유 설명

        운영자가 "왜 stable인데 weight 안 올라와?"에 바로 답할 수 있도록
        """
        if stats['low_rate'] < 0.2:
            return "HEALTHY - gate mostly open in stable zone"

        # low_rate가 높을 때 zone의 평균 residual_ema로 설명
        if zone_residual_mean > 0.6:
            return f"RESIDUAL_HIGH - avg_residual({zone_residual_mean:.2f}) still above gate_min, recovery delayed"
        elif zone_residual_mean > 0.4:
            return f"RESIDUAL_TRANSITIONING - avg_residual({zone_residual_mean:.2f}) in gate range, partial recovery"
        else:
            return f"RECOVERING - avg_residual({zone_residual_mean:.2f}) below gate_min, gate opening"

    def get_diagnostics(self) -> Dict:
        """전체 모니터링 진단"""
        return {
            'total_steps': self.total_steps,
            'steps_in_stable_zone': self.steps_in_stable_zone,
            'residual_ema': self.residual_ema,
            'early_recovery': self.get_early_recovery_summary(),
            'zone_distribution': self.get_score_zone_distribution(),
            'phase_analysis': self.get_phase_analysis(),
            # v5.13 추가 지표
            'lag_metric': self.get_lag_metric(),
            'premature_impact': self.get_premature_recovery_impact(),
            # v5.14 추가 지표
            'zone_openness': self.get_zone_openness_stats(),
        }

    def should_upgrade_to_v514(self) -> Tuple[bool, str]:
        """
        v5.14 업그레이드 필요성 판단 (2단계)

        1단계: 패턴 감지 (early/bad pattern)
        2단계: 실제 비용 발생 확인 (residual 상승, 효율 저하)

        둘 다 만족해야 업그레이드 권장
        """
        if self.total_steps < 100:
            return False, "insufficient_data"

        # 1단계: 패턴 감지
        early_rate = self.early_recovery_count / max(1, self.steps_in_stable_zone)
        phase = self.get_phase_analysis()
        lag = self.get_lag_metric()

        pattern_detected = False
        pattern_reason = ""

        if early_rate > 0.1:
            pattern_detected = True
            pattern_reason = f"high_early_recovery_rate: {early_rate:.2%}"
        elif phase.get('interpretation') == 'needs_v5.14':
            pattern_detected = True
            pattern_reason = f"bad_phase_pattern: {phase.get('bad_pattern_rate', 0):.2%}"
        elif lag.get('premature_rate', 0) > 0.3:
            pattern_detected = True
            pattern_reason = f"high_premature_rate: {lag.get('premature_rate', 0):.2%}"

        if not pattern_detected:
            return False, "v5.13_sufficient"

        # 2단계: 실제 비용 확인
        impact = self.get_premature_recovery_impact()
        cost_detected = impact.get('cost_detected', False)

        if cost_detected:
            return True, f"pattern+cost: {pattern_reason}, cost={impact.get('residual_ratio', 1):.2f}x"
        else:
            # 패턴은 있지만 비용 없음 → 관찰 계속, 아직 업그레이드 불필요
            return False, f"pattern_only: {pattern_reason} (no cost detected)"

    def reset(self):
        """모니터 리셋"""
        self.score_history.clear()
        self.residual_history.clear()
        self.convergence_history.clear()
        self.weight_history.clear()
        self.openness_history.clear()  # v5.14
        self.early_recovery_events.clear()
        self.total_steps = 0
        self.steps_in_stable_zone = 0
        self.early_recovery_count = 0
        self.residual_ema = 0.3
        # v5.14: zone-wise openness
        self.zone_openness = {'shock': [], 'transition': [], 'stable': []}
        self.zone_residual_ema = {'shock': [], 'transition': [], 'stable': []}


class DynamicPastRegime:
    """
    v5.13/v5.14: 동적 past_regime_weight 계산

    원칙:
    1. Shock에서는 거의 0 (wrong-confidence 방지)
    2. Stabilize 되면 서서히 복귀
    3. 연속 + 히스테리시스 (EMA)

    v5.14 추가:
    4. 복귀는 external stability만이 아니라 internal stability(residual)도 확인
    """

    def __init__(self, config: Optional[DynamicPastRegimeConfig] = None):
        self.config = config or DynamicPastRegimeConfig()
        self.w_applied = 0.1  # 현재 적용값 (v5.12 기본값에서 시작)
        self.w_target = 0.1
        self.steps_stable = 0  # 안정 구간 연속 스텝

        # v5.14: residual EMA for internal stability
        self.residual_ema = 0.15  # baseline residual

        # v5.13 모니터링 훅
        self.monitor = RecoveryMonitor()
        
    def compute_stability(self, regime_change_score: float) -> float:
        """
        stability = clamp((s_on - s) / (s_on - s_off), 0, 1)
        s >= s_on: stability = 0 (shock)
        s <= s_off: stability = 1 (stable)
        """
        cfg = self.config
        if cfg.s_off >= cfg.s_on:
            return 0.0  # Invalid config

        # Correct formula:
        # - score >= s_on (0.6) → stability = 0
        # - score <= s_off (0.2) → stability = 1
        stability = (cfg.s_on - regime_change_score) / (cfg.s_on - cfg.s_off)
        return max(0.0, min(1.0, stability))
    
    def compute_internal_stability(self) -> float:
        """
        v5.14: Internal stability from residual EMA

        - residual_ema < gate_min: stability = 1.0 (full recovery)
        - residual_ema > gate_max: stability = 0.0 (no recovery)
        - in between: linear interpolation

        This allows full recovery in stable scenarios (low residual)
        while blocking recovery when residual is high (internal unstable)
        """
        cfg = self.config
        gate_min = cfg.residual_gate_min
        gate_max = cfg.residual_gate_max

        if self.residual_ema <= gate_min:
            return 1.0  # Full recovery allowed
        elif self.residual_ema >= gate_max:
            return 0.0  # No recovery
        else:
            # Linear interpolation
            return 1.0 - (self.residual_ema - gate_min) / (gate_max - gate_min)

    def compute_target(
        self,
        regime_change_score: float,
        z_confidence: float,
        current_z: int
    ) -> float:
        """
        v5.13: w_target = w_max * stability_external * confidence_factor
        v5.14: w_target = w_max * stability_external * stability_internal * confidence_factor

        핵심: v5.14에서는 external이 안정이어도 internal이 불안정이면 복귀 안 함
        """
        cfg = self.config

        # External stability (from score)
        stability_external = self.compute_stability(regime_change_score)

        # v5.14: PC-gated recovery
        if cfg.use_pc_gate:
            stability_internal = self.compute_internal_stability()
            combined_stability = stability_external * stability_internal
        else:
            # v5.13: external only
            combined_stability = stability_external

        # Base target from combined stability and confidence
        confidence_factor = 0.5 + 0.5 * z_confidence
        w_target = cfg.w_max * combined_stability * confidence_factor

        # z=1 ceiling (exploring 모드에서는 과거 참조 제한)
        if current_z == 1:
            w_target = min(w_target, cfg.z1_ceiling)

        return max(cfg.w_min, min(cfg.w_max, w_target))
    
    def update(
        self,
        regime_change_score: float,
        z_confidence: float,
        current_z: int,
        residual_error: float = 0.0,  # v5.13 모니터링용, v5.14 게이트용
        convergence_cost: float = 0.0,  # v5.13 모니터링용
    ) -> float:
        """
        비대칭 EMA로 w_applied 업데이트

        Returns: w_applied (실제 적용할 past_regime_weight)
        """
        cfg = self.config
        w_before = self.w_applied  # 모니터링용

        # 0. v5.14: Update residual EMA (before computing target)
        if cfg.use_pc_gate:
            alpha = cfg.residual_ema_alpha
            self.residual_ema = (1 - alpha) * self.residual_ema + alpha * residual_error

        # 1. Compute target (v5.14에서는 residual_ema가 반영됨)
        self.w_target = self.compute_target(regime_change_score, z_confidence, current_z)

        # 2. Asymmetric EMA
        # Shock (score >= s_on): 빠르게 decay
        # Recovery (score < s_on): 느리게 복귀
        if regime_change_score >= cfg.s_on:
            alpha = cfg.alpha_decay
            self.steps_stable = 0
        else:
            alpha = cfg.alpha_recovery
            if regime_change_score <= cfg.s_off:
                self.steps_stable += 1
            else:
                self.steps_stable = max(0, self.steps_stable - 1)

        # 3. EMA update
        self.w_applied = (1 - alpha) * self.w_applied + alpha * self.w_target

        # 4. Anti-flap: 아직 충분히 안정되지 않았으면 복귀 억제
        if self.steps_stable < 30:  # 최소 30 steps 안정 필요
            recovery_suppression = self.steps_stable / 30.0
            max_allowed = cfg.z1_ceiling + (cfg.w_max - cfg.z1_ceiling) * recovery_suppression
            self.w_applied = min(self.w_applied, max_allowed)

        # 5. v5.13/v5.14 모니터링: 시계열 기록 + too-early recovery 감지
        openness = self.compute_internal_stability() if cfg.use_pc_gate else 1.0
        self.monitor.record(
            score=regime_change_score,
            residual_error=residual_error,
            convergence_cost=convergence_cost,
            weight_before=w_before,
            weight_after=self.w_applied,
            s_off=cfg.s_off,
            openness=openness,
        )

        return self.w_applied

    def get_diagnostics(self) -> Dict:
        cfg = self.config
        diag = {
            'w_target': self.w_target,
            'w_applied': self.w_applied,
            'steps_stable': self.steps_stable,
            'version': 'v5.14' if cfg.use_pc_gate else 'v5.13',
        }
        if cfg.use_pc_gate:
            diag['residual_ema'] = self.residual_ema
            diag['stability_internal'] = self.compute_internal_stability()
        return diag

    def get_monitoring_report(self) -> Dict:
        """v5.13 모니터링 리포트"""
        return self.monitor.get_diagnostics()

    def should_upgrade_to_v514(self) -> Tuple[bool, str]:
        """v5.14 업그레이드 필요성 판단 (모니터링 데이터 기반)"""
        return self.monitor.should_upgrade_to_v514()

    def reset(self):
        self.w_applied = 0.1
        self.w_target = 0.1
        self.steps_stable = 0
        self.residual_ema = 0.15  # v5.14
        self.monitor.reset()


@dataclass
class ZModulation:
    """Z가 PC에게 주는 조절 신호들"""
    # Precision/Gain 조절
    error_gain: float = 1.0         # ε에 곱해지는 gain (z=1이면 ↑)
    prior_precision: float = 1.0    # λ_prior에 곱해지는 계수 (z=1이면 ↓)

    # Iteration budget
    iteration_multiplier: float = 1.0  # max_iterations에 곱해지는 계수

    # Prior source blending
    current_regime_weight: float = 1.0  # 현재 regime prior 가중치
    past_regime_weight: float = 0.0     # 과거 regime prior 가중치

    @classmethod
    def from_z_state(cls, z: int, z_confidence: float = 1.0) -> 'ZModulation':
        """
        Z-state에서 modulation 계산

        z=0 (stable): 정상 작동
        z=1 (exploring): error gain↑, prior↓, iteration↑
        z=2 (reflecting): prior↑, iteration↓
        z=3 (fatigued): 연산량↓, 행동↓ (learn은 유지)
        """
        if z == 0:  # stable
            return cls(
                error_gain=1.0,
                prior_precision=1.0 + 0.1 * z_confidence,  # 약간 prior 강화
                iteration_multiplier=1.0,
                current_regime_weight=1.0,
                past_regime_weight=0.0,
            )
        elif z == 1:  # exploring
            return cls(
                # v5.12: Softer modulation to reduce degradation
                error_gain=1.2 + 0.1 * z_confidence,  # 1.3+0.2 -> 1.2+0.1 (softer)
                prior_precision=0.45 - 0.05 * z_confidence,  # 0.4-0.1 -> 0.45-0.05 (less extreme)
                iteration_multiplier=min(1.5, 1.3 + 0.2 * z_confidence),  # capped at 1.5
                current_regime_weight=1.0,
                # v5.12: Reduce past_regime_weight to prevent wrong-confidence
                # High regime uncertainty = don't trust past
                past_regime_weight=0.1,  # 0.2 -> 0.1 (more conservative)
            )
        elif z == 2:  # reflecting
            return cls(
                error_gain=0.8,  # error 민감도 약간 감소
                prior_precision=1.5 + 0.3 * z_confidence,  # prior 강화
                iteration_multiplier=0.7,  # 빠른 결정
                current_regime_weight=0.8,
                past_regime_weight=0.5,  # 과거 많이 참고
            )
        else:  # z=3 (fatigued)
            return cls(
                error_gain=1.0,  # gain 유지
                prior_precision=0.8,  # prior 약간 약화 (wrong confidence 방지)
                iteration_multiplier=0.5,  # 연산량 감소
                current_regime_weight=0.5,
                past_regime_weight=0.0,  # 과거 회상 억제
            )


@dataclass
class PCZDynamicsState:
    """PC-Z 동역학 상태"""
    # PC 신호 히스토리
    epsilon_history: deque = field(default_factory=lambda: deque(maxlen=50))
    convergence_history: deque = field(default_factory=lambda: deque(maxlen=50))
    error_history: deque = field(default_factory=lambda: deque(maxlen=50))

    # Spike 감지용
    epsilon_baseline: float = 0.5
    epsilon_baseline_samples: int = 0

    # v5.12: Residual decay acceleration
    residual_decay_rate: float = 0.08
    last_shock_step: int = -100
    
    # v5.13: Dynamic past_regime tracking
    regime_change_score_ema: float = 0.0

    # 현재 신호
    current_pc_signals: PCSignals = field(default_factory=PCSignals)
    current_z_modulation: ZModulation = field(default_factory=ZModulation)

    # 통계
    total_steps: int = 0
    z_from_pc_count: int = 0  # PC 신호로 z가 바뀐 횟수


class PCZDynamics:
    """
    PC ↔ Z 양방향 동역학

    v5.9 bridge와 다른 점:
    - PC 신호가 z-evidence에 직접 합류 (sensor fusion)
    - Z modulation이 PC의 gain/iteration을 직접 조절
    - "게이트"가 아닌 "연속 동역학"
    
    v5.13 추가:
    - Dynamic past_regime_weight (shock→0, stable→복귀)
    """

    def __init__(self, past_regime_config: Optional[DynamicPastRegimeConfig] = None):
        self.state = PCZDynamicsState()

        # v5.13/v5.14: Dynamic past_regime controller
        self.dynamic_past_regime = DynamicPastRegime(config=past_regime_config)

        # PC → z-evidence 가중치
        self.pc_evidence_weights = {
            'epsilon_spike': 0.35,
            'convergence_cost': 0.25,
            'residual_error': 0.20,
            'margin': 0.20,
        }

        # Spike 감지 임계값
        self.spike_threshold = 1.5  # baseline 대비 1.5배면 spike
        self.spike_ema_alpha = 0.1  # baseline EMA 업데이트 속도

    def compute_pc_signals(
        self,
        epsilon: np.ndarray,
        error_norm: float,
        iterations: int,
        max_iterations: int,
        converged: bool,
        prior_force_norm: float,
        data_force_norm: float,
        action_margin: float = 1.0,
    ) -> PCSignals:
        """
        PC 상태에서 Z-evidence용 신호 추출

        이 함수가 "PC의 감각 채널"을 정의
        """
        # 1. ε_spike 계산
        epsilon_norm = np.linalg.norm(epsilon)

        # Baseline 업데이트 (첫 50스텝은 baseline 구축)
        if self.state.epsilon_baseline_samples < 50:
            self.state.epsilon_baseline = (
                self.state.epsilon_baseline * self.state.epsilon_baseline_samples +
                epsilon_norm
            ) / (self.state.epsilon_baseline_samples + 1)
            self.state.epsilon_baseline_samples += 1
        else:
            # EMA로 천천히 업데이트
            self.state.epsilon_baseline = (
                self.spike_ema_alpha * epsilon_norm +
                (1 - self.spike_ema_alpha) * self.state.epsilon_baseline
            )

        # Spike = baseline 대비 급등 정도
        if self.state.epsilon_baseline > 0.01:
            spike_ratio = epsilon_norm / self.state.epsilon_baseline
            epsilon_spike = np.clip((spike_ratio - 1.0) / (self.spike_threshold - 1.0), 0, 1)
        else:
            epsilon_spike = 0.0

        # 2. Convergence cost
        convergence_cost = iterations / max(max_iterations, 1)
        if not converged:
            convergence_cost = 1.0  # 수렴 실패 = 최대 비용

        # 3. Residual error (수렴 후에도 남는 오차)
        # 정규화: 0.5를 기준으로
        residual_error = np.clip(error_norm / 0.5, 0, 1)

        # 4. Error velocity (오차 변화 속도)
        self.state.error_history.append(error_norm)
        if len(self.state.error_history) >= 5:
            recent = list(self.state.error_history)[-5:]
            error_velocity = (recent[-1] - recent[0]) / 5  # 양수 = 증가
        else:
            error_velocity = 0.0

        # 5. Prior pull ratio
        total_force = data_force_norm + prior_force_norm + 1e-10
        prior_pull_ratio = prior_force_norm / total_force

        signals = PCSignals(
            epsilon_spike=epsilon_spike,
            convergence_cost=convergence_cost,
            residual_error=residual_error,
            action_margin=action_margin,
            error_velocity=np.clip(error_velocity, -1, 1),
            prior_pull_ratio=prior_pull_ratio,
        )

        self.state.current_pc_signals = signals
        return signals

    def get_z_evidence_from_pc(self, signals: Optional[PCSignals] = None) -> Dict[int, float]:
        """
        PC 신호를 z-evidence로 변환

        기존 regime_change_score와 "sensor fusion"되어야 함
        """
        if signals is None:
            signals = self.state.current_pc_signals

        return signals.to_z_evidence(self.pc_evidence_weights)

    def get_modulation_for_pc(
        self,
        z: int,
        z_confidence: float = 1.0,
        regime_change_score: float = 0.0,
    ) -> ZModulation:
        """
        현재 Z-state에 따른 PC modulation 계산

        v5.13: regime_change_score를 사용해 past_regime_weight를 동적으로 조절
        """
        # 1. Base modulation from z-state
        modulation = ZModulation.from_z_state(z, z_confidence)

        # 2. v5.13: Dynamic past_regime_weight 적용
        # PC 신호를 모니터링에 함께 전달
        pc_signals = self.state.current_pc_signals
        dynamic_weight = self.dynamic_past_regime.update(
            regime_change_score=regime_change_score,
            z_confidence=z_confidence,
            current_z=z,
            residual_error=pc_signals.residual_error,
            convergence_cost=pc_signals.convergence_cost,
        )

        # 3. Override past_regime_weight with dynamic value
        # z=2 (reflecting)는 과거 참조가 의도적으로 높으므로 그대로 유지
        if z != 2:
            modulation = ZModulation(
                error_gain=modulation.error_gain,
                prior_precision=modulation.prior_precision,
                iteration_multiplier=modulation.iteration_multiplier,
                current_regime_weight=modulation.current_regime_weight,
                past_regime_weight=dynamic_weight,
            )

        # 4. Update regime_change_score EMA for tracking
        self.state.regime_change_score_ema = (
            0.1 * regime_change_score +
            0.9 * self.state.regime_change_score_ema
        )

        self.state.current_z_modulation = modulation
        return modulation

    def step(
        self,
        # PC 입력
        epsilon: np.ndarray,
        error_norm: float,
        iterations: int,
        max_iterations: int,
        converged: bool,
        prior_force_norm: float = 0.0,
        data_force_norm: float = 0.0,
        action_margin: float = 1.0,
        # Z 입력
        current_z: int = 0,
        z_confidence: float = 1.0,
        # v5.13: regime score for dynamic past_regime
        regime_change_score: float = 0.0,
    ) -> Tuple[Dict[int, float], ZModulation]:
        """
        한 스텝 양방향 업데이트

        Returns:
            (z_evidence_delta, z_modulation)
        """
        self.state.total_steps += 1

        # 1. PC → Z: 신호 계산 및 evidence 생성
        pc_signals = self.compute_pc_signals(
            epsilon=epsilon,
            error_norm=error_norm,
            iterations=iterations,
            max_iterations=max_iterations,
            converged=converged,
            prior_force_norm=prior_force_norm,
            data_force_norm=data_force_norm,
            action_margin=action_margin,
        )
        z_evidence = self.get_z_evidence_from_pc(pc_signals)

        # 2. Z → PC: modulation 계산 (v5.13: with regime_change_score)
        modulation = self.get_modulation_for_pc(
            current_z, z_confidence, regime_change_score
        )

        return z_evidence, modulation

    def get_diagnostics(self) -> Dict:
        """진단 정보"""
        signals = self.state.current_pc_signals
        mod = self.state.current_z_modulation
        dpr = self.dynamic_past_regime.get_diagnostics()

        return {
            'total_steps': self.state.total_steps,
            'epsilon_baseline': self.state.epsilon_baseline,
            'regime_change_score_ema': self.state.regime_change_score_ema,
            'pc_signals': {
                'epsilon_spike': signals.epsilon_spike,
                'convergence_cost': signals.convergence_cost,
                'residual_error': signals.residual_error,
                'action_margin': signals.action_margin,
                'error_velocity': signals.error_velocity,
            },
            'z_modulation': {
                'error_gain': mod.error_gain,
                'prior_precision': mod.prior_precision,
                'iteration_multiplier': mod.iteration_multiplier,
                'past_regime_weight': mod.past_regime_weight,
            },
            # v5.13: Dynamic past_regime diagnostics
            'dynamic_past_regime': dpr,
        }

    def get_monitoring_report(self) -> Dict:
        """
        v5.13/v5.14 모니터링 리포트

        Returns:
            {
                'time_series': {...},
                'early_recovery': {...},
                'zone_distribution': {...},
                'phase_analysis': {...},
                'v514_recommendation': (bool, reason),
                'pc_gate_status': {...},  # v5.14 only
            }
        """
        report = self.dynamic_past_regime.get_monitoring_report()
        should_upgrade, reason = self.dynamic_past_regime.should_upgrade_to_v514()
        report['v514_recommendation'] = {
            'should_upgrade': should_upgrade,
            'reason': reason,
        }

        # v5.14: PC gate openness 노출 (운영 대시보드용)
        dpr = self.dynamic_past_regime
        cfg = dpr.config
        if cfg.use_pc_gate:
            stability_internal = dpr.compute_internal_stability()
            report['pc_gate_status'] = {
                'version': 'v5.14',
                'gate_openness': stability_internal,  # 0~1, 1이면 full recovery 허용
                'residual_ema': dpr.residual_ema,
                'gate_min': cfg.residual_gate_min,
                'gate_max': cfg.residual_gate_max,
                'interpretation': self._interpret_gate_status(stability_internal, dpr.residual_ema),
            }
        else:
            report['pc_gate_status'] = {
                'version': 'v5.13',
                'gate_openness': 1.0,  # v5.13은 항상 열림
                'note': 'v5.13 uses external score only, no residual gating',
            }

        return report

    def _interpret_gate_status(self, openness: float, residual_ema: float) -> str:
        """PC gate 상태를 한 줄로 해석"""
        if openness >= 0.9:
            return f"OPEN - residual({residual_ema:.2f}) below min, full recovery allowed"
        elif openness <= 0.1:
            return f"CLOSED - residual({residual_ema:.2f}) above max, recovery blocked"
        else:
            return f"PARTIAL({openness:.0%}) - residual({residual_ema:.2f}) in transition zone"

    def reset(self):
        """상태 초기화"""
        self.state = PCZDynamicsState()
        self.dynamic_past_regime.reset()


# =============================================================================
# Test Functions
# =============================================================================

def test_pc_signals():
    """PC 신호 계산 테스트"""
    print("=== PC Signals Test ===")
    dynamics = PCZDynamics()

    # 정상 상태
    signals = dynamics.compute_pc_signals(
        epsilon=np.ones(8) * 0.3,
        error_norm=0.3,
        iterations=15,
        max_iterations=30,
        converged=True,
        prior_force_norm=0.2,
        data_force_norm=0.3,
        action_margin=0.8,
    )
    print(f"Normal: spike={signals.epsilon_spike:.3f}, "
          f"conv_cost={signals.convergence_cost:.3f}, "
          f"residual={signals.residual_error:.3f}")

    # 스파이크 상태 (갑자기 오차 급등)
    for _ in range(30):  # baseline 구축
        dynamics.compute_pc_signals(
            epsilon=np.ones(8) * 0.3,
            error_norm=0.3,
            iterations=15,
            max_iterations=30,
            converged=True,
            prior_force_norm=0.2,
            data_force_norm=0.3,
            action_margin=0.8,
        )

    signals_spike = dynamics.compute_pc_signals(
        epsilon=np.ones(8) * 0.9,  # 3배 급등
        error_norm=0.9,
        iterations=28,  # 거의 최대
        max_iterations=30,
        converged=True,
        prior_force_norm=0.1,
        data_force_norm=0.5,
        action_margin=0.2,  # 낮은 마진 (갈등)
    )
    print(f"Spike:  spike={signals_spike.epsilon_spike:.3f}, "
          f"conv_cost={signals_spike.convergence_cost:.3f}, "
          f"residual={signals_spike.residual_error:.3f}")

    # z-evidence 변환
    z_evidence = signals_spike.to_z_evidence()
    print(f"Z-evidence: z0={z_evidence[0]:.3f}, z1={z_evidence[1]:.3f}, "
          f"z2={z_evidence[2]:.3f}, z3={z_evidence[3]:.3f}")


def test_z_modulation():
    """Z → PC modulation 테스트"""
    print("\n=== Z Modulation Test ===")

    for z in range(4):
        mod = ZModulation.from_z_state(z, z_confidence=0.8)
        z_names = ['stable', 'exploring', 'reflecting', 'fatigued']
        print(f"z={z} ({z_names[z]}): "
              f"error_gain={mod.error_gain:.2f}, "
              f"prior_prec={mod.prior_precision:.2f}, "
              f"iter_mult={mod.iteration_multiplier:.2f}")


def test_bidirectional():
    """양방향 동역학 테스트"""
    print("\n=== Bidirectional Dynamics Test ===")
    dynamics = PCZDynamics()

    # 시뮬레이션
    z = 0
    for step in range(100):
        # PC 상태 생성 (점점 오차 증가하다가 spike)
        if step < 40:
            error = 0.3 + step * 0.005
        elif step < 50:
            error = 0.5 + (step - 40) * 0.05  # spike
        else:
            error = 0.8 - (step - 50) * 0.01  # 회복

        z_evidence, modulation = dynamics.step(
            epsilon=np.ones(8) * error,
            error_norm=error,
            iterations=int(15 + error * 10),
            max_iterations=30,
            converged=True,
            prior_force_norm=0.2,
            data_force_norm=error,
            action_margin=max(0.1, 1.0 - error),
            current_z=z,
            z_confidence=0.8,
        )

        # Z 전환 시뮬레이션 (단순화)
        if z_evidence[1] > z_evidence[0] + 0.1:
            z = 1
        elif z_evidence[0] > z_evidence[1] + 0.1:
            z = 0

        if step % 20 == 0:
            print(f"Step {step}: error={error:.2f}, z={z}, "
                  f"z1_ev={z_evidence[1]:.3f}, "
                  f"gain={modulation.error_gain:.2f}, "
                  f"prior={modulation.prior_precision:.2f}")

    print("\nDiagnostics:", dynamics.get_diagnostics())


if __name__ == "__main__":
    test_pc_signals()
    test_z_modulation()
    test_bidirectional()
