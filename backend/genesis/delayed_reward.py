"""
E3: Delayed Reward - Temporal Credit Assignment

목표:
- Delayed reward에서도 regime_score/PC residual이 '환경 변화'와 '보상 지연'을 구분
- Memory/prior/recall이 "도움될 때만" 개입 (wrong-confidence 재발 금지)

3종 시나리오:
- Delayed Food: 음식을 먹어도 reward가 N step 뒤에 반영 (N=5/10/20)
- Trap Credit: 당장은 좋아 보이나 10 step 후 손해로 돌아옴
- PO + Delay: Partial Observability와 결합

게이트:
- E3a: Adaptation - delay 커져도 적응/회복 패턴 유지
- E3b: Wrong-confidence - early_recovery_with_cost, phase order 위반 없음
- E3c: Memory Benefit - memory 있을 때가 없을 때보다 나음
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum


class DelayType(Enum):
    """Delay 종류"""
    DELAYED_FOOD = "delayed_food"      # 보상 지연
    TRAP_CREDIT = "trap_credit"        # 당장 좋으나 나중에 손해
    PO_DELAY = "po_delay"              # PO + Delay 결합


@dataclass
class DelayConfig:
    """Delay 설정"""
    delay_type: DelayType
    delay_steps: int = 10  # 보상 지연 스텝

    # Delayed Food 설정
    food_reward: float = 0.3  # 음식 먹을 때 보상

    # Trap Credit 설정
    trap_immediate_gain: float = 0.2   # 즉시 이득
    trap_delayed_cost: float = 0.5     # 지연 손해
    trap_trigger_prob: float = 0.1     # trap 발생 확률

    # PO + Delay 설정
    po_noise_std: float = 0.1  # PO 노이즈

    @classmethod
    def delayed_food(cls, n: int) -> 'DelayConfig':
        """Delayed Food 설정 (N=5/10/20)"""
        return cls(
            delay_type=DelayType.DELAYED_FOOD,
            delay_steps=n,
        )

    @classmethod
    def trap_credit(cls, n: int = 10) -> 'DelayConfig':
        """Trap Credit 설정"""
        return cls(
            delay_type=DelayType.TRAP_CREDIT,
            delay_steps=n,
        )

    @classmethod
    def po_delay(cls, n: int = 10, noise: float = 0.1) -> 'DelayConfig':
        """PO + Delay 결합 설정"""
        return cls(
            delay_type=DelayType.PO_DELAY,
            delay_steps=n,
            po_noise_std=noise,
        )


class DelayedRewardApplicator:
    """
    Delayed Reward 적용기

    보상을 N step 뒤에 반영하거나, trap credit을 시뮬레이션
    """

    def __init__(self, config: DelayConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(seed)

        # Delayed reward 버퍼
        self.reward_buffer: Deque[float] = deque(maxlen=config.delay_steps + 10)
        self.pending_rewards: List[Tuple[int, float]] = []  # (step_to_apply, reward)

        # Trap 상태 추적
        self.trap_active = False
        self.trap_countdown = 0
        self.trap_events: List[int] = []  # trap 발동한 step들

        # 스텝 카운터
        self.step = 0

    def apply(
        self,
        immediate_reward: float,
        action: int = 0,
    ) -> Tuple[float, bool, Dict]:
        """
        Delayed reward 적용

        Args:
            immediate_reward: 즉시 보상 (원래 시스템이 계산한)
            action: 선택한 행동

        Returns:
            effective_reward: 실제 적용될 보상
            is_delayed_event: 지연된 보상이 이번 스텝에 도착했는지
            info: 추가 정보
        """
        self.step += 1
        cfg = self.config

        if cfg.delay_type == DelayType.DELAYED_FOOD:
            return self._apply_delayed_food(immediate_reward)
        elif cfg.delay_type == DelayType.TRAP_CREDIT:
            return self._apply_trap_credit(immediate_reward, action)
        elif cfg.delay_type == DelayType.PO_DELAY:
            return self._apply_po_delay(immediate_reward)
        else:
            return immediate_reward, False, {}

    def _apply_delayed_food(self, immediate_reward: float) -> Tuple[float, bool, Dict]:
        """보상 지연 적용"""
        cfg = self.config

        # 보상을 버퍼에 추가 (N step 후 반영)
        if immediate_reward > 0.1:  # 의미있는 보상만 지연
            apply_at = self.step + cfg.delay_steps
            self.pending_rewards.append((apply_at, immediate_reward))

        # 현재 스텝에 도착한 지연 보상 확인
        delayed_reward = 0.0
        is_delayed_event = False
        arrived = [r for r in self.pending_rewards if r[0] <= self.step]

        for apply_at, reward in arrived:
            delayed_reward += reward
            is_delayed_event = True
            self.pending_rewards.remove((apply_at, reward))

        # 즉시 보상은 0으로, 지연 보상만 반환
        effective_reward = delayed_reward

        return effective_reward, is_delayed_event, {
            'pending_count': len(self.pending_rewards),
            'delayed_reward': delayed_reward,
        }

    def _apply_trap_credit(
        self,
        immediate_reward: float,
        action: int
    ) -> Tuple[float, bool, Dict]:
        """Trap credit 적용 - 즉시 이득, 나중에 손해"""
        cfg = self.config

        # delay_steps=0이면 trap 안 발동 (baseline)
        if cfg.delay_steps == 0:
            return immediate_reward, False, {}

        # Trap 발동 체크 (일정 확률로)
        if not self.trap_active and self.rng.random() < cfg.trap_trigger_prob:
            self.trap_active = True
            self.trap_countdown = cfg.delay_steps
            self.trap_events.append(self.step)

            # 즉시 이득 추가
            return immediate_reward + cfg.trap_immediate_gain, False, {
                'trap_triggered': True,
                'trap_countdown': self.trap_countdown,
            }

        # Trap 카운트다운
        if self.trap_active:
            self.trap_countdown -= 1

            if self.trap_countdown <= 0:
                # 지연 손해 적용
                self.trap_active = False
                return immediate_reward - cfg.trap_delayed_cost, True, {
                    'trap_cost_applied': True,
                    'cost': cfg.trap_delayed_cost,
                }

        return immediate_reward, False, {}

    def _apply_po_delay(self, immediate_reward: float) -> Tuple[float, bool, Dict]:
        """PO + Delay 결합"""
        cfg = self.config

        # 노이즈 추가 (PO 효과)
        noise = self.rng.randn() * cfg.po_noise_std
        noisy_reward = immediate_reward + noise

        # Delay 적용 (delayed_food 로직 재사용)
        if noisy_reward > 0.1:
            apply_at = self.step + cfg.delay_steps
            self.pending_rewards.append((apply_at, noisy_reward))

        # 지연 보상 확인
        delayed_reward = 0.0
        is_delayed_event = False
        arrived = [r for r in self.pending_rewards if r[0] <= self.step]

        for apply_at, reward in arrived:
            delayed_reward += reward
            is_delayed_event = True
            self.pending_rewards.remove((apply_at, reward))

        return delayed_reward, is_delayed_event, {
            'noise_applied': noise,
            'pending_count': len(self.pending_rewards),
        }

    def reset(self):
        """상태 리셋"""
        self.reward_buffer.clear()
        self.pending_rewards = []
        self.trap_active = False
        self.trap_countdown = 0
        self.trap_events = []
        self.step = 0


@dataclass
class E3RunStats:
    """E3 단일 실행 통계"""
    delay_type: str
    delay_steps: int
    seed: int
    use_memory: bool

    # 성과 지표
    total_reward: float
    avg_reward: float
    reward_variance: float

    # 적응 패턴
    adaptation_time: int  # 안정화까지 걸린 스텝
    recovery_success: bool  # shock 후 회복 성공 여부

    # Wrong-confidence 지표
    early_recovery_count: int
    phase_order_violations: int

    # PC 신호
    residual_mean: float
    epsilon_spike_rate: float


@dataclass
class E3GateResult:
    """E3 게이트 결과"""
    # E3a: Adaptation
    e3a_passed: bool
    adaptation_degradation: float  # delay에 따른 적응 시간 증가율
    recovery_rate: float  # 회복 성공률

    # E3b: Wrong-confidence
    e3b_passed: bool
    early_recovery_rate: float
    phase_violation_rate: float

    # E3c: Memory Benefit
    e3c_passed: bool
    memory_benefit_ratio: float  # memory 있을 때 / 없을 때

    # 전체 판정
    passed: bool
    reason: str


class E3Gate:
    """
    E3 Delayed Reward 게이트

    E3a: Adaptation - 적응/회복 패턴 유지
    E3b: Wrong-confidence - 조기 복귀 및 위상 위반 없음
    E3c: Memory Benefit - 메모리가 도움됨
    """

    def __init__(
        self,
        adaptation_degradation_max: float = 3.0,  # delay 10에서 적응 시간 3배까지 허용
        recovery_rate_min: float = 0.7,  # 70% 이상 회복 성공
        early_recovery_max: float = 0.05,  # 조기 복귀 5% 미만
        phase_violation_max: float = 0.1,  # 위상 위반 10% 미만
        memory_benefit_min: float = 1.0,  # 메모리 있을 때가 같거나 나음
    ):
        self.adaptation_degradation_max = adaptation_degradation_max
        self.recovery_rate_min = recovery_rate_min
        self.early_recovery_max = early_recovery_max
        self.phase_violation_max = phase_violation_max
        self.memory_benefit_min = memory_benefit_min

    def evaluate(
        self,
        stats_with_memory: Dict[int, List[E3RunStats]],  # delay -> stats list
        stats_without_memory: Dict[int, List[E3RunStats]],
        baseline_adaptation_time: float,  # delay=0에서의 적응 시간
    ) -> E3GateResult:
        """
        E3 게이트 평가

        Args:
            stats_with_memory: 메모리 활성화 상태의 통계
            stats_without_memory: 메모리 비활성화 상태의 통계
            baseline_adaptation_time: 기준 적응 시간 (delay=0)
        """
        # E3a: Adaptation
        max_degradation = 0.0
        recovery_rates = []

        for delay, stats_list in stats_with_memory.items():
            if delay == 0:
                continue

            avg_adaptation = np.mean([s.adaptation_time for s in stats_list])
            degradation = avg_adaptation / max(1, baseline_adaptation_time)
            max_degradation = max(max_degradation, degradation)

            recovery_rate = np.mean([1 if s.recovery_success else 0 for s in stats_list])
            recovery_rates.append(recovery_rate)

        avg_recovery_rate = np.mean(recovery_rates) if recovery_rates else 1.0
        e3a_passed = (
            max_degradation <= self.adaptation_degradation_max and
            avg_recovery_rate >= self.recovery_rate_min
        )

        # E3b: Wrong-confidence
        all_stats = []
        for stats_list in stats_with_memory.values():
            all_stats.extend(stats_list)

        total_runs = len(all_stats)
        total_early_recovery = sum(s.early_recovery_count for s in all_stats)
        total_phase_violations = sum(s.phase_order_violations for s in all_stats)

        early_recovery_rate = total_early_recovery / max(1, total_runs)
        phase_violation_rate = total_phase_violations / max(1, total_runs)

        e3b_passed = (
            early_recovery_rate <= self.early_recovery_max and
            phase_violation_rate <= self.phase_violation_max
        )

        # E3c: Memory Benefit
        memory_rewards = []
        no_memory_rewards = []

        for delay in stats_with_memory.keys():
            if delay in stats_without_memory:
                memory_rewards.extend([s.avg_reward for s in stats_with_memory[delay]])
                no_memory_rewards.extend([s.avg_reward for s in stats_without_memory[delay]])

        if memory_rewards and no_memory_rewards:
            avg_memory = np.mean(memory_rewards)
            avg_no_memory = np.mean(no_memory_rewards)
            memory_benefit_ratio = avg_memory / max(0.001, avg_no_memory)
        else:
            memory_benefit_ratio = 1.0

        e3c_passed = memory_benefit_ratio >= self.memory_benefit_min

        # 전체 판정
        passed = e3a_passed and e3b_passed and e3c_passed

        reasons = []
        if not e3a_passed:
            if max_degradation > self.adaptation_degradation_max:
                reasons.append(f"adaptation_degradation={max_degradation:.1f}x>{self.adaptation_degradation_max:.1f}x")
            if avg_recovery_rate < self.recovery_rate_min:
                reasons.append(f"recovery_rate={avg_recovery_rate:.1%}<{self.recovery_rate_min:.1%}")
        if not e3b_passed:
            if early_recovery_rate > self.early_recovery_max:
                reasons.append(f"early_recovery={early_recovery_rate:.1%}>{self.early_recovery_max:.1%}")
            if phase_violation_rate > self.phase_violation_max:
                reasons.append(f"phase_violations={phase_violation_rate:.1%}>{self.phase_violation_max:.1%}")
        if not e3c_passed:
            reasons.append(f"memory_benefit={memory_benefit_ratio:.2f}<{self.memory_benefit_min:.2f}")

        reason = "PASS" if passed else "; ".join(reasons)

        return E3GateResult(
            e3a_passed=e3a_passed,
            adaptation_degradation=max_degradation,
            recovery_rate=avg_recovery_rate,
            e3b_passed=e3b_passed,
            early_recovery_rate=early_recovery_rate,
            phase_violation_rate=phase_violation_rate,
            e3c_passed=e3c_passed,
            memory_benefit_ratio=memory_benefit_ratio,
            passed=passed,
            reason=reason,
        )


# Delay 레벨 정의
DELAY_LEVELS = {
    DelayType.DELAYED_FOOD: [0, 5, 10, 20],
    DelayType.TRAP_CREDIT: [0, 5, 10, 20],
    DelayType.PO_DELAY: [0, 5, 10, 20],
}


def create_delay_configs() -> List[DelayConfig]:
    """모든 Delay 설정 생성"""
    configs = []

    for n in DELAY_LEVELS[DelayType.DELAYED_FOOD]:
        configs.append(DelayConfig.delayed_food(n))

    for n in DELAY_LEVELS[DelayType.TRAP_CREDIT]:
        configs.append(DelayConfig.trap_credit(n))

    for n in DELAY_LEVELS[DelayType.PO_DELAY]:
        configs.append(DelayConfig.po_delay(n))

    return configs
