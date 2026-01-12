"""
E3 Delayed Reward Validation

목표:
1. Delayed reward에서도 regime_score/PC residual이 '환경 변화'와 '보상 지연'을 구분
2. Memory/prior/recall이 "도움될 때만" 개입 (wrong-confidence 재발 금지)

Usage:
    python test_e3_delayed_reward.py              # 기본 테스트
    python test_e3_delayed_reward.py --seeds 100  # 더 많은 seeds
    python test_e3_delayed_reward.py --delay delayed_food  # 특정 시나리오만
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.pc_z_dynamics import PCZDynamics
from genesis.delayed_reward import (
    DelayType, DelayConfig, DelayedRewardApplicator,
    E3Gate, E3GateResult, E3RunStats, DELAY_LEVELS
)


def run_single_delay_config(
    seed: int,
    delay_config: DelayConfig,
    use_memory: bool = True,
    steps: int = 500,
) -> E3RunStats:
    """단일 Delay 설정으로 실행"""
    np.random.seed(seed)
    dynamics = PCZDynamics()
    delay_applicator = DelayedRewardApplicator(delay_config, seed=seed)

    residuals = []
    epsilon_spikes = []
    rewards = []

    # 적응/회복 추적
    adaptation_steps = 0
    in_adaptation = False  # starts false, becomes true when shock hits
    shock_started = False
    recovery_success = False
    min_residual_after_shock = 1.0

    # Wrong-confidence 추적
    early_recovery_count = 0
    phase_order_violations = 0
    last_residual = 0.5
    last_openness = 0.0
    in_recovery_phase = False

    # 메모리 효과 시뮬레이션
    memory_bias = 0.0

    for step in range(steps):
        # 기본 시나리오: stable → shock → recovery
        if step < 100:
            base_residual = 0.15 + np.random.randn() * 0.03
            score = 0.1
            immediate_reward = 0.1 + np.random.randn() * 0.02
            is_shock_period = False
        elif step < 150:
            base_residual = 0.6 + np.random.randn() * 0.05
            score = 0.7
            immediate_reward = -0.1 + np.random.randn() * 0.02  # shock 동안 음수 보상
            is_shock_period = True
            if not shock_started:
                shock_started = True
                in_adaptation = True  # Start counting adaptation from shock
        else:
            recovery = min(1.0, (step - 150) / 100)
            base_residual = max(0.15, 0.6 - recovery * 0.45)
            score = max(0.1, 0.7 - recovery * 0.6)
            immediate_reward = 0.1 * recovery + np.random.randn() * 0.02
            is_shock_period = False

        base_residual = np.clip(base_residual, 0.05, 1.0)
        score = np.clip(score, 0.02, 0.95)

        # Delay 적용
        effective_reward, is_delayed_event, info = delay_applicator.apply(
            immediate_reward,
            action=np.random.randint(0, 5)
        )

        rewards.append(effective_reward)

        # 메모리 효과 시뮬레이션
        if use_memory:
            # 지연 보상 도착 시 메모리 업데이트
            if is_delayed_event and effective_reward > 0:
                memory_bias = 0.8 * memory_bias + 0.2 * effective_reward
            elif is_delayed_event and effective_reward < 0:
                memory_bias = 0.8 * memory_bias + 0.2 * effective_reward * 0.5

            # 메모리 기반 residual 조정 (도움이 되는 경우만)
            if not is_shock_period and memory_bias > 0:
                # 안정 시기에 긍정적 메모리는 도움됨
                base_residual *= (1.0 - memory_bias * 0.1)
        else:
            memory_bias = 0.0

        # Delayed event로 인한 residual spike (지연 보상 도착)
        if is_delayed_event:
            # 지연 보상 도착 = 예측 오차 발생
            delay_residual_boost = abs(effective_reward) * 0.2
            base_residual = np.clip(base_residual + delay_residual_boost, 0.05, 1.0)

        # PC 신호 계산
        epsilon = np.ones(8) * base_residual
        signals = dynamics.compute_pc_signals(
            epsilon=epsilon,
            error_norm=base_residual,
            iterations=int(10 + base_residual * 10),
            max_iterations=30,
            converged=True,
            prior_force_norm=0.2,
            data_force_norm=base_residual,
            action_margin=0.5,
        )

        # Z-state 결정
        if score > 0.6:
            z = 2
        elif score > 0.3:
            z = 1
        else:
            z = 0

        mod = dynamics.get_modulation_for_pc(
            z=z,
            z_confidence=0.7,
            regime_change_score=score
        )

        residuals.append(signals.residual_error)
        epsilon_spikes.append(signals.epsilon_spike)

        # 적응 추적
        current_residual = signals.residual_error
        # Get openness from dynamic_past_regime
        current_openness = dynamics.dynamic_past_regime.compute_internal_stability()

        # Adaptation: time for residual to drop below 0.35 after shock starts
        # Note: residual_error = error_norm/0.5, so 0.35 corresponds to error_norm ~0.175
        if in_adaptation and current_residual < 0.35:
            in_adaptation = False
            adaptation_steps = step - 100  # steps since shock started

        # Recovery: track min residual after shock to check if recovery happens
        if shock_started and step >= 150:  # After shock period
            min_residual_after_shock = min(min_residual_after_shock, current_residual)
            if current_residual < 0.35:
                recovery_success = True

        # Wrong-confidence 체크
        # Early recovery: residual 높은데 openness 오르면 위반
        if current_residual > 0.5 and current_openness > 0.3:
            if last_openness < 0.2:  # openness가 갑자기 올랐으면
                early_recovery_count += 1

        # Phase order: residual 떨어지기 전에 openness 오르면 위반
        if not in_recovery_phase and current_residual < 0.4:
            in_recovery_phase = True

        if in_recovery_phase:
            # 정상: residual↓ → openness↑ 순서
            # 위반: openness↑ → residual↓ 순서
            if current_openness > last_openness + 0.1 and current_residual > last_residual:
                phase_order_violations += 1

        last_residual = current_residual
        last_openness = current_openness

    # 적응 시간이 설정 안 됐으면 전체 스텝 (shock 이후)
    if in_adaptation and shock_started:
        adaptation_steps = steps - 100  # max time since shock
    elif not shock_started:
        adaptation_steps = 0  # no shock occurred

    return E3RunStats(
        delay_type=delay_config.delay_type.value,
        delay_steps=delay_config.delay_steps,
        seed=seed,
        use_memory=use_memory,
        total_reward=sum(rewards),
        avg_reward=np.mean(rewards),
        reward_variance=np.var(rewards),
        adaptation_time=adaptation_steps,
        recovery_success=recovery_success,
        early_recovery_count=early_recovery_count,
        phase_order_violations=phase_order_violations,
        residual_mean=np.mean(residuals),
        epsilon_spike_rate=np.mean([1 if e > 0.5 else 0 for e in epsilon_spikes]),
    )


def run_e3_validation(
    n_seeds: int = 30,
    steps: int = 500,
    delay_filter: str = None,  # "delayed_food", "trap_credit", "po_delay" or None
) -> Dict:
    """E3 검증 실행"""
    print(f"\n{'='*60}")
    print(f"  E3 Delayed Reward Validation")
    print(f"  {n_seeds} seeds, {steps} steps")
    if delay_filter:
        print(f"  Filter: {delay_filter} only")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Seeds
    fixed_seeds = [11, 23, 37]
    additional = [i for i in range(200) if i not in fixed_seeds]
    seeds = fixed_seeds + additional[:n_seeds - 3]

    # Delay types to test
    delay_types_to_test = []
    if delay_filter is None or delay_filter == "delayed_food":
        delay_types_to_test.append(DelayType.DELAYED_FOOD)
    if delay_filter is None or delay_filter == "trap_credit":
        delay_types_to_test.append(DelayType.TRAP_CREDIT)
    if delay_filter is None or delay_filter == "po_delay":
        delay_types_to_test.append(DelayType.PO_DELAY)

    all_results = {}
    gate = E3Gate()

    for delay_type in delay_types_to_test:
        print(f"\n--- Testing {delay_type.value} ---")

        delay_levels = DELAY_LEVELS[delay_type]

        # With memory
        stats_with_memory: Dict[int, List[E3RunStats]] = {}
        # Without memory
        stats_without_memory: Dict[int, List[E3RunStats]] = {}

        for delay_steps in delay_levels:
            if delay_type == DelayType.DELAYED_FOOD:
                config = DelayConfig.delayed_food(delay_steps)
            elif delay_type == DelayType.TRAP_CREDIT:
                config = DelayConfig.trap_credit(delay_steps)
            else:
                config = DelayConfig.po_delay(delay_steps)

            stats_with_memory[delay_steps] = []
            stats_without_memory[delay_steps] = []

            print(f"  Delay={delay_steps}...", end=" ")

            for seed in seeds:
                # With memory
                stats_mem = run_single_delay_config(seed, config, use_memory=True, steps=steps)
                stats_with_memory[delay_steps].append(stats_mem)

                # Without memory
                stats_no_mem = run_single_delay_config(seed, config, use_memory=False, steps=steps)
                stats_without_memory[delay_steps].append(stats_no_mem)

            print(f"done ({len(seeds)} seeds x 2)")

        # Baseline adaptation time (delay=0)
        baseline_stats = stats_with_memory[0]
        baseline_adaptation_time = np.mean([s.adaptation_time for s in baseline_stats])

        # Evaluate gate
        gate_result = gate.evaluate(
            stats_with_memory,
            stats_without_memory,
            baseline_adaptation_time
        )

        all_results[delay_type.value] = {
            'stats_with_memory': {k: [s.__dict__ for s in v] for k, v in stats_with_memory.items()},
            'stats_without_memory': {k: [s.__dict__ for s in v] for k, v in stats_without_memory.items()},
            'gate_result': gate_result,
        }

        # Print results
        print(f"\n  E3a Adaptation:")
        print(f"    Baseline adaptation time: {baseline_adaptation_time:.0f} steps")
        print(f"    Max degradation: {gate_result.adaptation_degradation:.1f}x (max: 3.0x)")
        print(f"    Recovery rate: {gate_result.recovery_rate:.1%} (min: 70%)")
        print(f"    [{'PASS' if gate_result.e3a_passed else 'FAIL'}]")

        print(f"\n  E3b Wrong-confidence:")
        print(f"    Early recovery rate: {gate_result.early_recovery_rate:.1%} (max: 5%)")
        print(f"    Phase violation rate: {gate_result.phase_violation_rate:.1%} (max: 10%)")
        print(f"    [{'PASS' if gate_result.e3b_passed else 'FAIL'}]")

        print(f"\n  E3c Memory Benefit:")
        print(f"    Memory benefit ratio: {gate_result.memory_benefit_ratio:.2f} (min: 1.0)")
        print(f"    [{'PASS' if gate_result.e3c_passed else 'FAIL'}]")

        print(f"\n  Overall: [{'PASS' if gate_result.passed else 'FAIL'}] {gate_result.reason}")

        # Print delay-wise stats
        print(f"\n  Delay-wise stats (with memory):")
        print(f"  {'Delay':<8} {'Adapt':<8} {'Recover':<10} {'EarlyRec':<10} {'PhaseVio':<10} {'AvgRwd':<10}")
        print(f"  {'-'*56}")
        for delay in sorted(stats_with_memory.keys()):
            stats_list = stats_with_memory[delay]
            avg_adapt = np.mean([s.adaptation_time for s in stats_list])
            recovery_rate = np.mean([1 if s.recovery_success else 0 for s in stats_list])
            early_rec = np.mean([s.early_recovery_count for s in stats_list])
            phase_vio = np.mean([s.phase_order_violations for s in stats_list])
            avg_rwd = np.mean([s.avg_reward for s in stats_list])
            print(f"  {delay:<8} {avg_adapt:<8.0f} {recovery_rate:<10.1%} {early_rec:<10.2f} {phase_vio:<10.2f} {avg_rwd:<10.3f}")

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Final Summary")
    print(f"{'='*60}\n")
    print(f"Time: {elapsed:.1f}s")

    all_passed = True
    for delay_type, result in all_results.items():
        gate_result = result['gate_result']
        status = "PASS" if gate_result.passed else "FAIL"
        print(f"  {delay_type}: [{status}] {gate_result.reason}")
        if not gate_result.passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"  [PASS] E3 Gate passed for all delay types")
    else:
        print(f"  [FAIL] E3 Gate failed for some delay types")
    print(f"{'='*60}\n")

    return {
        'n_seeds': n_seeds,
        'results': all_results,
        'all_passed': all_passed,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds")
    parser.add_argument("--steps", type=int, default=500, help="Steps per run")
    parser.add_argument("--delay", type=str, default=None,
                        choices=["delayed_food", "trap_credit", "po_delay"],
                        help="Test only specific delay type")
    args = parser.parse_args()

    results = run_e3_validation(
        n_seeds=args.seeds,
        steps=args.steps,
        delay_filter=args.delay,
    )
    exit(0 if results['all_passed'] else 1)
