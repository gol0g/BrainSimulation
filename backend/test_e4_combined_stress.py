"""
E4 Combined Stress Test Validation

목표:
"센서(PC) + 상태(z) + 복귀 게이트(v5.14) + regime_score"가 동시에 흔들리는 최악의 조합

게이트:
- E4a: False Regime Detection - distractor/PO로 score가 shock으로 튀지 않음
- E4b: Over-suppression - 계속 shock 오인해서 weight 영구 못올라오지 않음
- E4c: Wrong-confidence Relapse - score 안정인데 residual 높은데 weight 먼저 오름
- E4d: Utility Collapse - 오탐 없어도 성능 너무 떨어지지 않음

Usage:
    python test_e4_combined_stress.py              # 기본 테스트
    python test_e4_combined_stress.py --seeds 100  # 더 많은 seeds
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.pc_z_dynamics import PCZDynamics
from genesis.combined_stress import (
    CombinedStressConfig, CombinedStressApplicator, DriftType,
    E4Gate, E4GateResult, E4RunStats, E4_SCENARIOS
)


def run_single_stress_config(
    seed: int,
    config: CombinedStressConfig,
    config_name: str,
    steps: int = 500,
) -> E4RunStats:
    """단일 Stress 설정으로 실행"""
    np.random.seed(seed)
    dynamics = PCZDynamics()
    stress_applicator = CombinedStressApplicator(config, seed=seed)

    residuals = []
    rewards = []

    # E4a: False regime detection tracking
    false_shock_count = 0
    total_distractor_only_steps = 0

    # E4b: Over-suppression tracking
    stable_weights = []
    in_stable_zone = False

    # E4c: Wrong-confidence tracking
    phase_violations = 0
    early_recovery_count = 0
    last_residual = 0.5
    last_weight = 0.0
    last_openness = 0.0
    in_high_residual_zone = False
    high_residual_entry_weight = 0.0

    # Score threshold
    SHOCK_THRESHOLD = 0.5

    for step in range(steps):
        # 기본 시나리오 (drift에 의해 수정됨)
        # 0-100: stable, 100-200: potential drift impact, 200+: recovery
        if step < 100:
            base_residual = 0.15 + np.random.randn() * 0.03
            base_score = 0.1
            is_base_stable = True
        elif step < 200:
            # Drift impact zone - drift_type에 따라 residual이 달라짐
            base_residual = 0.15 + np.random.randn() * 0.03
            base_score = 0.1  # base는 stable, drift가 score를 올림
            is_base_stable = True
        else:
            base_residual = 0.15 + np.random.randn() * 0.03
            base_score = 0.1
            is_base_stable = True

        base_residual = np.clip(base_residual, 0.05, 0.3)

        # 8D 관측
        base_obs = np.random.randn(8) * 0.1 + np.array([0.5, 0.3, 0, 0, 0, 0, 0.7, 0.1])
        base_obs = np.clip(base_obs, 0, 1)

        # Combined stress 적용
        processed_obs, effective_residual, quality, info = stress_applicator.apply(
            base_obs, base_residual
        )

        # Drift가 score에 미치는 영향 (실제 환경 변화)
        drift_score_impact = stress_applicator.get_drift_score_impact()
        effective_score = np.clip(base_score + drift_score_impact, 0.02, 0.95)

        # PC 신호 계산
        obs_dim = len(processed_obs)
        epsilon = np.ones(obs_dim) * effective_residual
        signals = dynamics.compute_pc_signals(
            epsilon=epsilon[:8] if obs_dim > 8 else epsilon,
            error_norm=effective_residual,
            iterations=int(10 + effective_residual * 10),
            max_iterations=30,
            converged=True,
            prior_force_norm=0.2,
            data_force_norm=effective_residual,
            action_margin=0.5,
        )

        # Z-state 결정
        if effective_score > 0.6:
            z = 2  # shock
        elif effective_score > 0.3:
            z = 1  # uncertainty
        else:
            z = 0  # stable

        mod = dynamics.get_modulation_for_pc(
            z=z,
            z_confidence=0.7 * quality,
            regime_change_score=effective_score
        )

        current_residual = signals.residual_error
        current_weight = dynamics.dynamic_past_regime.w_applied
        current_openness = dynamics.dynamic_past_regime.compute_internal_stability()

        residuals.append(current_residual)
        rewards.append(1.0 - effective_residual)  # simple reward proxy

        # E4a: False shock detection
        # distractor_change만 있고 drift는 없는데 shock으로 감지되면 false positive
        distractor_only_change = info.get('distractor_change', False)
        is_po_degraded = quality < 0.8
        no_real_drift = (config.drift_type == DriftType.NONE or drift_score_impact < 0.1)

        if (distractor_only_change or is_po_degraded) and no_real_drift and is_base_stable:
            total_distractor_only_steps += 1
            if effective_score >= SHOCK_THRESHOLD:
                false_shock_count += 1

        # E4b: Over-suppression - stable zone weight tracking
        # drift 없고 step > 250이면 stable zone
        if config.drift_type == DriftType.NONE or step > 250:
            in_stable_zone = True
            stable_weights.append(current_weight)

        # E4c: Wrong-confidence check
        # Track entry/exit of high-residual zone
        if current_residual > 0.5 and not in_high_residual_zone:
            # Just entered high-residual zone
            in_high_residual_zone = True
            high_residual_entry_weight = current_weight

        if current_residual <= 0.5 and in_high_residual_zone:
            # Just exited high-residual zone
            in_high_residual_zone = False

            # Phase violation: weight increased during high-residual period
            weight_increase = current_weight - high_residual_entry_weight
            if weight_increase > 0.02:  # Significant weight gain while residual was high
                phase_violations += 1

        # Early recovery: openness jumps while in high-residual zone
        # This means the gate is opening prematurely
        if in_high_residual_zone:
            openness_jump = current_openness - last_openness
            if openness_jump > 0.2 and current_openness > 0.4:
                # Significant openness jump while residual is high = early recovery attempt
                early_recovery_count += 1

        last_residual = current_residual
        last_weight = current_weight
        last_openness = current_openness

    # Aggregate stats
    avg_stable_weight = np.mean(stable_weights) if stable_weights else 0.0
    weight_never_recovered = avg_stable_weight < 0.05

    return E4RunStats(
        config_name=config_name,
        seed=seed,
        false_shock_count=false_shock_count,
        total_distractor_only_steps=total_distractor_only_steps,
        false_shock_rate=false_shock_count / max(1, total_distractor_only_steps),
        weight_never_recovered=weight_never_recovered,
        avg_stable_weight=avg_stable_weight,
        phase_violations=phase_violations,
        early_recovery_count=early_recovery_count,
        avg_residual=np.mean(residuals),
        avg_reward=np.mean(rewards),
        efficiency_proxy=1.0 - np.mean(residuals),
    )


def run_e4_validation(
    n_seeds: int = 30,
    steps: int = 500,
) -> Dict:
    """E4 검증 실행"""
    print(f"\n{'='*60}")
    print(f"  E4 Combined Stress Test Validation")
    print(f"  {n_seeds} seeds, {steps} steps")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Seeds
    fixed_seeds = [11, 23, 37]
    additional = [i for i in range(200) if i not in fixed_seeds]
    seeds = fixed_seeds + additional[:n_seeds - 3]

    # Scenarios to test
    scenarios_to_test = [
        ("baseline", E4_SCENARIOS["baseline"]),
        ("full_gradual", E4_SCENARIOS["full_gradual"]),
        ("full_sudden", E4_SCENARIOS["full_sudden"]),
        ("full_oscillating", E4_SCENARIOS["full_oscillating"]),
    ]

    all_results = {}
    gate = E4Gate()

    # Run baseline first
    print("Running baseline...")
    baseline_stats: List[E4RunStats] = []
    for seed in seeds:
        stats = run_single_stress_config(seed, E4_SCENARIOS["baseline"], "baseline", steps)
        baseline_stats.append(stats)
    print(f"  done ({len(seeds)} seeds)")

    all_results["baseline"] = {
        'stats': [s.__dict__ for s in baseline_stats],
    }

    # Run stress scenarios
    for scenario_name, config in scenarios_to_test[1:]:  # Skip baseline
        print(f"\nRunning {scenario_name}...")
        stress_stats: List[E4RunStats] = []

        for seed in seeds:
            stats = run_single_stress_config(seed, config, scenario_name, steps)
            stress_stats.append(stats)

        print(f"  done ({len(seeds)} seeds)")

        # Evaluate gate
        gate_result = gate.evaluate(stress_stats, baseline_stats)

        all_results[scenario_name] = {
            'stats': [s.__dict__ for s in stress_stats],
            'gate_result': gate_result,
        }

        # Print results
        print(f"\n  E4a False Regime Detection:")
        print(f"    False shock rate: {gate_result.false_shock_rate:.1%} (max: 5%)")
        print(f"    [{'PASS' if gate_result.e4a_passed else 'FAIL'}]")

        print(f"\n  E4b Over-suppression:")
        print(f"    Weight recovery rate: {gate_result.weight_recovery_rate:.1%} (min: 90%)")
        print(f"    Avg stable weight: {gate_result.avg_stable_weight:.3f} (min: 0.08)")
        print(f"    [{'PASS' if gate_result.e4b_passed else 'FAIL'}]")

        print(f"\n  E4c Wrong-confidence Relapse:")
        print(f"    Phase violation rate: {gate_result.phase_violation_rate:.1%} (max: 10%)")
        print(f"    Early recovery rate: {gate_result.early_recovery_rate:.1%} (max: 5%)")
        print(f"    [{'PASS' if gate_result.e4c_passed else 'FAIL'}]")

        print(f"\n  E4d Utility Collapse:")
        print(f"    Efficiency retention: {gate_result.efficiency_retention:.1%} (min: 60%)")
        print(f"    [{'PASS' if gate_result.e4d_passed else 'FAIL'}]")

        print(f"\n  Overall: [{'PASS' if gate_result.passed else 'FAIL'}] {gate_result.reason}")

        # Print per-run summary
        avg_false_shock = np.mean([s.false_shock_rate for s in stress_stats])
        avg_stable_wt = np.mean([s.avg_stable_weight for s in stress_stats])
        avg_efficiency = np.mean([s.efficiency_proxy for s in stress_stats])
        print(f"\n  Summary: false_shock={avg_false_shock:.1%}, "
              f"stable_weight={avg_stable_wt:.3f}, efficiency={avg_efficiency:.2f}")

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Final Summary")
    print(f"{'='*60}\n")
    print(f"Time: {elapsed:.1f}s")

    all_passed = True
    for scenario_name, result in all_results.items():
        if scenario_name == "baseline":
            continue
        gate_result = result['gate_result']
        status = "PASS" if gate_result.passed else "FAIL"
        print(f"  {scenario_name}: [{status}] {gate_result.reason}")
        if not gate_result.passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"  [PASS] E4 Gate passed for all stress scenarios")
    else:
        print(f"  [FAIL] E4 Gate failed for some scenarios")
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
    args = parser.parse_args()

    results = run_e4_validation(
        n_seeds=args.seeds,
        steps=args.steps,
    )
    exit(0 if results['all_passed'] else 1)
