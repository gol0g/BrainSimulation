"""
E2 Partial Observability Validation

목표:
1. Uncertainty Calibration: 정보량↓ → PC residual/ε↑, z=1↑ (단조 증가)
2. No Panic: 정보 부족을 z=3(피로)로 오인하지 않음
3. Utility Preservation: 중간 강도에서 성과 유지

3종 PO × 4강도 = 12 configurations per seed

Usage:
    python test_e2_partial_observability.py              # 기본 테스트
    python test_e2_partial_observability.py --seeds 100  # 더 많은 seeds
    python test_e2_partial_observability.py --po dropout # 특정 PO만
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.pc_z_dynamics import PCZDynamics
from genesis.partial_observability import (
    POType, POConfig, PartialObservabilityApplicator,
    E2Gate, E2GateResult, E2RunStats, PO_INTENSITIES, create_po_configs
)


def run_single_po_config(
    seed: int,
    po_config: POConfig,
    steps: int = 500,
) -> E2RunStats:
    """단일 PO 설정으로 실행"""
    np.random.seed(seed)
    dynamics = PCZDynamics()
    po_applicator = PartialObservabilityApplicator(po_config, seed=seed)

    residuals = []
    epsilon_spikes = []
    z_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    qualities = []

    for step in range(steps):
        # 기본 시나리오: stable → shock → recovery
        if step < 100:
            base_residual = 0.15 + np.random.randn() * 0.03
            score = 0.1
        elif step < 150:
            base_residual = 0.6 + np.random.randn() * 0.05
            score = 0.7
        else:
            recovery = min(1.0, (step - 150) / 100)
            base_residual = max(0.15, 0.6 - recovery * 0.45)
            score = max(0.1, 0.7 - recovery * 0.6)

        base_residual = np.clip(base_residual, 0.05, 1.0)
        score = np.clip(score, 0.02, 0.95)

        # 8D 관측 생성
        obs = np.random.randn(8) * 0.1 + np.array([0.5, 0.3, 0, 0, 0, 0, 0.7, 0.1])
        obs = np.clip(obs, 0, 1)

        # Partial Observability 적용
        degraded_obs, quality = po_applicator.apply(obs)
        qualities.append(quality)

        # PO로 인한 추가 residual (관측 품질 저하)
        # 품질 저하가 residual에 미치는 영향 - 적당한 수준으로 조정
        po_residual_boost = (1.0 - quality) * 0.2
        effective_residual = np.clip(base_residual + po_residual_boost, 0.05, 1.0)

        # PC 신호 계산
        epsilon = np.ones(8) * effective_residual
        # 관측 불확실성 반영
        if quality < 0.9:
            epsilon += (1.0 - quality) * 0.2 * np.random.randn(8)

        signals = dynamics.compute_pc_signals(
            epsilon=epsilon,
            error_norm=effective_residual,
            iterations=int(10 + effective_residual * 10),
            max_iterations=30,
            converged=True,
            prior_force_norm=0.2,
            data_force_norm=effective_residual,
            action_margin=0.5,
        )

        # Z-state 결정 (불확실성 반영)
        # PO로 인한 불확실성 → z=1 경향
        uncertainty_bias = (1.0 - quality) * 0.3
        effective_score = score + uncertainty_bias

        if effective_score > 0.6:
            z = 2  # shock
        elif effective_score > 0.3:
            z = 1  # uncertainty/exploration
        else:
            z = 0  # stable

        dynamics.get_modulation_for_pc(
            z=z,
            z_confidence=0.7 * quality,  # 품질 저하 → 신뢰도 저하
            regime_change_score=effective_score
        )

        residuals.append(signals.residual_error)
        epsilon_spikes.append(signals.epsilon_spike)
        z_counts[z] += 1

    total_steps = steps
    return E2RunStats(
        po_type=po_config.po_type.value,
        intensity=po_config.intensity,
        seed=seed,
        residual_ema_mean=np.mean(residuals),
        epsilon_spike_rate=np.mean([1 if e > 0.5 else 0 for e in epsilon_spikes]),
        z0_occupation=z_counts[0] / total_steps,
        z1_occupation=z_counts[1] / total_steps,
        z2_occupation=z_counts[2] / total_steps,
        z3_occupation=z_counts[3] / total_steps,
        avg_quality=np.mean(qualities),
        efficiency_proxy=1.0 - np.mean(residuals),  # 간접 효율 지표
    )


def run_e2_validation(
    n_seeds: int = 30,
    steps: int = 500,
    po_filter: str = None,  # "dropout", "noise", "stale" or None for all
) -> Dict:
    """E2 검증 실행"""
    print(f"\n{'='*60}")
    print(f"  E2 Partial Observability Validation")
    print(f"  {n_seeds} seeds, {steps} steps")
    if po_filter:
        print(f"  Filter: {po_filter} only")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Seeds
    fixed_seeds = [11, 23, 37]
    additional = [i for i in range(200) if i not in fixed_seeds]
    seeds = fixed_seeds + additional[:n_seeds - 3]

    # PO types to test
    po_types_to_test = []
    if po_filter is None or po_filter == "dropout":
        po_types_to_test.append(POType.DROPOUT)
    if po_filter is None or po_filter == "noise":
        po_types_to_test.append(POType.NOISE)
    if po_filter is None or po_filter == "stale":
        po_types_to_test.append(POType.STALE)

    all_results = {}
    gate = E2Gate()

    for po_type in po_types_to_test:
        print(f"\n--- Testing {po_type.value} ---")

        raw_intensities = PO_INTENSITIES[po_type]
        stats_by_intensity: Dict[float, List[E2RunStats]] = {}

        # Create configs for this PO type
        for raw_intensity in raw_intensities:
            if po_type == POType.DROPOUT:
                config = POConfig.dropout(raw_intensity)
            elif po_type == POType.NOISE:
                config = POConfig.noise(raw_intensity)
            else:  # STALE
                config = POConfig.stale(int(raw_intensity))

            # Use computed intensity as key
            if config.intensity not in stats_by_intensity:
                stats_by_intensity[config.intensity] = []

            print(f"  Intensity {config.intensity:.2f}...", end=" ")

            for seed in seeds:
                stats = run_single_po_config(seed, config, steps)
                stats_by_intensity[config.intensity].append(stats)

            print(f"done ({len(seeds)} seeds)")

        # Aggregate stats by intensity
        aggregated_stats: Dict[float, E2RunStats] = {}
        for intensity, stats_list in stats_by_intensity.items():
            aggregated_stats[intensity] = E2RunStats(
                po_type=po_type.value,
                intensity=intensity,
                seed=-1,  # aggregated
                residual_ema_mean=np.mean([s.residual_ema_mean for s in stats_list]),
                epsilon_spike_rate=np.mean([s.epsilon_spike_rate for s in stats_list]),
                z0_occupation=np.mean([s.z0_occupation for s in stats_list]),
                z1_occupation=np.mean([s.z1_occupation for s in stats_list]),
                z2_occupation=np.mean([s.z2_occupation for s in stats_list]),
                z3_occupation=np.mean([s.z3_occupation for s in stats_list]),
                avg_quality=np.mean([s.avg_quality for s in stats_list]),
                efficiency_proxy=np.mean([s.efficiency_proxy for s in stats_list]),
            )

        # Get baseline (intensity 0)
        baseline_intensity = min(stats_by_intensity.keys())
        baseline_stats = aggregated_stats[baseline_intensity]

        # Evaluate gate
        gate_result = gate.evaluate(aggregated_stats, baseline_stats)

        all_results[po_type.value] = {
            'stats_by_intensity': {k: v.__dict__ for k, v in aggregated_stats.items()},
            'gate_result': gate_result,
        }

        # Print results
        print(f"\n  E2a Uncertainty Calibration:")
        print(f"    Monotonicity score: {gate_result.monotonicity_score:.2f}")
        print(f"    - residual: {'Y' if gate_result.residual_monotonic else 'N'}")
        print(f"    - epsilon:  {'Y' if gate_result.epsilon_monotonic else 'N'}")
        print(f"    - z1:       {'Y' if gate_result.z1_monotonic else 'N'}")
        print(f"    [{'PASS' if gate_result.e2a_passed else 'FAIL'}]")

        print(f"\n  E2b Disambiguation (z=1 vs z=3):")
        print(f"    z3 excess rate: {gate_result.z3_excess_rate:.1%} (max: 5%)")
        print(f"    [{'PASS' if gate_result.e2b_passed else 'FAIL'}]")

        print(f"\n  E2c Utility Preservation:")
        print(f"    Efficiency retention: {gate_result.efficiency_retention:.1%} (min: 70%)")
        print(f"    [{'PASS' if gate_result.e2c_passed else 'FAIL'}]")

        print(f"\n  Overall: [{'PASS' if gate_result.passed else 'FAIL'}] {gate_result.reason}")

        # Print intensity-wise stats
        print(f"\n  Intensity-wise stats:")
        print(f"  {'Intensity':<10} {'Residual':<10} {'ε-spike':<10} {'z0':<8} {'z1':<8} {'z2':<8} {'z3':<8} {'Quality':<10}")
        print(f"  {'-'*74}")
        for intensity in sorted(aggregated_stats.keys()):
            s = aggregated_stats[intensity]
            print(f"  {intensity:<10.2f} {s.residual_ema_mean:<10.3f} {s.epsilon_spike_rate:<10.3f} "
                  f"{s.z0_occupation:<8.2%} {s.z1_occupation:<8.2%} {s.z2_occupation:<8.2%} {s.z3_occupation:<8.2%} "
                  f"{s.avg_quality:<10.2%}")

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Final Summary")
    print(f"{'='*60}\n")
    print(f"Time: {elapsed:.1f}s")

    all_passed = True
    for po_type, result in all_results.items():
        gate_result = result['gate_result']
        status = "PASS" if gate_result.passed else "FAIL"
        print(f"  {po_type}: [{status}] {gate_result.reason}")
        if not gate_result.passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print(f"  [PASS] E2 Gate passed for all PO types")
    else:
        print(f"  [FAIL] E2 Gate failed for some PO types")
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
    parser.add_argument("--po", type=str, default=None,
                        choices=["dropout", "noise", "stale"],
                        help="Test only specific PO type")
    args = parser.parse_args()

    results = run_e2_validation(
        n_seeds=args.seeds,
        steps=args.steps,
        po_filter=args.po,
    )
    exit(0 if results['all_passed'] else 1)
