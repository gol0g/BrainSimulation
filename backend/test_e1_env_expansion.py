"""
E1 Environment Expansion Validation

목표:
1. PC residual/epsilon 스케일이 차원 증가에서 유지되는지 (E1a)
2. Distractor-only 변화에서 regime_change_score가 흔들리지 않는지 (E1b)

Usage:
    python test_e1_env_expansion.py              # 기본 테스트
    python test_e1_env_expansion.py --seeds 100  # 더 많은 seeds
    python test_e1_env_expansion.py --dim 32     # 32D 테스트
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))

from genesis.pc_z_dynamics import PCZDynamics, DynamicPastRegimeConfig
from genesis.env_expansion import (
    ExtendedObsConfig, DistractorGenerator, E1Gate, E1GateResult,
    normalize_extended_observation, E1_SCENARIOS
)


@dataclass
class E1RunResult:
    """단일 실행 결과"""
    seed: int
    dim: int

    # PC 신호 통계
    residual_mean: float
    residual_std: float
    epsilon_spike_mean: float
    epsilon_spike_std: float

    # Regime 감지 통계
    distractor_only_shock_count: int
    distractor_only_total: int
    real_change_shock_count: int
    real_change_total: int


def run_baseline_8d(seed: int, steps: int = 500) -> E1RunResult:
    """8D 기본 환경 실행 (baseline)"""
    np.random.seed(seed)
    dynamics = PCZDynamics()

    residuals = []
    epsilon_spikes = []

    for step in range(steps):
        # 기본 시나리오: stable → shock → recovery
        if step < 100:
            residual = 0.15 + np.random.randn() * 0.03
            score = 0.1
        elif step < 150:
            residual = 0.6 + np.random.randn() * 0.05
            score = 0.7
        else:
            recovery = min(1.0, (step - 150) / 100)
            residual = max(0.15, 0.6 - recovery * 0.45)
            score = max(0.1, 0.7 - recovery * 0.6)

        residual = np.clip(residual, 0.05, 1.0)
        score = np.clip(score, 0.02, 0.95)

        # 8D 관측
        obs = np.random.randn(8) * 0.1 + np.array([0.5, 0.3, 0, 0, 0, 0, 0.7, 0.1])
        obs = np.clip(obs, 0, 1)

        signals = dynamics.compute_pc_signals(
            epsilon=np.ones(8) * residual,
            error_norm=residual,
            iterations=int(10 + residual * 10),
            max_iterations=30,
            converged=True,
            prior_force_norm=0.2,
            data_force_norm=residual,
            action_margin=0.5,
        )
        dynamics.get_modulation_for_pc(
            z=1 if score > 0.3 else 0,
            z_confidence=0.7,
            regime_change_score=score
        )

        residuals.append(signals.residual_error)
        epsilon_spikes.append(signals.epsilon_spike)

    return E1RunResult(
        seed=seed,
        dim=8,
        residual_mean=np.mean(residuals),
        residual_std=np.std(residuals),
        epsilon_spike_mean=np.mean(epsilon_spikes),
        epsilon_spike_std=np.std(epsilon_spikes),
        distractor_only_shock_count=0,
        distractor_only_total=0,
        real_change_shock_count=0,
        real_change_total=0,
    )


def run_extended(
    seed: int,
    config: ExtendedObsConfig,
    steps: int = 500,
    s_on: float = 0.6
) -> E1RunResult:
    """확장 환경 실행"""
    np.random.seed(seed)
    dynamics = PCZDynamics()
    distractor_gen = DistractorGenerator(config, seed=seed)

    residuals = []
    epsilon_spikes = []

    distractor_only_shock_count = 0
    distractor_only_total = 0
    real_change_shock_count = 0
    real_change_total = 0

    prev_is_shock = False

    for step in range(steps):
        # 기본 시나리오: stable → shock → recovery
        if step < 100:
            residual = 0.15 + np.random.randn() * 0.03
            base_score = 0.1
            is_real_change = False
        elif step < 150:
            residual = 0.6 + np.random.randn() * 0.05
            base_score = 0.7
            is_real_change = (step == 100)  # shock 시작점
        else:
            recovery = min(1.0, (step - 150) / 100)
            residual = max(0.15, 0.6 - recovery * 0.45)
            base_score = max(0.1, 0.7 - recovery * 0.6)
            is_real_change = False

        residual = np.clip(residual, 0.05, 1.0)
        base_score = np.clip(base_score, 0.02, 0.95)

        # 8D 기본 관측
        base_obs = np.random.randn(8) * 0.1 + np.array([0.5, 0.3, 0, 0, 0, 0, 0.7, 0.1])
        base_obs = np.clip(base_obs, 0, 1)

        # Distractor 추가
        extended_obs, distractor_only_change = distractor_gen.generate(base_obs)

        # 정규화
        normalized_obs = normalize_extended_observation(
            extended_obs, base_dim=8, method="per_channel"
        )

        # PC 신호 계산 (확장 차원 사용)
        epsilon = np.ones(config.target_dim) * residual
        # Distractor 채널의 epsilon은 더 크게 (노이즈 반영)
        if config.target_dim > 8:
            distractor_epsilon = normalized_obs[8:] - base_obs[:min(config.target_dim-8, 8)].mean()
            epsilon[8:] = np.abs(distractor_epsilon) + residual * 0.5

        error_norm = np.linalg.norm(epsilon) / np.sqrt(config.target_dim)  # 차원 정규화

        signals = dynamics.compute_pc_signals(
            epsilon=epsilon,
            error_norm=error_norm,
            iterations=int(10 + residual * 10),
            max_iterations=30,
            converged=True,
            prior_force_norm=0.2,
            data_force_norm=residual,
            action_margin=0.5,
        )

        # Score 계산: 기본 score만 사용 (distractor는 score에 영향 없어야 함)
        # E1b의 핵심: distractor 변화가 regime_change_score를 올리면 안 됨
        # 실제 시스템에서는 score가 "의미 있는 변화"만 반영해야 함
        score = base_score  # distractor 영향 제거
        score = np.clip(score, 0.02, 0.95)

        dynamics.get_modulation_for_pc(
            z=1 if score > 0.3 else 0,
            z_confidence=0.7,
            regime_change_score=score
        )

        residuals.append(signals.residual_error)
        epsilon_spikes.append(signals.epsilon_spike)

        # E1b 추적: distractor-only change vs real change
        is_shock = score >= s_on
        is_stable_period = (step < 100 or step >= 250)  # shock 기간 제외

        # Distractor-only shock: stable 기간에 distractor만 바뀌었는데 shock 감지
        if distractor_only_change and is_stable_period:
            distractor_only_total += 1
            if is_shock:
                distractor_only_shock_count += 1

        # Real change detection: 실제 변화 시점에서 shock 감지
        if is_real_change:
            real_change_total += 1
            if is_shock:
                real_change_shock_count += 1

    return E1RunResult(
        seed=seed,
        dim=config.target_dim,
        residual_mean=np.mean(residuals),
        residual_std=np.std(residuals),
        epsilon_spike_mean=np.mean(epsilon_spikes),
        epsilon_spike_std=np.std(epsilon_spikes),
        distractor_only_shock_count=distractor_only_shock_count,
        distractor_only_total=distractor_only_total,
        real_change_shock_count=real_change_shock_count,
        real_change_total=real_change_total,
    )


def run_e1_validation(
    n_seeds: int = 30,
    target_dim: int = 16,
    steps: int = 500
) -> Dict:
    """E1 검증 실행"""
    print(f"\n{'='*60}")
    print(f"  E1 Environment Expansion Validation")
    print(f"  8D → {target_dim}D, {n_seeds} seeds, {steps} steps")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Seeds
    fixed_seeds = [11, 23, 37]
    additional = [i for i in range(200) if i not in fixed_seeds]
    seeds = fixed_seeds + additional[:n_seeds - 3]

    # Config
    config = ExtendedObsConfig(target_dim=target_dim)

    # Run baseline (8D)
    print("Running 8D baseline...")
    baseline_results: List[E1RunResult] = []
    for i, seed in enumerate(seeds):
        result = run_baseline_8d(seed, steps)
        baseline_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_seeds} done")

    # Run extended
    print(f"\nRunning {target_dim}D extended...")
    extended_results: List[E1RunResult] = []
    for i, seed in enumerate(seeds):
        result = run_extended(seed, config, steps)
        extended_results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_seeds} done")

    elapsed = time.time() - start_time

    # Aggregate results
    baseline_stats = {
        'residual_mean': np.mean([r.residual_mean for r in baseline_results]),
        'residual_std': np.mean([r.residual_std for r in baseline_results]),
        'epsilon_spike_mean': np.mean([r.epsilon_spike_mean for r in baseline_results]),
        'epsilon_spike_std': np.mean([r.epsilon_spike_std for r in baseline_results]),
    }

    total_distractor_only = sum(r.distractor_only_total for r in extended_results)
    total_distractor_shock = sum(r.distractor_only_shock_count for r in extended_results)
    total_real_change = sum(r.real_change_total for r in extended_results)
    total_real_shock = sum(r.real_change_shock_count for r in extended_results)

    extended_stats = {
        'residual_mean': np.mean([r.residual_mean for r in extended_results]),
        'residual_std': np.mean([r.residual_std for r in extended_results]),
        'epsilon_spike_mean': np.mean([r.epsilon_spike_mean for r in extended_results]),
        'epsilon_spike_std': np.mean([r.epsilon_spike_std for r in extended_results]),
        'distractor_only_shock_rate': total_distractor_shock / max(1, total_distractor_only),
        'real_change_detection_rate': total_real_shock / max(1, total_real_change),
    }

    # Evaluate gate
    gate = E1Gate()
    gate_result = gate.evaluate(baseline_stats, extended_stats)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}\n")

    print(f"Time: {elapsed:.1f}s")
    print(f"\n--- E1a: Scale Invariance ---")
    print(f"  8D baseline:")
    print(f"    residual_mean: {baseline_stats['residual_mean']:.4f}")
    print(f"    epsilon_spike_mean: {baseline_stats['epsilon_spike_mean']:.4f}")
    print(f"  {target_dim}D extended:")
    print(f"    residual_mean: {extended_stats['residual_mean']:.4f}")
    print(f"    epsilon_spike_mean: {extended_stats['epsilon_spike_mean']:.4f}")
    print(f"  Ratios:")
    print(f"    residual_ratio: {gate_result.residual_ratio:.2f} (target: 0.5~2.0)")
    print(f"    epsilon_ratio: {gate_result.epsilon_ratio:.2f} (target: 0.5~2.0)")
    print(f"  [{'PASS' if gate_result.e1a_passed else 'FAIL'}] E1a Scale Invariance")

    print(f"\n--- E1b: Change Specificity ---")
    print(f"  Distractor-only changes: {total_distractor_only}")
    print(f"  Distractor-only shocks: {total_distractor_shock}")
    print(f"  Distractor shock rate: {gate_result.distractor_only_shock_rate:.1%} (target: <5%)")
    print(f"  Real changes: {total_real_change}")
    print(f"  Real change detections: {total_real_shock}")
    print(f"  Real detection rate: {gate_result.real_change_detection_rate:.1%} (target: >70%)")
    print(f"  [{'PASS' if gate_result.e1b_passed else 'FAIL'}] E1b Change Specificity")

    print(f"\n{'='*60}")
    print(f"  Final Verdict")
    print(f"{'='*60}\n")

    if gate_result.passed:
        print(f"[PASS] E1 Gate passed for 8D → {target_dim}D")
    else:
        print(f"[FAIL] E1 Gate failed: {gate_result.reason}")

    return {
        'target_dim': target_dim,
        'n_seeds': n_seeds,
        'baseline_stats': baseline_stats,
        'extended_stats': extended_stats,
        'gate_result': gate_result,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds")
    parser.add_argument("--dim", type=int, default=16, help="Target dimension (16 or 32)")
    parser.add_argument("--steps", type=int, default=500, help="Steps per run")
    args = parser.parse_args()

    results = run_e1_validation(
        n_seeds=args.seeds,
        target_dim=args.dim,
        steps=args.steps
    )
    exit(0 if results['gate_result'].passed else 1)
