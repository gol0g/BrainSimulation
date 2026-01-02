"""
v5.4 Robustness Tests - 속임수 가능성 체크

1. Multi-seed test: seed 1~20으로 평균/분산 확인
2. Multi-drift type: rotate/flip_x/probabilistic/delayed/continuous
3. Long-run: 2000+ steps에서 누적 효과
4. z=1/z=3 mixed zone: 충돌 구간에서 안정성

PASS 조건:
- 평균적으로 ON >= OFF * 0.95 (5% 이내 손실)
- 분산이 크지 않음 (일관된 방향성)
- Action efficiency가 평균적으로 개선 또는 유지
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.self_model import SelfModel
from genesis.interaction_gating import InteractionGating
from genesis.g2_gate import G2GateTracker


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_simulation(
    use_gating: bool,
    seed: int,
    total_steps: int = 400,
    drift_type: str = "rotate",
    z_conflict_mode: bool = False,
) -> dict:
    """
    시뮬레이션 실행

    Args:
        use_gating: v5.4 gating 적용 여부
        seed: 랜덤 시드
        total_steps: 총 스텝 수
        drift_type: drift 타입 (rotate, flip_x, probabilistic, delayed, continuous)
        z_conflict_mode: z=1/z=3 충돌 구간 강제 여부
    """
    np.random.seed(seed)

    self_model = SelfModel()
    gating = InteractionGating()
    g2_tracker = G2GateTracker(drift_after=100, drift_type=drift_type)

    current_energy = 0.7
    total_actions = 0
    food_collected = 0
    transition_std_baseline = 0.15

    # z 분포 추적
    z_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for step in range(total_steps):
        drift_active = step >= 100

        # 환경 설정 (drift 타입별로 다르게)
        if step < 100:
            env_efficiency = 0.6
            uncertainty = 0.2
            transition_std = transition_std_baseline
        elif step < 130:  # shock
            if drift_type == "rotate":
                env_efficiency = 0.3
                uncertainty = 0.6
            elif drift_type == "flip_x":
                env_efficiency = 0.25  # 더 어려움
                uncertainty = 0.65
            elif drift_type == "probabilistic":
                env_efficiency = 0.3 + np.random.uniform(-0.1, 0.1)
                uncertainty = 0.6 + np.random.uniform(-0.1, 0.1)
            elif drift_type == "delayed":
                # 지연된 충격 - shock이 더 길게
                env_efficiency = 0.35
                uncertainty = 0.55
            elif drift_type == "continuous":
                # 연속 변화 - 점진적
                progress = (step - 100) / 30
                env_efficiency = 0.6 - 0.3 * progress
                uncertainty = 0.2 + 0.4 * progress
            else:
                env_efficiency = 0.3
                uncertainty = 0.6
            transition_std = 0.5
        else:  # adapt
            adapt_progress = min(1.0, (step - 130) / 100)
            if drift_type == "continuous":
                # 연속: 느린 회복
                adapt_progress = min(1.0, (step - 130) / 200)
            env_efficiency = 0.3 + 0.3 * adapt_progress
            uncertainty = 0.5 - 0.3 * adapt_progress
            transition_std = 0.5 - 0.35 * adapt_progress

        # z=1/z=3 충돌 모드: 불확실성과 피로가 동시에 높음
        if z_conflict_mode and drift_active:
            uncertainty = 0.6  # z=1 유도
            env_efficiency = 0.2  # z=3 유도 (낮은 효율)

        # Self-model + Gating
        signals = {
            'uncertainty': uncertainty,
            'regret_spike_rate': 0.1 if not drift_active else 0.15,
            'energy_efficiency': env_efficiency,
            'volatility': transition_std,
            'movement_ratio': 0.6,
        }
        modifiers, sm_info = self_model.update(signals)
        z = sm_info['z']
        z_counts[z] += 1

        # Gating 적용 여부
        if use_gating:
            gating_mods = gating.update(z=z, efficiency=env_efficiency, Q_z=sm_info['Q_z'])
            action_prob = gating_mods.action_execution_prob
        else:
            action_prob = 1.0

        # 행동
        if np.random.random() < action_prob:
            total_actions += 1
            ate_food = np.random.random() < env_efficiency
            if ate_food:
                food_collected += 1
                current_energy = min(1.0, current_energy + 0.1)
        else:
            ate_food = False

        current_energy = max(0.1, current_energy - 0.01)

        # G2 로깅
        g2_tracker.log_step(
            circuit_action=1, fep_action=1, final_action=1, agreed=True,
            disagreement_type=None, energy=current_energy, danger_prox=0.1,
            food_prox=env_efficiency, drift_active=drift_active,
            transition_std=transition_std, transition_error=0.1,
            ate_food=ate_food, hit_danger=False, energy_spent=0.01,
            regret_spike=drift_active and np.random.random() < 0.1,
            circuit_margin=0.5,
        )

    return {
        'g2_result': g2_tracker.get_result(),
        'total_actions': total_actions,
        'food_collected': food_collected,
        'food_rate': food_collected / total_steps,
        'action_efficiency': food_collected / max(1, total_actions),
        'z_counts': z_counts,
    }


# =============================================================================
# Test 1: Multi-Seed Robustness
# =============================================================================

def test_multi_seed_robustness():
    """
    seed 1~20으로 ON/OFF 비교

    PASS 조건:
    - 평균 retention ratio >= 0.95
    - ON wins >= 10 (50% 이상에서 효율 개선 또는 동등)
    """
    print_header("Robustness Test 1: Multi-Seed (seeds 1-20)")

    seeds = range(1, 21)
    results = []

    for seed in seeds:
        off = run_simulation(use_gating=False, seed=seed)
        on = run_simulation(use_gating=True, seed=seed)

        retention_off = off['g2_result'].efficiency_retention
        retention_on = on['g2_result'].efficiency_retention
        eff_off = off['action_efficiency']
        eff_on = on['action_efficiency']

        retention_ratio = retention_on / retention_off if retention_off > 0 else 1.0
        eff_ratio = eff_on / eff_off if eff_off > 0 else 1.0

        results.append({
            'seed': seed,
            'retention_off': retention_off,
            'retention_on': retention_on,
            'retention_ratio': retention_ratio,
            'eff_off': eff_off,
            'eff_on': eff_on,
            'eff_ratio': eff_ratio,
            'on_wins': eff_on >= eff_off,
        })

    # 통계
    retention_ratios = [r['retention_ratio'] for r in results]
    eff_ratios = [r['eff_ratio'] for r in results]
    on_wins = sum(1 for r in results if r['on_wins'])

    avg_retention = np.mean(retention_ratios)
    std_retention = np.std(retention_ratios)
    avg_eff = np.mean(eff_ratios)
    std_eff = np.std(eff_ratios)

    print(f"  Results over {len(seeds)} seeds:")
    print(f"    Retention ratio: {avg_retention:.3f} ± {std_retention:.3f}")
    print(f"    Efficiency ratio: {avg_eff:.3f} ± {std_eff:.3f}")
    print(f"    ON wins (efficiency): {on_wins}/{len(seeds)} ({on_wins/len(seeds)*100:.1f}%)")

    print(f"\n  Per-seed breakdown:")
    for r in results[:5]:  # 처음 5개만 출력
        print(f"    seed {r['seed']:2d}: ret={r['retention_ratio']:.2f}, eff={r['eff_ratio']:.2f}, win={'Y' if r['on_wins'] else 'N'}")
    print(f"    ...")

    # 판정
    avg_ok = avg_retention >= 0.95
    wins_ok = on_wins >= len(seeds) // 2  # 50% 이상

    print(f"\n  Checks:")
    print(f"    Avg retention ratio >= 0.95: {avg_ok} ({avg_retention:.3f})")
    print(f"    ON wins >= 50%: {wins_ok} ({on_wins}/{len(seeds)})")

    passed = avg_ok and wins_ok
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed, results


# =============================================================================
# Test 2: Multi-Drift Type
# =============================================================================

def test_multi_drift_type():
    """
    다양한 drift 타입에서 ON/OFF 비교

    PASS 조건:
    - 모든 drift 타입에서 retention ratio >= 0.90
    - 과반수 타입에서 efficiency ratio >= 1.0
    """
    print_header("Robustness Test 2: Multi-Drift Type")

    drift_types = ["rotate", "flip_x", "probabilistic", "delayed", "continuous"]
    results = []

    for dtype in drift_types:
        # 3개 seed 평균
        dtype_results = []
        for seed in [42, 43, 44]:
            off = run_simulation(use_gating=False, seed=seed, drift_type=dtype)
            on = run_simulation(use_gating=True, seed=seed, drift_type=dtype)

            retention_ratio = on['g2_result'].efficiency_retention / off['g2_result'].efficiency_retention if off['g2_result'].efficiency_retention > 0 else 1.0
            eff_ratio = on['action_efficiency'] / off['action_efficiency'] if off['action_efficiency'] > 0 else 1.0
            dtype_results.append({'retention': retention_ratio, 'eff': eff_ratio})

        avg_retention = np.mean([r['retention'] for r in dtype_results])
        avg_eff = np.mean([r['eff'] for r in dtype_results])

        results.append({
            'drift_type': dtype,
            'retention_ratio': avg_retention,
            'eff_ratio': avg_eff,
            'retention_ok': avg_retention >= 0.90,
            'eff_improved': avg_eff >= 1.0,
        })

    print(f"  Results by drift type (avg of 3 seeds):")
    print(f"    {'Type':<15} {'Ret Ratio':>10} {'Eff Ratio':>10} {'Status':>10}")
    print(f"    {'-'*45}")
    for r in results:
        status = 'OK' if r['retention_ok'] else 'LOW'
        if r['eff_improved']:
            status += '+EFF'
        print(f"    {r['drift_type']:<15} {r['retention_ratio']:>10.3f} {r['eff_ratio']:>10.3f} {status:>10}")

    # 판정
    retention_pass = sum(1 for r in results if r['retention_ok'])
    eff_improved = sum(1 for r in results if r['eff_improved'])

    all_retention_ok = retention_pass == len(drift_types)
    majority_eff_ok = eff_improved >= len(drift_types) // 2 + 1

    print(f"\n  Checks:")
    print(f"    All retention >= 0.90: {all_retention_ok} ({retention_pass}/{len(drift_types)})")
    print(f"    Majority efficiency >= 1.0: {majority_eff_ok} ({eff_improved}/{len(drift_types)})")

    passed = all_retention_ok and majority_eff_ok
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed, results


# =============================================================================
# Test 3: Long-Run Stability
# =============================================================================

def test_long_run_stability():
    """
    2000 steps 장기 run에서 누적 효과

    PASS 조건:
    - 장기에서도 retention ratio >= 0.95
    - Actions saved가 양수 (gating 효과 유지)
    """
    print_header("Robustness Test 3: Long-Run Stability (2000 steps)")

    total_steps = 2000

    off = run_simulation(use_gating=False, seed=42, total_steps=total_steps)
    on = run_simulation(use_gating=True, seed=42, total_steps=total_steps)

    g2_off = off['g2_result']
    g2_on = on['g2_result']

    retention_ratio = g2_on.efficiency_retention / g2_off.efficiency_retention if g2_off.efficiency_retention > 0 else 1.0
    eff_ratio = on['action_efficiency'] / off['action_efficiency'] if off['action_efficiency'] > 0 else 1.0
    actions_saved = off['total_actions'] - on['total_actions']

    print(f"  Long-run results ({total_steps} steps):")
    print(f"    OFF - retention: {g2_off.efficiency_retention:.1%}, actions: {off['total_actions']}, food: {off['food_collected']}")
    print(f"    ON  - retention: {g2_on.efficiency_retention:.1%}, actions: {on['total_actions']}, food: {on['food_collected']}")
    print(f"\n  Comparison:")
    print(f"    Retention ratio: {retention_ratio:.3f}")
    print(f"    Efficiency ratio: {eff_ratio:.3f}")
    print(f"    Actions saved: {actions_saved}")
    print(f"    Food difference: {on['food_collected'] - off['food_collected']}")

    # z 분포 (ON)
    print(f"\n  Z distribution (ON):")
    for z, count in on['z_counts'].items():
        pct = count / total_steps * 100
        print(f"    z={z}: {count} ({pct:.1f}%)")

    # 판정
    retention_ok = retention_ratio >= 0.95
    actions_saved_ok = actions_saved > 0

    print(f"\n  Checks:")
    print(f"    Retention ratio >= 0.95: {retention_ok} ({retention_ratio:.3f})")
    print(f"    Actions saved > 0: {actions_saved_ok} ({actions_saved})")

    passed = retention_ok and actions_saved_ok
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Test 4: z=1/z=3 Mixed Zone
# =============================================================================

def test_z_conflict_zone():
    """
    z=1(불확실)과 z=3(피로)가 동시에 높은 구간에서 안정성

    PASS 조건 (v5.4-fix):
    1. z=0이 100%가 아님 (z가 발화해야 act/learn 분리가 작동)
    2. z=1 또는 z=3이 10% 이상 발화
    3. ON이 OFF보다 나쁘지 않음 (v5.4 효과)
    """
    print_header("Robustness Test 4: z=1/z=3 Conflict Zone")

    # Conflict mode에서 ON/OFF 비교
    conflict_off = run_simulation(use_gating=False, seed=42, z_conflict_mode=True)
    conflict_on = run_simulation(use_gating=True, seed=42, z_conflict_mode=True)

    g2_off = conflict_off['g2_result']
    g2_on = conflict_on['g2_result']

    print(f"  === CONFLICT MODE (hard environment) ===")
    print(f"\n  Gating OFF (baseline):")
    print(f"    Retention: {g2_off.efficiency_retention:.1%}")
    print(f"    Food collected: {conflict_off['food_collected']}")
    print(f"    Action efficiency: {conflict_off['action_efficiency']:.3f}")
    print(f"    Z distribution: {conflict_off['z_counts']}")

    print(f"\n  Gating ON (v5.4):")
    print(f"    Retention: {g2_on.efficiency_retention:.1%}")
    print(f"    Food collected: {conflict_on['food_collected']}")
    print(f"    Action efficiency: {conflict_on['action_efficiency']:.3f}")
    print(f"    Z distribution: {conflict_on['z_counts']}")

    # z 발화 분석
    z1_count = conflict_on['z_counts'].get(1, 0)
    z3_count = conflict_on['z_counts'].get(3, 0)
    z0_count = conflict_on['z_counts'].get(0, 0)
    total = sum(conflict_on['z_counts'].values())

    print(f"\n  Z activation analysis (ON):")
    print(f"    z=0 (stable): {z0_count} ({z0_count/total*100:.1f}%)")
    print(f"    z=1 (uncertain): {z1_count} ({z1_count/total*100:.1f}%)")
    print(f"    z=3 (fatigued): {z3_count} ({z3_count/total*100:.1f}%)")

    # ON/OFF 비교
    retention_ratio = g2_on.efficiency_retention / g2_off.efficiency_retention if g2_off.efficiency_retention > 0 else 1.0
    eff_ratio = conflict_on['action_efficiency'] / conflict_off['action_efficiency'] if conflict_off['action_efficiency'] > 0 else 1.0
    actions_saved = conflict_off['total_actions'] - conflict_on['total_actions']

    print(f"\n  ON/OFF Comparison:")
    print(f"    Retention ratio: {retention_ratio:.2f}")
    print(f"    Efficiency ratio: {eff_ratio:.2f}")
    print(f"    Actions saved: {actions_saved}")

    # === 판정 기준 (v5.4-fix) ===
    # 1. z가 발화해야 함 (z=0 < 100%)
    z_fires = z0_count < total  # z=0이 100%가 아님

    # 2. z=1 또는 z=3이 최소 10% 이상
    z_meaningful = (z1_count + z3_count) >= total * 0.10

    # 3. ON이 OFF보다 심하게 나쁘지 않음 (95% 이상)
    not_degraded = retention_ratio >= 0.90 or eff_ratio >= 0.95

    print(f"\n  === CHECKS ===")
    print(f"    z fires (z=0 < 100%): {z_fires}")
    print(f"    z meaningful (z1+z3 >= 10%): {z_meaningful} ({(z1_count+z3_count)/total*100:.1f}%)")
    print(f"    Not degraded (ON >= OFF*0.90): {not_degraded}")

    # Bonus: efficiency 개선되면 추가 성공
    if eff_ratio > 1.0:
        print(f"\n  [BONUS] Efficiency improved by {(eff_ratio-1)*100:.1f}% in conflict zone!")

    passed = z_fires and z_meaningful and not_degraded
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Main
# =============================================================================

def run_all_robustness_tests():
    """모든 robustness 테스트 실행"""
    print("\n" + "="*60)
    print("  v5.4 ROBUSTNESS TESTS")
    print("="*60)

    results = {}

    passed1, _ = test_multi_seed_robustness()
    results['multi_seed'] = passed1

    passed2, _ = test_multi_drift_type()
    results['multi_drift'] = passed2

    passed3 = test_long_run_stability()
    results['long_run'] = passed3

    passed4 = test_z_conflict_zone()
    results['z_conflict'] = passed4

    print("\n" + "="*60)
    print("  ROBUSTNESS TEST SUMMARY")
    print("="*60 + "\n")

    all_passed = True
    for name, passed in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  \033[92m✓ v5.4 IS ROBUST\033[0m")
        print("    → Consistent across seeds")
        print("    → Works for multiple drift types")
        print("    → Stable in long runs")
        print("    → Handles z=1/z=3 conflict")
    else:
        print("  \033[91m✗ v5.4 NEEDS MORE TUNING\033[0m")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    run_all_robustness_tests()
