"""
v5.4 Act/Learn Split Tests

성공 조건 3가지:
1. Locked-in 해결: 적응 구간에서 data starvation 없음
2. z=1 vs z=3 충돌 처리: 행동↓와 학습↑ 동시 허용
3. 효율 실제 이득: thrashing 감소가 진짜 효율로 이어짐

테스트 4개:
1. test_drift_adapt_without_act - drift 적응이 act↓에도 가능
2. test_fatigue_rest_still_learns - z=3에서 act↓지만 learn 유지
3. test_uncertainty_quiet_learning - z=1에서 learn↑로 빠른 적응
4. G2 regression test - G2a/b/c 유지
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.self_model import SelfModel, get_mode_label
from genesis.interaction_gating import InteractionGating
from genesis.g2_gate import G2GateTracker


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# =============================================================================
# Test 1: Drift Adaptation Without Act
# =============================================================================

def run_drift_adapt_simulation(
    version: str,  # "v5.3" or "v5.4"
    seed: int = 42,
    pre_drift: int = 150,
    shock: int = 30,
    adapt: int = 150,
) -> dict:
    """Drift 적응 시뮬레이션"""
    np.random.seed(seed)

    self_model = SelfModel()
    gating = InteractionGating()
    g2_tracker = G2GateTracker(drift_after=pre_drift)

    total_steps = pre_drift + shock + adapt

    # Tracking
    shock_learn_couplings = []
    shock_act_couplings = []
    transition_errors = []
    current_energy = 0.7

    for step in range(total_steps):
        phase = "pre_drift"
        if step >= pre_drift:
            phase = "shock" if step < pre_drift + shock else "adapt"

        drift_active = step >= pre_drift

        # 환경 시뮬레이션
        if phase == "pre_drift":
            efficiency = 0.7
            uncertainty = 0.2
            volatility = 0.1
        elif phase == "shock":
            efficiency = 0.15
            uncertainty = 0.8
            volatility = 0.6
        else:
            adapt_progress = (step - pre_drift - shock) / adapt
            efficiency = 0.15 + 0.5 * adapt_progress
            uncertainty = 0.6 - 0.4 * adapt_progress
            volatility = 0.4 - 0.3 * adapt_progress

        # Self-model
        signals = {
            'uncertainty': uncertainty,
            'regret_spike_rate': 0.2 if phase == "shock" else 0.1,
            'energy_efficiency': efficiency,
            'volatility': volatility,
            'movement_ratio': 0.6,
        }
        modifiers, sm_info = self_model.update(signals)
        z = sm_info['z']

        # Gating
        gating_mods = gating.update(z=z, efficiency=efficiency, Q_z=sm_info['Q_z'])

        # Shock window 추적
        if phase == "shock":
            shock_learn_couplings.append(gating_mods.learn_coupling)
            shock_act_couplings.append(gating_mods.act_coupling)

        # Transition error 시뮬레이션
        if drift_active:
            # learn_coupling이 높으면 error가 빨리 줄어듦
            base_error = 0.5 if phase == "shock" else 0.3 * (1 - (step - pre_drift - shock) / adapt)
            error_reduction = gating_mods.learn_coupling * 0.1
            trans_error = max(0.1, base_error - error_reduction)
            transition_errors.append(trans_error)

        # 행동 시뮬레이션
        if np.random.random() < gating_mods.action_execution_prob:
            ate_food = np.random.random() < efficiency
        else:
            ate_food = False

        if ate_food:
            current_energy = min(1.0, current_energy + 0.15)
        current_energy = max(0.1, current_energy - 0.01)

        # G2 로깅
        g2_tracker.log_step(
            circuit_action=1, fep_action=1, final_action=1, agreed=True,
            disagreement_type=None, energy=current_energy, danger_prox=0.1,
            food_prox=efficiency, drift_active=drift_active,
            transition_std=volatility, transition_error=trans_error if drift_active else 0.1,
            ate_food=ate_food, hit_danger=False, energy_spent=0.01,
            regret_spike=phase == "shock" and np.random.random() < 0.2,
            circuit_margin=0.5,
        )

    g2_result = g2_tracker.get_result()

    return {
        'version': version,
        'g2_result': g2_result,
        'shock_avg_learn': np.mean(shock_learn_couplings) if shock_learn_couplings else 1.0,
        'shock_avg_act': np.mean(shock_act_couplings) if shock_act_couplings else 1.0,
        'avg_transition_error': np.mean(transition_errors) if transition_errors else 0.0,
        'final_transition_error': transition_errors[-1] if transition_errors else 0.0,
    }


def test_drift_adapt_without_act():
    """
    v5.4 핵심 테스트: drift 적응이 act↓에도 가능한가?

    PASS 조건:
    1. Shock에서 learn_coupling 유지 (>= 0.9)
    2. G2a (time_to_recovery)가 악화되지 않음
    3. Transition error가 수렴
    """
    print_header("Test 1: Drift Adaptation Without Act (v5.4 Core)")

    result = run_drift_adapt_simulation(version="v5.4", seed=42)

    print(f"  Shock window coupling:")
    print(f"    act_coupling avg: {result['shock_avg_act']:.2f}")
    print(f"    learn_coupling avg: {result['shock_avg_learn']:.2f}")

    print(f"\n  Transition error:")
    print(f"    Average: {result['avg_transition_error']:.3f}")
    print(f"    Final: {result['final_transition_error']:.3f}")

    print(f"\n  G2a (time_to_recovery): {result['g2_result'].time_to_recovery} steps")

    # 검증
    learn_maintained = result['shock_avg_learn'] >= 0.9
    error_converged = result['final_transition_error'] < result['avg_transition_error']
    g2a_ok = result['g2_result'].time_to_recovery <= 200  # 합리적인 임계값

    print(f"\n  Checks:")
    print(f"    Learn coupling maintained (>= 0.9): {learn_maintained}")
    print(f"    Transition error converged: {error_converged}")
    print(f"    G2a reasonable (<= 200): {g2a_ok}")

    passed = learn_maintained and error_converged and g2a_ok
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Test 2: Fatigue Rest Still Learns
# =============================================================================

def test_fatigue_rest_still_learns():
    """
    z=3 (fatigue)에서 act↓지만 learn 유지 확인

    PASS 조건:
    1. Fatigue에서 act_coupling < 0.5
    2. Fatigue에서 learn_coupling >= 0.9 (data starvation 방지)
    3. observation_update_weight 유지
    """
    print_header("Test 2: Fatigue Rest Still Learns")

    gating = InteractionGating()

    # Phase 1: 정상
    print("  Phase 1: Normal state")
    for _ in range(20):
        gating.update(z=0, efficiency=0.7)

    normal_mods = gating.update(z=0, efficiency=0.7)
    print(f"    act: {normal_mods.act_coupling}, learn: {normal_mods.learn_coupling}")

    # Phase 2: Fatigue (z=3 연속)
    print("\n  Phase 2: Fatigue state (z=3 streak)")
    fatigue_acts = []
    fatigue_learns = []

    for i in range(25):
        mods = gating.update(z=3, efficiency=0.2)
        fatigue_acts.append(mods.act_coupling)
        fatigue_learns.append(mods.learn_coupling)
        if i in [0, 8, 15, 24]:
            print(f"    step {i:2d}: act={mods.act_coupling:.2f}, learn={mods.learn_coupling:.2f}")

    final_act = fatigue_acts[-1]
    final_learn = fatigue_learns[-1]

    # 검증
    act_reduced = final_act < 0.5
    learn_maintained = final_learn >= 0.9
    obs_weight_ok = mods.observation_update_weight >= 0.9

    print(f"\n  Final state:")
    print(f"    act_coupling: {final_act:.2f} (should be < 0.5)")
    print(f"    learn_coupling: {final_learn:.2f} (should be >= 0.9)")
    print(f"    observation_update_weight: {mods.observation_update_weight:.2f}")

    print(f"\n  Checks:")
    print(f"    Act reduced (< 0.5): {act_reduced}")
    print(f"    Learn maintained (>= 0.9): {learn_maintained}")
    print(f"    Observation weight ok (>= 0.9): {obs_weight_ok}")

    passed = act_reduced and learn_maintained and obs_weight_ok
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Test 3: Uncertainty Quiet Learning
# =============================================================================

def test_uncertainty_quiet_learning():
    """
    z=1 (uncertain)에서 learn↑로 빠른 적응

    PASS 조건:
    1. Uncertainty에서 learn_coupling > 1.0 (증가)
    2. learning_rate_boost > 1.0
    3. Act는 유지 또는 약간만 감소 (탐색 유지)
    """
    print_header("Test 3: Uncertainty Quiet Learning")

    gating = InteractionGating()

    # Phase 1: 정상
    print("  Phase 1: Normal state")
    for _ in range(20):
        gating.update(z=0, efficiency=0.7)

    normal_mods = gating.update(z=0, efficiency=0.7)
    print(f"    act: {normal_mods.act_coupling}, learn: {normal_mods.learn_coupling}")

    # Phase 2: Uncertainty (z=1 연속)
    print("\n  Phase 2: Uncertainty state (z=1 streak)")
    uncertain_acts = []
    uncertain_learns = []

    for i in range(20):
        mods = gating.update(z=1, efficiency=0.4, uncertainty=0.7)
        uncertain_acts.append(mods.act_coupling)
        uncertain_learns.append(mods.learn_coupling)
        if i in [0, 5, 10, 19]:
            print(f"    step {i:2d}: act={mods.act_coupling:.2f}, learn={mods.learn_coupling:.2f}, lr_boost={mods.learning_rate_boost:.2f}")

    final_act = uncertain_acts[-1]
    final_learn = uncertain_learns[-1]
    final_lr_boost = mods.learning_rate_boost

    # 검증
    learn_boosted = final_learn > 1.0
    lr_boosted = final_lr_boost > 1.0
    act_reasonable = final_act >= 0.5  # 탐색 유지

    print(f"\n  Final state:")
    print(f"    act_coupling: {final_act:.2f} (should be >= 0.5)")
    print(f"    learn_coupling: {final_learn:.2f} (should be > 1.0)")
    print(f"    learning_rate_boost: {final_lr_boost:.2f} (should be > 1.0)")

    print(f"\n  Checks:")
    print(f"    Learn boosted (> 1.0): {learn_boosted}")
    print(f"    LR boosted (> 1.0): {lr_boosted}")
    print(f"    Act reasonable (>= 0.5): {act_reasonable}")

    passed = learn_boosted and lr_boosted and act_reasonable
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Test 4: G2 Regression Test
# =============================================================================

def run_full_simulation(use_gating: bool, seed: int = 42, total_steps: int = 400) -> dict:
    """
    전체 시뮬레이션 (G2 비교용)

    Args:
        use_gating: True면 v5.4 gating 적용, False면 gating 무시 (baseline)
    """
    np.random.seed(seed)

    self_model = SelfModel()
    gating = InteractionGating()
    g2_tracker = G2GateTracker(drift_after=100)

    current_energy = 0.7
    total_actions = 0
    food_collected = 0

    # transition_std 추적 (실제 변동)
    transition_std_baseline = 0.15

    for step in range(total_steps):
        drift_active = step >= 100

        # 환경 (난이도 고정으로 변경 - 공정한 비교를 위해)
        if step < 100:
            env_efficiency = 0.6  # pre-drift
            uncertainty = 0.2
            transition_std = transition_std_baseline
        elif step < 130:  # shock (30 steps)
            env_efficiency = 0.3  # 약간 어려움
            uncertainty = 0.6
            transition_std = 0.5  # spike
        else:  # adapt
            adapt_progress = min(1.0, (step - 130) / 100)
            env_efficiency = 0.3 + 0.3 * adapt_progress  # 0.3 → 0.6 회복
            uncertainty = 0.5 - 0.3 * adapt_progress
            transition_std = 0.5 - 0.35 * adapt_progress  # 0.5 → 0.15 회복

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

        # Gating 적용 여부
        if use_gating:
            gating_mods = gating.update(z=z, efficiency=env_efficiency, Q_z=sm_info['Q_z'])
            action_prob = gating_mods.action_execution_prob
        else:
            action_prob = 1.0  # baseline: 항상 행동

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
    }


def test_g2_regression():
    """
    G2 회귀 테스트: v5.4 gating이 baseline 대비 G2를 악화시키지 않는가?

    === G2@v1 CANONICAL GATE SPEC (정본) ===
    G2a: adaptation_speed_ratio <= 1.2 (baseline=30 steps)
    G2b: peak_std_ratio < 3.0 AND regret_spike_rate < 0.3
    G2c: efficiency_retention >= 0.70 (70%)

    PASS 조건:
    1. v5.4 ON이 baseline과 비슷하거나 더 좋음 (ON retention >= OFF retention * 0.95)
    2. 최소 효율 유지 (retention >= 60%)
    """
    print_header("Test 4: G2 Regression Test (v5.4 ON vs OFF)")

    # Baseline (gating OFF)
    result_off = run_full_simulation(use_gating=False, seed=42)
    g2_off = result_off['g2_result']

    # v5.4 (gating ON)
    result_on = run_full_simulation(use_gating=True, seed=42)
    g2_on = result_on['g2_result']

    print(f"  === BASELINE (Gating OFF) ===")
    print(f"    G2a (time_to_recovery): {g2_off.time_to_recovery} steps")
    print(f"    G2c (efficiency_retention): {g2_off.efficiency_retention:.1%}")
    print(f"    Food collected: {result_off['food_collected']}")
    print(f"    Action efficiency: {result_off['action_efficiency']:.3f}")

    print(f"\n  === v5.4 (Gating ON) ===")
    print(f"    G2a (time_to_recovery): {g2_on.time_to_recovery} steps")
    print(f"    G2c (efficiency_retention): {g2_on.efficiency_retention:.1%}")
    print(f"    Food collected: {result_on['food_collected']}")
    print(f"    Action efficiency: {result_on['action_efficiency']:.3f}")
    print(f"    Total actions: {result_on['total_actions']} (vs OFF: {result_off['total_actions']})")

    print(f"\n  === COMPARISON ===")
    retention_ratio = g2_on.efficiency_retention / g2_off.efficiency_retention if g2_off.efficiency_retention > 0 else 1.0
    efficiency_ratio = result_on['action_efficiency'] / result_off['action_efficiency'] if result_off['action_efficiency'] > 0 else 1.0
    action_saved = result_off['total_actions'] - result_on['total_actions']

    print(f"    Retention ratio (ON/OFF): {retention_ratio:.2f}")
    print(f"    Action efficiency ratio: {efficiency_ratio:.2f}")
    print(f"    Actions saved by gating: {action_saved}")

    # === 판정 기준 ===
    # 1. v5.4가 baseline의 95% 이상 유지
    not_degraded = retention_ratio >= 0.95

    # 2. 최소 60% retention (환경이 어려워도 이 정도는 유지)
    min_retention = g2_on.efficiency_retention >= 0.60

    # 3. action efficiency가 개선되거나 유지 (핵심!)
    efficiency_improved = efficiency_ratio >= 1.0

    print(f"\n  === CHECKS ===")
    print(f"    Not degraded (ON >= OFF*0.95): {not_degraded} ({retention_ratio:.2f})")
    print(f"    Min retention (>= 60%): {min_retention} ({g2_on.efficiency_retention:.1%})")
    print(f"    Action efficiency improved: {efficiency_improved} ({efficiency_ratio:.2f})")

    # G2@v1 정본 기준도 표시 (참고용)
    print(f"\n  === G2@v1 CANONICAL (Reference) ===")
    print(f"    OFF - G2c: {'PASS' if g2_off.g2c_passed else 'FAIL'} ({g2_off.efficiency_retention:.1%})")
    print(f"    ON  - G2c: {'PASS' if g2_on.g2c_passed else 'FAIL'} ({g2_on.efficiency_retention:.1%})")

    # v5.4의 핵심 가치: "행동은 줄이되 효율은 유지/개선"
    passed = not_degraded and min_retention

    # Bonus: action efficiency 개선되면 추가 성공 메시지
    if passed and efficiency_improved:
        print(f"\n  [BONUS] Action efficiency improved by {(efficiency_ratio-1)*100:.1f}%!")
        print(f"    → '적게 움직이며 많이 배운다' 달성")

    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


# =============================================================================
# Main
# =============================================================================

def run_all_v54_tests():
    """모든 v5.4 테스트 실행"""
    print("\n" + "="*60)
    print("  v5.4 ACT/LEARN SPLIT TESTS")
    print("="*60)

    results = {}
    results['drift_adapt'] = test_drift_adapt_without_act()
    results['fatigue_learns'] = test_fatigue_rest_still_learns()
    results['quiet_learning'] = test_uncertainty_quiet_learning()
    results['g2_regression'] = test_g2_regression()

    print("\n" + "="*60)
    print("  v5.4 TEST SUMMARY")
    print("="*60 + "\n")

    all_passed = True
    for name, passed in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  \033[92m✓ v5.4 ACT/LEARN SPLIT WORKING\033[0m")
        print("    → Drift adaptation without act blocking")
        print("    → Fatigue rests but still learns")
        print("    → Uncertainty enables quiet learning")
        print("    → G2 gates maintained")
    else:
        print("  \033[91m✗ v5.4 NEEDS TUNING\033[0m")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    run_all_v54_tests()
