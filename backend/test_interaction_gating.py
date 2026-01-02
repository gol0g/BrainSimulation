"""
v5.3-2 Interaction Gating Test

검증 항목:
1. Fatigue streak → external coupling 감소
2. Fatigue streak → internal coupling 증가
3. Recovery phase → 점진적 복귀
4. Fatigue cycle 완료 → efficiency 회복

성공 조건:
- fatigue 구간에서 gating 동작 확인
- recovery 후 coupling 복귀 확인
- 순환 구조 (fatigue → gating → consolidation → recovery) 작동
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.interaction_gating import InteractionGating, InteractionGatingConfig


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_fatigue_gating():
    """Fatigue streak → gating 동작 테스트"""
    print_header("Test 1: Fatigue-Driven Gating")

    gating = InteractionGating()

    # Phase 1: 정상 상태 (z=0)
    print("  Phase 1: Stable state (z=0)")
    for _ in range(10):
        modifiers = gating.update(z=0, efficiency=0.7)

    print(f"    external_coupling: {modifiers.external_coupling}")
    print(f"    internal_coupling: {modifiers.internal_coupling}")
    stable_external = modifiers.external_coupling

    # Phase 2: Fatigue 시작 (z=3 연속)
    print("\n  Phase 2: Fatigue starts (z=3 streak)")
    for i in range(20):
        modifiers = gating.update(z=3, efficiency=0.3 - i * 0.01)
        if i in [0, 5, 10, 15, 19]:
            print(f"    step {i:2d}: external={modifiers.external_coupling:.2f}, internal={modifiers.internal_coupling:.2f}")

    fatigue_external = modifiers.external_coupling
    fatigue_internal = modifiers.internal_coupling

    # 검증
    external_decreased = fatigue_external < stable_external
    internal_increased = fatigue_internal > 1.0

    print(f"\n  Verification:")
    print(f"    External coupling decreased: {external_decreased} ({stable_external:.2f} → {fatigue_external:.2f})")
    print(f"    Internal coupling increased: {internal_increased} ({1.0} → {fatigue_internal:.2f})")

    passed = external_decreased and internal_increased
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_recovery_phase():
    """Recovery phase 동작 테스트"""
    print_header("Test 2: Recovery Phase")

    gating = InteractionGating()

    # Fatigue 구간 생성
    print("  Creating fatigue period...")
    for _ in range(15):
        gating.update(z=3, efficiency=0.2)

    print(f"    After fatigue: external={gating.state.current_external_coupling:.2f}")

    # Recovery 시작 (z != 3)
    print("\n  Recovery phase (z=0)...")
    recovery_couplings = []
    for i in range(20):
        modifiers = gating.update(z=0, efficiency=0.4 + i * 0.02)
        recovery_couplings.append(modifiers.external_coupling)
        if i in [0, 5, 10, 15, 19]:
            print(f"    step {i:2d}: external={modifiers.external_coupling:.2f}, recovery_phase={gating.state.recovery_phase}")

    # 검증: external coupling이 점진적으로 증가
    initial_recovery = recovery_couplings[0]
    final_recovery = recovery_couplings[-1]
    gradual_increase = final_recovery >= initial_recovery

    print(f"\n  Verification:")
    print(f"    Gradual external increase: {gradual_increase} ({initial_recovery:.2f} → {final_recovery:.2f})")

    passed = gradual_increase and final_recovery >= 0.9
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_full_cycle():
    """완전한 fatigue → recovery 사이클 테스트"""
    print_header("Test 3: Full Fatigue-Recovery Cycle")

    gating = InteractionGating()

    # 시뮬레이션: stable → fatigue → recovery → stable
    phases = [
        ("stable", 0, 0.7, 15),
        ("fatigue", 3, 0.2, 20),
        ("recovery", 0, 0.6, 20),
        ("stable_again", 0, 0.8, 15),
    ]

    phase_stats = []

    for phase_name, z, base_eff, steps in phases:
        phase_external = []
        phase_internal = []

        for i in range(steps):
            eff = base_eff + np.random.uniform(-0.05, 0.05)
            modifiers = gating.update(z=z, efficiency=eff)
            phase_external.append(modifiers.external_coupling)
            phase_internal.append(modifiers.internal_coupling)

        avg_ext = np.mean(phase_external)
        avg_int = np.mean(phase_internal)
        phase_stats.append((phase_name, avg_ext, avg_int))
        print(f"  {phase_name:15s}: avg_external={avg_ext:.2f}, avg_internal={avg_int:.2f}")

    # 검증
    stable_ext = phase_stats[0][1]
    fatigue_ext = phase_stats[1][1]
    recovery_ext = phase_stats[2][1]
    stable_again_ext = phase_stats[3][1]

    fatigue_int = phase_stats[1][2]

    # 패턴 확인
    fatigue_reduced = fatigue_ext < stable_ext
    recovery_increased = recovery_ext > fatigue_ext
    returned_to_normal = stable_again_ext >= 0.95
    internal_boosted = fatigue_int > 1.5

    print(f"\n  Cycle verification:")
    print(f"    Fatigue reduced external: {fatigue_reduced}")
    print(f"    Recovery increased external: {recovery_increased}")
    print(f"    Returned to normal: {returned_to_normal}")
    print(f"    Internal boosted during fatigue: {internal_boosted}")

    passed = fatigue_reduced and recovery_increased and returned_to_normal
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_gating_modifiers():
    """Gating modifier 파생 값 테스트"""
    print_header("Test 4: Gating Modifier Derivatives")

    gating = InteractionGating()

    # 정상 상태
    for _ in range(5):
        modifiers = gating.update(z=0, efficiency=0.7)

    print("  Normal state modifiers:")
    print(f"    observation_update_weight: {modifiers.observation_update_weight}")
    print(f"    action_execution_prob: {modifiers.action_execution_prob}")
    print(f"    consolidation_boost: {modifiers.consolidation_boost}")
    print(f"    replay_intensity: {modifiers.replay_intensity}")

    normal_obs_weight = modifiers.observation_update_weight
    normal_action_prob = modifiers.action_execution_prob

    # Fatigue 상태
    for _ in range(15):
        modifiers = gating.update(z=3, efficiency=0.2)

    print("\n  Fatigue state modifiers:")
    print(f"    observation_update_weight: {modifiers.observation_update_weight}")
    print(f"    action_execution_prob: {modifiers.action_execution_prob}")
    print(f"    consolidation_boost: {modifiers.consolidation_boost}")
    print(f"    replay_intensity: {modifiers.replay_intensity}")

    fatigue_obs_weight = modifiers.observation_update_weight
    fatigue_action_prob = modifiers.action_execution_prob
    fatigue_consolidation = modifiers.consolidation_boost
    fatigue_replay = modifiers.replay_intensity

    # 검증
    obs_reduced = fatigue_obs_weight < normal_obs_weight
    action_reduced = fatigue_action_prob < normal_action_prob
    consolidation_boosted = fatigue_consolidation > 1.0
    replay_boosted = fatigue_replay > 1.0

    print(f"\n  Verification:")
    print(f"    Observation update reduced: {obs_reduced}")
    print(f"    Action probability reduced: {action_reduced}")
    print(f"    Consolidation boosted: {consolidation_boosted}")
    print(f"    Replay intensity boosted: {replay_boosted}")

    passed = obs_reduced and action_reduced and consolidation_boosted and replay_boosted
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def test_z1_exception():
    """z=1 (uncertain) 상태에서 외부 결합 유지 테스트"""
    print_header("Test 5: Uncertain State External Coupling")

    gating = InteractionGating()

    # z=1에서 외부 결합이 약간 강화되어야 함
    for _ in range(10):
        modifiers = gating.update(z=1, efficiency=0.5, uncertainty=0.7)

    print(f"  z=1 (uncertain) state:")
    print(f"    external_coupling: {modifiers.external_coupling}")
    print(f"    internal_coupling: {modifiers.internal_coupling}")

    # 검증: 불확실할 때는 외부 신호가 더 중요
    external_maintained = modifiers.external_coupling >= 1.0

    print(f"\n  Verification:")
    print(f"    External coupling maintained/boosted: {external_maintained}")

    passed = external_maintained
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("  v5.3-2 INTERACTION GATING TESTS")
    print("="*60)

    results = {}
    results['fatigue_gating'] = test_fatigue_gating()
    results['recovery_phase'] = test_recovery_phase()
    results['full_cycle'] = test_full_cycle()
    results['gating_modifiers'] = test_gating_modifiers()
    results['z1_exception'] = test_z1_exception()

    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60 + "\n")

    all_passed = True
    for name, passed in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  \033[92m✓ INTERACTION GATING READY\033[0m")
        print("    → Fatigue-driven gating working")
        print("    → Recovery cycle complete")
        print("    → Ready for G2c integration test")
    else:
        print("  \033[91m✗ NEEDS TUNING\033[0m")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
