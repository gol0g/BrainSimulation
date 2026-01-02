"""
v5.2 Self-model Test - Completion Criteria Validation

완료 조건 4가지:
1. 모드 전이 관성: z가 매 스텝 flip하지 않음 (stability_score > 0.7)
2. 자원-불확실성 정합: uncertainty↑ → 자동 자원 재배치
3. 피로/비효율 억제: z3 활성화 시 exploration 감소
4. (별도 테스트) G2 유지/개선

핵심 검증:
- Self-model이 action을 고르지 않음 (output은 modifier만)
- 라벨은 테스트/디버깅용일 뿐
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.self_model import SelfModel, SelfModelConfig, get_mode_label


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_mode_transition_inertia():
    """
    완료 조건 1: 모드 전이 관성

    z가 매 스텝 flip하지 않고, 상황 변화에서만 전이
    """
    print_header("Test 1: Mode Transition Inertia")

    model = SelfModel()

    # 시나리오: 안정 → 불확실 → 후회 → 피로 → 안정
    scenarios = [
        # (name, steps, signals)
        ("stable_phase", 50, {'uncertainty': 0.2, 'regret_spike_rate': 0.05, 'energy_efficiency': 0.8, 'volatility': 0.1, 'movement_ratio': 0.5}),
        ("uncertain_phase", 40, {'uncertainty': 0.7, 'regret_spike_rate': 0.05, 'energy_efficiency': 0.6, 'volatility': 0.3, 'movement_ratio': 0.6}),
        ("regret_phase", 30, {'uncertainty': 0.5, 'regret_spike_rate': 0.6, 'energy_efficiency': 0.5, 'volatility': 0.4, 'movement_ratio': 0.5}),
        ("fatigue_phase", 40, {'uncertainty': 0.3, 'regret_spike_rate': 0.1, 'energy_efficiency': 0.2, 'volatility': 0.2, 'movement_ratio': 0.9}),
        ("recovery_phase", 40, {'uncertainty': 0.2, 'regret_spike_rate': 0.05, 'energy_efficiency': 0.7, 'volatility': 0.1, 'movement_ratio': 0.4}),
    ]

    phase_modes = []

    for name, steps, signals in scenarios:
        mode_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _ in range(steps):
            # 약간의 노이즈 추가
            noisy_signals = {
                k: v + np.random.normal(0, 0.05)
                for k, v in signals.items()
            }
            noisy_signals = {k: max(0, min(1, v)) for k, v in noisy_signals.items()}

            modifiers, info = model.update(noisy_signals)
            mode_counts[info['z']] += 1

        dominant_mode = max(mode_counts, key=mode_counts.get)
        phase_modes.append((name, dominant_mode, mode_counts))
        print(f"  {name}: z={dominant_mode} ({get_mode_label(dominant_mode)})")
        print(f"    distribution: {mode_counts}")

    # 안정성 분석
    stability = model.get_mode_stability()
    print(f"\n  Stability Analysis:")
    print(f"    Avg steps per mode: {stability['avg_steps_per_mode']:.1f}")
    print(f"    Flip-flop count: {stability['flip_flop_count']}")
    print(f"    Stability score: {stability['stability_score']:.3f}")

    # 통과 기준: stability_score > 0.7
    passed = stability['stability_score'] >= 0.7
    print(f"\n  Result: {'PASS' if passed else 'FAIL'} (threshold: stability >= 0.7)")

    return passed


def test_resource_uncertainty_alignment():
    """
    완료 조건 2: 자원-불확실성 정합

    uncertainty↑ → THINK/학습/억제가 자동으로 바뀌는 패턴
    """
    print_header("Test 2: Resource-Uncertainty Alignment")

    model = SelfModel()

    # 불확실성 레벨별 테스트
    uncertainty_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = []
    for u in uncertainty_levels:
        # 안정화를 위해 20스텝 실행
        for _ in range(20):
            signals = {
                'uncertainty': u,
                'regret_spike_rate': 0.1,
                'energy_efficiency': 0.5,
                'volatility': u * 0.5,  # uncertainty와 약간 연동
                'movement_ratio': 0.5,
            }
            modifiers, info = model.update(signals)

        results.append({
            'uncertainty': u,
            'z': info['z'],
            'think_budget': modifiers.think_budget,
            'recall_weight': modifiers.recall_weight,
            'learning_rate': modifiers.learning_rate,
            'prior_strength': modifiers.prior_strength,
        })

    print("  Uncertainty → Resource Mapping:")
    print(f"  {'U':>5} | {'z':>2} | {'THINK':>6} | {'Recall':>6} | {'LR':>6} | {'Prior':>6}")
    print("  " + "-" * 50)

    for r in results:
        print(f"  {r['uncertainty']:>5.1f} | {r['z']:>2} | {r['think_budget']:>6.2f} | {r['recall_weight']:>6.2f} | {r['learning_rate']:>6.2f} | {r['prior_strength']:>6.2f}")

    # 검증: uncertainty 높을수록 THINK↑, prior↓ 경향
    low_u = results[0]
    high_u = results[-1]

    think_increases = high_u['think_budget'] > low_u['think_budget']
    prior_decreases = high_u['prior_strength'] < low_u['prior_strength']
    recall_decreases = high_u['recall_weight'] < low_u['recall_weight']

    alignment_score = sum([think_increases, prior_decreases, recall_decreases]) / 3

    print(f"\n  Alignment check:")
    print(f"    THINK increases with uncertainty: {think_increases}")
    print(f"    Prior decreases with uncertainty: {prior_decreases}")
    print(f"    Recall decreases with uncertainty: {recall_decreases}")
    print(f"    Alignment score: {alignment_score:.1%}")

    passed = alignment_score >= 0.67  # 2/3 이상
    print(f"\n  Result: {'PASS' if passed else 'FAIL'} (threshold: 2/3 aligned)")

    return passed


def test_fatigue_suppression():
    """
    완료 조건 3: 피로/비효율 억제

    energy_efficiency↓ + high movement → 탐색 억제, 통합 증가
    """
    print_header("Test 3: Fatigue/Inefficiency Suppression")

    model = SelfModel()

    # Phase 1: 효율적 탐색
    print("  Phase 1: Efficient exploration (high efficiency)")
    for _ in range(30):
        signals = {
            'uncertainty': 0.4,
            'regret_spike_rate': 0.05,
            'energy_efficiency': 0.8,  # 높은 효율
            'volatility': 0.1,
            'movement_ratio': 0.7,  # 적극적 이동
        }
        modifiers, info = model.update(signals)

    efficient_modifiers = modifiers
    efficient_z = info['z']
    print(f"    z={efficient_z} ({get_mode_label(efficient_z)})")
    print(f"    think_budget={modifiers.think_budget:.2f}, sleep_prob={modifiers.sleep_prob:.2f}")

    # Phase 2: 비효율적 탐색 (헛움직임)
    print("\n  Phase 2: Inefficient exploration (low efficiency, high movement)")
    for _ in range(40):
        signals = {
            'uncertainty': 0.3,
            'regret_spike_rate': 0.1,
            'energy_efficiency': 0.15,  # 낮은 효율
            'volatility': 0.2,
            'movement_ratio': 0.9,  # 과도한 이동
        }
        modifiers, info = model.update(signals)

    inefficient_modifiers = modifiers
    inefficient_z = info['z']
    print(f"    z={inefficient_z} ({get_mode_label(inefficient_z)})")
    print(f"    think_budget={modifiers.think_budget:.2f}, sleep_prob={modifiers.sleep_prob:.2f}")

    # 검증
    print("\n  Fatigue suppression check:")

    # 피로 상태에서:
    # - think_budget 감소 (과도한 deliberation 줄임)
    # - sleep_prob 증가 (통합/휴식 강화)
    # - prior_strength 증가 (습관 모드)
    think_reduced = inefficient_modifiers.think_budget < efficient_modifiers.think_budget
    sleep_increased = inefficient_modifiers.sleep_prob > efficient_modifiers.sleep_prob
    prior_increased = inefficient_modifiers.prior_strength > efficient_modifiers.prior_strength

    print(f"    THINK budget reduced: {think_reduced} ({efficient_modifiers.think_budget:.2f} → {inefficient_modifiers.think_budget:.2f})")
    print(f"    Sleep prob increased: {sleep_increased} ({efficient_modifiers.sleep_prob:.2f} → {inefficient_modifiers.sleep_prob:.2f})")
    print(f"    Prior strength increased: {prior_increased} ({efficient_modifiers.prior_strength:.2f} → {inefficient_modifiers.prior_strength:.2f})")

    suppression_score = sum([think_reduced, sleep_increased, prior_increased]) / 3

    passed = suppression_score >= 0.67
    print(f"\n  Result: {'PASS' if passed else 'FAIL'} (threshold: 2/3 suppression effects)")

    return passed


def test_output_is_modifiers_only():
    """
    핵심 원칙 검증: Self-model은 action을 고르지 않음

    출력이 ResourceModifiers만인지 확인
    """
    print_header("Test: Output is Modifiers Only (No Action Selection)")

    model = SelfModel()

    signals = {
        'uncertainty': 0.5,
        'regret_spike_rate': 0.1,
        'energy_efficiency': 0.5,
        'volatility': 0.2,
        'movement_ratio': 0.5,
    }

    modifiers, info = model.update(signals)

    # 검증: modifiers에 action 관련 필드 없음
    modifier_fields = set(modifiers.to_dict().keys())
    expected_fields = {'think_budget', 'recall_weight', 'sleep_prob', 'learning_rate', 'prior_strength'}

    print(f"  Modifier fields: {modifier_fields}")
    print(f"  Expected fields: {expected_fields}")

    # action, choice, selected 등의 필드가 없어야 함
    forbidden_patterns = ['action', 'choice', 'select', 'move', 'direction']
    has_action_field = any(
        pattern in str(modifier_fields).lower()
        for pattern in forbidden_patterns
    )

    fields_correct = modifier_fields == expected_fields
    no_action = not has_action_field

    print(f"\n  Fields match expected: {fields_correct}")
    print(f"  No action-related fields: {no_action}")

    passed = fields_correct and no_action
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def run_all_self_model_tests():
    """모든 Self-model 테스트 실행"""
    print("\n" + "="*60)
    print("  v5.2 SELF-MODEL COMPLETION CRITERIA TESTS")
    print("="*60)

    results = {}
    results['output_only_modifiers'] = test_output_is_modifiers_only()
    results['mode_inertia'] = test_mode_transition_inertia()
    results['uncertainty_alignment'] = test_resource_uncertainty_alignment()
    results['fatigue_suppression'] = test_fatigue_suppression()

    print("\n" + "="*60)
    print("  SELF-MODEL TEST SUMMARY")
    print("="*60 + "\n")

    all_passed = True
    for name, passed in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  \033[92m✓ SELF-MODEL COMPLETION CRITERIA MET\033[0m")
        print("    → Ready for G2 integration test")
    else:
        print("  \033[91m✗ SELF-MODEL NEEDS TUNING\033[0m")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    run_all_self_model_tests()
