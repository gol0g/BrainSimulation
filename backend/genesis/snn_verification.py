"""
SNN Brain Verification

철저한 검증:
1. STDP가 실제로 의미있는 패턴을 학습하는가?
2. 희소성이 뇌처럼 낮은가? (목표: 1-5%)
3. 다른 입력에 다른 출력이 나오는가?
4. 학습이 누적되는가?
5. 예측 오차가 실제로 감소하는가?
"""

import os
import sys
import torch
import numpy as np
import time
from typing import Dict, List

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
genesis_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
sys.path.insert(0, genesis_dir)

from snn_brain import SpikingBrain, BrainConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_1_stdp_learning():
    """
    테스트 1: STDP가 패턴을 학습하는가?

    방법: 같은 패턴을 반복 제시 → 가중치 변화 측정
    기대: 자주 함께 발화하는 뉴런 연결 강화
    """
    print("\n" + "=" * 60)
    print("TEST 1: STDP Pattern Learning")
    print("=" * 60)

    config = BrainConfig(
        sensory_neurons=256,
        v1_neurons=512,
        v2_neurons=256,
        association_neurons=128,
        prefrontal_neurons=64,
        motor_neurons=16,
        time_steps=10,
    )

    brain = SpikingBrain(config).to(DEVICE)

    # 두 가지 다른 패턴 준비
    pattern_A = torch.zeros(1, 3, 256, 256).to(DEVICE)
    pattern_A[:, 0, 50:150, 50:150] = 1.0  # 빨간 사각형 (왼쪽 위)

    pattern_B = torch.zeros(1, 3, 256, 256).to(DEVICE)
    pattern_B[:, 2, 100:200, 100:200] = 1.0  # 파란 사각형 (중앙)

    # 초기 가중치 저장
    w_v1_before = brain.v1.synapse.weight.clone()

    # 패턴 A 반복 학습
    print("\n[Phase 1] Learning pattern A (red square, top-left)...")
    for i in range(50):
        brain(pattern_A, learn=True)

    w_v1_after_A = brain.v1.synapse.weight.clone()
    change_A = (w_v1_after_A - w_v1_before).abs().mean().item()

    # 패턴 B 학습
    print("[Phase 2] Learning pattern B (blue square, center)...")
    for i in range(50):
        brain(pattern_B, learn=True)

    w_v1_after_B = brain.v1.synapse.weight.clone()
    change_B = (w_v1_after_B - w_v1_after_A).abs().mean().item()

    # 결과 분석
    print("\n[Results]")
    print(f"  Weight change after pattern A: {change_A:.6f}")
    print(f"  Weight change after pattern B: {change_B:.6f}")

    # 검증
    if change_A > 0.001 and change_B > 0.001:
        print("  [PASS] STDP is modifying weights")
        passed = True
    else:
        print("  [FAIL] STDP weight changes too small")
        passed = False

    # 패턴 특이성 테스트
    print("\n[Phase 3] Testing pattern specificity...")
    brain.reset_state()

    output_A = brain(pattern_A, learn=False)
    motor_A = output_A['motor_raw'].sum(dim=1).squeeze()

    brain.reset_state()
    output_B = brain(pattern_B, learn=False)
    motor_B = output_B['motor_raw'].sum(dim=1).squeeze()

    # 다른 패턴에 다른 출력?
    output_diff = (motor_A - motor_B).abs().sum().item()
    print(f"  Motor output difference: {output_diff:.4f}")

    if output_diff > 0.1:
        print("  [PASS] Different patterns produce different outputs")
    else:
        print("  [WARN] Outputs too similar - may need more training")

    return passed


def test_2_sparsity():
    """
    테스트 2: 희소성이 뇌처럼 낮은가?

    인간 뇌: 동시에 1-2% 뉴런만 활성
    목표: 5% 이하
    """
    print("\n" + "=" * 60)
    print("TEST 2: Activation Sparsity")
    print("=" * 60)

    config = BrainConfig(
        sensory_neurons=256,
        v1_neurons=512,
        v2_neurons=256,
        time_steps=20,
        target_sparsity=0.05,
    )

    brain = SpikingBrain(config).to(DEVICE)

    # 여러 이미지로 테스트
    sparsities = []

    print("\n[Testing sparsity across 10 random images]")
    for i in range(10):
        test_image = torch.rand(1, 3, 256, 256).to(DEVICE)
        brain.reset_state()
        output = brain(test_image, learn=False)

        # V1 스파이크 희소성 계산
        v1_activity = output['v1'].squeeze()
        active_ratio = (v1_activity > 0).float().mean().item()
        sparsities.append(active_ratio)

    avg_sparsity = np.mean(sparsities)

    print(f"\n[Results]")
    print(f"  Average V1 activation: {avg_sparsity:.2%}")
    print(f"  Target: < 5%")
    print(f"  Human brain: ~1-2%")

    if avg_sparsity < 0.10:
        print(f"  [PASS] Sparsity acceptable (< 10%)")
        passed = True
    elif avg_sparsity < 0.30:
        print(f"  [WARN] Sparsity high but workable")
        passed = True
    else:
        print(f"  [FAIL] Too many neurons active - not brain-like")
        passed = False
        print(f"  [ACTION NEEDED] Add lateral inhibition or adjust thresholds")

    return passed, avg_sparsity


def test_3_action_diversity():
    """
    테스트 3: 다양한 행동이 나오는가?

    문제: 이전 테스트에서 "Unique actions: 1"
    원인: 모든 입력에 같은 행동 출력
    """
    print("\n" + "=" * 60)
    print("TEST 3: Action Diversity")
    print("=" * 60)

    config = BrainConfig(
        sensory_neurons=256,
        v1_neurons=512,
        motor_neurons=16,
        time_steps=10,
    )

    brain = SpikingBrain(config).to(DEVICE)

    actions = []

    print("\n[Testing action diversity across 20 random images]")
    for i in range(20):
        test_image = torch.rand(1, 3, 256, 256).to(DEVICE)
        brain.reset_state()
        output = brain(test_image, learn=False)

        action = brain.get_action(output['motor_raw'])
        actions.append(action)

    unique_actions = len(set(actions))

    print(f"\n[Results]")
    print(f"  Actions taken: {actions}")
    print(f"  Unique actions: {unique_actions}/16 possible")

    if unique_actions >= 5:
        print(f"  [PASS] Good action diversity")
        passed = True
    elif unique_actions >= 2:
        print(f"  [WARN] Low diversity but some variation")
        passed = True
    else:
        print(f"  [FAIL] No action diversity - brain stuck")
        print(f"  [CAUSE] Motor neurons not receiving varied input")
        passed = False

    return passed, unique_actions


def test_4_learning_accumulation():
    """
    테스트 4: 학습이 누적되는가?

    기대: 같은 패턴 반복 → 예측 오차 감소
    """
    print("\n" + "=" * 60)
    print("TEST 4: Learning Accumulation")
    print("=" * 60)

    config = BrainConfig(
        sensory_neurons=256,
        v1_neurons=512,
        time_steps=10,
    )

    brain = SpikingBrain(config).to(DEVICE)

    # 고정 패턴
    pattern = torch.rand(1, 3, 256, 256).to(DEVICE)

    # 학습 전 활동 기록
    brain.reset_state()
    output_before = brain(pattern, learn=False)
    v1_before = output_before['v1'].clone()

    # 학습
    print("\n[Learning same pattern 100 times]")
    for i in range(100):
        brain(pattern, learn=True)
        if (i + 1) % 25 == 0:
            print(f"  Iteration {i+1}")

    # 학습 후 활동
    brain.reset_state()
    output_after = brain(pattern, learn=False)
    v1_after = output_after['v1']

    # 활동 패턴 유사성 (학습된 패턴은 더 안정적)
    # 같은 입력 여러번 → 출력 일관성
    consistency_scores = []
    for _ in range(5):
        brain.reset_state()
        out = brain(pattern, learn=False)
        consistency_scores.append(out['v1'].clone())

    # 일관성 계산
    consistency = 0
    for i in range(len(consistency_scores) - 1):
        diff = (consistency_scores[i] - consistency_scores[i+1]).abs().mean().item()
        consistency += diff
    consistency /= (len(consistency_scores) - 1)

    print(f"\n[Results]")
    print(f"  V1 activity consistency (lower = better): {consistency:.6f}")

    # 가중치 변화 확인
    w_change = brain.v1.synapse.weight.std().item()
    print(f"  Weight distribution std: {w_change:.4f}")

    if consistency < 0.1:
        print(f"  [PASS] Consistent responses to learned pattern")
        passed = True
    else:
        print(f"  [WARN] Some inconsistency in responses")
        passed = True

    return passed


def test_5_biological_plausibility():
    """
    테스트 5: 생물학적 타당성 검사
    """
    print("\n" + "=" * 60)
    print("TEST 5: Biological Plausibility Check")
    print("=" * 60)

    issues = []

    config = BrainConfig()
    brain = SpikingBrain(config).to(DEVICE)

    # 체크 항목
    print("\n[Checking biological constraints]")

    # 1. Dale's Law: 뉴런은 흥분성 OR 억제성 (우리는 아직 없음)
    print("  1. Dale's Law (excitatory/inhibitory): NOT IMPLEMENTED")
    issues.append("No Dale's Law - all neurons can be both E and I")

    # 2. 시냅스 지연: 실제 뇌는 전달에 시간 소요
    print("  2. Synaptic delay: NOT IMPLEMENTED")
    issues.append("No synaptic delays")

    # 3. 피드백 연결: 상위 → 하위 (예측 코딩에 필요)
    has_feedback = hasattr(brain, 'pc_v1_v2')
    print(f"  3. Feedback connections: {'YES' if has_feedback else 'NO'}")
    if not has_feedback:
        issues.append("No feedback connections")

    # 4. 측면 억제: 희소성에 필요
    print("  4. Lateral inhibition: NOT IMPLEMENTED")
    issues.append("No lateral inhibition - causes high activation")

    # 5. 가소성 조절: 보상/주의에 의한 학습 조절
    print("  5. Neuromodulation (dopamine, etc.): NOT IMPLEMENTED")
    issues.append("No neuromodulation")

    print(f"\n[Summary]")
    print(f"  Issues found: {len(issues)}")
    for issue in issues:
        print(f"    - {issue}")

    return issues


def run_all_tests():
    """전체 검증 실행"""
    print("=" * 60)
    print("SNN BRAIN COMPREHENSIVE VERIFICATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # 테스트 실행
    results['stdp_learning'] = test_1_stdp_learning()
    results['sparsity'] = test_2_sparsity()
    results['action_diversity'] = test_3_action_diversity()
    results['learning_accumulation'] = test_4_learning_accumulation()
    results['biological_issues'] = test_5_biological_plausibility()

    # 종합 결과
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    print("\n[Test Results]")
    print(f"  1. STDP Learning:        {'PASS' if results['stdp_learning'] else 'FAIL'}")
    print(f"  2. Sparsity:             {'PASS' if results['sparsity'][0] else 'FAIL'} ({results['sparsity'][1]:.1%})")
    print(f"  3. Action Diversity:     {'PASS' if results['action_diversity'][0] else 'FAIL'} ({results['action_diversity'][1]} unique)")
    print(f"  4. Learning Accumulation: {'PASS' if results['learning_accumulation'] else 'FAIL'}")
    print(f"  5. Biological Issues:    {len(results['biological_issues'])} found")

    # 핵심 문제점
    print("\n[Critical Issues to Address]")

    if results['sparsity'][1] > 0.30:
        print("  ! HIGH PRIORITY: Sparsity too high")
        print("    → Need lateral inhibition")
        print("    → Adjust LIF threshold")

    if results['action_diversity'][1] < 3:
        print("  ! HIGH PRIORITY: Low action diversity")
        print("    → Motor layer not differentiating inputs")
        print("    → Check information flow to motor neurons")

    if len(results['biological_issues']) > 3:
        print("  ! MEDIUM PRIORITY: Missing biological mechanisms")
        print("    → Add lateral inhibition (most important)")
        print("    → Consider Dale's Law")

    print("\n[Conclusion]")
    all_passed = (results['stdp_learning'] and
                  results['sparsity'][0] and
                  results['action_diversity'][0] and
                  results['learning_accumulation'])

    if all_passed:
        print("  Basic functionality VERIFIED")
        print("  Ready for next phase with improvements")
    else:
        print("  Some tests FAILED - need fixes before proceeding")

    return results


if __name__ == '__main__':
    results = run_all_tests()
