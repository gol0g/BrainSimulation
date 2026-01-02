"""
Self-model Activation Smoke Test

목적: z=1/z=3이 "필요한 순간"에 실제로 발화하는지 검증
- v5.4 검증이 비어버리는 것 방지
- flip-flop 없이 적절한 전환만 확인

시나리오:
1. Extreme shock: prediction error/volatility 강제 증가 → z=1 발화 확인
2. Extreme fatigue: energy decay↑ + food sparse → z=3 발화 확인
3. Stability check: 안정 구간에서 z=0 유지 확인 (flip-flop 없음)
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.self_model import SelfModel, get_mode_label


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_extreme_shock_activates_z1():
    """
    Extreme shock에서 z=1(불확실) 발화 확인

    조건: uncertainty 0.9+, volatility 0.8+, regret 낮음
    기대: z=1이 최소 10% 이상 발화
    """
    print_header("Smoke Test 1: Extreme Shock → z=1 Activation")

    model = SelfModel()

    # Warm-up: 안정 상태
    for _ in range(30):
        signals = {
            'uncertainty': 0.2,
            'regret_spike_rate': 0.05,
            'energy_efficiency': 0.7,
            'volatility': 0.1,
            'movement_ratio': 0.5,
        }
        model.update(signals)

    # Extreme shock: 매우 강한 불확실성 신호
    z_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    shock_steps = 50

    for step in range(shock_steps):
        signals = {
            'uncertainty': 0.95,       # 극도로 높은 불확실성
            'regret_spike_rate': 0.05, # regret 낮음 (z=2 억제)
            'energy_efficiency': 0.3,  # 효율 낮음 (새 환경)
            'volatility': 0.85,        # 극도로 높은 변동성
            'movement_ratio': 0.6,     # 적당한 움직임
        }

        modifiers, info = model.update(signals)
        z = info['z']
        z_counts[z] += 1

        if step < 5 or step % 10 == 0:
            Q_str = [f'{q:.2f}' for q in info['Q_z']]
            print(f"    step {step:2d}: z={z}, Q={Q_str}")

    print(f"\n  Z distribution over {shock_steps} shock steps:")
    for z, count in z_counts.items():
        pct = count / shock_steps * 100
        print(f"    z={z} ({get_mode_label(z):10s}): {count:3d} ({pct:5.1f}%)")

    # 검증: z=1이 최소 10% (5회) 이상
    z1_rate = z_counts[1] / shock_steps
    passed = z1_rate >= 0.10

    print(f"\n  z=1 activation rate: {z1_rate:.1%}")
    print(f"  Threshold: >= 10%")
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed, z_counts


def test_extreme_fatigue_activates_z3():
    """
    Extreme fatigue에서 z=3(피로) 발화 확인

    조건: efficiency 0.1 이하, movement 0.9+, uncertainty/regret 낮음
    기대: z=3이 최소 10% 이상 발화
    """
    print_header("Smoke Test 2: Extreme Fatigue → z=3 Activation")

    model = SelfModel()

    # Warm-up: 안정 상태
    for _ in range(30):
        signals = {
            'uncertainty': 0.2,
            'regret_spike_rate': 0.05,
            'energy_efficiency': 0.7,
            'volatility': 0.1,
            'movement_ratio': 0.5,
        }
        model.update(signals)

    # Extreme fatigue: 매우 낮은 효율 + 높은 움직임
    z_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    fatigue_steps = 50

    for step in range(fatigue_steps):
        signals = {
            'uncertainty': 0.15,        # 불확실성 낮음 (z=1 억제)
            'regret_spike_rate': 0.02,  # regret 매우 낮음 (z=2 억제)
            'energy_efficiency': 0.05,  # 극도로 낮은 효율
            'volatility': 0.15,         # 변동성 낮음
            'movement_ratio': 0.95,     # 극도로 높은 움직임 (헛발질)
        }

        modifiers, info = model.update(signals)
        z = info['z']
        z_counts[z] += 1

        if step < 5 or step % 10 == 0:
            Q_str = [f'{q:.2f}' for q in info['Q_z']]
            print(f"    step {step:2d}: z={z}, Q={Q_str}")

    print(f"\n  Z distribution over {fatigue_steps} fatigue steps:")
    for z, count in z_counts.items():
        pct = count / fatigue_steps * 100
        print(f"    z={z} ({get_mode_label(z):10s}): {count:3d} ({pct:5.1f}%)")

    # 검증: z=3이 최소 10% (5회) 이상
    z3_rate = z_counts[3] / fatigue_steps
    passed = z3_rate >= 0.10

    print(f"\n  z=3 activation rate: {z3_rate:.1%}")
    print(f"  Threshold: >= 10%")
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed, z_counts


def test_stability_no_flipflop():
    """
    안정 구간에서 z=0 유지, flip-flop 없음 확인

    조건: 안정적 신호 (낮은 uncertainty, 높은 efficiency)
    기대: z=0이 95% 이상, 전환 횟수 3회 이하
    """
    print_header("Smoke Test 3: Stability → No Flip-Flop")

    model = SelfModel()

    # 장기 안정 구간
    z_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    stable_steps = 100

    for step in range(stable_steps):
        # 약간의 노이즈가 있는 안정 신호
        signals = {
            'uncertainty': 0.2 + np.random.uniform(-0.05, 0.05),
            'regret_spike_rate': 0.05 + np.random.uniform(-0.02, 0.02),
            'energy_efficiency': 0.7 + np.random.uniform(-0.1, 0.1),
            'volatility': 0.1 + np.random.uniform(-0.03, 0.03),
            'movement_ratio': 0.5 + np.random.uniform(-0.1, 0.1),
        }

        modifiers, info = model.update(signals)
        z = info['z']
        z_counts[z] += 1

    stability = model.get_mode_stability()

    print(f"  Z distribution over {stable_steps} stable steps:")
    for z, count in z_counts.items():
        pct = count / stable_steps * 100
        print(f"    z={z} ({get_mode_label(z):10s}): {count:3d} ({pct:5.1f}%)")

    print(f"\n  Mode stability:")
    print(f"    Avg steps per mode: {stability['avg_steps_per_mode']:.1f}")
    print(f"    Flip-flop count: {stability['flip_flop_count']}")
    print(f"    Stability score: {stability['stability_score']:.3f}")

    # 검증
    z0_rate = z_counts[0] / stable_steps
    z0_dominant = z0_rate >= 0.95
    no_flipflop = stability['flip_flop_count'] <= 3

    print(f"\n  Checks:")
    print(f"    z=0 >= 95%: {z0_dominant} ({z0_rate:.1%})")
    print(f"    Flip-flop <= 3: {no_flipflop}")

    passed = z0_dominant and no_flipflop
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


def run_activation_smoke_tests():
    """모든 activation smoke 테스트 실행"""
    print("\n" + "="*60)
    print("  SELF-MODEL ACTIVATION SMOKE TESTS")
    print("="*60)

    results = {}

    passed1, z1_counts = test_extreme_shock_activates_z1()
    results['shock_z1'] = passed1

    passed2, z3_counts = test_extreme_fatigue_activates_z3()
    results['fatigue_z3'] = passed2

    passed3 = test_stability_no_flipflop()
    results['stability'] = passed3

    print("\n" + "="*60)
    print("  ACTIVATION SMOKE TEST SUMMARY")
    print("="*60 + "\n")

    all_passed = True
    for name, passed in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  \033[92m✓ SELF-MODEL ACTIVATES PROPERLY\033[0m")
        print("    → z=1 fires on extreme shock")
        print("    → z=3 fires on extreme fatigue")
        print("    → z=0 stable without flip-flop")
    else:
        print("  \033[91m✗ SELF-MODEL NEEDS TUNING\033[0m")
        if not results.get('shock_z1'):
            print("    → z=1 not responding to shock signals")
        if not results.get('fatigue_z3'):
            print("    → z=3 not responding to fatigue signals")
        if not results.get('stability'):
            print("    → Stability issues detected")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    run_activation_smoke_tests()
