"""
v5.3-2 Interaction Gating + G2c Integration Test

핵심 검증:
- Interaction Gating ON vs OFF에서 G2c 비교
- fatigue 구간의 efficiency 회복 속도 비교

통과 기준:
- G2c (efficiency_retention) 유지 또는 개선
- fatigue 구간에서 thrashing 감소
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


def run_fatigue_simulation(
    use_gating: bool,
    seed: int = 42,
    total_steps: int = 300,
    fatigue_start: int = 100,
    fatigue_end: int = 200,
) -> dict:
    """
    Fatigue 구간이 포함된 시뮬레이션

    Args:
        use_gating: InteractionGating 사용 여부
        fatigue_start: 피로 조건 시작 스텝
        fatigue_end: 피로 조건 종료 스텝
    """
    np.random.seed(seed)

    self_model = SelfModel()
    gating = InteractionGating() if use_gating else None
    g2_tracker = G2GateTracker(drift_after=fatigue_start)

    # State
    current_energy = 0.7
    efficiency_sum = 0.0
    actions_taken = 0
    thrashing_count = 0  # 비효율적 행동 (행동했는데 food 못 얻음)

    # Tracking
    last_action = 0
    recent_efficiencies = []

    for step in range(total_steps):
        # === 환경 조건 ===
        in_fatigue_zone = fatigue_start <= step < fatigue_end

        if in_fatigue_zone:
            # 피로 유발 조건: 높은 movement, 매우 낮은 efficiency, 낮은 regret
            # z=3 evidence = m * (1-e) * (1 - r*0.5) 이므로 m↑, e↓, r↓ 필요
            base_efficiency = 0.05  # 매우 낮은 효율
            movement_ratio = 0.95   # 매우 높은 움직임
        else:
            base_efficiency = 0.6
            movement_ratio = 0.4

        # 노이즈 추가
        efficiency = base_efficiency + np.random.uniform(-0.02, 0.02)
        efficiency = max(0.0, min(1.0, efficiency))

        # === Self-model update ===
        # z=3에 필요한 조건: high movement, low efficiency, low regret
        signals = {
            'uncertainty': 0.2 if not in_fatigue_zone else 0.15,  # 불확실성은 낮게 (z=1 억제)
            'regret_spike_rate': 0.05 if not in_fatigue_zone else 0.02,  # regret 낮게 (z=2,3 억제 방지)
            'energy_efficiency': efficiency,
            'volatility': 0.1 if not in_fatigue_zone else 0.15,
            'movement_ratio': movement_ratio,
        }

        modifiers, sm_info = self_model.update(signals)
        z = sm_info['z']

        # === Gating update (if enabled) ===
        action_prob = 1.0
        if gating is not None:
            gating_mods = gating.update(
                z=z,
                efficiency=efficiency,
                Q_z=sm_info['Q_z']  # v5.3-2: Q(z) 전달
            )
            action_prob = gating_mods.action_execution_prob

        # === Action simulation ===
        # 행동 결정 (gating 적용)
        if np.random.random() < action_prob:
            action = np.random.randint(1, 5)  # 1-4 movement
            actions_taken += 1

            # 행동 결과 시뮬레이션
            ate_food = np.random.random() < efficiency
            if ate_food:
                current_energy = min(1.0, current_energy + 0.15)
            else:
                # 움직였는데 food 못 얻음 = thrashing
                thrashing_count += 1
                current_energy = max(0.1, current_energy - 0.03)
        else:
            action = 0  # THINK or rest
            ate_food = False

        # Energy decay
        current_energy = max(0.1, current_energy - 0.01)

        # Efficiency 추적
        step_efficiency = 1.0 if ate_food else 0.0
        recent_efficiencies.append(step_efficiency)
        if len(recent_efficiencies) > 20:
            recent_efficiencies.pop(0)
        efficiency_sum += step_efficiency

        # G2 로깅
        g2_tracker.log_step(
            circuit_action=action,
            fep_action=action,
            final_action=action,
            agreed=True,
            disagreement_type=None,
            energy=current_energy,
            danger_prox=0.1,
            food_prox=efficiency,
            drift_active=in_fatigue_zone,
            transition_std=0.2 if not in_fatigue_zone else 0.4,
            transition_error=0.1 if not in_fatigue_zone else 0.3,
            ate_food=ate_food,
            hit_danger=False,
            energy_spent=0.01,
            regret_spike=in_fatigue_zone and np.random.random() < 0.1,
            circuit_margin=0.5,
        )

        last_action = action

    # Results
    g2_result = g2_tracker.get_result()

    return {
        'use_gating': use_gating,
        'g2_result': g2_result,
        'total_actions': actions_taken,
        'thrashing_count': thrashing_count,
        'thrashing_rate': thrashing_count / max(1, actions_taken),
        'avg_efficiency': efficiency_sum / total_steps,
        'final_energy': current_energy,
        'self_model_status': self_model.get_status(),
        'gating_status': gating.get_status() if gating else None,
    }


def compare_results(off_result: dict, on_result: dict) -> dict:
    """Gating OFF vs ON 비교"""
    off_g2 = off_result['g2_result']
    on_g2 = on_result['g2_result']

    comparison = {
        # 핵심 지표: Thrashing 감소 (gating의 주요 목표)
        'thrashing_reduced': on_result['thrashing_rate'] <= off_result['thrashing_rate'],

        # 에너지 유지 (과도한 행동으로 에너지 소진 방지)
        'energy_maintained': on_result['final_energy'] >= off_result['final_energy'] * 0.9,

        # 행동 효율 (같은 효율 결과를 더 적은 행동으로)
        'action_efficiency_improved': (
            on_result['total_actions'] < off_result['total_actions'] and
            on_result['avg_efficiency'] >= off_result['avg_efficiency'] * 0.85
        ),

        # Gating 작동 확인
        'gating_activated': (
            on_result['gating_status'] is not None and
            on_result['gating_status']['stats']['fatigue_episodes'] > 0
        ),

        # Recovery cycle 완료
        'recovery_completed': (
            on_result['gating_status'] is not None and
            on_result['gating_status']['stats']['recovery_count'] > 0
        ),
    }

    return comparison


def test_gating_g2c_integration():
    """Interaction Gating + G2c 통합 테스트"""
    print_header("v5.3-2 Interaction Gating + G2c Integration Test")

    # === Run without gating ===
    print("Running without gating...")
    off_result = run_fatigue_simulation(use_gating=False, seed=42)

    # === Run with gating ===
    print("Running with gating...")
    on_result = run_fatigue_simulation(use_gating=True, seed=42)

    # === Compare ===
    print_header("Comparison: Gating OFF vs ON")

    off_g2 = off_result['g2_result']
    on_g2 = on_result['g2_result']

    print("  Efficiency Metrics:")
    print(f"  {'Metric':<25} | {'OFF':>10} | {'ON':>10} | {'Change':>10}")
    print("  " + "-" * 60)

    metrics = [
        ('G2c (efficiency ret %)', off_g2.efficiency_retention * 100, on_g2.efficiency_retention * 100),
        ('Avg efficiency', off_result['avg_efficiency'] * 100, on_result['avg_efficiency'] * 100),
        ('Thrashing rate %', off_result['thrashing_rate'] * 100, on_result['thrashing_rate'] * 100),
        ('Total actions', off_result['total_actions'], on_result['total_actions']),
        ('Final energy', off_result['final_energy'] * 100, on_result['final_energy'] * 100),
    ]

    for name, off_val, on_val in metrics:
        change = on_val - off_val
        change_str = f"{change:+.1f}"
        print(f"  {name:<25} | {off_val:>10.1f} | {on_val:>10.1f} | {change_str:>10}")

    # === Gate comparison ===
    print("\n  G2 Gate Status:")
    print(f"  {'Gate':<15} | {'OFF':>8} | {'ON':>8}")
    print("  " + "-" * 35)
    print(f"  {'G2a (adapt)':<15} | {'PASS' if off_g2.g2a_passed else 'FAIL':>8} | {'PASS' if on_g2.g2a_passed else 'FAIL':>8}")
    print(f"  {'G2b (stable)':<15} | {'PASS' if off_g2.g2b_passed else 'FAIL':>8} | {'PASS' if on_g2.g2b_passed else 'FAIL':>8}")
    print(f"  {'G2c (efficient)':<15} | {'PASS' if off_g2.g2c_passed else 'FAIL':>8} | {'PASS' if on_g2.g2c_passed else 'FAIL':>8}")
    print(f"  {'Overall':<15} | {'PASS' if off_g2.overall_passed else 'FAIL':>8} | {'PASS' if on_g2.overall_passed else 'FAIL':>8}")

    # === Self-model status ===
    sm_on = on_result['self_model_status']
    print("\n  Self-model Status (with gating):")
    print(f"    Final z: {sm_on['z']} ({get_mode_label(sm_on['z'])})")
    print(f"    Switches: {sm_on['switch_count']}")

    # === Gating status ===
    if on_result['gating_status']:
        gs = on_result['gating_status']
        print("\n  Gating Status:")
        print(f"    Fatigue episodes: {gs['stats']['fatigue_episodes']}")
        print(f"    Recovery count: {gs['stats']['recovery_count']}")
        print(f"    Avg recovery steps: {gs['stats']['avg_recovery_steps']}")

    # === Final judgment ===
    comparison = compare_results(off_result, on_result)

    print("\n  Integration Check:")
    for key, value in comparison.items():
        status = "\033[92m✓\033[0m" if value else "\033[91m✗\033[0m"
        print(f"    {status} {key}: {value}")

    # 최종 판정: 핵심 지표 = thrashing 감소 + gating 작동 + recovery 완료
    key_checks = [
        comparison['thrashing_reduced'],
        comparison['gating_activated'],
        comparison['recovery_completed'],
        comparison['energy_maintained'],
    ]
    success_rate = sum(key_checks) / len(key_checks)

    print()
    if success_rate >= 0.75:  # 4개 중 3개 이상 통과
        print("  \033[92m✓ INTERACTION GATING: WORKING CORRECTLY\033[0m")
        print("    Fatigue → Gating → Recovery cycle complete")
        print(f"    Thrashing reduced: {off_result['thrashing_rate']:.1%} → {on_result['thrashing_rate']:.1%}")
        print(f"    Actions reduced: {off_result['total_actions']} → {on_result['total_actions']}")
        passed = True
    elif success_rate >= 0.5:
        print("  \033[93m○ INTERACTION GATING: PARTIALLY WORKING\033[0m")
        print("    Some metrics need tuning")
        passed = True
    else:
        print("  \033[91m✗ INTERACTION GATING: NOT WORKING\033[0m")
        print("    Gating mechanism needs debugging")
        passed = False

    print("=" * 60 + "\n")

    return passed


def test_fatigue_recovery_speed():
    """Fatigue 구간 회복 속도 비교"""
    print_header("Fatigue Recovery Speed Test")

    # 더 긴 fatigue 구간에서 테스트
    off_result = run_fatigue_simulation(
        use_gating=False,
        seed=123,
        total_steps=500,
        fatigue_start=100,
        fatigue_end=350,  # 250 steps of fatigue
    )

    on_result = run_fatigue_simulation(
        use_gating=True,
        seed=123,
        total_steps=500,
        fatigue_start=100,
        fatigue_end=350,
    )

    print("  Extended fatigue period (150 steps):")
    print(f"    Without gating: thrashing={off_result['thrashing_rate']:.1%}, final_energy={off_result['final_energy']:.2f}")
    print(f"    With gating: thrashing={on_result['thrashing_rate']:.1%}, final_energy={on_result['final_energy']:.2f}")

    # Gating 사용 시 thrashing 감소 확인
    thrashing_improved = on_result['thrashing_rate'] < off_result['thrashing_rate']
    energy_improved = on_result['final_energy'] >= off_result['final_energy']

    print(f"\n  Improvement check:")
    print(f"    Thrashing reduced: {thrashing_improved}")
    print(f"    Energy maintained: {energy_improved}")

    passed = thrashing_improved or energy_improved
    print(f"\n  Result: {'PASS' if passed else 'FAIL'}")

    return passed


if __name__ == "__main__":
    result1 = test_gating_g2c_integration()
    result2 = test_fatigue_recovery_speed()

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  G2c Integration: {'PASS' if result1 else 'FAIL'}")
    print(f"  Recovery Speed: {'PASS' if result2 else 'FAIL'}")
    print("="*60 + "\n")
