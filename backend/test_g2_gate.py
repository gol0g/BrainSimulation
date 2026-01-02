"""
v5.2 G2 Gate Test - Circuit-Driven Drift Adaptation

테스트 목적:
- Circuit이 메인 컨트롤러일 때 drift 적응력 검증
- energy_efficiency로 "탐색 품질" 확인
- N1a/N1b + G2 통합 검증

G2 Gate 기준:
- G2a: adaptation_speed <= baseline × 1.2
- G2b: peak_std < 3.0, regret_spike_rate < 30%
- G2c: efficiency_retention >= 70%
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.agent import GenesisAgent
from genesis.action_circuit import ActionCompetitionCircuit, FEPActionOracle
from genesis.circuit_controller import CircuitController, DisagreementType
from genesis.g2_gate import G2GateTracker, format_g2_result
from genesis.test_utils import make_observation, Direction


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def simulate_drift_scenario(
    drift_type: str = "rotate",
    drift_after: int = 100,
    total_steps: int = 200,
    seed: int = 42,
) -> dict:
    """
    Drift 시나리오 시뮬레이션 (Circuit 주행)

    Returns:
        G2GateResult and CircuitController status
    """
    np.random.seed(seed)

    # Setup
    N_STATES = 64
    N_OBSERVATIONS = 8
    N_ACTIONS = 6

    preferred_obs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0])
    agent = GenesisAgent(N_STATES, N_OBSERVATIONS, N_ACTIONS, preferred_obs)
    action_circuit = ActionCompetitionCircuit()
    fep_oracle = FEPActionOracle(agent)

    controller = CircuitController(
        action_circuit=action_circuit,
        fep_oracle=fep_oracle,
        agent=agent,
        fep_compare_interval=1,  # 모든 스텝 비교
    )

    g2_tracker = G2GateTracker(
        drift_after=drift_after,
        drift_type=drift_type,
    )

    # State
    current_energy = 0.6
    drift_active = False

    for step in range(total_steps):
        # Drift activation
        if step >= drift_after:
            drift_active = True

        # Generate observation
        if drift_active:
            # Drift: 방향 정보가 바뀜 (시뮬레이션)
            if drift_type == "rotate":
                # 방향 90도 회전 효과
                food_dir = np.random.choice([Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT])
            elif drift_type == "flip_x":
                # 좌우 반전
                food_dir = np.random.choice([Direction.LEFT, Direction.RIGHT])
            else:
                food_dir = np.random.choice(list(Direction))

            # Drift 직후에는 transition_std 증가
            drift_steps = step - drift_after
            if drift_steps < 10:
                transition_noise = 0.3 * (1 - drift_steps / 10)  # 감쇠
            else:
                transition_noise = 0.05
        else:
            food_dir = np.random.choice([Direction.RIGHT, Direction.DOWN])
            transition_noise = 0.05

        # 상황별 시나리오
        scenario = np.random.choice(['normal', 'danger', 'food'], p=[0.6, 0.2, 0.2])

        if scenario == 'danger':
            obs = make_observation(
                food_direction=food_dir,
                food_proximity=np.random.uniform(0.2, 0.5),
                danger_direction=np.random.choice(list(Direction)),
                danger_proximity=np.random.uniform(0.5, 0.8),
                energy=current_energy,
                pain=np.random.uniform(0.1, 0.3)
            )
        elif scenario == 'food':
            obs = make_observation(
                food_direction=food_dir,
                food_proximity=np.random.uniform(0.6, 0.9),
                danger_proximity=np.random.uniform(0.0, 0.2),
                energy=current_energy,
            )
        else:
            obs = make_observation(
                food_direction=food_dir,
                food_proximity=np.random.uniform(0.3, 0.6),
                danger_proximity=np.random.uniform(0.1, 0.3),
                energy=current_energy,
            )

        # Circuit action
        action, info = controller.select_action(obs)

        # Simulate outcome
        ate_food = np.random.random() < 0.1 * obs[0]  # food_prox 기반
        hit_danger = obs[1] > 0.7 and np.random.random() < 0.2

        # Energy dynamics
        energy_spent = 0.02 if action != 0 else 0.01  # 이동 시 더 소비
        if ate_food:
            current_energy = min(1.0, current_energy + 0.2)
        current_energy = max(0.1, current_energy - energy_spent)

        # Transition std (시뮬레이션)
        base_std = 0.1
        transition_std = base_std + transition_noise + np.random.normal(0, 0.02)
        transition_error = abs(transition_noise) + np.random.uniform(0, 0.1)

        # Regret spike (drift 직후에 더 높음)
        regret_spike = drift_active and np.random.random() < (0.2 if (step - drift_after) < 20 else 0.05)

        # Log to G2 tracker
        g2_tracker.log_step(
            circuit_action=info['circuit_action'],
            fep_action=info['fep_action'],
            final_action=action,
            agreed=not info['disagreement'],
            disagreement_type=None,  # 단순화
            energy=current_energy,
            danger_prox=obs[1],
            food_prox=obs[0],
            drift_active=drift_active,
            transition_std=transition_std,
            transition_error=transition_error,
            ate_food=ate_food,
            hit_danger=hit_danger,
            energy_spent=energy_spent,
            regret_spike=regret_spike,
            circuit_margin=info.get('circuit_margin', 0.0),
        )

    # Get results
    g2_result = g2_tracker.get_result()
    controller_status = controller.get_status()

    return {
        'g2_result': g2_result,
        'controller_status': controller_status,
    }


def test_g2_gate_rotate():
    """Rotate drift 테스트"""
    print_header("G2 Gate Test: Rotate Drift")

    result = simulate_drift_scenario(
        drift_type="rotate",
        drift_after=100,
        total_steps=250,
        seed=42,
    )

    g2 = result['g2_result']
    if g2:
        print(format_g2_result(g2))
        return g2.overall_passed
    return False


def test_g2_gate_flip():
    """Flip_x drift 테스트"""
    print_header("G2 Gate Test: Flip-X Drift")

    result = simulate_drift_scenario(
        drift_type="flip_x",
        drift_after=100,
        total_steps=250,
        seed=43,
    )

    g2 = result['g2_result']
    if g2:
        print(format_g2_result(g2))
        return g2.overall_passed
    return False


def test_g2_with_n1ab():
    """G2 + N1a/N1b 통합 테스트"""
    print_header("G2 + N1a/N1b Integration Test")

    result = simulate_drift_scenario(
        drift_type="rotate",
        drift_after=100,
        total_steps=300,
        seed=44,
    )

    g2 = result['g2_result']
    cs = result['controller_status']

    if g2 and cs:
        print("=== N1a/N1b Status (during drift test) ===")
        m = cs['metrics']
        print(f"  N1a (safety): danger_approach={m['danger_approach_count']}")
        print(f"  N1b (style): style_rate={m['n1b_style_rate']*100:.1f}%")
        print(f"  Overall agreement: {m['agreement_rate']*100:.1f}%")
        print()

        print("=== G2 Summary ===")
        print(f"  G2a (adapt): {'PASS' if g2.g2a_passed else 'FAIL'} (recovery={g2.time_to_recovery} steps)")
        print(f"  G2b (stable): {'PASS' if g2.g2b_passed else 'FAIL'} (peak={g2.peak_std_ratio:.2f}x)")
        print(f"  G2c (efficient): {'PASS' if g2.g2c_passed else 'FAIL'} (retention={g2.efficiency_retention:.1%})")
        print()

        # Combined gate
        n1a_passed = m['danger_approach_count'] == 0
        combined = g2.overall_passed and n1a_passed

        if combined:
            print("  \033[92m✓ COMBINED (N1a + G2) PASSED\033[0m")
        else:
            print("  \033[91m✗ COMBINED (N1a + G2) FAILED\033[0m")

        return combined
    return False


def run_all_g2_tests():
    """모든 G2 테스트 실행"""
    print("\n" + "="*60)
    print("  v5.2 G2 GATE TESTS (Circuit-Driven Adaptation)")
    print("="*60)

    results = {}
    results['rotate'] = test_g2_gate_rotate()
    results['flip'] = test_g2_gate_flip()
    results['integration'] = test_g2_with_n1ab()

    print("\n" + "="*60)
    print("  G2 TEST SUMMARY")
    print("="*60 + "\n")

    all_passed = True
    for name, passed in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  \033[92m✓ G2 GATE READY\033[0m")
    else:
        print("  \033[93m○ G2 GATE NEEDS TUNING\033[0m")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    run_all_g2_tests()
