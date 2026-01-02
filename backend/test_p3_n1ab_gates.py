"""
v5.2 P3: N1a/N1b Gate Verification

P3-2 결과 검증:
- N1a (Safety-critical): danger_approach=0, safety_rate >= 95%
- N1b (Style): style_rate >= 50% (soft target)

Goal: N1a PASS는 필수, N1b는 참고용
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.agent import GenesisAgent
from genesis.action_circuit import ActionCompetitionCircuit, FEPActionOracle
from genesis.circuit_controller import CircuitController, DisagreementType
from genesis.test_utils import make_observation, Direction


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_n1ab_test(n_steps: int = 500):
    """N1a/N1b 게이트 검증"""
    print_header("P3: N1a/N1b Gate Verification")

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

    # === Run diverse scenarios ===
    np.random.seed(42)

    scenario_counts = {
        'safe': 0,
        'danger': 0,
        'low_energy_food': 0,
        'exploration': 0
    }

    print(f"Running {n_steps} steps with diverse scenarios...")

    for i in range(n_steps):
        # 다양한 시나리오 생성
        scenario_type = np.random.choice(
            ['safe', 'danger', 'low_energy_food', 'exploration'],
            p=[0.4, 0.2, 0.15, 0.25]
        )

        if scenario_type == 'safe':
            obs = make_observation(
                food_direction=np.random.choice(list(Direction)),
                food_proximity=np.random.uniform(0.3, 0.8),
                danger_proximity=np.random.uniform(0.0, 0.3),  # Low danger
                energy=np.random.uniform(0.5, 1.0),
            )
            scenario_counts['safe'] += 1

        elif scenario_type == 'danger':
            # Safety-critical: High danger
            obs = make_observation(
                food_direction=np.random.choice(list(Direction)),
                food_proximity=np.random.uniform(0.2, 0.6),
                danger_direction=np.random.choice(list(Direction)),
                danger_proximity=np.random.uniform(0.5, 0.9),  # High danger
                energy=np.random.uniform(0.4, 0.8),
                pain=np.random.uniform(0.1, 0.4)
            )
            scenario_counts['danger'] += 1

        elif scenario_type == 'low_energy_food':
            # Safety-critical: Low energy + food nearby
            obs = make_observation(
                food_direction=np.random.choice(list(Direction)),
                food_proximity=np.random.uniform(0.5, 0.9),  # Food nearby
                danger_proximity=np.random.uniform(0.0, 0.3),
                energy=np.random.uniform(0.1, 0.4),  # Low energy
            )
            scenario_counts['low_energy_food'] += 1

        else:  # exploration
            obs = make_observation(
                food_direction=np.random.choice(list(Direction)),
                food_proximity=np.random.uniform(0.1, 0.5),
                danger_proximity=np.random.uniform(0.0, 0.4),
                energy=np.random.uniform(0.4, 0.8),
            )
            scenario_counts['exploration'] += 1

        controller.select_action(obs)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{n_steps}...")

    # === Results ===
    print_header("N1a/N1b Results")

    status = controller.get_status()
    metrics = status['metrics']

    print(f"Scenario distribution:")
    for name, count in scenario_counts.items():
        print(f"  {name}: {count} ({count/n_steps*100:.1f}%)")

    print(f"\n=== Overall Metrics ===")
    print(f"  Total comparisons: {metrics['agreements'] + metrics['disagreements']}")
    print(f"  Overall agreement rate: {metrics['agreement_rate']*100:.1f}%")

    print(f"\n=== N1a: Safety-Critical (Danger Avoidance) ===")
    print(f"  Danger approach count: {metrics['danger_approach_count']} (must be 0)")
    print(f"  → N1a = danger_approach == 0")

    print(f"\n=== N1a-food: Food Seeking (Reference Only) ===")
    safety_total = metrics['safety_agreements'] + metrics['safety_disagreements']
    print(f"  Safety contexts: {safety_total}")
    print(f"  Safety rate: {metrics['n1a_safety_rate']*100:.1f}%")
    print(f"  Food ignore count: {metrics['food_ignore_count']} (not a hard gate)")

    print(f"\n=== N1b: Style Alignment ===")
    style_total = metrics['style_agreements'] + metrics['style_disagreements']
    print(f"  Style contexts: {style_total}")
    print(f"  Style agreements: {metrics['style_agreements']}")
    print(f"  Style disagreements: {metrics['style_disagreements']}")
    print(f"  Style rate: {metrics['n1b_style_rate']*100:.1f}%")

    print(f"\n=== Disagreement Type Distribution ===")
    for dtype, count in status['type_distribution'].items():
        print(f"  {dtype}: {count}")

    # === Gate Results ===
    print_header("Gate Results")

    n1a_passed = status['n1a_passed']
    n1b_passed = status['n1b_passed']
    n1_legacy = status['n1_passed']

    print(f"N1 (legacy): {format_gate(n1_legacy, metrics['agreement_rate']*100, '70-95%')}")
    da_count = metrics['danger_approach_count']
    print(f"N1a (safety): {format_gate(n1a_passed, 100.0 if da_count == 0 else 0.0, 'danger_approach=0')}")
    print(f"N1b (style): {format_gate(n1b_passed, metrics['n1b_style_rate']*100, '≥50%')}")

    print()
    if n1a_passed:
        print("  \033[92m✓ N1a PASSED - No danger approach (safety-critical)\033[0m")
    else:
        print("  \033[91m✗ N1a FAILED - Circuit approached danger\033[0m")
        print(f"    → danger_approach_count = {metrics['danger_approach_count']} (should be 0)")

    if n1b_passed:
        print("  \033[92m✓ N1b PASSED - Style alignment acceptable\033[0m")
    else:
        print("  \033[93m○ N1b SOFT FAIL - Style alignment low (exploration difference)\033[0m")

    print("="*60 + "\n")

    return {
        'n1a_passed': n1a_passed,
        'n1b_passed': n1b_passed,
        'metrics': metrics,
        'status': status
    }


def format_gate(passed, value, threshold):
    if passed is None:
        return f"[PENDING] (need more data)"
    elif passed:
        return f"\033[92m[PASS]\033[0m {value:.1f}% (threshold: {threshold})"
    else:
        return f"\033[91m[FAIL]\033[0m {value:.1f}% (threshold: {threshold})"


if __name__ == "__main__":
    result = run_n1ab_test(n_steps=500)
