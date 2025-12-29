"""
v5.1 Action Competition Circuit Test

3 Gates:
A. Selection consistency: same state -> stable selection
B. Quality: matches/approaches FEP oracle
C. Resource tradeoff: THINK activates when needed, saves resources otherwise
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.action_circuit import (
    ActionCompetitionCircuit, ActionCircuitConfig,
    test_selection_consistency, test_quality_vs_oracle, test_deliberation_tradeoff
)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, result: dict, gate: str):
    status = "PASS" if result['passed'] else "FAIL"
    color = "\033[92m" if result['passed'] else "\033[91m"
    reset = "\033[0m"

    print(f"Gate {gate}: {name}")
    print(f"  Status: {color}{status}{reset}")
    for k, v in result.items():
        if k != 'passed':
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    print()


def test_basic_competition():
    """Test basic competition dynamics"""
    print_header("Basic Competition Test")

    circuit = ActionCompetitionCircuit()

    # Test observation: agent near food
    obs = np.array([0.8, 0.1, 0.3, 0.2, -0.5, 0.0, 0.6, 0.0])

    print(f"Input observation: {obs.round(2)}")
    print()

    # Run competition
    action, result = circuit.select_action(obs)

    print(f"Selected action: {action}")
    print(f"  (0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=THINK)")
    print()

    print("Action energies:")
    for ae in sorted(result.action_energies, key=lambda x: x.action):
        winner = " <-- WINNER" if ae.action == action else ""
        print(f"  Action {ae.action}: E={ae.energy:.4f} "
              f"(error={ae.error_component:.4f}, prior={ae.prior_component:.4f}){winner}")

    print()
    print(f"Margin: {result.margin:.4f}")
    print(f"Extended deliberation: {result.extended_deliberation}")
    print(f"Total iterations: {result.competition_iterations}")

    return action in [0, 1, 2, 3, 4, 5]  # Valid action


def test_uncertainty_effect():
    """Test how uncertainty affects selection"""
    print_header("Uncertainty Effect Test")

    circuit = ActionCompetitionCircuit()

    # Ambiguous observation
    obs = np.array([0.5, 0.4, 0.1, -0.1, 0.1, 0.1, 0.5, 0.2])

    print("Testing same observation with different uncertainty levels:")
    print()

    for uncertainty in [0.0, 0.3, 0.6, 0.9]:
        circuit.pc_core.reset()
        action, result = circuit.select_action(obs, uncertainty=uncertainty)

        print(f"  Uncertainty={uncertainty:.1f}: action={action}, "
              f"margin={result.margin:.4f}, extended={result.extended_deliberation}")

    print()
    print("Expected: Higher uncertainty -> smaller margins, more extended deliberation")


def test_prior_influence():
    """Test how prior affects selection"""
    print_header("Prior Influence Test")

    circuit = ActionCompetitionCircuit()
    obs = np.array([0.5, 0.3, 0.2, 0.1, -0.2, 0.0, 0.6, 0.1])

    print("Testing with different prior precisions:")
    print()

    # Create a prior that favors a specific state
    mu_prior = np.zeros(16)
    mu_prior[0:4] = 0.5  # Bias toward certain state

    for lambda_prior in [0.0, 0.5, 1.0, 2.0]:
        circuit.pc_core.reset()
        action, result = circuit.select_action(
            obs,
            mu_prior=mu_prior,
            lambda_prior=lambda_prior
        )

        winner = result.action_energies[0]
        print(f"  lambda_prior={lambda_prior:.1f}: action={action}, "
              f"prior_component={winner.prior_component:.4f}")

    print()
    print("Expected: Higher lambda_prior -> larger prior component in energy")


def test_transition_learning():
    """Test transition model learning"""
    print_header("Transition Model Learning Test")

    circuit = ActionCompetitionCircuit()

    # Initial prediction for action 4 (RIGHT)
    obs = np.array([0.5, 0.2, 0.3, 0.0, -0.3, 0.0, 0.7, 0.0])
    initial_imagined = circuit.imagine_observation(obs, 4)

    print(f"Current obs:      {obs.round(2)}")
    print(f"Initial imagined: {initial_imagined.round(2)}")

    # Actual outcome (food got closer)
    actual_next = np.array([0.6, 0.2, 0.2, 0.0, -0.3, 0.0, 0.69, 0.0])

    # Learn from experience
    for _ in range(10):
        circuit.update_transition_model(obs, 4, actual_next, learning_rate=0.1)

    # Check updated prediction
    updated_imagined = circuit.imagine_observation(obs, 4)

    print(f"Actual next:      {actual_next.round(2)}")
    print(f"Updated imagined: {updated_imagined.round(2)}")

    # Check improvement
    initial_error = np.linalg.norm(initial_imagined - actual_next)
    updated_error = np.linalg.norm(updated_imagined - actual_next)

    print()
    print(f"Initial prediction error: {initial_error:.4f}")
    print(f"Updated prediction error: {updated_error:.4f}")
    print(f"Improvement: {initial_error - updated_error:.4f}")

    return updated_error < initial_error


def run_gate_tests():
    """Run all 3 gate tests"""
    print_header("v5.1 Gate Tests")

    circuit = ActionCompetitionCircuit()
    results = {}

    # Gate A: Selection consistency
    print("Running Gate A: Selection Consistency...")
    results['consistency'] = test_selection_consistency(circuit, n_tests=20)
    print_result("Selection Consistency", results['consistency'], "A")

    # Gate B: Quality vs oracle
    print("Running Gate B: Quality vs Oracle...")
    circuit_b = ActionCompetitionCircuit()  # Fresh circuit
    results['quality'] = test_quality_vs_oracle(circuit_b, n_tests=50)
    print_result("Quality vs Oracle", results['quality'], "B")

    # Gate C: Deliberation tradeoff
    print("Running Gate C: Deliberation Tradeoff...")
    circuit_c = ActionCompetitionCircuit()  # Fresh circuit
    results['tradeoff'] = test_deliberation_tradeoff(circuit_c, n_tests=30)
    print_result("Deliberation Tradeoff", results['tradeoff'], "C")

    # Overall
    all_passed = all(r['passed'] for r in results.values())

    print("="*60)
    print(f"  Overall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")
    print("="*60)

    return all_passed, results


def visualize_competition():
    """Visualize competition dynamics"""
    print_header("Competition Dynamics Visualization")

    circuit = ActionCompetitionCircuit()

    # Sequence of observations (agent moving toward food)
    observations = [
        np.array([0.3, 0.1, 0.5, 0.3, -0.3, 0.0, 0.8, 0.0]),   # Far from food
        np.array([0.5, 0.1, 0.3, 0.2, -0.3, 0.0, 0.7, 0.0]),   # Closer
        np.array([0.7, 0.1, 0.1, 0.1, -0.3, 0.0, 0.6, 0.0]),   # Very close
        np.array([1.0, 0.1, 0.0, 0.0, -0.3, 0.0, 1.0, 0.0]),   # On food
    ]

    print("Observation sequence (approaching food):")
    print()

    for i, obs in enumerate(observations):
        circuit.pc_core.reset()
        action, result = circuit.select_action(obs)

        action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'THINK']
        print(f"Step {i+1}: food_prox={obs[0]:.1f}, energy={obs[6]:.1f}")
        print(f"  -> {action_names[action]} (margin={result.margin:.3f})")

    # Check diagnostics
    diag = circuit.get_diagnostics()
    print()
    print("Diagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  v5.1 Action Competition Circuit - Test Suite")
    print("="*60)

    # Basic tests
    test_basic_competition()
    test_uncertainty_effect()
    test_prior_influence()
    test_transition_learning()

    # Gate tests
    all_passed, results = run_gate_tests()

    # Visualization
    visualize_competition()

    print("\n" + "="*60)
    if all_passed:
        print("  v5.1 MVP: READY FOR INTEGRATION")
    else:
        print("  v5.1 MVP: NEEDS REFINEMENT")
    print("="*60 + "\n")
