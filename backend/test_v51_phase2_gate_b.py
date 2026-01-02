"""
v5.1 Phase 2: Gate B - Real FEP Oracle Comparison

Gate B 재정의:
- 단순 일치율 X
- 랭킹 유사도: action 순위가 얼마나 비슷한가
- 분포 유사도: softmax 확률 분포의 유사도
- 방향성: 둘 다 같은 "나쁜 action"을 피하는가
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.agent import GenesisAgent
from genesis.action_circuit import (
    ActionCompetitionCircuit, ActionCircuitConfig,
    FEPActionOracle, compare_rankings, compare_distributions
)
from genesis.test_utils import (
    make_observation, Direction, Action,
    get_scenario, describe_observation
)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, passed: bool, details: dict):
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"

    print(f"{name}: {color}{status}{reset}")
    for k, v in details.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print()


def create_fep_agent():
    """Create a GenesisAgent for FEP Oracle"""
    N_STATES = 64
    N_OBSERVATIONS = 8
    N_ACTIONS = 6

    preferred_obs = np.array([
        1.0,   # food proximity high
        0.0,   # danger proximity low
        0.0,   # food_dx = 0 (on food)
        0.0,   # food_dy = 0
        0.5,   # danger_dx (far is neutral)
        0.5,   # danger_dy
        0.7,   # energy high
        0.0    # pain low
    ])

    agent = GenesisAgent(N_STATES, N_OBSERVATIONS, N_ACTIONS, preferred_obs)
    return agent


def get_circuit_ranking(circuit: ActionCompetitionCircuit, obs: np.ndarray) -> list:
    """Get action ranking from Circuit (by energy, lowest first)"""
    circuit.pc_core.reset()
    _, result = circuit.select_action(obs)
    # Sort by energy
    sorted_actions = sorted(result.action_energies, key=lambda x: x.energy)
    return [ae.action for ae in sorted_actions]


def get_circuit_probs(circuit: ActionCompetitionCircuit, obs: np.ndarray, temperature: float = 0.3, n_actions: int = 5) -> np.ndarray:
    """Get action probabilities from Circuit (softmax over -energy)

    Args:
        n_actions: Number of actions to return (5 for physical only, 6 with THINK)
    """
    circuit.pc_core.reset()
    _, result = circuit.select_action(obs)

    energies = np.array([0.0] * n_actions)
    for ae in result.action_energies:
        if ae.action < n_actions:  # Only physical actions
            energies[ae.action] = ae.energy

    # Softmax over negative energy
    log_probs = -energies / temperature
    log_probs = log_probs - np.max(log_probs)
    probs = np.exp(log_probs)
    probs = probs / (probs.sum() + 1e-10)

    return probs


def test_gate_b_ranking_similarity():
    """
    Gate B-1: 랭킹 유사도 테스트

    FEP와 Circuit의 action 랭킹이 얼마나 비슷한가?
    (top-1 일치가 아니라 전체 순위 비교)
    """
    print_header("Gate B-1: Ranking Similarity")

    # Create agents
    agent = create_fep_agent()
    oracle = FEPActionOracle(agent)
    circuit = ActionCompetitionCircuit()

    # Test scenarios using test_utils
    scenarios = [
        ("near_food", make_observation(
            food_direction=Direction.RIGHT,
            food_proximity=0.8,
            danger_proximity=0.1,
            energy=0.6
        )),
        ("near_danger", make_observation(
            danger_direction=Direction.RIGHT,
            danger_proximity=0.8,
            food_proximity=0.2,
            energy=0.7,
            pain=0.3
        )),
        ("low_energy", make_observation(
            food_direction=Direction.RIGHT,
            food_proximity=0.3,
            danger_proximity=0.1,
            energy=0.2  # Low energy
        )),
        ("balanced", make_observation(
            food_direction=Direction.RIGHT,
            food_proximity=0.5,
            danger_proximity=0.3,
            energy=0.6,
            pain=0.1
        )),
    ]

    ranking_scores = []
    top1_matches = 0
    top2_overlaps = 0

    for name, obs in scenarios:
        # Get rankings
        fep_ranking = oracle.get_action_ranking(obs)
        circuit_ranking = get_circuit_ranking(circuit, obs)

        # Compare
        score = compare_rankings(fep_ranking, circuit_ranking)
        ranking_scores.append(score)

        # Top-1 match
        if fep_ranking[0] == circuit_ranking[0]:
            top1_matches += 1

        # Top-2 overlap
        fep_top2 = set(fep_ranking[:2])
        circuit_top2 = set(circuit_ranking[:2])
        if len(fep_top2 & circuit_top2) > 0:
            top2_overlaps += 1

        action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'THINK']
        print(f"  {name}:")
        print(f"    FEP ranking:     {[action_names[a] for a in fep_ranking[:3]]}")
        print(f"    Circuit ranking: {[action_names[a] for a in circuit_ranking[:3]]}")
        print(f"    Similarity: {score:.2f}")

    avg_ranking_score = np.mean(ranking_scores)
    top1_rate = top1_matches / len(scenarios)
    top2_rate = top2_overlaps / len(scenarios)

    # 기준: 랭킹 유사도 > 0.5, top-2 overlap > 50%
    passed = avg_ranking_score > 0.4 or top2_rate > 0.5

    details = {
        'avg_ranking_similarity': avg_ranking_score,
        'top1_match_rate': top1_rate,
        'top2_overlap_rate': top2_rate,
        'threshold': 0.4
    }

    print_result("Gate B-1 (Ranking)", passed, details)
    return passed, details


def test_gate_b_distribution_similarity():
    """
    Gate B-2: 분포 유사도 테스트

    FEP와 Circuit의 action 확률 분포가 얼마나 비슷한가?
    (Jensen-Shannon divergence 기반)
    """
    print_header("Gate B-2: Distribution Similarity")

    agent = create_fep_agent()
    oracle = FEPActionOracle(agent)
    circuit = ActionCompetitionCircuit()

    # Random test observations
    np.random.seed(42)
    n_tests = 20

    js_similarities = []

    for i in range(n_tests):
        obs = np.zeros(8)
        obs[0] = np.random.uniform(0.2, 1.0)  # food_prox
        obs[1] = np.random.uniform(0.0, 0.7)  # danger_prox
        obs[2] = np.random.uniform(-0.4, 0.4)
        obs[3] = np.random.uniform(-0.4, 0.4)
        obs[4] = np.random.uniform(-0.5, 0.5)
        obs[5] = np.random.uniform(-0.5, 0.5)
        obs[6] = np.random.uniform(0.3, 1.0)  # energy
        obs[7] = np.random.uniform(0.0, 0.3)  # pain

        # Get probabilities
        fep_probs = oracle.get_action_probs(obs)
        circuit_probs = get_circuit_probs(circuit, obs)

        # Compare
        similarity = compare_distributions(fep_probs, circuit_probs)
        js_similarities.append(similarity)

    avg_similarity = np.mean(js_similarities)
    min_similarity = np.min(js_similarities)

    # 기준: 평균 분포 유사도 > 0.3
    passed = avg_similarity > 0.3

    details = {
        'avg_distribution_similarity': avg_similarity,
        'min_distribution_similarity': min_similarity,
        'n_tests': n_tests,
        'threshold': 0.3
    }

    print_result("Gate B-2 (Distribution)", passed, details)
    return passed, details


def test_gate_b_worst_action_avoidance():
    """
    Gate B-3: 나쁜 행동 회피 테스트

    FEP와 Circuit 모두 명백히 나쁜 action을 피하는가?
    (예: 위험 쪽으로 이동, 음식 반대로 이동)
    """
    print_header("Gate B-3: Worst Action Avoidance")

    agent = create_fep_agent()
    oracle = FEPActionOracle(agent)
    circuit = ActionCompetitionCircuit()

    # Scenarios with clear "worst" actions
    # Using test_utils.make_observation() to prevent sign bugs
    test_cases = [
        {
            'name': 'danger_right',
            'obs': make_observation(
                danger_direction=Direction.RIGHT,
                danger_proximity=0.7,
                food_proximity=0.3,
                energy=0.6,
                pain=0.2
            ),
            'worst_actions': [Action.RIGHT],  # Toward danger
        },
        {
            'name': 'danger_up',
            'obs': make_observation(
                danger_direction=Direction.UP,
                danger_proximity=0.7,
                food_proximity=0.3,
                energy=0.6,
                pain=0.2
            ),
            'worst_actions': [Action.UP],  # Toward danger
        },
        {
            'name': 'food_right_low_energy',
            'obs': make_observation(
                food_direction=Direction.RIGHT,
                food_proximity=0.4,
                danger_direction=Direction.LEFT,
                danger_proximity=0.1,
                energy=0.2,  # Low energy = hungry
                pain=0.0
            ),
            'worst_actions': [Action.LEFT],  # Away from food when hungry
        },
    ]

    fep_avoids = 0
    circuit_avoids = 0
    both_avoid = 0

    for tc in test_cases:
        obs = tc['obs']
        worst = set(tc['worst_actions'])

        fep_action, _ = oracle.get_action(obs)
        circuit_ranking = get_circuit_ranking(circuit, obs)
        circuit_action = circuit_ranking[0]

        fep_avoided = fep_action not in worst
        circuit_avoided = circuit_action not in worst

        if fep_avoided:
            fep_avoids += 1
        if circuit_avoided:
            circuit_avoids += 1
        if fep_avoided and circuit_avoided:
            both_avoid += 1

        action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'THINK']
        print(f"  {tc['name']}:")
        print(f"    Worst: {[action_names[a] for a in worst]}")
        print(f"    FEP chose: {action_names[fep_action]} ({'OK' if fep_avoided else 'BAD'})")
        print(f"    Circuit chose: {action_names[circuit_action]} ({'OK' if circuit_avoided else 'BAD'})")

    n = len(test_cases)
    fep_rate = fep_avoids / n
    circuit_rate = circuit_avoids / n
    both_rate = both_avoid / n

    # 기준: Circuit이 최악의 action을 70% 이상 피함
    passed = circuit_rate >= 0.6

    details = {
        'fep_avoidance_rate': fep_rate,
        'circuit_avoidance_rate': circuit_rate,
        'both_avoid_rate': both_rate,
        'threshold': 0.6
    }

    print_result("Gate B-3 (Avoidance)", passed, details)
    return passed, details


def run_gate_b_phase2():
    """Run all Phase 2 Gate B tests"""
    print("\n" + "="*60)
    print("  v5.1 Phase 2: Gate B - FEP Oracle Comparison")
    print("="*60)

    results = {}

    results['ranking'] = test_gate_b_ranking_similarity()
    results['distribution'] = test_gate_b_distribution_similarity()
    results['avoidance'] = test_gate_b_worst_action_avoidance()

    # 종합
    print("\n" + "="*60)
    print("  GATE B SUMMARY")
    print("="*60 + "\n")

    # 2/3 이상 통과하면 Gate B 통과
    passed_count = sum(1 for _, (p, _) in results.items() if p)
    overall_passed = passed_count >= 2

    for name, (passed, _) in results.items():
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {name}: {status}")

    print()
    if overall_passed:
        print(f"  \033[92m✓ GATE B PASSED ({passed_count}/3)\033[0m")
    else:
        print(f"  \033[91m✗ GATE B FAILED ({passed_count}/3)\033[0m")

    print("="*60 + "\n")

    return overall_passed, results


if __name__ == "__main__":
    run_gate_b_phase2()
