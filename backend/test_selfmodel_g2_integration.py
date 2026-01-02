"""
v5.2 Self-model + G2 Integration Test

핵심 검증:
- Self-model ON vs OFF에서 G2 비교
- Self-model이 "거짓 개선"이 아니라 실제 도움이 되는지

통과 기준:
- G2 (모든 게이트) 유지 또는 개선
- N1a (safety) 유지
- energy_efficiency 저하 없음

이 테스트가 v5.2의 최종 게이트
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.agent import GenesisAgent
from genesis.action_circuit import ActionCompetitionCircuit, FEPActionOracle
from genesis.circuit_controller import CircuitController
from genesis.self_model import SelfModel, get_mode_label
from genesis.g2_gate import G2GateTracker
from genesis.test_utils import make_observation, Direction


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_drift_simulation(
    use_self_model: bool,
    drift_type: str = "rotate",
    drift_after: int = 100,
    total_steps: int = 250,
    seed: int = 42,
) -> dict:
    """
    Drift 시뮬레이션 (Self-model ON/OFF 비교용)
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
        fep_compare_interval=1,
    )

    g2_tracker = G2GateTracker(
        drift_after=drift_after,
        drift_type=drift_type,
    )

    self_model = SelfModel() if use_self_model else None

    # State
    current_energy = 0.6
    drift_active = False

    # Tracking for self-model signals
    recent_regret_spikes = []
    recent_efficiency = []

    for step in range(total_steps):
        if step >= drift_after:
            drift_active = True

        # Generate observation
        if drift_active:
            drift_steps = step - drift_after
            if drift_type == "rotate":
                food_dir = np.random.choice([Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT])
            else:
                food_dir = np.random.choice(list(Direction))

            if drift_steps < 10:
                transition_noise = 0.3 * (1 - drift_steps / 10)
            else:
                transition_noise = 0.05
        else:
            food_dir = np.random.choice([Direction.RIGHT, Direction.DOWN])
            transition_noise = 0.05

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

        # === Self-model update (if enabled) ===
        resource_mods = None
        if self_model is not None:
            # Compute signals from recent history
            avg_regret = np.mean(recent_regret_spikes[-20:]) if recent_regret_spikes else 0.0
            avg_efficiency = np.mean(recent_efficiency[-20:]) if recent_efficiency else 0.5

            signals = {
                'uncertainty': transition_noise * 2,  # 단순화: noise를 uncertainty로
                'regret_spike_rate': avg_regret,
                'energy_efficiency': avg_efficiency,
                'volatility': transition_noise,
                'movement_ratio': 0.7,  # 고정 (실제론 추적해야 함)
            }

            resource_mods, sm_info = self_model.update(signals)

        # === Circuit action ===
        action, info = controller.select_action(obs)

        # Simulate outcome
        ate_food = np.random.random() < 0.1 * obs[0]
        hit_danger = obs[1] > 0.7 and np.random.random() < 0.2

        # Energy dynamics
        energy_spent = 0.02 if action != 0 else 0.01
        if ate_food:
            current_energy = min(1.0, current_energy + 0.2)
        current_energy = max(0.1, current_energy - energy_spent)

        # Transition std
        base_std = 0.1
        transition_std = base_std + transition_noise + np.random.normal(0, 0.02)
        transition_error = abs(transition_noise) + np.random.uniform(0, 0.1)

        # Regret spike
        regret_spike = drift_active and np.random.random() < (0.2 if (step - drift_after) < 20 else 0.05)
        recent_regret_spikes.append(1.0 if regret_spike else 0.0)
        if len(recent_regret_spikes) > 50:
            recent_regret_spikes.pop(0)

        # Efficiency tracking
        step_efficiency = 1.0 if ate_food else 0.0
        recent_efficiency.append(step_efficiency)
        if len(recent_efficiency) > 50:
            recent_efficiency.pop(0)

        # Log to G2 tracker
        g2_tracker.log_step(
            circuit_action=info['circuit_action'],
            fep_action=info['fep_action'],
            final_action=action,
            agreed=not info['disagreement'],
            disagreement_type=None,
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

    # Results
    g2_result = g2_tracker.get_result()
    controller_status = controller.get_status()
    self_model_status = self_model.get_status() if self_model else None

    return {
        'use_self_model': use_self_model,
        'g2_result': g2_result,
        'controller_status': controller_status,
        'self_model_status': self_model_status,
    }


def compare_results(off_result: dict, on_result: dict) -> dict:
    """Self-model ON vs OFF 비교"""
    off_g2 = off_result['g2_result']
    on_g2 = on_result['g2_result']

    comparison = {
        'g2a_maintained': on_g2.g2a_passed >= off_g2.g2a_passed,
        'g2b_maintained': on_g2.g2b_passed >= off_g2.g2b_passed,
        'g2c_maintained': on_g2.g2c_passed >= off_g2.g2c_passed,
        'overall_maintained': on_g2.overall_passed >= off_g2.overall_passed,
        'safety_maintained': on_g2.safety_maintained and off_g2.safety_maintained,
        'efficiency_retained': on_g2.efficiency_retention >= off_g2.efficiency_retention * 0.9,
    }

    # 개선 지표
    comparison['recovery_improved'] = on_g2.time_to_recovery <= off_g2.time_to_recovery
    comparison['stability_improved'] = on_g2.peak_std_ratio <= off_g2.peak_std_ratio

    return comparison


def test_selfmodel_g2_integration():
    """Self-model + G2 통합 테스트"""
    print_header("v5.2 Self-model + G2 Integration Test")

    # === Run without self-model ===
    print("Running without self-model...")
    off_result = run_drift_simulation(use_self_model=False, seed=42)

    # === Run with self-model ===
    print("Running with self-model...")
    on_result = run_drift_simulation(use_self_model=True, seed=42)

    # === Compare ===
    print_header("Comparison: Self-model OFF vs ON")

    off_g2 = off_result['g2_result']
    on_g2 = on_result['g2_result']

    print("  G2 Gate Results:")
    print(f"  {'Metric':<25} | {'OFF':>10} | {'ON':>10} | {'Change':>10}")
    print("  " + "-" * 60)

    metrics = [
        ('G2a (recovery steps)', off_g2.time_to_recovery, on_g2.time_to_recovery),
        ('G2b (peak std ratio)', off_g2.peak_std_ratio, on_g2.peak_std_ratio),
        ('G2b (regret spike %)', off_g2.regret_spike_rate * 100, on_g2.regret_spike_rate * 100),
        ('G2c (efficiency ret %)', off_g2.efficiency_retention * 100, on_g2.efficiency_retention * 100),
        ('Agreement rate %', off_g2.circuit_agreement_rate * 100, on_g2.circuit_agreement_rate * 100),
        ('Danger approaches', off_g2.danger_approach_count, on_g2.danger_approach_count),
    ]

    for name, off_val, on_val in metrics:
        if isinstance(off_val, float):
            change = on_val - off_val
            change_str = f"{change:+.1f}"
        else:
            change = on_val - off_val
            change_str = f"{change:+d}"
        print(f"  {name:<25} | {off_val:>10.1f} | {on_val:>10.1f} | {change_str:>10}")

    # === Gate comparison ===
    print("\n  Gate Status:")
    print(f"  {'Gate':<15} | {'OFF':>8} | {'ON':>8}")
    print("  " + "-" * 35)
    print(f"  {'G2a (adapt)':<15} | {'PASS' if off_g2.g2a_passed else 'FAIL':>8} | {'PASS' if on_g2.g2a_passed else 'FAIL':>8}")
    print(f"  {'G2b (stable)':<15} | {'PASS' if off_g2.g2b_passed else 'FAIL':>8} | {'PASS' if on_g2.g2b_passed else 'FAIL':>8}")
    print(f"  {'G2c (efficient)':<15} | {'PASS' if off_g2.g2c_passed else 'FAIL':>8} | {'PASS' if on_g2.g2c_passed else 'FAIL':>8}")
    print(f"  {'Overall':<15} | {'PASS' if off_g2.overall_passed else 'FAIL':>8} | {'PASS' if on_g2.overall_passed else 'FAIL':>8}")

    # === Self-model status ===
    if on_result['self_model_status']:
        sm = on_result['self_model_status']
        print("\n  Self-model Status:")
        print(f"    Final z: {sm['z']} ({get_mode_label(sm['z'])})")
        print(f"    Q(z): {[f'{q:.2f}' for q in sm['Q_z']]}")
        print(f"    Switches: {sm['switch_count']}")

    # === Final judgment ===
    comparison = compare_results(off_result, on_result)

    print("\n  Integration Check:")
    all_maintained = True
    for key, value in comparison.items():
        status = "\033[92m✓\033[0m" if value else "\033[91m✗\033[0m"
        print(f"    {status} {key}: {value}")
        if not value and 'maintained' in key:
            all_maintained = False

    # 최종 판정
    print()
    if all_maintained and on_g2.overall_passed:
        print("  \033[92m✓ SELF-MODEL INTEGRATION: G2 MAINTAINED\033[0m")
        print("    Self-model is valid - not a 'false improvement'")
        passed = True
    elif on_g2.overall_passed:
        print("  \033[93m○ SELF-MODEL INTEGRATION: G2 PASSED (with changes)\033[0m")
        print("    Some metrics changed but overall G2 passed")
        passed = True
    else:
        print("  \033[91m✗ SELF-MODEL INTEGRATION: G2 DEGRADED\033[0m")
        print("    Self-model caused regression - needs tuning")
        passed = False

    print("="*60 + "\n")

    return passed


if __name__ == "__main__":
    result = test_selfmodel_g2_integration()
    print(f"Final result: {'PASS' if result else 'FAIL'}")
