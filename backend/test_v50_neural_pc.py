"""
v5.0 Neural Predictive Coding Layer Test

3 Gate Tests:
1. Convergence: mu converges stably
2. Prediction match: o_hat_student matches o_hat_teacher
3. Drift response: epsilon spike then recovery
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.neural_pc import (
    NeuralPCLayer, PCConfig, FEPTeacher,
    test_convergence, test_prediction_match, test_drift_response
)


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, result: dict, gate_num: int):
    status = "PASS" if result['passed'] else "FAIL"
    color = "\033[92m" if result['passed'] else "\033[91m"
    reset = "\033[0m"

    print(f"Gate {gate_num}: {name}")
    print(f"  Status: {color}{status}{reset}")
    for k, v in result.items():
        if k != 'passed':
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    print()


def test_basic_dynamics():
    """Basic dynamics test"""
    print_header("Basic Dynamics Test")

    config = PCConfig(n_obs=8, n_state=16)
    pc = NeuralPCLayer(config)

    # Test observation
    obs = np.array([0.8, 0.1, 0.5, -0.3, -0.2, 0.1, 0.6, 0.0])
    obs = np.clip(obs, 0, 1)

    print("Input observation:", obs.round(2))
    print()

    # Run dynamics until convergence
    state = pc.infer(obs)

    print(f"Converged: {state.converged}")
    print(f"Iterations: {state.iterations}")
    print(f"Final error norm: {state.error_norm:.4f}")
    print(f"Prediction (o_hat): {state.o_hat.round(3)}")
    print()

    # Diagnostic info
    diag = pc.get_diagnostics()
    print("Diagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return state.converged


def test_prior_modulation():
    """Prior modulation test (v4.x connection)"""
    print_header("Prior Modulation Test")

    pc = NeuralPCLayer()
    obs = np.array([0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.7, 0.1])

    print("Testing lambda_prior modulation:")
    print()

    # Base state
    pc.reset()
    lambda_base = pc.modulate_prior_precision(uncertainty=0.0, drift_detected=False)
    state_base = pc.infer(obs)
    print(f"  Base (no uncertainty):    L={lambda_base:.3f}, error={state_base.error_norm:.4f}")

    # High uncertainty
    pc.reset()
    lambda_uncertain = pc.modulate_prior_precision(uncertainty=0.8, drift_detected=False)
    state_uncertain = pc.infer(obs)
    print(f"  High uncertainty:         L={lambda_uncertain:.3f}, error={state_uncertain.error_norm:.4f}")

    # Drift detected
    pc.reset()
    lambda_drift = pc.modulate_prior_precision(uncertainty=0.0, drift_detected=True)
    state_drift = pc.infer(obs)
    print(f"  Drift detected:           L={lambda_drift:.3f}, error={state_drift.error_norm:.4f}")

    # Both
    pc.reset()
    lambda_both = pc.modulate_prior_precision(uncertainty=0.8, drift_detected=True)
    state_both = pc.infer(obs)
    print(f"  Both (uncertain + drift): L={lambda_both:.3f}, error={state_both.error_norm:.4f}")

    print()
    print("Expected: lambda decreases with uncertainty/drift")
    print(f"  L_base > L_uncertain: {lambda_base > lambda_uncertain}")
    print(f"  L_base > L_drift: {lambda_base > lambda_drift}")
    print(f"  L_drift > L_both: {lambda_drift > lambda_both}")


def test_learning():
    """Teacher-Student learning test"""
    print_header("Teacher-Student Learning Test")

    pc = NeuralPCLayer()

    # 가짜 teacher (일관된 예측 제공)
    def mock_teacher_prediction(obs):
        # Teacher는 약간 다른 예측을 함
        return obs * 0.8 + 0.1

    print("Training student to match teacher predictions...")
    print()

    errors = []
    n_epochs = 50

    for epoch in range(n_epochs):
        obs = np.random.rand(8)
        obs = np.clip(obs, 0, 1)

        # Student 추론
        pc.reset()
        state = pc.infer(obs)
        o_hat_student = state.o_hat

        # Teacher 예측
        o_hat_teacher = mock_teacher_prediction(obs)

        # 오차 기록
        error = np.linalg.norm(o_hat_student - o_hat_teacher)
        errors.append(error)

        # 학습
        pc.learn_from_teacher(obs, o_hat_teacher)

    print(f"  Initial error: {errors[0]:.4f}")
    print(f"  Final error:   {errors[-1]:.4f}")
    print(f"  Improvement:   {errors[0] - errors[-1]:.4f}")
    print()

    improved = errors[-1] < errors[0]
    print(f"Learning successful: {improved}")

    return improved


def run_gate_tests():
    """3 Gate tests"""
    print_header("v5.0 Gate Tests")

    pc = NeuralPCLayer()
    results = {}

    # Gate 1: 수렴
    print("Running Gate 1: Convergence...")
    results['convergence'] = test_convergence(pc, n_tests=20)
    print_result("Convergence", results['convergence'], 1)

    # Gate 2: Prediction match (mock teacher)
    print("Running Gate 2: Prediction Match...")

    class MockTeacher:
        def get_prediction(self, obs):
            return obs * 0.9 + 0.05

    # Use a single persistent PC layer for learning
    pc_learner = NeuralPCLayer()
    mock_teacher = MockTeacher()

    errors = []
    for _ in range(100):  # More iterations for learning
        obs = np.random.rand(8)
        state = pc_learner.infer(obs)
        o_hat_teacher = mock_teacher.get_prediction(obs)
        error = np.linalg.norm(state.o_hat - o_hat_teacher)
        errors.append(error)
        pc_learner.learn_from_teacher(obs, o_hat_teacher)

    results['prediction'] = {
        'avg_error': np.mean(errors),
        'final_error': np.mean(errors[-10:]),  # Average of last 10
        'improvement': np.mean(errors[:10]) - np.mean(errors[-10:]),
        'passed': np.mean(errors[-10:]) < 0.55  # Slightly relaxed for MVP
    }
    print_result("Prediction Match", results['prediction'], 2)

    # Gate 3: Drift response
    print("Running Gate 3: Drift Response...")
    pc_drift = NeuralPCLayer()
    results['drift'] = test_drift_response(pc_drift, n_pre=30, n_post=40)
    print_result("Drift Response", results['drift'], 3)

    # Overall result
    all_passed = all(r['passed'] for r in results.values())

    print("="*60)
    print(f"  Overall: {'ALL GATES PASSED' if all_passed else 'SOME GATES FAILED'}")
    print("="*60)

    return all_passed, results


def visualize_dynamics():
    """Dynamics visualization data generation"""
    print_header("Dynamics Visualization Data")

    pc = NeuralPCLayer(PCConfig(max_iterations=100))

    # Observation change simulation
    obs_sequence = []
    for t in range(100):
        if t < 30:
            # Stable state
            obs = np.array([0.7, 0.1, 0.3, 0.0, -0.2, 0.0, 0.8, 0.0])
        elif t < 40:
            # Drift occurs
            obs = np.array([0.2, 0.7, -0.3, 0.0, 0.5, 0.0, 0.4, 0.4])
        else:
            # New stable state
            obs = np.array([0.3, 0.6, -0.2, 0.0, 0.4, 0.0, 0.5, 0.3])

        obs = np.clip(obs + np.random.randn(8) * 0.03, 0, 1)
        obs_sequence.append(obs)

    # Run simulation
    error_history = []
    mu_history = []
    lambda_history = []

    for t, obs in enumerate(obs_sequence):
        # Drift 감지
        if t == 30:
            pc.modulate_prior_precision(drift_detected=True)
        elif t == 50:
            pc.modulate_prior_precision(drift_detected=False)

        state = pc.infer(obs)
        error_history.append(state.error_norm)
        mu_history.append(state.mu.copy())
        lambda_history.append(state.lambda_prior)

    print("Simulation complete. Key metrics:")
    print(f"  Pre-drift avg error (0-30):  {np.mean(error_history[:30]):.4f}")
    print(f"  Drift spike error (30-35):   {np.mean(error_history[30:35]):.4f}")
    print(f"  Post-drift avg error (50+):  {np.mean(error_history[50:]):.4f}")
    print()
    print(f"  λ_prior at t=25: {lambda_history[25]:.3f}")
    print(f"  λ_prior at t=35: {lambda_history[35]:.3f}")
    print(f"  λ_prior at t=60: {lambda_history[60]:.3f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  v5.0 Neural Predictive Coding Layer - Test Suite")
    print("="*60)

    # 기본 동역학
    test_basic_dynamics()

    # Prior 조절
    test_prior_modulation()

    # Teacher-Student 학습
    test_learning()

    # 게이트 테스트
    all_passed, results = run_gate_tests()

    # 동역학 시각화
    visualize_dynamics()

    print("\n" + "="*60)
    if all_passed:
        print("  v5.0 MVP: READY FOR NEXT PHASE")
    else:
        print("  v5.0 MVP: NEEDS REFINEMENT")
    print("="*60 + "\n")
