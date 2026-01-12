"""
Test Safety Gates - Verify all safety constraints are enforced.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.safety_gates import SafetyGates, SafetyConfig, ViolationType, create_continuous_gates


def test_nan_inf_gate():
    """Test NaN/Inf detection and replacement."""
    print("Testing NaN/Inf gate...")

    gates = SafetyGates(SafetyConfig(nan_replacement=0.0))

    # Test NaN
    action = np.array([np.nan, 0.5])
    safe = gates.filter(action)
    assert np.isfinite(safe).all(), "NaN not replaced"
    assert gates.last_violation == ViolationType.NAN_INF

    # Test Inf
    action = np.array([np.inf, -np.inf])
    safe = gates.filter(action)
    assert np.isfinite(safe).all(), "Inf not replaced"

    # Test normal values
    gates.reset()
    action = np.array([0.5, 0.5])
    safe = gates.filter(action)
    assert gates.last_violation is None

    print("  [PASS] NaN/Inf gate works correctly")


def test_action_clamp():
    """Test action magnitude clamping."""
    print("Testing action clamp...")

    gates = SafetyGates(SafetyConfig(
        action_max=1.0,
        action_component_max=0.8
    ))

    # Test component clamp
    action = np.array([1.5, 0.3])
    safe = gates.filter(action)
    assert np.abs(safe[0]) <= 0.8, "Component not clamped"
    assert gates.last_violation == ViolationType.ACTION_CLAMP

    # Test magnitude clamp
    gates.reset()
    action = np.array([0.8, 0.8])  # magnitude ~1.13
    safe = gates.filter(action)
    assert np.linalg.norm(safe) <= 1.0 + 1e-6, "Magnitude not clamped"

    # Test normal action
    gates.reset()
    action = np.array([0.3, 0.3])
    safe = gates.filter(action)
    np.testing.assert_array_almost_equal(safe, action)

    print("  [PASS] Action clamp works correctly")


def test_rate_limit():
    """Test action rate limiting."""
    print("Testing rate limit...")

    gates = SafetyGates(SafetyConfig(
        delta_max=0.3,
        delta_component_max=0.2
    ))

    # First action (no rate limit)
    action1 = np.array([0.0, 0.0])
    safe1 = gates.filter(action1)

    # Large jump (should be limited)
    action2 = np.array([1.0, 1.0])
    safe2 = gates.filter(action2)
    delta = safe2 - safe1
    assert np.linalg.norm(delta) <= 0.3 + 1e-6, f"Delta too large: {np.linalg.norm(delta)}"
    assert gates.last_violation == ViolationType.RATE_LIMIT

    # Small step (no violation)
    action3 = safe2 + np.array([0.1, 0.1])
    safe3 = gates.filter(action3)
    # Should not violate if within delta_max

    print("  [PASS] Rate limit works correctly")


def test_emergency_brake():
    """Test emergency brake activation."""
    print("Testing emergency brake...")

    gates = SafetyGates(SafetyConfig(
        emergency_distance=2.0,
        emergency_action=np.array([0.0, 0.0])
    ))

    # Normal distance - no brake
    action = np.array([0.5, 0.5])
    safe = gates.filter(action, danger_dist=5.0)
    np.testing.assert_array_almost_equal(safe, action)
    assert gates.last_violation != ViolationType.EMERGENCY_BRAKE

    # Emergency distance - brake activated
    gates.reset()
    action = np.array([0.5, 0.5])
    safe = gates.filter(action, danger_dist=1.0)
    np.testing.assert_array_almost_equal(safe, np.array([0.0, 0.0]))
    assert gates.last_violation == ViolationType.EMERGENCY_BRAKE

    print("  [PASS] Emergency brake works correctly")


def test_combined_scenario():
    """Test all gates in a realistic scenario."""
    print("Testing combined scenario...")

    gates = create_continuous_gates(
        action_max=1.0,
        delta_max=0.3,
        emergency_dist=0.5
    )

    # Simulate 100 steps with various inputs
    np.random.seed(42)

    for step in range(100):
        # Generate random action (some may be extreme)
        raw_action = np.random.randn(2) * 2.0

        # Random danger distance
        danger_dist = np.random.uniform(0.3, 5.0)

        # Occasionally inject bad values
        if step % 20 == 0:
            raw_action[0] = np.nan

        # Apply gates
        safe_action = gates.filter(raw_action, danger_dist=danger_dist, step=step)

        # Verify safety invariants
        assert np.isfinite(safe_action).all(), f"Step {step}: Non-finite action"
        assert np.linalg.norm(safe_action) <= 1.0 + 1e-6, f"Step {step}: Action too large"

    stats = gates.get_stats()
    print(f"  Steps: {stats['steps']}")
    print(f"  Violations: {stats['total_violations']} ({stats['violation_rate']:.1%})")
    print(f"  By type: {stats['by_type']}")
    print("  [PASS] All safety invariants maintained")


def test_adversarial_inputs():
    """Test against adversarial/malicious inputs."""
    print("Testing adversarial inputs...")

    gates = SafetyGates(SafetyConfig(
        action_max=1.0,
        action_component_max=1.0,
        delta_max=0.5,
        emergency_distance=1.0
    ))

    adversarial_cases = [
        ("Very large values", np.array([1e10, -1e10])),
        ("Very small values", np.array([1e-100, 1e-100])),
        ("Mixed NaN/Inf", np.array([np.nan, np.inf])),
        ("All NaN", np.array([np.nan, np.nan])),
        ("Negative Inf", np.array([-np.inf, -np.inf])),
    ]

    for name, action in adversarial_cases:
        gates.reset()
        safe = gates.filter(action)

        # All outputs must be safe
        assert np.isfinite(safe).all(), f"{name}: Non-finite output"
        assert np.linalg.norm(safe) <= 1.0 + 1e-6, f"{name}: Output too large"
        print(f"  [{name}] SAFE -> {safe}")

    print("  [PASS] All adversarial inputs handled safely")


if __name__ == "__main__":
    print("=" * 60)
    print("SAFETY GATES TEST SUITE")
    print("=" * 60)
    print()

    test_nan_inf_gate()
    test_action_clamp()
    test_rate_limit()
    test_emergency_brake()
    test_combined_scenario()
    test_adversarial_inputs()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
