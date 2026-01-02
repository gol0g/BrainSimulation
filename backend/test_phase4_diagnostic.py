"""
Phase 4 Diagnostic: B-3 Failure Case Analysis

목표: FEP가 위험 방향으로 가는 이유를 분석
1. RIGHT를 하면 위험이 '멀어질 것으로' 예측되고 있나?
2. 위험 관련 차원이 Risk에 실제로 들어가고 있나?
3. 위험 항이 존재해도 energy/pain에 눌리고 있나?
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from genesis.agent import GenesisAgent
from genesis.action_selection import ActionSelector, GDecomposition
from genesis.action_circuit import FEPActionOracle


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def analyze_failure_case(name: str, obs: np.ndarray, worst_actions: list, selector: ActionSelector):
    """
    하나의 실패 케이스에 대해 전체 분석 수행
    """
    print_header(f"Case: {name}")

    action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT']

    # 관측 해석
    print("Current Observation:")
    print(f"  food_prox:   {obs[0]:.2f}")
    print(f"  danger_prox: {obs[1]:.2f} {'⚠️ HIGH' if obs[1] > 0.5 else ''}")
    print(f"  food_dx:     {obs[2]:.2f} (food is {'RIGHT' if obs[2] > 0 else 'LEFT' if obs[2] < 0 else 'HERE'})")
    print(f"  food_dy:     {obs[3]:.2f} (food is {'DOWN' if obs[3] > 0 else 'UP' if obs[3] < 0 else 'HERE'})")
    print(f"  danger_dx:   {obs[4]:.2f} (danger is {'RIGHT' if obs[4] > 0 else 'LEFT' if obs[4] < 0 else 'HERE'})")
    print(f"  danger_dy:   {obs[5]:.2f} (danger is {'DOWN' if obs[5] > 0 else 'UP' if obs[5] < 0 else 'HERE'})")
    print(f"  energy:      {obs[6]:.2f}")
    print(f"  pain:        {obs[7]:.2f}")
    print()

    print(f"⚠️  Worst actions to avoid: {[action_names[a] for a in worst_actions]}")
    print()

    # G 계산
    G_all = selector.compute_G(current_obs=obs)

    # 예측 관측 직접 계산 (selector 내부 로직 재현)
    print("-" * 70)
    print("Per-Action Analysis:")
    print("-" * 70)

    for a in range(5):  # Physical actions only
        print(f"\n[Action {a}: {action_names[a]}]")

        # === 예측 관측 계산 (physics prior) ===
        delta_prox_base = 0.1
        danger_delta_scale = obs[1] * 0.4  # Current danger_prox * 0.4

        delta_food_prox = 0.0
        delta_danger_prox = 0.0

        food_dx, food_dy = obs[2], obs[3]
        danger_dx, danger_dy = obs[4], obs[5]

        if a == 1:  # UP
            delta_food_prox = delta_prox_base if food_dy < 0 else (-delta_prox_base if food_dy > 0 else 0)
            delta_danger_prox = danger_delta_scale if danger_dy < 0 else (-danger_delta_scale if danger_dy > 0 else 0)
        elif a == 2:  # DOWN
            delta_food_prox = delta_prox_base if food_dy > 0 else (-delta_prox_base if food_dy < 0 else 0)
            delta_danger_prox = danger_delta_scale if danger_dy > 0 else (-danger_delta_scale if danger_dy < 0 else 0)
        elif a == 3:  # LEFT
            delta_food_prox = delta_prox_base if food_dx < 0 else (-delta_prox_base if food_dx > 0 else 0)
            delta_danger_prox = danger_delta_scale if danger_dx < 0 else (-danger_delta_scale if danger_dx > 0 else 0)
        elif a == 4:  # RIGHT
            delta_food_prox = delta_prox_base if food_dx > 0 else (-delta_prox_base if food_dx < 0 else 0)
            delta_danger_prox = danger_delta_scale if danger_dx > 0 else (-danger_delta_scale if danger_dx < 0 else 0)

        pred_food_prox = np.clip(obs[0] + delta_food_prox, 0, 1)
        pred_danger_prox = np.clip(obs[1] + delta_danger_prox, 0, 1)

        print(f"  Predicted δ_food_prox:   {delta_food_prox:+.3f} → {pred_food_prox:.3f}")
        print(f"  Predicted δ_danger_prox: {delta_danger_prox:+.3f} → {pred_danger_prox:.3f}")

        # G 분해
        G = G_all[a]
        print(f"  G breakdown:")
        print(f"    Risk:       {G.risk:.4f}")
        print(f"    Ambiguity:  {G.ambiguity:.4f}")
        print(f"    Complexity: {G.complexity:.4f}")
        print(f"    TOTAL G:    {G.G:.4f}")

        if a in worst_actions:
            print(f"  ⚠️  THIS IS A 'WORST' ACTION - should have HIGH G!")

    # === 결론 ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    G_sorted = sorted([(a, G_all[a].G) for a in range(5)], key=lambda x: x[1])
    best_action = G_sorted[0][0]

    print(f"\nAction ranking (by G, lower=better):")
    for rank, (a, g) in enumerate(G_sorted):
        marker = "← CHOSEN (lowest G)" if rank == 0 else ""
        worst_marker = "⚠️ WORST" if a in worst_actions else ""
        print(f"  {rank+1}. {action_names[a]:6s} G={g:.4f} {marker} {worst_marker}")

    # 진단
    print("\n" + "-" * 70)
    print("DIAGNOSIS:")
    print("-" * 70)

    if best_action in worst_actions:
        print(f"\n❌ FAILURE: FEP chose {action_names[best_action]} which is in worst_actions!")

        # 원인 분석
        worst_G = G_all[best_action]

        # 1. 예측 관측 체크
        if best_action == 4:  # RIGHT
            if obs[4] > 0:  # danger_dx > 0 means danger is right
                print(f"\n[1] Prediction Check:")
                print(f"    danger_dx={obs[4]:.2f} > 0 → danger is to the RIGHT")
                print(f"    RIGHT action → model predicts danger gets CLOSER")
                print(f"    → Prediction is CORRECT (danger proximity should increase)")
            elif obs[4] < 0:
                print(f"\n[1] Prediction Check:")
                print(f"    danger_dx={obs[4]:.2f} < 0 → danger is to the LEFT")
                print(f"    RIGHT action → model predicts danger gets FARTHER")
                print(f"    → ⚠️ TEST CASE BUG? 'danger_right' but danger_dx < 0!")

        # 2. Risk 기여 체크
        print(f"\n[2] Risk Contribution Check:")
        risk_values = [G_all[a].risk for a in range(5)]
        print(f"    Risks: {[f'{r:.3f}' for r in risk_values]}")
        risk_diff = max(risk_values) - min(risk_values)
        print(f"    Risk difference (max-min): {risk_diff:.4f}")
        if risk_diff < 0.1:
            print(f"    → ⚠️ Danger dimension might NOT be contributing to Risk!")

        # 3. 스케일 체크
        print(f"\n[3] Scale Check (energy/pain vs danger):")
        print(f"    Chosen action {action_names[best_action]}:")
        print(f"      Risk:       {worst_G.risk:.4f}")
        print(f"      Ambiguity:  {worst_G.ambiguity:.4f}")
        print(f"      Complexity: {worst_G.complexity:.4f}")

        # internal_pref_weight 확인
        print(f"\n    internal_pref_weight (λ): {selector.preferences.internal_pref_weight}")
        ext_weight = 1 - selector.preferences.internal_pref_weight
        int_weight = selector.preferences.internal_pref_weight
        print(f"    External risk weight: {ext_weight:.2f}")
        print(f"    Internal risk weight: {int_weight:.2f}")

        if int_weight > ext_weight:
            print(f"    → ⚠️ Internal preferences (energy/pain) may dominate external (danger)!")
    else:
        print(f"\n✓ SUCCESS: FEP correctly avoided worst actions")
        print(f"  Chose: {action_names[best_action]}")

    return best_action, G_all


def main():
    print_header("Phase 4 Diagnostic: B-3 Failure Analysis")

    # Create agent and selector
    N_STATES = 64
    N_OBSERVATIONS = 8
    N_ACTIONS = 6  # Including THINK

    preferred_obs = np.array([
        1.0,   # food_prox high
        0.0,   # danger_prox low
        0.0, 0.0,  # direction neutral
        0.0, 0.0,
        0.7,   # energy ~0.6-0.7
        0.0    # pain low
    ])

    agent = GenesisAgent(N_STATES, N_OBSERVATIONS, N_ACTIONS, preferred_obs)
    selector = agent.action_selector

    # B-3 failure cases (from test)
    failure_cases = [
        {
            'name': 'danger_right (original)',
            'obs': np.array([0.3, 0.7, 0.0, 0.0, -0.3, 0.0, 0.6, 0.2]),
            'worst_actions': [4],  # RIGHT
        },
        {
            'name': 'danger_up (original)',
            'obs': np.array([0.3, 0.7, 0.0, 0.0, 0.0, 0.3, 0.6, 0.2]),
            'worst_actions': [1],  # UP
        },
        # Corrected test cases (danger direction matches name)
        {
            'name': 'danger_right (CORRECTED)',
            'obs': np.array([0.3, 0.7, 0.0, 0.0, 0.3, 0.0, 0.6, 0.2]),  # danger_dx=+0.3 (right)
            'worst_actions': [4],  # RIGHT
        },
        {
            'name': 'danger_up (CORRECTED)',
            'obs': np.array([0.3, 0.7, 0.0, 0.0, 0.0, -0.3, 0.6, 0.2]),  # danger_dy=-0.3 (up)
            'worst_actions': [1],  # UP
        },
    ]

    results = []
    for case in failure_cases:
        chosen, G_all = analyze_failure_case(
            case['name'],
            case['obs'],
            case['worst_actions'],
            selector
        )
        results.append({
            'name': case['name'],
            'chosen': chosen,
            'worst': case['worst_actions'],
            'failed': chosen in case['worst_actions']
        })

    # Final Summary
    print_header("FINAL SUMMARY")

    for r in results:
        action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        status = "❌ FAIL" if r['failed'] else "✓ PASS"
        print(f"  {r['name']:30s}: {status} (chose {action_names[r['chosen']]})")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    original_failed = sum(1 for r in results if 'original' in r['name'] and r['failed'])
    corrected_failed = sum(1 for r in results if 'CORRECTED' in r['name'] and r['failed'])

    if original_failed > 0 and corrected_failed == 0:
        print("""
🔍 ROOT CAUSE IDENTIFIED: TEST CASE BUG (Sign/Direction Mismatch)

The original test cases have WRONG danger_dx/dy signs!
- "danger_right" has danger_dx=-0.3 (danger is actually LEFT)
- "danger_up" has danger_dy=+0.3 (danger is actually DOWN)

When danger direction signs are CORRECTED:
- FEP correctly avoids moving toward danger

ACTION REQUIRED:
1. Fix test case observation vectors in test_v51_phase2_gate_b.py
2. This is NOT an FEP bug - the physics prior is working correctly
""")
    elif corrected_failed > 0:
        print("""
🔍 ROOT CAUSE: FEP Danger Modeling Issue

Even with corrected test cases, FEP fails to avoid danger.
This indicates a deeper problem in:
1. Risk calculation weight
2. Transition model prediction
3. Preference distribution scale

See detailed analysis above.
""")
    else:
        print("""
✓ All cases passed. No obvious issues detected.
""")


if __name__ == "__main__":
    main()
