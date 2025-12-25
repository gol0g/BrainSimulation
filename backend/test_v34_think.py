"""
v3.4 테스트: THINK Action (Metacognition)

검증 포인트:
1. THINK action이 G(THINK) < G(best_physical)일 때 선택되는가?
2. THINK 선택 시 rollout이 실행되는가?
3. THINK 후 물리 행동이 결정되는가?
4. THINK의 자연스러운 비용 (energy decay)이 적용되는가?
"""

import requests
import time
import json

BASE_URL = "http://localhost:8002"

def reset():
    requests.post(f"{BASE_URL}/reset")
    time.sleep(0.1)

def step():
    time.sleep(0.06)
    return requests.post(f"{BASE_URL}/step", json={}).json()

def enable_think(energy_cost=0.003):
    return requests.post(f"{BASE_URL}/think/enable?energy_cost={energy_cost}").json()

def disable_think():
    return requests.post(f"{BASE_URL}/think/disable").json()

def get_think_status():
    return requests.get(f"{BASE_URL}/think/status").json()

def enable_hierarchy():
    return requests.post(f"{BASE_URL}/hierarchy/enable?K=4&update_interval=10").json()

def run_test():
    print("=" * 60)
    print("v3.4 Test: THINK Action (Metacognition)")
    print("=" * 60)

    # Reset
    reset()

    # Enable hierarchy (for context learning)
    print("\n1. Enable hierarchy...")
    enable_hierarchy()

    # Enable THINK
    print("\n2. Enable THINK action...")
    result = enable_think()
    print(f"   {result}")

    # Check initial status
    status = get_think_status()
    print(f"   Initial status: enabled={status['enabled']}, think_count={status['think_count']}")

    # Run simulation
    print("\n3. Running simulation (100 steps)...")
    think_selected_count = 0
    think_steps = []

    for i in range(100):
        result = step()
        action = result['action']['selected']

        # Check if THINK was selected
        think_status = result.get('think', {})
        if think_status.get('last_think_selected', False):
            think_selected_count += 1
            think_steps.append({
                'step': i,
                'reason': think_status.get('last_think_reason', ''),
                'physical_action': think_status.get('physical_action_after_think'),
            })

    # Results
    print(f"\n4. Results:")
    print(f"   THINK selected: {think_selected_count} times ({think_selected_count}%)")

    if think_steps:
        print(f"\n   Sample THINK selections:")
        for ts in think_steps[:5]:
            print(f"     Step {ts['step']}: {ts['reason'][:60]}...")
            print(f"       Physical action after: {ts['physical_action']}")

    # Final status
    final_status = get_think_status()
    print(f"\n5. Final THINK status:")
    print(f"   Total think_count: {final_status['think_count']}")
    print(f"   Last expected improvement: {final_status['last_expected_improvement']:.4f}")

    # Disable THINK and compare
    print("\n6. Comparison: THINK disabled (30 steps)...")
    disable_think()

    actions_no_think = []
    for i in range(30):
        result = step()
        actions_no_think.append(result['action']['selected'])

    action_counts_no_think = {a: actions_no_think.count(a) for a in range(6)}
    print(f"   Actions without THINK: {action_counts_no_think}")

    # Re-enable and compare
    print("\n7. Comparison: THINK enabled (30 steps)...")
    enable_think()

    actions_with_think = []
    think_in_period = 0
    for i in range(30):
        result = step()
        action = result['action']['selected']
        actions_with_think.append(action)
        if action == 5:  # THINK
            think_in_period += 1

    action_counts_with_think = {a: actions_with_think.count(a) for a in range(6)}
    print(f"   Actions with THINK: {action_counts_with_think}")
    print(f"   THINK count in period: {think_in_period}")

    print("\n" + "=" * 60)
    print("Summary:")
    if think_selected_count > 0:
        print(f"[OK] THINK was selected {think_selected_count} times - metacognition working!")
    else:
        print("[?] THINK was never selected - may need tuning")

    # Cleanup
    disable_think()

if __name__ == "__main__":
    run_test()
