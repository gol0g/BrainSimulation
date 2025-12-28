#!/usr/bin/env python
"""
G1 Gate Enhanced Report Test v2
- Calculates metrics directly from step responses
- Compares configs with food_rate, recovery, and adaptation metrics
"""
import requests
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

BASE_URL = 'http://localhost:8002'

@dataclass
class G1Metrics:
    """Enhanced G1 Gate metrics"""
    # Basic counts
    pre_food: int
    post_food: int
    total_food: int

    # Food rates
    food_rate_pre: float
    food_rate_shock: float  # first 20 steps after drift
    food_rate_adapt: float  # after shock phase

    # Recovery
    time_to_recovery: int  # steps until food rate recovers
    recovery_ratio: float  # post/pre food ratio

    # Detection
    pre_G: float
    post_G: float
    G_spike_ratio: float

def setup_config(config_name: str):
    """Setup a specific configuration"""
    requests.post(f'{BASE_URL}/reset')

    if config_name == 'baseline':
        requests.post(f'{BASE_URL}/memory/disable')
        requests.post(f'{BASE_URL}/regret/disable')
    elif config_name == 'mem_full':
        requests.post(f'{BASE_URL}/memory/enable')
        requests.post(f'{BASE_URL}/memory/mode', params={'store_enabled': True, 'recall_enabled': True})
        requests.post(f'{BASE_URL}/memory/drift_suppression/disable')
        requests.post(f'{BASE_URL}/regret/disable')
    elif config_name == 'mem+supp+regret':
        requests.post(f'{BASE_URL}/memory/enable')
        requests.post(f'{BASE_URL}/memory/mode', params={'store_enabled': True, 'recall_enabled': True})
        requests.post(f'{BASE_URL}/regret/enable')
        requests.post(f'{BASE_URL}/memory/drift_suppression/enable', params={'spike_threshold': 1.5, 'use_regret': True})

def run_g1_test(config_name: str, drift_type: str = 'rotate', n_pre: int = 100, n_post: int = 100) -> G1Metrics:
    """Run test and calculate G1 metrics directly"""
    setup_config(config_name)

    pre_logs = []
    post_logs = []

    # Pre-drift phase
    for _ in range(n_pre):
        resp = requests.post(f'{BASE_URL}/step', json={}).json()
        pre_logs.append({
            'ate_food': resp['outcome']['ate_food'],
            'G': resp['action']['G'].get(str(resp['action']['selected']), 0)
        })

    # Enable drift
    requests.post(f'{BASE_URL}/drift/enable', params={'drift_type': drift_type})

    # Post-drift phase
    for _ in range(n_post):
        resp = requests.post(f'{BASE_URL}/step', json={}).json()
        post_logs.append({
            'ate_food': resp['outcome']['ate_food'],
            'G': resp['action']['G'].get(str(resp['action']['selected']), 0)
        })

    # Calculate metrics
    pre_food = sum(1 for l in pre_logs if l['ate_food'])
    post_food = sum(1 for l in post_logs if l['ate_food'])

    food_rate_pre = pre_food / len(pre_logs) if pre_logs else 0

    # Shock phase (first 20 steps)
    shock_end = min(20, len(post_logs))
    shock_food = sum(1 for l in post_logs[:shock_end] if l['ate_food'])
    food_rate_shock = shock_food / shock_end if shock_end > 0 else 0

    # Adapt phase (after shock)
    adapt_logs = post_logs[shock_end:]
    adapt_food = sum(1 for l in adapt_logs if l['ate_food'])
    food_rate_adapt = adapt_food / len(adapt_logs) if adapt_logs else 0

    # Recovery time: first step where rolling 10-step food rate >= 50% of pre
    time_to_recovery = n_post
    recovery_threshold = food_rate_pre * 0.5
    for i in range(10, len(post_logs)):
        window_food = sum(1 for l in post_logs[i-10:i] if l['ate_food'])
        window_rate = window_food / 10
        if window_rate >= recovery_threshold:
            time_to_recovery = i
            break

    # G values
    pre_G = np.mean([l['G'] for l in pre_logs]) if pre_logs else 0
    post_G = np.mean([l['G'] for l in post_logs[:20]]) if post_logs else 0
    G_spike_ratio = post_G / pre_G if pre_G > 0 else 1.0

    return G1Metrics(
        pre_food=pre_food,
        post_food=post_food,
        total_food=pre_food + post_food,
        food_rate_pre=round(food_rate_pre, 4),
        food_rate_shock=round(food_rate_shock, 4),
        food_rate_adapt=round(food_rate_adapt, 4),
        time_to_recovery=time_to_recovery,
        recovery_ratio=round(post_food / pre_food if pre_food > 0 else 0, 3),
        pre_G=round(pre_G, 3),
        post_G=round(post_G, 3),
        G_spike_ratio=round(G_spike_ratio, 3)
    )

def print_metrics(m: G1Metrics, name: str):
    """Print formatted metrics"""
    print(f"\n  === {name} ===")
    print(f"  Food:     pre={m.pre_food:3}, post={m.post_food:3}, total={m.total_food:3}")
    print(f"  Rates:    pre={m.food_rate_pre:.3f}, shock={m.food_rate_shock:.3f}, adapt={m.food_rate_adapt:.3f}")
    print(f"  Recovery: time={m.time_to_recovery} steps, ratio={m.recovery_ratio:.2f}")
    print(f"  G:        pre={m.pre_G:.2f}, post={m.post_G:.2f}, spike={m.G_spike_ratio:.2f}x")

if __name__ == '__main__':
    N_PRE = 100
    N_POST = 100
    DRIFT_TYPE = 'rotate'

    configs = ['baseline', 'mem_full', 'mem+supp+regret']

    print(f"G1 GATE ENHANCED METRICS TEST")
    print(f"Drift: {DRIFT_TYPE}, Pre: {N_PRE} steps, Post: {N_POST} steps")
    print("=" * 70)

    results = {}
    for config in configs:
        print(f"\nRunning {config}...", end=' ', flush=True)
        metrics = run_g1_test(config, drift_type=DRIFT_TYPE, n_pre=N_PRE, n_post=N_POST)
        results[config] = metrics
        print("Done")
        print_metrics(metrics, config)

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("-" * 70)
    print(f"{'Config':<18} | {'Pre Rate':>8} | {'Shock':>6} | {'Adapt':>6} | {'Recovery':>8} | {'Post Food':>9}")
    print("-" * 70)

    for name, m in results.items():
        print(f"{name:<18} | {m.food_rate_pre:>8.3f} | {m.food_rate_shock:>6.3f} | {m.food_rate_adapt:>6.3f} | {m.time_to_recovery:>8} | {m.post_food:>9}")

    print("-" * 70)

    # Analysis
    print("\nKEY INSIGHTS:")
    bl = results['baseline']
    ms = results['mem+supp+regret']

    # Adaptation rate comparison
    adapt_diff = ms.food_rate_adapt - bl.food_rate_adapt
    sign = '+' if adapt_diff >= 0 else ''
    print(f"  Adapt rate: mem+supp+regret {sign}{adapt_diff:.3f} vs baseline")

    # Recovery time comparison
    recovery_diff = bl.time_to_recovery - ms.time_to_recovery
    if recovery_diff > 0:
        print(f"  Recovery:   mem+supp+regret {recovery_diff} steps faster")
    elif recovery_diff < 0:
        print(f"  Recovery:   mem+supp+regret {-recovery_diff} steps slower")
    else:
        print(f"  Recovery:   same as baseline")

    # Best config
    best_adapt = max(results.items(), key=lambda x: x[1].food_rate_adapt)
    best_recovery = min(results.items(), key=lambda x: x[1].time_to_recovery)
    print(f"\n  Best adapt_rate: {best_adapt[0]} ({best_adapt[1].food_rate_adapt:.3f})")
    print(f"  Best recovery:   {best_recovery[0]} ({best_recovery[1].time_to_recovery} steps)")
