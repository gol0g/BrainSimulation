#!/usr/bin/env python
"""
v4.7 Regime-tagged Memory Test

핵심 검증: post-drift food rate가 개선되는가?

비교 설정:
1. baseline: 메모리 없음
2. mem+supp+regret: v4.6.2 (단일 메모리 뱅크)
3. regime_memory: v4.7 (레짐별 분리 메모리)

기대 결과:
- regime_memory의 post-drift food rate > mem+supp+regret
- regime_memory의 adapt rate > 0 (기존은 거의 0)
"""
import requests
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

BASE_URL = 'http://localhost:8002'


@dataclass
class TestMetrics:
    """Test metrics for comparison"""
    name: str
    pre_food: int
    post_food: int
    total_food: int
    food_rate_pre: float
    food_rate_shock: float  # first 20 steps after drift
    food_rate_adapt: float  # after shock
    time_to_recovery: int
    regime_switches: int = 0
    final_regime: int = 0


def setup_baseline():
    """No memory"""
    requests.post(f'{BASE_URL}/memory/disable')
    requests.post(f'{BASE_URL}/regret/disable')
    requests.post(f'{BASE_URL}/regime/disable')


def setup_mem_supp_regret():
    """v4.6.2: Memory + Suppression + Regret (single bank)"""
    requests.post(f'{BASE_URL}/regime/disable')
    requests.post(f'{BASE_URL}/memory/enable')
    requests.post(f'{BASE_URL}/memory/mode', params={'store_enabled': True, 'recall_enabled': True})
    requests.post(f'{BASE_URL}/regret/enable')
    requests.post(f'{BASE_URL}/memory/drift_suppression/enable',
                  params={'spike_threshold': 1.5, 'use_regret': True})


def setup_regime_memory():
    """v4.7: Regime-tagged Memory (separated banks)"""
    requests.post(f'{BASE_URL}/memory/disable')  # disable old memory
    requests.post(f'{BASE_URL}/regret/enable')
    requests.post(f'{BASE_URL}/regime/enable',
                  params={
                      'n_regimes': 2,
                      'spike_threshold': 2.0,
                      'persistence_required': 5,
                      'grace_period_length': 15
                  })
    # v4.6 drift suppression도 함께 활성화
    requests.post(f'{BASE_URL}/memory/drift_suppression/enable',
                  params={'spike_threshold': 1.5, 'use_regret': True})


def run_test(config_name: str, setup_fn, drift_type: str = 'rotate',
             n_pre: int = 100, n_post: int = 100) -> TestMetrics:
    """Run test and calculate metrics"""
    requests.post(f'{BASE_URL}/reset')
    setup_fn()

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

    # Get regime info (v4.7 only)
    regime_status = requests.get(f'{BASE_URL}/regime/status').json()
    regime_switches = regime_status.get('switch_count', 0)
    final_regime = regime_status.get('current_regime', 0)

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

    # Recovery time
    time_to_recovery = n_post
    recovery_threshold = food_rate_pre * 0.5
    for i in range(10, len(post_logs)):
        window_food = sum(1 for l in post_logs[i-10:i] if l['ate_food'])
        window_rate = window_food / 10
        if window_rate >= recovery_threshold:
            time_to_recovery = i
            break

    return TestMetrics(
        name=config_name,
        pre_food=pre_food,
        post_food=post_food,
        total_food=pre_food + post_food,
        food_rate_pre=round(food_rate_pre, 4),
        food_rate_shock=round(food_rate_shock, 4),
        food_rate_adapt=round(food_rate_adapt, 4),
        time_to_recovery=time_to_recovery,
        regime_switches=regime_switches,
        final_regime=final_regime
    )


def print_metrics(m: TestMetrics):
    """Print formatted metrics"""
    print(f"\n  === {m.name} ===")
    print(f"  Food:     pre={m.pre_food:3}, post={m.post_food:3}, total={m.total_food:3}")
    print(f"  Rates:    pre={m.food_rate_pre:.3f}, shock={m.food_rate_shock:.3f}, adapt={m.food_rate_adapt:.3f}")
    print(f"  Recovery: {m.time_to_recovery} steps")
    if m.regime_switches > 0:
        print(f"  Regime:   switches={m.regime_switches}, final={m.final_regime}")


if __name__ == '__main__':
    N_TRIALS = 3
    N_PRE = 100
    N_POST = 100
    DRIFT_TYPE = 'rotate'

    configs = [
        ('baseline', setup_baseline),
        ('mem+supp+regret', setup_mem_supp_regret),
        ('regime_memory', setup_regime_memory),
    ]

    print(f"v4.7 REGIME-TAGGED MEMORY TEST")
    print(f"Drift: {DRIFT_TYPE}, Pre: {N_PRE} steps, Post: {N_POST} steps, Trials: {N_TRIALS}")
    print("=" * 70)

    all_results = {name: [] for name, _ in configs}

    for trial in range(N_TRIALS):
        print(f"\nTrial {trial+1}/{N_TRIALS}:")
        for name, fn in configs:
            print(f"  Running {name}...", end=' ', flush=True)
            metrics = run_test(name, fn, drift_type=DRIFT_TYPE, n_pre=N_PRE, n_post=N_POST)
            all_results[name].append(metrics)
            print(f"pre={metrics.pre_food}, post={metrics.post_food}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (averages)")
    print("-" * 70)
    print(f"{'Config':<18} | {'Pre':>5} | {'Post':>5} | {'Total':>5} | {'Adapt':>6} | {'Recovery':>8}")
    print("-" * 70)

    summary = {}
    for name, results in all_results.items():
        avg_pre = np.mean([r.pre_food for r in results])
        avg_post = np.mean([r.post_food for r in results])
        avg_total = np.mean([r.total_food for r in results])
        avg_adapt = np.mean([r.food_rate_adapt for r in results])
        avg_recovery = np.mean([r.time_to_recovery for r in results])

        summary[name] = {
            'pre': avg_pre,
            'post': avg_post,
            'total': avg_total,
            'adapt': avg_adapt,
            'recovery': avg_recovery
        }

        print(f"{name:<18} | {avg_pre:5.1f} | {avg_post:5.1f} | {avg_total:5.1f} | {avg_adapt:6.3f} | {avg_recovery:8.1f}")

    print("-" * 70)

    # Analysis
    print("\nKEY ANALYSIS:")

    bl = summary['baseline']
    ms = summary['mem+supp+regret']
    rm = summary['regime_memory']

    # Post-drift comparison (핵심!)
    print("\n  Post-drift food (핵심 지표):")
    print(f"    baseline:        {bl['post']:.1f}")
    print(f"    mem+supp+regret: {ms['post']:.1f} ({'+' if ms['post'] > bl['post'] else ''}{ms['post'] - bl['post']:.1f} vs baseline)")
    print(f"    regime_memory:   {rm['post']:.1f} ({'+' if rm['post'] > bl['post'] else ''}{rm['post'] - bl['post']:.1f} vs baseline)")

    # Adapt rate comparison
    print("\n  Adapt rate (post-drift adaptation):")
    print(f"    baseline:        {bl['adapt']:.3f}")
    print(f"    mem+supp+regret: {ms['adapt']:.3f}")
    print(f"    regime_memory:   {rm['adapt']:.3f}")

    # v4.7 성공 여부 판정
    print("\n  v4.7 SUCCESS CRITERIA:")
    success_post = rm['post'] > ms['post']
    success_adapt = rm['adapt'] > ms['adapt']
    success_total = rm['total'] >= bl['total']

    print(f"    [{'OK' if success_post else 'X'}] regime_memory post > mem+supp+regret post")
    print(f"    [{'OK' if success_adapt else 'X'}] regime_memory adapt > mem+supp+regret adapt")
    print(f"    [{'OK' if success_total else 'X'}] regime_memory total >= baseline total")

    if success_post and success_adapt:
        print("\n  VERDICT: v4.7 Regime-tagged Memory SUCCESS!")
    elif success_post or success_adapt:
        print("\n  VERDICT: v4.7 shows improvement, needs tuning")
    else:
        print("\n  VERDICT: v4.7 needs investigation")
