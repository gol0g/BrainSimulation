#!/usr/bin/env python
"""v4.6.1 Drift Suppression Test"""
import requests
import sys

BASE_URL = 'http://localhost:8002'

def run_exp(name, setup_fn, n_pre=50, n_post=50, drift_type='rotate'):
    print(f'  Running {name}...', flush=True)
    requests.post(f'{BASE_URL}/reset')
    setup_fn()

    pre = 0
    for _ in range(n_pre):
        resp = requests.post(f'{BASE_URL}/step', json={}).json()
        if resp['outcome']['ate_food']:
            pre += 1

    requests.post(f'{BASE_URL}/drift/enable', params={'drift_type': drift_type})

    post = 0
    for _ in range(n_post):
        resp = requests.post(f'{BASE_URL}/step', json={}).json()
        if resp['outcome']['ate_food']:
            post += 1

    return pre, post

def baseline():
    requests.post(f'{BASE_URL}/memory/disable')
    requests.post(f'{BASE_URL}/regret/disable')

def mem_full():
    requests.post(f'{BASE_URL}/memory/enable')
    requests.post(f'{BASE_URL}/memory/mode', params={'store_enabled': True, 'recall_enabled': True})
    requests.post(f'{BASE_URL}/memory/drift_suppression/disable')
    requests.post(f'{BASE_URL}/regret/disable')

def mem_supp():
    """Memory + Suppression (without regret)"""
    requests.post(f'{BASE_URL}/memory/enable')
    requests.post(f'{BASE_URL}/memory/mode', params={'store_enabled': True, 'recall_enabled': True})
    requests.post(f'{BASE_URL}/memory/drift_suppression/enable', params={'spike_threshold': 1.5, 'use_regret': False})
    requests.post(f'{BASE_URL}/regret/disable')

def mem_supp_regret():
    """Memory + Suppression + Regret (v4.6.2 full combo)"""
    requests.post(f'{BASE_URL}/memory/enable')
    requests.post(f'{BASE_URL}/memory/mode', params={'store_enabled': True, 'recall_enabled': True})
    requests.post(f'{BASE_URL}/regret/enable')
    requests.post(f'{BASE_URL}/memory/drift_suppression/enable', params={'spike_threshold': 1.5, 'use_regret': True})

if __name__ == '__main__':
    N_TRIALS = 5
    N_PRE = 100
    N_POST = 100

    print(f'v4.6.2 SUPPRESSION + REGRET TEST ({N_PRE}+{N_POST} steps x {N_TRIALS} trials)', flush=True)
    print('=' * 60, flush=True)

    configs = [
        ('baseline', baseline),
        ('mem_full', mem_full),
        ('mem+supp', mem_supp),
        ('mem+supp+regret', mem_supp_regret),  # v4.6.2: Regret+Suppression combo
    ]
    all_results = {name: [] for name, _ in configs}

    for trial in range(N_TRIALS):
        print(f'\nTrial {trial+1}/{N_TRIALS}:', flush=True)
        for name, fn in configs:
            pre, post = run_exp(name, fn, n_pre=N_PRE, n_post=N_POST)
            all_results[name].append((pre, post))
            print(f'  {name:18} pre={pre:2} post={post:2} total={pre+post:2}', flush=True)

    print('\n' + '=' * 60, flush=True)
    print('SUMMARY (averages):', flush=True)
    print('-' * 60, flush=True)
    summary = {}
    for name, _ in configs:
        avg_pre = sum(r[0] for r in all_results[name]) / N_TRIALS
        avg_post = sum(r[1] for r in all_results[name]) / N_TRIALS
        avg_total = avg_pre + avg_post
        summary[name] = (avg_pre, avg_post, avg_total)
        print(f'{name:18} pre={avg_pre:5.1f} post={avg_post:5.1f} total={avg_total:5.1f}', flush=True)

    print('\nAnalysis:', flush=True)
    if summary['mem_full'][2] < summary['baseline'][2]:
        diff = summary['baseline'][2] - summary['mem_full'][2]
        print(f'  - mem_full < baseline by {diff:.1f}: WRONG CONFIDENCE PROBLEM confirmed', flush=True)
    else:
        diff = summary['mem_full'][2] - summary['baseline'][2]
        print(f'  - mem_full >= baseline by {diff:.1f}: No wrong confidence problem', flush=True)

    if summary['mem+supp'][2] > summary['mem_full'][2]:
        diff = summary['mem+supp'][2] - summary['mem_full'][2]
        print(f'  - mem+supp > mem_full by {diff:.1f}: SUPPRESSION HELPS!', flush=True)
    else:
        print('  - mem+supp does not help vs mem_full', flush=True)

    # v4.6.2: Regret + Suppression combo analysis
    if summary['mem+supp+regret'][2] > summary['mem+supp'][2]:
        diff = summary['mem+supp+regret'][2] - summary['mem+supp'][2]
        print(f'  - mem+supp+regret > mem+supp by {diff:.1f}: REGRET BOOST HELPS!', flush=True)
    elif summary['mem+supp+regret'][2] > summary['baseline'][2]:
        diff = summary['mem+supp+regret'][2] - summary['baseline'][2]
        print(f'  - mem+supp+regret > baseline by {diff:.1f}: Better than baseline', flush=True)
    else:
        print('  - Regret boost does not add extra benefit', flush=True)

    # Post-drift comparison (most important for drift adaptation)
    print('\nPost-drift comparison (drift adaptation ability):', flush=True)
    for name in ['baseline', 'mem_full', 'mem+supp', 'mem+supp+regret']:
        print(f'  {name:18} post={summary[name][1]:.1f}', flush=True)
