#!/usr/bin/env python
"""v4.6.2 Multi-Drift Type Test - rotate, flip_x, flip_y, reverse, probabilistic"""
import requests

BASE_URL = 'http://localhost:8002'

def run_exp(name, setup_fn, n_pre=50, n_post=50, drift_type='rotate'):
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

def mem_supp_regret():
    """Best config from v4.6.2 test"""
    requests.post(f'{BASE_URL}/memory/enable')
    requests.post(f'{BASE_URL}/memory/mode', params={'store_enabled': True, 'recall_enabled': True})
    requests.post(f'{BASE_URL}/regret/enable')
    requests.post(f'{BASE_URL}/memory/drift_suppression/enable', params={'spike_threshold': 1.5, 'use_regret': True})

if __name__ == '__main__':
    N_TRIALS = 3
    N_PRE = 100
    N_POST = 100

    drift_types = ['rotate', 'flip_x', 'flip_y', 'reverse', 'probabilistic']
    configs = [
        ('baseline', baseline),
        ('mem+supp+regret', mem_supp_regret),
    ]

    print(f'MULTI-DRIFT TYPE TEST ({N_PRE}+{N_POST} steps x {N_TRIALS} trials)')
    print('=' * 70)

    results = {dt: {name: [] for name, _ in configs} for dt in drift_types}

    for drift_type in drift_types:
        print(f'\n--- Drift Type: {drift_type} ---')
        for trial in range(N_TRIALS):
            print(f'  Trial {trial+1}/{N_TRIALS}:', end=' ', flush=True)
            for name, fn in configs:
                pre, post = run_exp(name, fn, n_pre=N_PRE, n_post=N_POST, drift_type=drift_type)
                results[drift_type][name].append((pre, post))
                print(f'{name}={pre+post}', end=' ', flush=True)
            print()

    print('\n' + '=' * 70)
    print('SUMMARY BY DRIFT TYPE')
    print('-' * 70)
    print(f'{"Drift Type":15} | {"baseline":20} | {"mem+supp+regret":20} | {"Δ":8}')
    print(f'{"":15} | {"pre":>6} {"post":>6} {"total":>6} | {"pre":>6} {"post":>6} {"total":>6} | {"":>8}')
    print('-' * 70)

    for drift_type in drift_types:
        bl = results[drift_type]['baseline']
        ms = results[drift_type]['mem+supp+regret']

        bl_pre = sum(r[0] for r in bl) / N_TRIALS
        bl_post = sum(r[1] for r in bl) / N_TRIALS
        bl_total = bl_pre + bl_post

        ms_pre = sum(r[0] for r in ms) / N_TRIALS
        ms_post = sum(r[1] for r in ms) / N_TRIALS
        ms_total = ms_pre + ms_post

        delta = ms_total - bl_total
        sign = '+' if delta >= 0 else ''

        print(f'{drift_type:15} | {bl_pre:6.1f} {bl_post:6.1f} {bl_total:6.1f} | {ms_pre:6.1f} {ms_post:6.1f} {ms_total:6.1f} | {sign}{delta:6.1f}')

    print('-' * 70)
    print('\nPost-drift adaptation (mem+supp+regret vs baseline):')
    for drift_type in drift_types:
        bl_post = sum(r[1] for r in results[drift_type]['baseline']) / N_TRIALS
        ms_post = sum(r[1] for r in results[drift_type]['mem+supp+regret']) / N_TRIALS
        diff = ms_post - bl_post
        sign = '+' if diff >= 0 else ''
        status = 'OK' if diff >= 0 else 'X'
        print(f'  {drift_type:15}: baseline={bl_post:.1f}, mem+supp+regret={ms_post:.1f} ({sign}{diff:.1f}) {status}')
