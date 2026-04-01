#!/usr/bin/env python3
"""KC_auditory→D1 sparsity sweep: 0.10, 0.15, 0.20"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig
from evaluate_concepts import test_call_semantics, diagnose_auditory
import numpy as np

results = []

for sp in [0.10, 0.15, 0.20]:
    print(f"\n{'='*60}")
    print(f"  SWEEP: kc_auditory_to_d1_sparsity = {sp}")
    print(f"  Expected connections: 500 × {sp} = {int(500*sp)} per D1 (vs visual 2000×0.05=100)")
    print(f"{'='*60}")

    config = ForagerBrainConfig()
    config.kc_auditory_to_d1_sparsity = sp
    brain = ForagerBrain(config)

    env_config = ForagerConfig()
    env = ForagerGym(env_config)

    # 20ep training
    for ep in range(20):
        obs = env.reset()
        done = False
        while not done:
            angle, info = brain.process(obs)
            obs, reward, done, step_info = env.step((angle,))

            # Food eating triggers
            if step_info.get("food_eaten"):
                eaten_type = step_info.get("eaten_food_type", -1)
                food_pos = (obs["position_x"], obs["position_y"])
                if eaten_type == 0:
                    brain.learn_food_location(food_position=food_pos)
                    brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
                elif eaten_type == 1:
                    if config.dopamine_dip_enabled:
                        brain.release_dopamine(reward_magnitude=-config.dopamine_dip_magnitude)
                    if config.taste_aversion_learning_enabled:
                        brain.trigger_taste_aversion()

            brain.decay_dopamine()

        # SWR replay
        if config.swr_replay_enabled:
            brain.replay_swr()

    # Diagnose
    print(f"\n  --- Diagnosis (sparsity={sp}) ---")
    diag = diagnose_auditory(brain, n_trials=15)

    # Call Semantics
    call = test_call_semantics(brain, n_trials=30)
    print(f"\n  Call Semantics: {call['score']:.1f}% ({'PASS' if call['pass'] else 'FAIL'})")

    results.append({
        "sparsity": sp,
        "connections": int(500 * sp),
        "call_score": call["score"],
        "call_pass": call["pass"],
        "decode": diag.get("decode_pass", False),
        "bias": diag.get("bias_pass", False),
    })

print(f"\n{'='*60}")
print(f"  SWEEP SUMMARY")
print(f"{'='*60}")
print(f"  {'Sparsity':>10} | {'Connections':>12} | {'Call %':>8} | {'Pass':>6} | {'Decode':>8} | {'Bias':>6}")
print(f"  {'-'*10} | {'-'*12} | {'-'*8} | {'-'*6} | {'-'*8} | {'-'*6}")
for r in results:
    print(f"  {r['sparsity']:>10.2f} | {r['connections']:>12} | {r['call_score']:>7.1f}% | {'✓' if r['call_pass'] else '✗':>6} | {'✓' if r['decode'] else '✗':>8} | {'✓' if r['bias'] else '✗':>6}")
print(f"{'='*60}")
