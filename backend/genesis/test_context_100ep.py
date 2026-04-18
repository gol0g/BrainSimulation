#!/usr/bin/env python3
"""M4 v9b Context Hard Gate — typed food I_input + learnable scales

v9b 핵심:
- D1_ctx I_input = good_food * scale_good + bad_food * scale_bad (typed!)
- context별 scale이 DA에 의해 학습됨
- Zone A: good_scale stays high, bad_scale stays low
- Zone B: good_scale decreases (type 0=bad), bad_scale increases (type 1=good)
- update_context_food_scales() called at food eating time
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

env_config = ForagerConfig()
env_config.context_rules_enabled = True
env_config.contingency_reversal_enabled = False

brain_config = ForagerBrainConfig()
brain_config.context_hard_gate_enabled = True

env = ForagerGym(env_config, render_mode="none")
brain = ForagerBrain(brain_config)

print("\n=== M4 v9b: Context Hard Gate + Typed Food Scales ===")
print(f"  D1_ctx neurons: {brain_config.n_ctx_d1}×4 = {brain_config.n_ctx_d1*4}")
print(f"  Hard gate: {brain._context_hard_gate_active}")
print(f"  Initial scales: {brain._ctx_food_scale}")
print()

for batch in range(5):
    total_good = 0
    total_bad = 0
    for ep in range(20):
        obs = env.reset()
        done = False
        step_count = 0
        prev_food = env.total_food_eaten
        while not done:
            angle, info = brain.process(obs)
            step_count += 1
            if step_count % 10 == 0:
                brain.decay_dopamine()
            obs, reward, done, step_info = env.step((angle,))

            # Dopamine + context food scale update on food eating
            curr_food = env.total_food_eaten
            if curr_food > prev_food:
                food_type = env._food_eaten_types[-1] if env._food_eaten_types else 0
                ctx = brain._current_ctx
                # Context-dependent DA
                if env_config.context_rules_enabled and ctx == "b":
                    da_mag = -0.5 if food_type == 0 else 1.0
                else:
                    da_mag = 1.0 if food_type == 0 else -0.5
                brain.release_dopamine(reward_magnitude=da_mag, primary_reward=True)
                # v9b: Update typed food scales for current context
                brain.update_context_food_scales(food_type, da_mag)
            prev_food = curr_food

        total_good += env.good_food_eaten
        total_bad += env.bad_food_eaten
        if brain_config.swr_replay_enabled:
            brain.replay_swr()

    sel = total_good / max(total_good + total_bad, 1)
    s = brain._ctx_food_scale
    # Divergence: how different are A and B food scales
    a_diff = s["a_good"] - s["a_bad"]  # A: good should be >> bad
    b_diff = s["b_good"] - s["b_bad"]  # B: good should be << bad (flipped)
    scale_div = abs(a_diff - b_diff)

    ep_num = (batch + 1) * 20
    print(f"ep {ep_num}: sel={sel:.3f} good={total_good} bad={total_bad} scale_div={scale_div:.2f}")
    print(f"  Scales: A_good={s['a_good']:.1f} A_bad={s['a_bad']:.1f} | "
          f"B_good={s['b_good']:.1f} B_bad={s['b_bad']:.1f}")
    print(f"  A_diff={a_diff:.1f} B_diff={b_diff:.1f}")

print("\n=== Target: sel > 0.55, scale_div > 5.0 ===")
