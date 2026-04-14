#!/usr/bin/env python3
"""M4 context-dependent 100ep learning trajectory test"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

env_config = ForagerConfig()
env_config.context_rules_enabled = True
env_config.contingency_reversal_enabled = False
brain_config = ForagerBrainConfig()
env = ForagerGym(env_config, render_mode="none")
brain = ForagerBrain(brain_config)

for batch in range(5):
    total_good = 0
    total_bad = 0
    for ep in range(20):
        obs = env.reset()
        done = False
        step_count = 0
        prev_food = 0
        while not done:
            angle, info = brain.process(obs)
            step_count += 1
            if step_count % 10 == 0:
                brain.decay_dopamine()
            obs, reward, done, step_info = env.step((angle,))
            # M4: food 이벤트 시 context-specific value update
            curr_food = env.total_food_eaten
            if curr_food > prev_food and brain_config.context_gate_enabled:
                ctx = brain._current_ctx
                food_type = env._food_eaten_types[-1] if env._food_eaten_types else 0
                reward_sign = 1.0 if food_type == 0 else -1.0
                for side in ["l", "r"]:
                    key = f"{ctx}_{side}"
                    brain._ctxval_w[key] += 0.15 * reward_sign
                    np.clip(brain._ctxval_w[key], 0.1, 8.0, out=brain._ctxval_w[key])
            prev_food = curr_food
        total_good += env.good_food_eaten
        total_bad += env.bad_food_eaten
        if brain_config.swr_replay_enabled:
            brain.replay_swr()
    sel = total_good / max(total_good + total_bad, 1)
    wa = brain._ctxval_w.get("a_l", np.array([3.0]))
    wb = brain._ctxval_w.get("b_l", np.array([3.0]))
    diff = float(np.mean(np.abs(wa - wb)))
    print(f"ep {(batch+1)*20}: sel={sel:.3f} divergence={diff:.4f}")
