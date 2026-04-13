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
        while not done:
            angle, info = brain.process(obs)
            step_count += 1
            if step_count % 10 == 0:  # 매 10스텝마다 (GPU 부하 절감)
                brain.decay_dopamine()
            obs, reward, done, step_info = env.step((angle,))
        total_good += env.good_food_eaten
        total_bad += env.bad_food_eaten
        if brain_config.swr_replay_enabled:
            brain.replay_swr()
    sel = total_good / max(total_good + total_bad, 1)
    wa = getattr(brain, '_ctx_a_kc_to_d1_l', np.array([0]))
    wb = getattr(brain, '_ctx_b_kc_to_d1_l', np.array([0]))
    diff = float(np.mean(np.abs(wa - wb)))
    print(f"ep {(batch+1)*20}: sel={sel:.3f} divergence={diff:.4f}")
