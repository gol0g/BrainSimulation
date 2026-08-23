#!/usr/bin/env python3
"""C14b: d1 억제로 포화 해소되나 + 개념(good/bad 변별)에 영향."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

inh = float(sys.argv[1])
cfg = ForagerBrainConfig(); cfg.d1_inhibition = inh
b = ForagerBrain(cfg); env = ForagerGym(ForagerConfig())
obs = env.reset()
for _ in range(15):
    a, i = b.process(obs); obs, _, d, _ = env.step((a,))
    if d: obs = env.reset()
nh = env.config.n_rays // 2
res = {}
for lvl in (0.0, 0.9):
    o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    for k in ("food_rays_left","good_food_rays_left","food_rays_right","good_food_rays_right"):
        o[k] = np.zeros(nh)
    o["food_rays_left"] = np.ones(nh)*lvl; o["good_food_rays_left"] = np.ones(nh)*lvl
    s = []
    for _ in range(5):
        b.process(o); s.append(len(b.d1_left.spike_recording_data[0][0]))
    res[lvl] = np.mean(s)
print(f"D1RESULT inhib={inh}: rest={res[0.0]:.0f} food={res[0.9]:.0f} diff={res[0.9]-res[0.0]:+.0f}")
