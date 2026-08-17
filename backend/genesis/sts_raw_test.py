#!/usr/bin/env python3
"""C9b: sts_social이 최대 사회입력(agent_rays 1.0)에도 무반응인지 원시 스파이크로 확정."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

b = ForagerBrain(ForagerBrainConfig())
env = ForagerGym(ForagerConfig())
obs = env.reset()
for _ in range(20):
    a, i = b.process(obs)
    obs, _, d, _ = env.step((a,))
    if d:
        obs = env.reset()

nh = env.config.n_rays // 2
for lvl in (0.0, 1.0):
    o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    o["agent_rays_left"] = np.ones(nh) * lvl
    o["agent_rays_right"] = np.zeros(nh)
    o["agent_sound_left"] = lvl
    o["social_proximity"] = lvl
    cnt = []
    for _ in range(4):
        b.process(o)
        cnt.append(len(b.sts_social.spike_recording_data[0][0]))
    print(f"RESULT agent_rays={lvl}: sts_social raw spikes/step = {cnt} (n={b.config.n_sts_social})")
