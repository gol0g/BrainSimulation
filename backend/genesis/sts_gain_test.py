#!/usr/bin/env python3
"""C9d: agent_eye→sts_social 게인 스윕. 사회 입력이 tonic drive를 이기게 되는 지점 탐색."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

for gain in (15.0, 60.0, 150.0):
    cfg = ForagerBrainConfig()
    cfg.agent_eye_to_sts_social_weight = gain
    b = ForagerBrain(cfg); env = ForagerGym(ForagerConfig())
    obs = env.reset()
    for _ in range(15):
        a, i = b.process(obs); obs, _, d, _ = env.step((a,))
        if d: obs = env.reset()
    nh = env.config.n_rays // 2
    res = {}
    for lvl in (0.0, 1.0):
        o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        o["agent_rays_left"] = np.ones(nh) * lvl
        o["agent_rays_right"] = np.zeros(nh)
        o["agent_sound_left"] = lvl; o["social_proximity"] = lvl
        s = []
        for _ in range(4):
            b.process(o)
            s.append(len(b.sts_social.spike_recording_data[0][0]))
        res[lvl] = np.mean(s)
    print(f"GAINRESULT gain={gain}: sts_social 0.0={res[0.0]:.0f} 1.0={res[1.0]:.0f} diff={res[1.0]-res[0.0]:+.0f}")
