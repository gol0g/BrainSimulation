#!/usr/bin/env python3
"""KC가 발화하는가 (E081b 후속).

용량-반응에서 kc_to_d1 가중치를 0→20(40배)으로 올려도 D1 발화가 126.8→129.4(+2%)뿐이었다.
기본값 0.5에서는 w=0과 소수점까지 동일했다 → KC→D1 기여가 0.
원인 후보: (1) KC가 거의 발화하지 않는다  (2) D1이 다른 입력에 포화돼 KC 몫이 묻힌다.
d1_diversity_probe에서 kc_left가 '측정불가'로 나온 이유도 여기서 같이 잡는다.
"""
import sys, os, argparse, random, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--d1-inhib", type=float, default=-400.0,
                help="INV-A4. 기본값 0.0은 D1 포화 상태다 — 불변식은 -400을 요구한다.")
ap.add_argument("--direct-inhib", type=float, default=-100.0,
                help="INV-A5.")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--kc-rstdp", action="store_true")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed)
cfg = ForagerBrainConfig()
if a.kc_rstdp: cfg.kc_rstdp = True
if a.d1_inhib: cfg.d1_inhibition = a.d1_inhib
if a.direct_inhib: cfg.direct_inhibition = a.direct_inhib
b = ForagerBrain(cfg)
env = ForagerGym(ForagerConfig()); obs = env.reset()
for _ in range(20):
    act, _ = b.process(obs); obs, _, d, _ = env.step((act,))
    if d: obs = env.reset()
nh = env.config.n_rays // 2

o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
o["good_food_rays_left"] = np.ones(nh)*0.9; o["good_food_rays_right"] = np.zeros(nh)
o["food_rays_left"] = np.ones(nh)*0.9;      o["food_rays_right"] = np.zeros(nh)

POPS = ["kc_left", "kc_right", "kc_inh_left", "d1_left",
        "food_eye_left", "good_food_eye_left", "d1_inh"]
tot = {p: 0 for p in POPS}
size = {}
errs = {}
STEPS = 40
for _ in range(STEPS):
    b.process(o)
    for nm in POPS:
        p = getattr(b, nm, None)
        if p is None:
            errs[nm] = "속성 없음"; continue
        size[nm] = getattr(p, "size", None)
        try:
            tot[nm] += len(p.spike_recording_data[0][1])
        except Exception as e:
            errs[nm] = "%s: %s" % (type(e).__name__, e)

print("\n%-20s %8s %12s %12s" % ("집단", "크기", "총스파이크", "뉴런당/스텝"))
print("-"*56)
for nm in POPS:
    if nm in errs and tot[nm] == 0:
        print("%-20s %8s %12s  %s" % (nm, size.get(nm, "?"), "-", errs[nm]))
        continue
    n = size.get(nm) or 1
    print("%-20s %8s %12d %12.3f" % (nm, size.get(nm, "?"), tot[nm], tot[nm]/float(n)/STEPS))
print("-"*56)
print("해석: KC 뉴런당/스텝이 0에 가까우면 KC는 침묵 — 가중치를 아무리 키워도 D1에 도달하지 않는다.")
