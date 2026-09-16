#!/usr/bin/env python3
"""D1 뉴런들이 서로 다른 입력을 받는가 (E081 전제 확인).

E080에서 측면억제가 실패했다. 가설: D1 100개 뉴런이 FixedProbability로 연결돼
**통계적으로 동일한 입력**을 받아 사실상 1개처럼 작동한다.
FlyWire의 MBON 96개는 각자 **다른 KC 조합**을 받는다(MBON당 441개씩 서로 다르게).

측정: 같은 자극에서 D1 뉴런별 발화율의 분산. 0에 가까우면 모두 동일 = 분할해도 소용없다.
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--kc-gamma", action="store_true", help="E081/H015: KC→D1 가중치 감마분포")
ap.add_argument("--kc-rstdp", action="store_true")
ap.add_argument("--kc-d1-w", type=float, default=None,
                help="E081후속: KC→D1 초기 가중치(기본 0.5). 0/0.5/5/20 용량-반응으로 경로 도달 확인.")
ap.add_argument("--d1-inhib", type=float, default=-400.0,
                help="INV-A4. 기본값 0.0은 D1 포화 상태다 — 불변식은 -400을 요구한다.")
ap.add_argument("--direct-inhib", type=float, default=-100.0,
                help="INV-A5.")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

import random
random.seed(a.seed); np.random.seed(a.seed)
cfg = ForagerBrainConfig()
if a.kc_rstdp: cfg.kc_rstdp = True
if getattr(a, "kc_gamma", False): cfg.kc_weight_gamma = True
if a.kc_d1_w is not None: cfg.kc_to_d1_init_w = a.kc_d1_w
if a.d1_inhib: cfg.d1_inhibition = a.d1_inhib
if a.direct_inhib: cfg.direct_inhibition = a.direct_inhib
b = ForagerBrain(cfg)
env = ForagerGym(ForagerConfig()); obs = env.reset()
for _ in range(20):
    act, _ = b.process(obs); obs, _, d, _ = env.step((act,))
    if d: obs = env.reset()
nh = env.config.n_rays // 2

def stim(side):
    o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    L = 0.9 if side == "left" else 0.0
    R = 0.9 if side == "right" else 0.0
    o["good_food_rays_left"] = np.ones(nh)*L; o["good_food_rays_right"] = np.ones(nh)*R
    o["food_rays_left"] = np.ones(nh)*L;      o["food_rays_right"] = np.ones(nh)*R
    return o

def _pop_size(p):
    for attr in ("num_neurons", "size", "n_neurons"):
        v = getattr(p, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    raise RuntimeError("집단 크기를 알 수 없다 — 추정 금지")


def per_neuron_rate(pop_name, side, steps=40):
    p = getattr(b, pop_name, None)
    if p is None: return None
    counts = None
    o = stim(side)
    for _ in range(steps):
        b.process(o)
        try:
            ids = p.spike_recording_data[0][1]
            # 집단 크기는 **고정값**이어야 한다. max(ids)+1로 추정하면 스텝마다 길이가 달라져
            # 누적이 깨진다. 포화 상태(매 스텝 전 뉴런 발화)에서만 우연히 맞았다 — 2026-09-16 적발.
            n = _pop_size(p)
            c = np.bincount(np.asarray(ids, dtype=int), minlength=n)[:n]
            counts = c if counts is None else counts + c
        except Exception as _e:
            # 조용한 None 반환이 '측정불가'만 남기고 원인을 숨겼다(2026-09-16).
            print("  [%s 측정실패] %s: %s" % (pop_name, type(_e).__name__, _e))
            return None
    return counts

print("=== D1 뉴런별 발화 분산 (같은 자극에서 뉴런들이 서로 다른가) ===")
for pop in ["d1_left", "d1_right", "kc_left"]:
    c = per_neuron_rate(pop, "left")
    if c is None:
        print("  %-10s 측정불가" % pop); continue
    nz = (c > 0).mean()*100
    cv = (c.std()/c.mean()) if c.mean() > 0 else float("nan")
    print("  %-10s n=%d  평균 %.1f  std %.1f  **변동계수 %.3f**  발화뉴런 %.0f%%"
          % (pop, len(c), c.mean(), c.std(), cv, nz))
print("\n해석: 변동계수(CV)가 0에 가까우면 모든 뉴런이 동일하게 반응 = 집단이 1개처럼 작동.")
print("      FlyWire MBON은 각자 다른 KC 조합을 받으므로 CV가 커야 정상이다.")
