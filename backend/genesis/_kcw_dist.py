"""E081 조작검증 0단계: kc_to_d1 가중치가 실제로 감마 롱테일인가.

b가 scale이면 평균 ≈ kc_d1_w, rate면 평균 ≈ kc_d1_w * shape^2 로 크게 어긋난다.
분포 자체를 찍어서 확정한다.
"""
import sys, os, argparse, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig

ap = argparse.ArgumentParser()
ap.add_argument("--kc-gamma", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = ForagerBrainConfig()
cfg.kc_rstdp = True
if a.kc_gamma:
    cfg.kc_weight_gamma = True
random.seed(a.seed); np.random.seed(a.seed)
brain = ForagerBrain(cfg)

for nm in ("kc_to_d1_l", "kc_to_d1_r"):
    s = getattr(brain, nm, None)
    if s is None:
        print(nm, "없음"); continue
    s.vars["g"].pull_from_device()
    v = s.vars["g"].values
    if v is None or (hasattr(v, "size") and v.size == 0):
        v = s.vars["g"].view
    w = np.asarray(v, dtype=np.float64).ravel()
    w = w[np.isfinite(w)]
    print("%s n=%d 평균 %.3f std %.3f 중앙 %.3f p99 %.3f max %.3f"
          % (nm, w.size, w.mean(), w.std(), np.median(w),
             np.percentile(w, 99), w.max()))
