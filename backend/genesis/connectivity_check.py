#!/usr/bin/env python3
"""네 칸의 **연결 구조가 동일한가** (E084 식별가능성 전제).

`--kc-rstdp`는 kc_to_d1 의 시냅스 모델을 StaticPulse ↔ R-STDP 로 바꾼다.
같은 genn_seed 라도 모델이 다르면 **난수 소비 순서가 달라져 연결이 달라질 수 있다.**
그러면 A/B 칸의 차이는 학습이 아니라 **배선 차이**가 된다 — 사전등록 6번이 잡아야 할 교란이다.

`transplant_eval.push()` 는 **크기 불일치만** 잡는다. 크기가 같은데 연결이 다르면
이식이 조용히 엉뚱한 시냅스에 꽂힌다. 그래서 연결 인덱스 자체를 비교한다.
"""
import sys, os, argparse, random, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig

SYNS = ("kc_to_d1_l", "kc_to_d1_r", "food_to_d1_l", "food_to_d1_r",
        "good_food_to_motor_l", "good_food_to_motor_r")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--d1-inhib", type=float, default=-400.0)
ap.add_argument("--direct-inhib", type=float, default=-100.0)
a = ap.parse_args()


def fingerprint(kc_rstdp, kc_d1_w):
    random.seed(a.seed); np.random.seed(a.seed)
    cfg = ForagerBrainConfig()
    cfg.d1_inhibition = a.d1_inhib
    cfg.direct_inhibition = a.direct_inhib
    cfg.genn_seed = 12345 + a.seed
    cfg.kc_real_rstdp_w_max = 750.0
    cfg.kc_to_d1_init_w = kc_d1_w
    if kc_rstdp:
        cfg.kc_rstdp = True
    b = ForagerBrain(cfg)
    out = {}
    for nm in SYNS:
        s = getattr(b, nm, None)
        if s is None:
            out[nm] = ("없음", 0); continue
        try:
            s.pull_connectivity_from_device()
        except Exception:
            pass
        try:
            ind = np.asarray(s.get_sparse_post_inds())
            rl = np.asarray(s.get_sparse_pre_inds())
            h = hashlib.sha256(ind.tobytes() + rl.tobytes()).hexdigest()[:16]
            out[nm] = (h, int(ind.size))
        except Exception as e:
            out[nm] = ("측정불가:%s" % type(e).__name__, -1)
    return out


CELLS = [("A w0.5 KC끔", False, 0.5), ("B w0.5 KC켬", True, 0.5),
         ("C w150 KC끔", False, 150.0), ("D w150 KC켬", True, 150.0)]
res = {}
for label, kr, kw in CELLS:
    res[label] = fingerprint(kr, kw)
    print("[완료] %s" % label)

print("\n%-22s %-18s %-18s %-18s %-18s" % ("시냅스", *[c[0] for c in CELLS]))
print("-" * 98)
allsame = True
for nm in SYNS:
    vals = [res[c[0]][nm] for c in CELLS]
    same = len(set(v[0] for v in vals)) == 1
    allsame = allsame and same
    print("%-22s %-18s %-18s %-18s %-18s  %s"
          % (nm, *[("%s/%d" % (v[0][:8], v[1])) for v in vals],
             "동일" if same else "**다름**"))
print("-" * 98)
print("판정: %s" % ("네 칸의 연결 구조가 동일하다 — 이식·비교 가능" if allsame
                 else "**연결이 다르다. 칸 간 차이에 배선 차이가 섞인다 — 설계를 고쳐야 한다**"))
