#!/usr/bin/env python3
"""E070 조작 검증: sparsity 설정이 실제 시냅스 수를 바꾸는가.

사전등록 2번이 요구하는 예비 확인. 본 실험 전에 통과해야 한다.
두 조건의 n이 같으면 조작 무효 — 이 코드베이스에서 4회 발생한 패턴이다.
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig

TARGETS = ["food_to_d1_l", "food_to_d1_r", "good_food_to_motor_l", "good_food_to_motor_r",
           "food_to_d1_cross_lr", "food_to_d1_cross_rl"]   # E071: 교차경로 실재 확인

ap = argparse.ArgumentParser()
ap.add_argument("--learn-sparsity", type=float, default=None)
ap.add_argument("--reflex-sparsity", type=float, default=None)
ap.add_argument("--crossed", action="store_true")
a = ap.parse_args()

cfg = ForagerBrainConfig()
cfg.real_rstdp = True
if a.learn_sparsity is not None:
    cfg.learn_path_sparsity = a.learn_sparsity
if a.reflex_sparsity is not None:
    cfg.reflex_sparsity = a.reflex_sparsity
if a.crossed:
    cfg.rstdp_crossed = True
b = ForagerBrain(cfg)

print("SPARSITY learn=%s reflex=%s"
      % (getattr(cfg, "learn_path_sparsity", "?"), getattr(cfg, "reflex_sparsity", "?")))
for nm in TARGETS:
    s = getattr(b, nm, None)
    if s is None:
        print("  %-24s 없음" % nm); continue
    try:
        s.pull_connectivity_from_device()
        s.vars["g"].pull_from_device()
        v = s.vars["g"].values
        if v is None or (hasattr(v, "size") and v.size == 0):
            v = s.vars["g"].view
        arr = np.array(v, dtype=np.float64)
        print("  %-24s n=%-7d 평균g=%.3f" % (nm, arr.size, float(arr.mean())))
    except Exception as e:
        print("  %-24s 조회실패(%s)" % (nm, e))
