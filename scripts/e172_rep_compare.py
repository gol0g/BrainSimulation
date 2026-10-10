#!/usr/bin/env python3
"""E172 재현성 비교 — 같은 rev 400(뇌 15) 학습의 KC→motor 가중치 차이: 현재 코드끼리(E172 경로 검사 대 rep1) 와 코드 사이(E163 경로 검사 대 E172·rep1).
같은 코드끼리도 비슷한 크기로 갈리면 E163 대 E172 의 0.0002 차이는 비결정성(비트 단위 비재현 — current-state §9, 2026-10-03)으로 본다.
실행: python3 scripts/e172_rep_compare.py (저장소 루트에서)"""
import os
import sys

import numpy as np

T = "research/experiments/traces"
RUNS = {"E163": "E163/pathcheck/w_rev_b15.npz", "E172": "E172/pathcheck/w_rev_b15.npz", "rep1": "E172/pathcheck/rep1/w_rev_b15.npz"}
KEYS = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")


def diff(a, b):
    A, B = np.load(os.path.join(T, RUNS[a])), np.load(os.path.join(T, RUNS[b]))
    nd = tot = 0
    mx = 0.0
    other = 0
    for k in A.files:
        x, y = A[k].astype(float), B[k].astype(float)
        d = np.abs(x - y)
        if k in KEYS:
            nd += int((d > 0).sum()); tot += x.size; mx = max(mx, float(d.max()))
        else:
            other += int((d > 0).sum())
    return nd, tot, mx, other


for a, b in (("E172", "rep1"), ("E163", "E172"), ("E163", "rep1")):
    nd, tot, mx, other = diff(a, b)
    print("%s 대 %s: KC→motor 다른 시냅스 %d/%d (%.3f%%), 최대 |Δ| %.3g, 그 밖 집단 다른 원소 %d" % (a, b, nd, tot, 100.0 * nd / tot, mx, other))
sys.exit(0)
