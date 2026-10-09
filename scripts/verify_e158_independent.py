#!/usr/bin/env python3
"""E158 독립 대조 — judge_e158.py 를 쓰지 않고 런별 원 로그(logs/E158/{F500,F1500}_b*.log)·추적(traces/E158/tr_*_b*.npz)에서 다시 계산한다.
기본 [사전]은 E142 F500 원 로그, 재현 기준은 E157 r25 Fk 원 로그에서 직접 읽는다(판정 코드의 상수를 쓰지 않는다).
실행: python3 scripts/verify_e158_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = (("F500", 500), ("F1500", 1500))


def q(x):
    sgn = -1 if x.startswith("-") else 1
    a, b = x.lstrip("+-").split(".")
    return sgn * (int(a) * 10000 + int((b + "0000")[:4]))


def pre_of(*p):
    txt = open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()
    return q(re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d{4})", txt, re.M).group(1))


def run(arm, b, n_exp):
    txt = open(os.path.join(EXP, "logs", "E158", "%s_b%d.log" % (arm, b)), encoding="utf-8", errors="replace").read()
    pre = q(re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d{4})", txt, re.M).group(1))
    post = q(re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d{4})", txt, re.M).group(1))
    refl = re.findall(r"^\[반사가중치\] good_food_to_motor_[lr]\s+n=\d+ w_mean (\S+)→(\S+)", txt, re.M)
    nld = sum(1 for ln in txt.splitlines() if ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln)
    nsc = sum(1 for ln in txt.splitlines() if ln.startswith("[E157 종류 입력 배율]") and "검증 일치" in ln)
    R = np.load(os.path.join(EXP, "traces", "E158", "tr_%s_b%d.npz" % (arm, b)))["rows"]
    rstar = (11.0 / 12.0) ** 20
    num = den = 0.0
    alive = 0
    for i in range(len(R)):
        for k in range(4):
            num += abs(R[i, 21 + k] - rstar * R[i, 13 + k])
            den += abs(R[i, 13 + k])
        alive += (abs(R[i, 13]) + abs(R[i, 14])) > 1.0
    prer = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
    ok = (len(R) == n_exp and num / den <= 1e-3 and alive / len(R) >= 0.9 and prer <= 1e-3 and len(refl) == 2
          and all(r == ("25.0000", "25.0000") for r in refl) and nld >= 2 and nsc >= 2
          and abs(pre - pre_of("logs", "E157", "r25_Fk_b%d.log" % b)) <= 20)
    return pre, post, ok


def main():
    try:
        D = {(a, b): run(a, b, n) for a, n in ARMS for b in BRAINS}
        base = {b: pre_of("logs", "E142", "F500_b%d.log" % b) for b in BRAINS}
    except (FileNotFoundError, AttributeError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    ok = all(D[k][2] for k in D) and all(abs(D[("F500", b)][0] - D[("F1500", b)][0]) <= 20 for b in BRAINS)
    l2 = sum(D[("F1500", b)][1] <= -200 for b in BRAINS)
    opp = sum(D[("F500", b)][1] - D[("F500", b)][0] <= -1000 for b in BRAINS)
    nul = sum(abs(D[("F500", b)][1] - D[("F500", b)][0]) < 300 for b in BRAINS)
    strict = sum(D[("F1500", b)][1] - D[("F1500", b)][0] <= -base[b] for b in BRAINS)
    for b in BRAINS:
        print("b%d 사전 %+d | F500 사후 %+d e %+d | F1500 사후 %+d e %+d | 기본 사전 %+d | 조작 %s·%s" % (
            b, D[("F500", b)][0], D[("F500", b)][1], D[("F500", b)][1] - D[("F500", b)][0], D[("F1500", b)][1],
            D[("F1500", b)][1] - D[("F1500", b)][0], base[b], "✓" if D[("F500", b)][2] else "✗", "✓" if D[("F1500", b)][2] else "✗"))
    print("조작검증 %s | L2 %d/5 거스름 %d/5 효과 없음 %d/5 엄격 %d/5" % ("통과" if ok else "실패", l2, opp, nul, strict))
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    else:
        v1 = ("L2 달성(H081)" if l2 >= 4 else "반사를 거스름(H081-partial)" if opp == 5 else "효과 없음(H081-null)" if nul >= 4 else "보류")
        v2 = "엄격 L2 달성" if strict >= 4 else "엄격 L2 아님"
    print("독립 판정 1: %s" % v1)
    print("독립 판정 2: %s" % v2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
