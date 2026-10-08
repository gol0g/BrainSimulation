#!/usr/bin/env python3
"""E156 독립 대조 — judge_e156.py 를 쓰지 않고 런별 원 로그(logs/E156/{F500,F1500}_b*.log)와 추적(traces/E156/tr_*_b*.npz)에서 다시 계산한다.
[사전]/[사후] 변조폭 줄(4자리 문자열 → 정수), [반사가중치] 25→25, 적재 검증 줄 수, 추적 행마다 동결 잔차·되돌림·도파민 전 변화.
실행: python3 scripts/verify_e156_independent.py (저장소 루트에서)"""
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


def wstar():
    """수정 1: 보정 결과(logs/E156/wstar.txt) 'W*=정수 ...' — 없거나 '보정 실패'면 None."""
    p = os.path.join(EXP, "logs", "E156", "wstar.txt")
    if not os.path.exists(p):
        return None
    m = re.search(r"W\*=(\d+)", open(p, encoding="utf-8").read())
    return int(m.group(1)) if m else None


def base_pre(b):
    """같은 뇌 기본 표현 반사 25 [사전] — E142 F500 원 로그에서 직접(판정 코드의 상수를 쓰지 않는다)."""
    txt = open(os.path.join(EXP, "logs", "E142", "F500_b%d.log" % b), encoding="utf-8", errors="replace").read()
    return q(re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d{4})", txt, re.M).group(1))


def run(arm, b, n_exp, w="25.0000"):
    txt = open(os.path.join(EXP, "logs", "E156", "%s_b%d.log" % (arm, b)), encoding="utf-8", errors="replace").read()
    pre = q(re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d{4})", txt, re.M).group(1))
    post = q(re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d{4})", txt, re.M).group(1))
    refl = re.findall(r"^\[반사가중치\] good_food_to_motor_[lr]\s+n=\d+ w_mean (\S+)→(\S+)", txt, re.M)
    nld = sum(1 for ln in txt.splitlines() if ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln)
    R = np.load(os.path.join(EXP, "traces", "E156", "tr_%s_b%d.npz" % (arm, b)))["rows"]
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
          and all(r == (w, w) for r in refl) and nld >= 2)
    return pre, post, ok


def main():
    D = {(a, b): run(a, b, n) for a, n in ARMS for b in BRAINS}
    ok = all(D[k][2] for k in D) and all(abs(D[("F500", b)][0] - D[("F1500", b)][0]) <= 20 for b in BRAINS)
    l2 = sum(D[("F1500", b)][1] <= -200 for b in BRAINS)
    opp = sum(D[("F500", b)][1] - D[("F500", b)][0] <= -1000 for b in BRAINS)
    nul = sum(abs(D[("F500", b)][1] - D[("F500", b)][0]) < 300 for b in BRAINS)
    for b in BRAINS:
        print("b%d 사전 %+d | F500 사후 %+d e %+d | F1500 사후 %+d e %+d | 조작 %s·%s" % (
            b, D[("F500", b)][0], D[("F500", b)][1], D[("F500", b)][1] - D[("F500", b)][0], D[("F1500", b)][1],
            D[("F1500", b)][1] - D[("F1500", b)][0], "✓" if D[("F500", b)][2] else "✗", "✓" if D[("F1500", b)][2] else "✗"))
    print("조작검증 %s | L2 %d/5 거스름 %d/5 효과 없음 %d/5" % ("통과" if ok else "실패", l2, opp, nul))
    if not ok:
        v = "보류(조작검증 실패)"
    elif l2 >= 4:
        v = "L2 달성(H079)"
    elif opp == 5:
        v = "반사를 거스름(H079-partial)"
    elif nul >= 4:
        v = "효과 없음(H079-null)"
    else:
        v = "보류"
    print("독립 판정: %s" % v)
    # 수정 1 — 판정 2(맞춤): W1500 조작검증 + MW([사전] − 기본 [사전] ∈ [−500, +1000] 1e-4), m ≤ −200 ≥4/5
    W = wstar()
    if W is None:
        v2 = "없음(보정 실패)"
    else:
        try:
            DW = {b: run("W1500", b, 1500, "%.4f" % W) for b in BRAINS}
        except (FileNotFoundError, AttributeError):
            DW = None
        if DW is None:
            v2 = "보류(결측)"
        else:
            okw = all(DW[b][2] and -500 <= DW[b][0] - base_pre(b) <= 1000 for b in BRAINS)
            nwin = sum(DW[b][1] <= -200 for b in BRAINS)
            for b in BRAINS:
                print("b%d W1500(W*=%d) 사전 %+d (기본 %+d) 사후 %+d e %+d | 조작 %s" % (
                    b, W, DW[b][0], base_pre(b), DW[b][1], DW[b][1] - DW[b][0], "✓" if DW[b][2] else "✗"))
            v2 = ("보류(조작검증 실패)" if not okw else
                  ("반사 발현을 맞춰도 학습이 이김" if nwin >= 4 else "반사 발현을 맞추면 L2 아님"))
    print("독립 판정 2: %s" % v2)
    if v2.startswith("없음") or v2.startswith("보류"):
        c = "판정 1(%s) + 식별 불가" % v
    elif v.startswith("L2 달성"):
        c = "H079 지지" if v2.endswith("이김") else "H079 부분"
    else:
        c = "보류(예측 밖)" if v2.endswith("이김") else "판정 1 그대로(%s) — L2 미해결" % v
    print("독립 종합: %s" % c)


if __name__ == "__main__":
    sys.exit(main())
