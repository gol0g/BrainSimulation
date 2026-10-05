#!/usr/bin/env python3
"""judge_e140.py 합성 시험(조건 1): 지지·기각·보류, 경계(평균 정확히 −0.10, 차 정확히 0.01), 조작검증 실패 3종, 결측, 줄 파싱.
실행: python3 scripts/test_judge_e140.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e140 as J


def S_(ml=0.0, mr=0.0, n=500, pre=0.0, ba=0.2, cp=0.2):
    return {"n": n, "ml": ml, "mr": mr, "pre_ratio": pre, "BA": ba, "CP": cp, "eda": -1.0, "eend": -0.5}


def build(effects, Skw=None, pre_shift=None):
    T, S = {}, {}
    for i, b in enumerate(J.BRAINS):
        pre = J.E119_PRE[b] + (pre_shift.get(b, 0.0) if pre_shift else 0.0)
        T[b] = {"pre": round(pre, 4), "post": round(pre + effects[i], 4), "rew": 300}
        S[b] = S_(**(Skw(b) if Skw else {}))
    return T, S


E = [J.E119_EFF[b] for b in J.BRAINS]
cases = [
    ("지지", build([-0.20, -0.15, -0.12, -0.13, -0.10]), "지지(H063)"),
    ("기각(같음)", build([x + 0.004 for x in E]), "기각(H063-null)"),
    ("보류(커졌으나 평균 −0.09)", build([-0.12, -0.08, -0.08, -0.09, -0.08]), "보류"),
    ("보류(4/5 만 큼)", build([-0.20, -0.15, -0.12, -0.13, -0.05]), "보류"),
    ("M1 침묵 실패", build([-0.20, -0.15, -0.12, -0.13, -0.10], Skw=lambda b: {"ml": 0.3} if b == 12 else {}), "보류(조작검증 실패)"),
    ("M2 출발점 다름", build([-0.20, -0.15, -0.12, -0.13, -0.10], pre_shift={11: 0.003}), "보류(조작검증 실패)"),
    ("M3 도파민 전 변화", build([-0.20, -0.15, -0.12, -0.13, -0.10], Skw=lambda b: {"pre": 0.01} if b == 14 else {}), "보류(조작검증 실패)"),
]
# 경계: 평균 정확히 −0.10(소수 4자리 값), 모두 E119 보다 큼
cases.append(("경계 평균 −0.10", build([-0.11, -0.10, -0.10, -0.10, -0.09]), "지지(H063)"))
# 경계: |차| 정확히 0.01 은 '같음' 아님(< 0.01) → 기각 조건 4/5 미달
cases.append(("경계 차 0.01", build([x + 0.01 for x in E]), "보류"))
ok_all = True
for name, (T, S), want in cases:
    c, r = J.judge(T, S)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-26s 기대 %-16s → %s %s" % (name, want, r["verdict"][:30] if r else None, "✓" if good else "✗"))
T, S = build([-0.2] * 5); del S[13]
c, r = J.judge(T, S)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-26s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E140.log"), "w", encoding="utf-8").write("  e140 b10: => 사전 +0.0195 사후 -0.1500 보상 280 || => KCTRACE 시행 500 ...\n")
    J.EXP = td
    Tp, Sp = J.load()
good = Tp[10] == {"pre": 0.0195, "post": -0.15, "rew": 280}; ok_all &= good
print("%-26s 기대 줄 파싱 → %s" % ("줄 파싱", "✓" if good else "✗ %s" % Tp))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
