#!/usr/bin/env python3
"""judge_e141.py 합성 시험(조건 1): 지지·반대·기각·보류, 경계(평균 정확히 −0.10, d 정확히 ±0.01), 조작검증 실패 4종, 결측, 줄 파싱,
그리고 stats() 를 답을 아는 합성 행으로(동결 흔적 → 잔차 0, 무동결 → 잔차 큼, B/A·C/P 부호, ΔD).
실행: python3 scripts/test_judge_e141.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e141 as J


def S_(res=1e-7, eda=1.0e4, n=500, pre=0.0, ba=-0.8, cp=-0.9, dD=2.0e6):
    return {"n": n, "res": res, "eda_abs": eda, "pre_ratio": pre, "BA": ba, "CP": cp, "dD": dD}


S139 = {b: S_(res=3.0, eda=1.6e4, ba=0.9, cp=0.9, dD=1.0e6) for b in J.BRAINS}


def build(effects, Skw=None, pre_shift=None):
    T, S = {}, {}
    for i, b in enumerate(J.BRAINS):
        pre = J.E119_PRE[b] + (pre_shift.get(b, 0.0) if pre_shift else 0.0)
        T[b] = {"pre": round(pre, 4), "post": round(pre + effects[i], 4), "rew": 300}
        S[b] = S_(**(Skw(b) if Skw else {}))
    return T, S


E = [J.E119_EFF[b] for b in J.BRAINS]
GOOD = [-0.20, -0.15, -0.12, -0.13, -0.10]
cases = [
    ("지지", build(GOOD), "지지(H064)"),
    ("기각(같음)", build([x + 0.004 for x in E]), "기각(H064-null)"),
    ("반대(작아짐)", build([x + 0.02 for x in E]), "반대(H064-rev)"),
    ("반대 경계 d=+0.01 4/5", build([E[0] + 0.01, E[1] + 0.01, E[2] + 0.01, E[3] + 0.01, E[4] - 0.03]), "반대(H064-rev)"),
    ("보류(평균 −0.09)", build([-0.12, -0.08, -0.08, -0.09, -0.08]), "보류"),
    ("보류(4/5 만 큼)", build([-0.20, -0.15, -0.12, -0.13, -0.05]), "보류"),
    ("경계 평균 −0.10", build([-0.11, -0.10, -0.10, -0.10, -0.09]), "지지(H064)"),
    ("경계 d=−0.01 은 같음 아님", build([x - 0.01 for x in E]), "보류"),
    ("M1 동결 실패", build(GOOD, Skw=lambda b: {"res": 3.0} if b == 12 else {}), "보류(조작검증 실패)"),
    ("M1b 되돌림 실패", build(GOOD, Skw=lambda b: {"eda": 10.0} if b == 13 else {}), "보류(조작검증 실패)"),
    ("M2 출발점 다름", build(GOOD, pre_shift={11: 0.003}), "보류(조작검증 실패)"),
    ("M3 도파민 전 변화", build(GOOD, Skw=lambda b: {"pre": 0.01} if b == 14 else {}), "보류(조작검증 실패)"),
]
ok_all = True
for name, (T, S), want in cases:
    c, r = J.judge(T, S, S139)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-26s 기대 %-18s → %s %s" % (name, want, r["verdict"][:30] if r else None, "✓" if good else "✗"))
T, S = build(GOOD); del S[13]
c, r = J.judge(T, S, S139)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-26s 기대 결측 → %s" % ("결측(E141 추적)", "✓" if good else "✗"))
T, S = build(GOOD); S139b = dict(S139); del S139b[10]
c, r = J.judge(T, S, S139b)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-26s 기대 결측 → %s" % ("결측(E139 비교)", "✓" if good else "✗"))

# stats(): 답을 아는 합성 행. 보상 시행(짝수) 교차 dg +3·같은쪽 −2, 처벌(홀수) 교차 +1·같은쪽 −1 → A 750 B −500 C 250 P −250,
# B/A −0.667, C/P −1.0, ΔD = (750+250) − (−500−250) = 1750. 동결 흔적: e_end = r*·e_da → 잔차 0. 무동결: e_end = 4·e_da → 잔차 |4 − r*|.
rng = np.random.default_rng(0)
rows = np.zeros((500, 27))
rows[:, 7] = (np.arange(500) % 2 == 0)
rows[:, 8] = np.where(rows[:, 7] == 1, 3.0, 1.0)
rows[:, 9] = np.where(rows[:, 7] == 1, -2.0, -1.0)
rows[:, 12] = rows[:, 8:12].sum(1)
rows[:, 13:17] = rng.normal(0, 1000, (500, 4))
rows[:, 21:25] = J.R_STAR * rows[:, 13:17]
s = J.stats(rows)
good = (s["n"] == 500 and s["res"] < 1e-12 and abs(s["BA"] + 500 / 750) < 1e-12 and abs(s["CP"] + 1.0) < 1e-12
        and abs(s["dD"] - 1750) < 1e-9 and s["pre_ratio"] == 0.0); ok_all &= good
print("%-26s 기대 잔차 0·B/A −0.667·C/P −1·ΔD 1750 → res %.1e B/A %.3f C/P %.3f ΔD %.0f %s"
      % ("stats 동결 행", s["res"], s["BA"], s["CP"], s["dD"], "✓" if good else "✗"))
rows2 = rows.copy(); rows2[:, 21:25] = 4.0 * rows[:, 13:17]
s2 = J.stats(rows2)
good = abs(s2["res"] - (4.0 - J.R_STAR)) < 1e-9; ok_all &= good
print("%-26s 기대 잔차 %.4f → %.4f %s" % ("stats 무동결 행", 4.0 - J.R_STAR, s2["res"], "✓" if good else "✗"))
rows3 = rows.copy(); rows3[:, 17] = 0.01 * rows3[:, 12]
s3 = J.stats(rows3)
good = abs(s3["pre_ratio"] - 0.01) < 1e-12; ok_all &= good
print("%-26s 기대 0.01 → %.4f %s" % ("stats 도파민 전 변화", s3["pre_ratio"], "✓" if good else "✗"))
good = abs(J.R_STAR - 0.175480) < 5e-7; ok_all &= good
print("%-26s 기대 0.175480 → %.6f %s" % ("r*", J.R_STAR, "✓" if good else "✗"))

with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E141.log"), "w", encoding="utf-8").write(
        "  e141 b10: => 사전 +0.0195 사후 -0.1500 보상 280 || => KCTRACE 시행 500 ...\n  e140 b11: => 사전 +0.0150 사후 -0.1 보상 1\n")
    J.EXP = td
    Tp, Sp, S139p = J.load()
good = Tp == {10: {"pre": 0.0195, "post": -0.15, "rew": 280}}; ok_all &= good
print("%-26s 기대 줄 파싱(e140 줄 무시) → %s" % ("줄 파싱", "✓" if good else "✗ %s" % Tp))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
