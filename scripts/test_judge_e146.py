#!/usr/bin/env python3
"""judge_e146.py 합성 시험(조건 1): 충족·일반화 실패·용량 포화·부분, 경계(e −0.05, 몫 0.5), 측정 검증 실패 4종, 결측, 줄 파싱.
실행: python3 scripts/test_judge_e146.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e146 as J


def build(e500=None, e1500=None, frac=None, post1500=-0.35, none_shift=0.0, e141_shift=0.0, tr_res=1e-8, rf=True, tr_n=1500):
    """none 변조폭 = 사전(+ 변형별 0), E141 효과 = e500[s], R0F1500 효과 = e1500[s]."""
    e500 = e500 or {"base": -0.24, "int05": -0.15, "int07": -0.20, "occ": -0.14, "noise": -0.18}
    e1500 = e1500 or {s: e500[s] - 0.05 for s in J.STIMS}
    TR, EV, TS, RF = {}, {}, {}, {}
    for b in J.BRAINS:
        nb = round(J.E119_PRE0[b] + none_shift, 4)
        base500 = round(J.E141_POST[b] + e141_shift, 4)
        for s in J.STIMS:
            n_ = nb if s == "base" else round(nb + 0.01, 4)
            EV[(b, "none", s)] = n_
            EV[(b, "E141", s)] = base500 if s == "base" else round(n_ + (frac[s] * (base500 - nb) if frac else e500[s]), 4)
            EV[(b, "R0F1500", s)] = round(post1500, 4) if s == "base" else round(n_ + e1500[s], 4)
        TR[b] = {"pre": J.E119_PRE0[b], "post": round(post1500, 4), "rew": 900}
        TS[b] = {"n": tr_n, "res": tr_res, "blk": [5e5, 2e5]}
        RF[b] = rf
    return TR, EV, TS, RF


cases = [
    ("충족", build(), "충족(H069)"),
    ("일반화 실패(2 변형)", build(frac={"int05": 0.3, "int07": 0.8, "occ": 0.4, "noise": 0.8}), "일반화 실패(H069-spec)"),
    ("용량 포화", build(e1500={"base": -0.2, "int05": -0.10, "int07": -0.15, "occ": -0.10, "noise": -0.12}, post1500=-0.18), "용량 포화(H069-sat)"),
    ("부분(C1 경계 밖)", build(e500={"base": -0.24, "int05": -0.0499, "int07": -0.20, "occ": -0.14, "noise": -0.18}), "부분(보류)"),
    ("V1 실패", build(e141_shift=0.003), "보류(측정 검증 실패)"),
    ("V2 실패", build(none_shift=0.003), "보류(측정 검증 실패)"),
    ("V4 실패(잔차)", build(tr_res=0.1), "보류(측정 검증 실패)"),
    ("V4 실패(반사 줄)", build(rf=False), "보류(측정 검증 실패)"),
]
ok_all = True
for name, (TR, EV, TS, RF), want in cases:
    c, r = J.judge(TR, EV, TS, RF)
    good = r is not None and r["verdict"].startswith(want)
    ok_all &= good
    print("%-20s 기대 %-22s → %s %s" % (name, want, r["verdict"][:26] if r else c[0][:40], "✓" if good else "✗"))
# 경계: 몫 정확히 0.5 — e_base 를 짝수(1e-4 단위)로 만들려고 none 을 0.0001 옮기고(V2 허용 ±0.002 안), e_v = e_base/2 정확히
TR, EV, TS, RF = build()
for b in J.BRAINS:
    nb = EV[(b, "none", "base")]
    if (int(round(EV[(b, "E141", "base")] * 1e4)) - int(round(nb * 1e4))) % 2:
        nb = round(nb + 0.0001, 4)
        for s_ in J.STIMS:
            EV[(b, "none", s_)] = nb if s_ == "base" else round(nb + 0.01, 4)
    eb = int(round(EV[(b, "E141", "base")] * 1e4)) - int(round(nb * 1e4))
    for v_ in J.VARS:
        EV[(b, "E141", v_)] = round((int(round(EV[(b, "none", v_)] * 1e4)) + eb // 2) / 1e4, 4)
c, r = J.judge(TR, EV, TS, RF)
good = r is not None and r["verdict"].startswith("충족(H069)"); ok_all &= good
print("%-20s 기대 충족(H069) → %s %s" % ("일반화 경계 몫 0.5", r["verdict"][:26] if r else c[0][:40], "✓" if good else "✗"))
for b in J.BRAINS:   # 한 단위 모자라면(몫 0.5 미만) 실패해야 한다
    for v_ in J.VARS:
        EV[(b, "E141", v_)] = round(EV[(b, "E141", v_)] + 0.0001, 4)
c, r = J.judge(TR, EV, TS, RF)
good = r is not None and r["verdict"].startswith("일반화 실패(H069-spec)"); ok_all &= good
print("%-20s 기대 일반화 실패 → %s %s" % ("경계 한 단위 밖", r["verdict"][:26] if r else c[0][:40], "✓" if good else "✗"))
TR, EV, TS, RF = build(); del EV[(12, "R0F1500", "occ")]
c, r = J.judge(TR, EV, TS, RF)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-20s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
# V3: 학습 사후와 base·R0F1500 이 다르면 실패
TR, EV, TS, RF = build(); TR[10]["post"] = round(TR[10]["post"] + 0.01, 4)
c, r = J.judge(TR, EV, TS, RF)
good = r is not None and r["verdict"].startswith("보류(측정 검증 실패)"); ok_all &= good
print("%-20s 기대 보류(측정 검증 실패) → %s" % ("V3 실패", "✓" if good else "✗"))
R = np.zeros((1500, 27)); R[:, 13:17] = 1.0; R[:, 21:25] = J.R20; R[:, 8] = 1.0
s = J.train_stats(R)
good = s["res"] < 1e-12 and s["n"] == 1500 and len(s["blk"]) == 15 and abs(s["blk"][0] - 100.0) < 1e-9; ok_all &= good
print("%-20s 기대 잔차 0·블록 15 → %s" % ("train_stats", "✓" if good else "✗ %s" % s))
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E146"))
    open(os.path.join(td, "E146.log"), "w", encoding="utf-8").write(
        "  e146 train b10: => 사전 +0.0195 사후 -0.3000 보상 900 || x\n  e146 b10 E141 int05: => mod -0.1500\n  e146 b10 none base: => mod +0.0195\n")
    open(os.path.join(td, "logs", "E146", "train_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")
    J.EXP = td
    TRp, EVp, TSp, RFp = J.load()
good = (TRp == {10: {"pre": 0.0195, "post": -0.3, "rew": 900}} and EVp == {(10, "E141", "int05"): -0.15, (10, "none", "base"): 0.0195}
        and RFp[10] is True and RFp[11] is None); ok_all &= good
print("%-20s 기대 학습·평가 줄·반사 0 줄 → %s" % ("줄 파싱", "✓" if good else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
