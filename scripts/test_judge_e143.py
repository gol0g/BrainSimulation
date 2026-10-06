#!/usr/bin/env python3
"""judge_e143.py 합성 시험(조건 1): L2·결정 단계 원인·무관·반대·보류, 경계(m 정확히 −0.02, d 정확히 −0.03·+0.03), 조작검증 실패 6종,
결측, 열 부족(35 미만 → M1d 실패), stats() 를 답을 아는 합성 행으로, 요약 줄·[반사가중치] 파싱.
실행: python3 scripts/test_judge_e143.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e143 as J


def S_(**kw):
    s = {"n": 1500, "ncol": 35, "res": 1e-8, "resd": 1e-8, "alive": 1.0, "pre_ratio": 0.0, "BA": -1.0, "CP": -0.8, "dD": 3e6, "blk": [4e5, 2e5]}
    s.update(kw)
    return s


def build(post=None, d=None, Skw=None, pre_shift=None, rf_bad=None):
    T, S, RF = {}, {}, {}
    for i, b in enumerate(J.BRAINS):
        pre = round(J.E119_PRE[b] + (pre_shift.get(b, 0.0) if pre_shift else 0.0), 4)
        if post is not None:
            po = post[i]
        else:
            po = J.E119_PRE[b] + J.F1500_EFF[b] + d[i]
        T[b] = {"pre": pre, "post": round(po, 4), "rew": 500}
        S[b] = S_(**(Skw(b) if Skw else {}))
        RF[b] = not (rf_bad and b in rf_bad)
    return T, S, RF


cases = [
    ("L2 달성", build(post=[-0.05, -0.03, -0.10, -0.02, 0.05]), "L2 달성(H066)"),
    ("L2 경계 m=−0.02", build(post=[-0.02, -0.02, -0.02, -0.02, 0.10]), "L2 달성(H066)"),
    ("결정 단계 원인", build(d=[-0.06, -0.05, -0.04, -0.05, -0.07]), "결정 단계 흔적이 정체 원인(H066-dec)"),
    ("원인 경계 d=−0.03", build(d=[-0.03] * 5), "결정 단계 흔적이 정체 원인(H066-dec)"),
    ("원인 4/5 → 보류", build(d=[-0.06, -0.05, -0.04, -0.05, -0.01]), "보류"),
    ("무관", build(d=[0.01, -0.02, 0.0, 0.029, -0.029]), "결정 단계 흔적 무관(H066-null)"),
    ("반대", build(d=[0.05, 0.04, 0.03, 0.06, 0.0]), "반대(H066-rev)"),
    ("M1 실패", build(d=[-0.06] * 5, Skw=lambda b: {"res": 0.1} if b == 10 else {}), "보류(조작검증 실패)"),
    ("M1d 실패", build(d=[-0.06] * 5, Skw=lambda b: {"resd": 0.5} if b == 11 else {}), "보류(조작검증 실패)"),
    ("M1d 열 없음(nan)", build(d=[-0.06] * 5, Skw=lambda b: {"resd": float("nan")} if b == 12 else {}), "보류(조작검증 실패)"),
    ("M1b 실패", build(d=[-0.06] * 5, Skw=lambda b: {"alive": 0.5} if b == 13 else {}), "보류(조작검증 실패)"),
    ("M2 실패", build(d=[-0.06] * 5, pre_shift={14: 0.003}), "보류(조작검증 실패)"),
    ("M3 실패", build(d=[-0.06] * 5, Skw=lambda b: {"n": 500} if b == 10 else {}), "보류(조작검증 실패)"),
    ("M4 실패", build(d=[-0.06] * 5, rf_bad={11}), "보류(조작검증 실패)"),
]
ok_all = True
for name, (T, S, RF), want in cases:
    c, r = J.judge(T, S, RF)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-18s 기대 %-30s → %s %s" % (name, want, r["verdict"][:30] if r else None, "✓" if good else "✗"))
T, S, RF = build(d=[-0.06] * 5); del S[12]
c, r = J.judge(T, S, RF)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-18s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))

# stats: 결정 단계 동결 행(e_dec = r30·e_t0) 잔차 0, 열 27(옛 형식)이면 resd nan
rng = np.random.default_rng(1)
R = np.zeros((1500, 35)); R[:, 7] = (np.arange(1500) % 3 == 0)
R[:, 8] = np.where(R[:, 7] == 1, 2.0, 1.0); R[:, 9] = np.where(R[:, 7] == 1, -1.0, -2.0); R[:, 12] = R[:, 8:12].sum(1)
R[:, 13:17] = rng.normal(0, 1000, (1500, 4)); R[:, 21:25] = J.R20 * R[:, 13:17]
R[:, 27:31] = rng.normal(0, 50, (1500, 4)); R[:, 31:35] = J.R30 * R[:, 27:31]
s = J.stats(R)
good = s["res"] < 1e-12 and s["resd"] < 1e-12 and s["n"] == 1500 and len(s["blk"]) == 15; ok_all &= good
print("%-18s 기대 잔차 0·0·블록 15 → res %.1e resd %.1e 블록 %d %s" % ("stats 동결 행", s["res"], s["resd"], len(s["blk"]), "✓" if good else "✗"))
R2 = R.copy(); R2[:, 31:35] = R2[:, 27:31] + 100.0
s2 = J.stats(R2)
good = s2["resd"] > 1e-3; ok_all &= good
print("%-18s 기대 잔차 > 1e-3 → %.2e %s" % ("stats 무동결 결정", s2["resd"], "✓" if good else "✗"))
s3 = J.stats(R[:, :27])
good = s3["resd"] != s3["resd"]; ok_all &= good
print("%-18s 기대 nan → %s %s" % ("stats 옛 27열", s3["resd"], "✓" if good else "✗"))
good = abs(J.R30 - 0.073509) < 5e-7 and abs(J.R20 - 0.175480) < 5e-7; ok_all &= good
print("%-18s 기대 0.073509·0.175480 → %.6f·%.6f %s" % ("r30·r20", J.R30, J.R20, "✓" if good else "✗"))

with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E143")); os.makedirs(os.path.join(td, "traces", "E143"))
    open(os.path.join(td, "E143.log"), "wb").write(
        "  e143 b10: => 사전 +0.4148 사후 -0.0500 보상 600 || => KCTRACE 블\n  e142 F1500 b10: => 사전 +0.4148 사후 +0.1 보상 1\n".encode("utf-8")[:-2] + b"\xeb\xa1\n")
    open(os.path.join(td, "logs", "E143", "b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=15139 w_mean 25.0000→25.0000 (학습 뇌)\n[반사가중치] good_food_to_motor_r   n=14818 w_mean 25.0000→25.0000 (학습 뇌)\n")
    J.EXP = td
    Tp, Sp, RFp = J.load()
good = Tp == {10: {"pre": 0.4148, "post": -0.05, "rew": 600}} and RFp[10] is True and RFp[11] is None; ok_all &= good
print("%-18s 기대 e143 줄만·잘린 바이트 허용·반사 줄 → %s" % ("줄 파싱", "✓" if good else "✗ %s %s" % (Tp, RFp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
