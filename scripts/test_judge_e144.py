#!/usr/bin/env python3
"""judge_e144.py 합성 시험(조건 1): L2·부분·무관·반대·보류, 경계(m −0.02, d ±0.03, 반대쪽 = R0+0.01), 조작검증 실패 7종, 결측·R0 없음,
stats() 합성 행(열 35·36, 보상 시행 같은 쪽 흔적), 요약 줄 파싱(잘린 바이트 허용).
실행: python3 scripts/test_judge_e144.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e144 as J

R0 = 0.02


def S_(**kw):
    s = {"n": 1500, "ncol": 37, "res": 1e-8, "alive": 1.0, "pre_ratio": 0.0, "aw_ex": 3.0, "aw_ot": 0.0, "same_rw": -1e6, "cross_rw": 1e6,
         "BA": -1.0, "CP": -0.8, "dD": 3e6, "blk": [4e5, 2e5]}
    s.update(kw)
    return s


def build(post=None, d=None, Skw=None, pre_shift=None, rf_bad=None):
    T, S, RF = {}, {}, {}
    for i, b in enumerate(J.BRAINS):
        pre = round(J.E119_PRE[b] + (pre_shift.get(b, 0.0) if pre_shift else 0.0), 4)
        po = post[i] if post is not None else J.E119_PRE[b] + J.F1500_EFF[b] + d[i]
        T[b] = {"pre": pre, "post": round(po, 4), "rew": 500}
        S[b] = S_(**(Skw(b) if Skw else {}))
        RF[b] = not (rf_bad and b in rf_bad)
    return T, S, RF


G = [-0.06] * 5
cases = [
    ("L2", build(post=[-0.05, -0.03, -0.10, -0.02, 0.05]), "L2 달성(H067)"),
    ("부분", build(d=[-0.06, -0.05, -0.04, -0.05, -0.07]), "반대쪽 발화가 정체 원인(H067-partial)"),
    ("부분 경계 d=−0.03", build(d=[-0.03] * 5), "반대쪽 발화가 정체 원인(H067-partial)"),
    ("무관", build(d=[0.01, -0.02, 0.0, 0.029, -0.029]), "무관(H067-null)"),
    ("반대", build(d=[0.05, 0.04, 0.03, 0.06, 0.0]), "반대(H067-rev)"),
    ("보류", build(d=[-0.06, -0.06, 0.0, 0.05, 0.05]), "보류"),
    ("M5 경계 = R0+0.01 통과", build(d=G, Skw=lambda b: {"aw_ot": R0 + 0.01}), "반대쪽 발화가 정체 원인(H067-partial)"),
    ("M5 실패", build(d=G, Skw=lambda b: {"aw_ot": R0 + 0.02} if b == 10 else {}), "보류(조작검증 실패)"),
    ("M5 열 없음(nan)", build(d=G, Skw=lambda b: {"aw_ot": float("nan")} if b == 11 else {}), "보류(조작검증 실패)"),
    ("M6 실패", build(d=G, Skw=lambda b: {"same_rw": 5e4} if b == 12 else {}), "보류(조작검증 실패)"),
    ("M1 실패", build(d=G, Skw=lambda b: {"res": 0.1} if b == 13 else {}), "보류(조작검증 실패)"),
    ("M1b 실패", build(d=G, Skw=lambda b: {"alive": 0.5} if b == 14 else {}), "보류(조작검증 실패)"),
    ("M2 실패", build(d=G, pre_shift={10: 0.003}), "보류(조작검증 실패)"),
    ("M3 실패", build(d=G, Skw=lambda b: {"n": 500} if b == 11 else {}), "보류(조작검증 실패)"),
    ("M4 실패", build(d=G, rf_bad={12}), "보류(조작검증 실패)"),
]
ok_all = True
for name, (T, S, RF), want in cases:
    c, r = J.judge(T, S, RF, R0)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-22s 기대 %-30s → %s %s" % (name, want, r["verdict"][:30] if r else None, "✓" if good else "✗"))
T, S, RF = build(d=G); del S[13]
c, r = J.judge(T, S, RF, R0)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-22s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
T, S, RF = build(d=G)
c, r = J.judge(T, S, RF, None)
good = r is None and "R0" in c[0]; ok_all &= good
print("%-22s 기대 R0 없음 → %s" % ("R0 없음", "✓" if good else "✗"))

R = np.zeros((1500, 37)); R[:, 7] = (np.arange(1500) % 2 == 0)
R[:, 13] = np.where(R[:, 7] == 1, 100.0, -50.0); R[:, 14] = np.where(R[:, 7] == 1, -80.0, 60.0)
R[:, 21] = J.R20 * R[:, 13]; R[:, 22] = J.R20 * R[:, 14]
R[:, 35] = 2.5; R[:, 36] = 0.004
s = J.stats(R)
good = (abs(s["aw_ex"] - 2.5) < 1e-12 and abs(s["aw_ot"] - 0.004) < 1e-12 and abs(s["same_rw"] + 80.0 * 750) < 1e-6
        and abs(s["cross_rw"] - 100.0 * 750) < 1e-6 and s["res"] < 1e-12 and len(s["blk"]) == 15); ok_all &= good
print("%-22s 기대 실행 2.5·반대 0.004·같은쪽 −60000·교차 +75000 → %.3f %.3f %+.0f %+.0f %s"
      % ("stats", s["aw_ex"], s["aw_ot"], s["same_rw"], s["cross_rw"], "✓" if good else "✗"))
s2 = J.stats(R[:, :35])
good = s2["aw_ot"] != s2["aw_ot"]; ok_all &= good
print("%-22s 기대 nan → %s %s" % ("stats 옛 35열", s2["aw_ot"], "✓" if good else "✗"))

with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E144")); os.makedirs(os.path.join(td, "traces", "E144", "calib"))
    open(os.path.join(td, "E144.log"), "wb").write("  e144 b10: => 사전 +0.4148 사후 -0.0500 보상 600 || => KCTRACE 블\n".encode("utf-8")[:-2] + b"\xeb\xa1\n")
    Rc = np.zeros((100, 37)); Rc[:, 36] = 0.03
    np.savez_compressed(os.path.join(td, "traces", "E144", "calib", "tr_R0_5000_b15.npz"), rows=Rc)
    J.EXP = td
    Tp, Sp, RFp, r0p = J.load()
good = Tp == {10: {"pre": 0.4148, "post": -0.05, "rew": 600}} and abs(r0p - 0.03) < 1e-12; ok_all &= good
print("%-22s 기대 e144 줄·R0 0.03 → %s" % ("줄·R0 파싱", "✓" if good else "✗ %s %s" % (Tp, r0p)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
