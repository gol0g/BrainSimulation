#!/usr/bin/env python3
"""judge_e147.py 합성 시험(조건 1): 성공·부분·고착·보류, 경계(m +0.02, Δ +0.10·+0.05), 조작검증 실패 5종, 결측, stats(규칙 일치)·줄 파싱.
실행: python3 scripts/test_judge_e147.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e147 as J


def S_(**kw):
    s = {"n": 3000, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0, "rew_blk": [50] * 30, "dD_blk": [1e5] * 30}
    s.update(kw)
    return s


def build(post=None, d=None, Skw=None, rc_bad=None, pre_shift=0.0):
    T, S, RC = {}, {}, {}
    for i, b in enumerate(J.BRAINS):
        po = post[i] if post is not None else J.REF[b] + d[i]
        T[b] = {"pre": round(J.PRE0[b] + (pre_shift if b == 12 else 0.0), 4), "post": round(po, 4), "rew": 2000}
        S[b] = S_(**(Skw(b) if Skw else {}))
        RC[b] = (not (rc_bad == ("rev", b)), not (rc_bad == ("refl", b)))
    return T, S, RC


cases = [
    ("성공", build(post=[0.05, 0.10, 0.03, 0.02, -0.10]), "반전 성공(H070)"),
    ("성공 경계 +0.02 4/5", build(post=[0.02, 0.02, 0.02, 0.02, -0.30]), "반전 성공(H070)"),
    ("부분", build(d=[0.15, 0.20, 0.12, 0.10, 0.30]), "부분(H070-partial)"),
    ("부분 경계 Δ=+0.10", build(d=[0.10] * 5), "부분(H070-partial)"),
    ("고착", build(d=[0.01, 0.02, 0.0, 0.049, 0.20]), "고착(H070-stuck)"),
    ("보류", build(d=[0.08, 0.08, 0.06, 0.12, 0.30]), "보류"),
    ("M1 반전 줄 없음", build(d=[0.15] * 5, rc_bad=("rev", 10)), "보류(조작검증 실패)"),
    ("M2 규칙 불일치", build(d=[0.15] * 5, Skw=lambda b: {"agree": 0.99} if b == 11 else {}), "보류(조작검증 실패)"),
    ("M3 동결 실패", build(d=[0.15] * 5, Skw=lambda b: {"res": 0.1} if b == 12 else {}), "보류(조작검증 실패)"),
    ("M4 반사 변함", build(d=[0.15] * 5, rc_bad=("refl", 13)), "보류(조작검증 실패)"),
    ("M5 출발점 다름", build(d=[0.15] * 5, pre_shift=0.003), "보류(조작검증 실패)"),
]
ok_all = True
for name, (T, S, RC), want in cases:
    c, r = J.judge(T, S, RC)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-20s 기대 %-22s → %s %s" % (name, want, r["verdict"][:22] if r else None, "✓" if good else "✗"))
T, S, RC = build(d=[0.15] * 5); del S[14]
c, r = J.judge(T, S, RC)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-20s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
# stats: 앞 1500 은 정답 ⇔ 실행 ≠ 자극, 뒤 1500 은 정답 ⇔ 실행 = 자극 → 일치율 1.0; 한 행 어기면 < 1
rng = np.random.default_rng(0)
R = np.zeros((3000, 27)); R[:, 2] = rng.integers(0, 2, 3000); R[:, 6] = rng.integers(0, 2, 3000)
idx = np.arange(3000)
R[:, 7] = np.where(idx < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]).astype(float)
R[:, 13:17] = 1.0; R[:, 21:25] = J.R20
s = J.stats(R)
good = s["agree"] == 1.0 and s["n"] == 3000 and len(s["rew_blk"]) == 30 and s["res"] < 1e-12; ok_all &= good
R2 = R.copy(); R2[1600, 7] = 1 - R2[1600, 7]
s2 = J.stats(R2)
good2 = s2["agree"] < 1.0; ok_all &= good2
print("%-20s 기대 일치 1.0·한 행 어기면 < 1 → %s %s" % ("stats 규칙 일치", (s["agree"], round(s2["agree"], 5)), "✓" if good and good2 else "✗"))
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E147"))
    open(os.path.join(td, "E147.log"), "w", encoding="utf-8").write("  e147 b10: => 사전 +0.0195 사후 +0.0500 보상 2100 || x\n  e146 train b10: => 사전 +0.0195 사후 -0.3 보상 1\n")
    open(os.path.join(td, "logs", "E147", "b10.log"), "w", encoding="utf-8").write(
        "[반전] 시행 1500 부터 정답 = 같은 쪽(good 쪽)\n[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")
    J.EXP = td
    Tp, Sp, RCp = J.load()
good = Tp == {10: {"pre": 0.0195, "post": 0.05, "rew": 2100}} and RCp[10] == (True, True) and RCp[11] is None; ok_all &= good
print("%-20s 기대 e147 줄·반전 줄·반사 0 → %s" % ("줄 파싱", "✓" if good else "✗ %s %s" % (Tp, RCp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
