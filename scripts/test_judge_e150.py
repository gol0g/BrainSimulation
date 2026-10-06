#!/usr/bin/env python3
"""judge_e150.py 합성 시험(조건 1): 유지·간섭 유지·보류, 전제 P1·P2 실패, 경계(rA 0.5 — 홀수 eA1 은 floor), 조작검증 실패, 결측, stats·줄 파싱.
실행: python3 scripts/test_judge_e150.py (저장소 루트에서)"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e150 as J


def build(eA1=-0.30, rA=0.8, eB=0.10, eA1_list=None, rA_list=None, eB_list=None, s_bad=None, rc_bad=None, floor=False):
    TR, EV, S, RC = {}, {}, {}, {}
    for i, b in enumerate(J.BRAINS):
        nb, nbad = 0.0150, 0.0200
        a1 = int(round((eA1_list[i] if eA1_list else eA1) * 1e4))
        r = rA_list[i] if rA_list else rA
        ea = int(math.floor(r * a1)) if floor else int(round(r * a1))
        EV[(b, "none", "base")] = nb; EV[(b, "none", "bad")] = nbad
        EV[(b, "A", "base")] = round(nb + a1 / 1e4, 4)
        EV[(b, "AB", "base")] = round(nb + ea / 1e4, 4)
        EV[(b, "AB", "bad")] = round(nbad + (eB_list[i] if eB_list else eB), 4)
        for a in ("A", "AB"):
            TR[(a, b)] = {"pre": nb, "post": -0.1, "rew": 900}
            s = {"n": 1500 if a == "A" else 3000, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0}
            if s_bad and s_bad[0] == (a, b):
                s.update(s_bad[1])
            S[(a, b)] = s
            RC[(a, b)] = (not (rc_bad == ("b", a, b)), not (rc_bad == ("refl", a, b)))
    return TR, EV, S, RC


cases = [
    ("유지", build(rA=0.8), "유지(H073)"),
    ("유지 경계 rA=0.5(floor)", build(rA_list=[0.5] * 5, floor=True), "유지(H073)"),
    ("간섭 유지", build(rA=0.2), "간섭 유지(H073-null)"),
    ("간섭(뒤집힘)", build(rA=-0.3), "간섭 유지(H073-null)"),
    ("보류(섞임)", build(rA_list=[0.8, 0.8, 0.8, 0.2, 0.2]), "보류"),
    ("P1 과제 A 미학습", build(eA1_list=[-0.30, -0.30, -0.30, -0.04, -0.02]), "보류(차단 상태 과제 A 미학습)"),
    ("P2 과제 B 미학습", build(eB_list=[0.10, 0.10, 0.10, 0.04, 0.02]), "보류(과제 B 미학습)"),
    ("조작: 동결 실패", build(s_bad=(("AB", 12), {"res": 0.1})), "보류(조작검증 실패)"),
    ("조작: 규칙 불일치", build(s_bad=(("A", 11), {"agree": 0.99})), "보류(조작검증 실패)"),
    ("조작: 과제 B 줄 없음", build(rc_bad=("b", "AB", 10)), "보류(조작검증 실패)"),
    ("조작: 반사 변함", build(rc_bad=("refl", "A", 13)), "보류(조작검증 실패)"),
]
ok_all = True
for name, (TR, EV, S, RC), want in cases:
    c, r = J.judge(TR, EV, S, RC)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-24s 기대 %-26s → %s %s" % (name, want, r["verdict"][:26] if r else c[0][:30], "✓" if good else "✗"))
TR, EV, S, RC = build(); del EV[(14, "AB", "bad")]
c, r = J.judge(TR, EV, S, RC)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-24s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
R = np.zeros((3000, 27)); rng = np.random.default_rng(5); R[:, 2] = rng.integers(0, 2, 3000); R[:, 6] = rng.integers(0, 2, 3000)
R[:, 7] = np.where(np.arange(3000) < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]).astype(float); R[:, 13:17] = 1.0; R[:, 21:25] = J.R20
sAB = J.stats(R, 1500); sA = J.stats(R[:1500], None)
good = sAB["agree"] == 1.0 and sA["agree"] == 1.0 and sAB["n"] == 3000; ok_all &= good
print("%-24s 기대 일치 1.0·1.0 → %s %s %s" % ("stats", sAB["agree"], sA["agree"], "✓" if good else "✗"))
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E150"))
    open(os.path.join(td, "E150.log"), "w", encoding="utf-8").write(
        "  e150 train A b10: => 사전 +0.0150 사후 -0.2000 보상 900 || x\n  e150 train AB b10: => 사전 +0.0150 사후 +0.1000 보상 1900 || x\n  e150 b10 AB bad: => mod +0.3000\n")
    open(os.path.join(td, "logs", "E150", "train_AB_b10.log"), "w", encoding="utf-8").write(
        "[과제 B] 시행 1500 부터 자극 = bad food, 정답 = 같은 쪽\n[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")
    open(os.path.join(td, "logs", "E150", "train_A_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")
    J.EXP = td
    TRp, EVp, Sp, RCp = J.load()
good = (TRp[("A", 10)]["post"] == -0.2 and TRp[("AB", 10)]["rew"] == 1900 and EVp == {(10, "AB", "bad"): 0.3}
        and RCp[("AB", 10)] == (True, True) and RCp[("A", 10)] == (True, True) and RCp[("A", 11)] is None); ok_all &= good
print("%-24s 기대 학습·평가 줄·과제 B 줄(A 팔엔 없어야) → %s" % ("줄 파싱", "✓" if good else "✗ %s %s" % (TRp, RCp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
