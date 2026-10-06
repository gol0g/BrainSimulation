#!/usr/bin/env python3
"""judge_e148.py 합성 시험(조건 1): 유지·간섭·보류, 과제 B 미학습, 경계(rA 0.5·eB 0.05), 조작검증 실패, 결측, stats·줄 파싱.
실행: python3 scripts/test_judge_e148.py (저장소 루트에서)"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e148 as J


def build(rA=0.8, eB=0.10, none_bad=0.01, Skw=None, rc_bad=None, eB_list=None, rA_list=None):
    TR, EV, S, RC = {}, {}, {}, {}
    for i, b in enumerate(J.BRAINS):
        nb = J.PRE0[b]
        eA1 = int(round(J.F1500_POST[b] * 1e4)) - int(round(nb * 1e4))
        r = rA_list[i] if rA_list else rA
        # 경계 시험: 0.5 배가 4자리 정수로 안 떨어지는 홀수 eA1 이면 floor(더 음수) = 표현 가능한 몫 ≥ 0.5 의 최솟값
        eA = int(math.floor(r * eA1)) if rA_list else int(round(r * eA1))
        EV[(b, "none", "base")] = round(nb, 4)
        EV[(b, "learn", "base")] = round(nb + eA / 1e4, 4)
        EV[(b, "none", "bad")] = round(none_bad, 4)
        EV[(b, "learn", "bad")] = round(none_bad + (eB_list[i] if eB_list else eB), 4)
        TR[b] = {"pre": nb, "post": -0.1, "rew": 1900}
        s = {"n": 3000, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0, "rew_blk": [60] * 30}
        if Skw:
            s.update(Skw(b))
        S[b] = s
        RC[b] = (not (rc_bad == ("b", b)), not (rc_bad == ("refl", b)))
    return TR, EV, S, RC


cases = [
    ("유지", build(rA=0.8), "유지(H071)"),
    ("유지 경계 rA=0.5", build(rA_list=[0.5] * 5), "유지(H071)"),
    ("간섭", build(rA=0.2), "간섭(H071-int)"),
    ("간섭(뒤집힘)", build(rA=-0.3), "간섭(H071-int)"),
    ("보류(섞임)", build(rA_list=[0.8, 0.8, 0.8, 0.2, 0.2]), "보류"),
    ("과제 B 미학습", build(rA=0.9, eB_list=[0.10, 0.10, 0.10, 0.04, 0.02]), "보류(과제 B 미학습"),
    ("eB 경계 +0.05", build(rA=0.9, eB_list=[0.05] * 5), "유지(H071)"),
    ("M1 과제 B 줄 없음", build(rc_bad=("b", 10)), "보류(조작검증 실패)"),
    ("M2 규칙 불일치", build(Skw=lambda b: {"agree": 0.999} if b == 11 else {}), "보류(조작검증 실패)"),
    ("M3 동결 실패", build(Skw=lambda b: {"res": 0.1} if b == 12 else {}), "보류(조작검증 실패)"),
    ("M4 반사 변함", build(rc_bad=("refl", 13)), "보류(조작검증 실패)"),
]
ok_all = True
for name, (TR, EV, S, RC), want in cases:
    c, r = J.judge(TR, EV, S, RC)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-18s 기대 %-22s → %s %s" % (name, want, r["verdict"][:22] if r else c[0][:30], "✓" if good else "✗"))
TR, EV, S, RC = build(rA_list=[0.5] * 5)
for b in J.BRAINS:   # 한 단위 덜 음수(몫 0.5 바로 아래)면 유지 아님
    EV[(b, "learn", "base")] = round(EV[(b, "learn", "base")] + 0.0001, 4)
c, r = J.judge(TR, EV, S, RC)
good = r is not None and r["keep"] <= 1; ok_all &= good
print("%-18s 기대 유지 ≤ 1/5 → keep %s %s" % ("경계 한 단위 밖", r["keep"] if r else None, "✓" if good else "✗"))
TR, EV, S, RC = build(); EV[(12, "none", "base")] = round(EV[(12, "none", "base")] + 0.003, 4)
c, r = J.judge(TR, EV, S, RC)
good = r is not None and r["verdict"].startswith("보류(조작검증 실패)"); ok_all &= good
print("%-18s 기대 보류(조작검증 실패) → %s" % ("M6 무학습 재현 실패", "✓" if good else "✗"))
TR, EV, S, RC = build(); del EV[(14, "learn", "bad")]
c, r = J.judge(TR, EV, S, RC)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-18s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
rng = np.random.default_rng(1)
R = np.zeros((3000, 27)); R[:, 2] = rng.integers(0, 2, 3000); R[:, 6] = rng.integers(0, 2, 3000)
R[:, 7] = np.where(np.arange(3000) < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]).astype(float)
R[:, 13:17] = 1.0; R[:, 21:25] = J.R20
s = J.stats(R)
good = s["agree"] == 1.0 and len(s["rew_blk"]) == 30; ok_all &= good
print("%-18s 기대 일치 1.0 → %s %s" % ("stats", s["agree"], "✓" if good else "✗"))
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E148"))
    open(os.path.join(td, "E148.log"), "w", encoding="utf-8").write(
        "  e148 train b10: => 사전 +0.0195 사후 -0.1000 보상 1900 || x\n  e148 b10 learn bad: => mod +0.1000\n  e148 b10 none base: => mod +0.0195\n")
    open(os.path.join(td, "logs", "E148", "train_b10.log"), "w", encoding="utf-8").write(
        "[과제 B] 시행 1500 부터 자극 = bad food, 정답 = 같은 쪽\n[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")
    J.EXP = td
    TRp, EVp, Sp, RCp = J.load()
good = (TRp == {10: {"pre": 0.0195, "post": -0.1, "rew": 1900}} and EVp == {(10, "learn", "bad"): 0.1, (10, "none", "base"): 0.0195}
        and RCp[10] == (True, True) and RCp[11] is None); ok_all &= good
print("%-18s 기대 학습·평가 줄·과제 B 줄 → %s" % ("줄 파싱", "✓" if good else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
