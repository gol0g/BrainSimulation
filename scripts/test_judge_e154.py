#!/usr/bin/env python3
"""judge_e154.py 합성 시험(조건 1): 유지·간섭·보류, 전제 P1·P2(새 문턱 −0.20·+0.15), 경계(rA 0.5 — floor), 조작검증(동결·규칙·과제 B 줄·반사·학습 적재·평가 적재), 결측, 줄 파싱.
실행: python3 scripts/test_judge_e154.py (저장소 루트에서)"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e154 as J


def build(eA1=-0.30, rA=0.8, eB=0.25, eA1_list=None, rA_list=None, eB_list=None, s_bad=None, rc_bad=None, el_bad=None, floor=False):
    TR, EV, S, RC, EL = {}, {}, {}, {}, {}
    for i, b in enumerate(J.BRAINS):
        nb, nbad = 0.0150, 0.0200
        a1 = int(round((eA1_list[i] if eA1_list else eA1) * 1e4))
        r = rA_list[i] if rA_list else rA
        ea = int(math.floor(r * a1)) if floor else int(round(r * a1))
        EV[(b, "none", "base")] = nb; EV[(b, "none", "bad")] = nbad
        EV[(b, "A", "base")] = round(nb + a1 / 1e4, 4)
        EV[(b, "AB", "base")] = round(nb + ea / 1e4, 4)
        EV[(b, "AB", "bad")] = round(nbad + (eB_list[i] if eB_list else eB), 4)
        for (w, s) in J.NEEDED:
            EL[(b, w, s)] = 0 if el_bad == (b, w, s) else 1
        for a in ("A", "AB"):
            TR[(a, b)] = {"pre": nb, "post": -0.1, "rew": 900}
            st = {"n": 1500 if a == "A" else 3000, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0}
            if s_bad and s_bad[0] == (a, b):
                st.update(s_bad[1])
            S[(a, b)] = st
            RC[(a, b)] = (not (rc_bad == ("b", a, b)), not (rc_bad == ("refl", a, b)), 1 if rc_bad == ("load", a, b) else 2)
    return TR, EV, S, RC, EL


cases = [
    ("유지", build(rA=0.8), "유지(H077)"),
    ("유지 경계 rA=0.5(floor)", build(rA_list=[0.5] * 5, floor=True), "유지(H077)"),
    ("간섭", build(rA=0.2), "간섭(H077-null)"),
    ("간섭(뒤집힘)", build(rA=-0.3), "간섭(H077-null)"),
    ("보류(섞임)", build(rA_list=[0.8, 0.8, 0.8, 0.2, 0.2]), "보류"),
    ("P1 −0.19 두 뇌", build(eA1_list=[-0.30, -0.30, -0.30, -0.19, -0.19]), "보류(형성 표현 과제 A 전체 강도 미학습)"),
    ("P1 경계 −0.20 정확", build(eA1_list=[-0.20] * 5), "유지(H077)"),
    ("P2 +0.14 두 뇌", build(eB_list=[0.25, 0.25, 0.25, 0.14, 0.14]), "보류(과제 B 미학습)"),
    ("P2 경계 +0.15 정확", build(eB_list=[0.15] * 5), "유지(H077)"),
    ("조작: 동결 실패", build(s_bad=(("AB", 12), {"res": 0.1})), "보류(조작검증 실패)"),
    ("조작: 규칙 불일치", build(s_bad=(("A", 11), {"agree": 0.99})), "보류(조작검증 실패)"),
    ("조작: 과제 B 줄 없음", build(rc_bad=("b", "AB", 10)), "보류(조작검증 실패)"),
    ("조작: 반사 변함", build(rc_bad=("refl", "A", 13)), "보류(조작검증 실패)"),
    ("조작: 학습 적재 1줄", build(rc_bad=("load", "AB", 14)), "보류(조작검증 실패)"),
    ("조작: 평가 적재 0", build(el_bad=(12, "none", "bad")), "보류(조작검증 실패)"),
]
ok_all = True
for name, data, want in cases:
    c, r = J.judge(*data)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-24s 기대 %-30s → %s %s" % (name, want[:30], r["verdict"][:30] if r else c[0][:30], "✓" if good else "✗"))
TR, EV, S, RC, EL = build(); del EV[(14, "AB", "bad")]
c, r = J.judge(TR, EV, S, RC, EL)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-24s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
TR, EV, S, RC, EL = build(rA=0.6, eB=0.20)
c, r = J.judge(TR, EV, S, RC, EL)
good = abs(r["T"][10] - (0.12 / 0.20)) < 1e-9; ok_all &= good    # eA1 −0.30, eA −0.18 → d 0.12, T 0.6
print("%-24s 기대 T 0.60 → %.4f %s" % ("T 계산", r["T"][10], "✓" if good else "✗"))
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E154"))
    open(os.path.join(td, "E154.log"), "w", encoding="utf-8").write(
        "  e154 train A b10: => 사전 +0.0150 사후 -0.3000 보상 1000 || x\n  e154 b10 AB bad: => mod +0.3000\n")
    ld = "[E153 종류 입력 적재] k.npz 검증 일치 — x\n"
    refl = "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n"
    open(os.path.join(td, "logs", "E154", "train_A_b10.log"), "w", encoding="utf-8").write(ld + refl + ld)
    open(os.path.join(td, "logs", "E154", "train_AB_b10.log"), "w", encoding="utf-8").write("[과제 B] 시행 1500 부터 자극 = bad food\n" + ld + refl)
    open(os.path.join(td, "logs", "E154", "ev_b10_AB_bad.log"), "w", encoding="utf-8").write(ld + "=> DECOMP mode=all mod=+0.3000\n")
    J.EXP = td
    TRp, EVp, Sp, RCp, ELp = J.load()
good = (TRp[("A", 10)]["post"] == -0.3 and EVp == {(10, "AB", "bad"): 0.3} and RCp[("A", 10)] == (True, True, 2)
        and RCp[("AB", 10)] == (True, True, 1) and RCp[("A", 11)] is None and ELp[(10, "AB", "bad")] == 1 and ELp[(10, "none", "base")] is None)
ok_all &= good
print("%-24s → %s" % ("줄 파싱(적재 수·반사·과제 B)", "✓" if good else "✗ %s %s %s" % (TRp, RCp, ELp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
