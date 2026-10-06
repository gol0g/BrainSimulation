#!/usr/bin/env python3
"""judge_e151.py 합성 시험(조건 1): 기전(T)·능력(rA) 각 판정, 경계(T 0.8·1.0, rA 0.5), 전제 P1'·P2', 조작검증 4종(동결·eta 줄·E150 재현·Σ|Δg|),
결측, η* 없음, stats·줄 파싱.
실행: python3 scripts/test_judge_e151.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e151 as J

NB, NBAD = 0.0300, 0.0250


def build(eA1=-0.30, d=None, T=0.5, eB=0.25, eA1_list=None, T_list=None, eB_list=None, d_list=None,
          s_bad=None, rc_bad=None, ref_bad=None, s150=1.0):
    """d = eA − eA1 (1e-4 정수로 직접 줄 수도 있음). 기본은 d = T·eB."""
    TR, EV, S, RC, REF, S150 = {}, {}, {}, {}, {}, {}
    for i, b in enumerate(J.BRAINS):
        a1 = int(round((eA1_list[i] if eA1_list else eA1) * 1e4))
        eb = int(round((eB_list[i] if eB_list else eB) * 1e4))
        if d_list is not None:
            dd = d_list[i]
        elif d is not None:
            dd = d
        else:
            dd = int(round((T_list[i] if T_list else T) * eb))
        ea = a1 + dd
        EV[(b, "none", "base")] = NB; EV[(b, "none", "bad")] = NBAD
        EV[(b, "A", "base")] = round(NB + a1 / 1e4, 4)
        EV[(b, "AB", "base")] = round(NB + ea / 1e4, 4)
        EV[(b, "AB", "bad")] = round(NBAD + eb / 1e4, 4)
        REF[(b, "base")] = NB; REF[(b, "bad")] = NBAD
        if ref_bad == b:
            REF[(b, "bad")] = NBAD + 0.0001
        S150[b] = s150
        for a in ("A", "AB"):
            TR[(a, b)] = {"pre": NB, "post": -0.2, "rew": 900}
            s = {"n": 1500 if a == "A" else 3000, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0, "sabs": 3.0}
            if s_bad and s_bad[0] == (a, b):
                s.update(s_bad[1])
            S[(a, b)] = s
            RC[(a, b)] = (not (rc_bad == ("b", a, b)), not (rc_bad == ("refl", a, b)), not (rc_bad == ("eta", a, b)))
    return TR, EV, S, RC, REF, S150


def run(name, args, eta, want1, want2):
    c, r = J.judge(*args, eta)
    got1 = r["v1"] if r else c[0]
    got2 = r["v2"] if r else c[0]
    good = got1.startswith(want1) and got2.startswith(want2) and (want1 != "보류" or got1 == "보류") and (want2 != "보류" or got2 == "보류")
    print("%-28s 기대 %-22s / %-14s → %s / %s %s" % (name, want1[:22], want2[:14], got1[:22], got2[:14], "✓" if good else "✗"))
    return good


ok = True
# T 0.5, eA1 −0.30, eB 0.25 → d 0.125, eA −0.175, rA 0.58
ok &= run("전이 감소 + 유지", build(), 0.9, "H074(전이 감소)", "전체 강도 유지")
# T 0.7, eA1 −0.25, eB 0.25 → d 0.175, eA −0.075, rA 0.30
ok &= run("전이 감소 + 간섭", build(eA1=-0.25, T=0.7), 0.9, "H074(전이 감소)", "전체 강도 간섭")
# T 1.2, eA1 −0.30, eB 0.30 → d 0.36, rA −0.2
ok &= run("기본 수준 전이 + 간섭", build(T=1.2, eB=0.30), 0.9, "H074-mag", "전체 강도 간섭")
ok &= run("경계 T=0.8(정수 정확)", build(eB=0.25, d=2000), 0.9, "H074(전이 감소)", "전체 강도 간섭")
ok &= run("경계 T=1.0(정수 정확)", build(eB=0.25, d=2500), 0.9, "H074-mag", "전체 강도 간섭")
ok &= run("T 0.8 바로 위 → 보류", build(eB=0.25, d=2001), 0.9, "보류", "전체 강도 간섭")
ok &= run("경계 rA=0.5·T=1.0(eA −0.15)", build(eB=0.15, d=1500), 0.9, "H074-mag", "전체 강도 유지")
ok &= run("rA 0.5 바로 아래", build(eB=0.15, d=1501), 0.9, "H074-mag", "전체 강도 간섭")
ok &= run("섞임 → 둘 다 보류", build(T_list=[0.5, 0.5, 0.5, 1.2, 1.2]), 0.9, "보류", "보류")
ok &= run("P1' 실패", build(eA1_list=[-0.30, -0.30, -0.30, -0.19, -0.10]), 0.9, "보류(학습 크기 미회복 — 과제 A)", "보류(학습 크기 미회복 — 과제 A)")
ok &= run("P2' 실패", build(eB_list=[0.25, 0.25, 0.25, 0.14, 0.05]), 0.9, "보류(학습 크기 미회복 — 과제 B)", "보류(학습 크기 미회복 — 과제 B)")
ok &= run("eB<0.15 뇌는 T 집계 제외", build(eB_list=[0.25, 0.25, 0.25, 0.25, 0.10], T=0.5), 0.9, "H074(전이 감소)", "전체 강도 유지")
ok &= run("조작: 동결 실패", build(s_bad=(("AB", 12), {"res": 0.1})), 0.9, "보류(조작검증 실패)", "보류(조작검증 실패)")
ok &= run("조작: eta 줄 불일치", build(rc_bad=("eta", "A", 11)), 0.9, "보류(조작검증 실패)", "보류(조작검증 실패)")
ok &= run("조작: 무학습 ≠ E150", build(ref_bad=13), 0.9, "보류(조작검증 실패)", "보류(조작검증 실패)")
ok &= run("조작: Σ|Δg| 안 커짐", build(s150=3.0), 0.9, "보류(조작검증 실패)", "보류(조작검증 실패)")
ok &= run("조작: 과제 B 줄 없음", build(rc_bad=("b", "AB", 10)), 0.9, "보류(조작검증 실패)", "보류(조작검증 실패)")
# 결측·η* 없음·파일 없음
args = list(build()); del args[1][(14, "AB", "bad")]
c, r = J.judge(*args, 0.9); g = r is None and "결측" in c[0]; ok &= g; print("%-28s → %s" % ("결측", "✓" if g else "✗"))
c, r = J.judge(*build(), None); g = r is None and "학습 크기 미회복" in c[0]; ok &= g; print("%-28s → %s" % ("η* 없음", "✓" if g else "✗"))
c, r = J.judge(*build(), "missing"); g = r is None and "eta_star.txt 없음" in c[0]; ok &= g; print("%-28s → %s" % ("eta_star 파일 없음", "✓" if g else "✗"))
# stats
R = np.zeros((3000, 27)); rng = np.random.default_rng(5); R[:, 2] = rng.integers(0, 2, 3000); R[:, 6] = rng.integers(0, 2, 3000)
R[:, 7] = np.where(np.arange(3000) < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]).astype(float); R[:, 13:17] = 1.0; R[:, 21:25] = J.R20
R[:, 12] = np.where(np.arange(3000) % 2 == 0, 2.0, -1.0)
sAB = J.stats(R, 1500); sA = J.stats(R[:1500], None)
g = sAB["agree"] == 1.0 and sA["agree"] == 1.0 and sAB["n"] == 3000 and abs(sA["sabs"] - 2250.0) < 1e-9; ok &= g
print("%-28s → %s" % ("stats(일치·Σ|Δg|)", "✓" if g else "✗ %s" % sA))
# 줄 파싱·eta 줄·E150 참조
with tempfile.TemporaryDirectory() as td:
    for d_ in ("logs/E151", "traces/E150", "traces/E151"):
        os.makedirs(os.path.join(td, d_))
    open(os.path.join(td, "logs", "E151", "eta_star.txt"), "w", encoding="utf-8").write("eta_star 0.9\n")
    open(os.path.join(td, "E151.log"), "w", encoding="utf-8").write(
        "  e151 train A b10: => 사전 +0.0300 사후 -0.3000 보상 900 || x\n  e151 b10 AB bad: => mod +0.3000\n")
    open(os.path.join(td, "E150.log"), "w", encoding="utf-8").write("  e150 b10 none base: => mod +0.0321\n  e150 b10 none bad: => mod +0.0271\n  e150 b10 A base: => mod -0.0600\n")
    refl = "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n"
    eta_ok = "    KC→motor [E109 R-STDP 4방향]: init_w=150.0, w_max=300.0, eta=0.9, tau_e=12.0, sparsity=0.25\n"
    eta_bad = "    KC→motor [E109 R-STDP 4방향]: init_w=150.0, w_max=300.0, eta=0.15, tau_e=12.0, sparsity=0.25\n"
    open(os.path.join(td, "logs", "E151", "train_A_b10.log"), "w", encoding="utf-8").write(eta_ok + refl + eta_ok)
    open(os.path.join(td, "logs", "E151", "train_AB_b10.log"), "w", encoding="utf-8").write("[과제 B] 시행 1500 부터 자극 = bad food, 정답 = 같은 쪽\n" + eta_ok + refl + eta_bad)
    np.savez_compressed(os.path.join(td, "traces", "E150", "tr_A_b10.npz"), rows=R[:1500])
    J.EXP = td
    TRp, EVp, Sp, RCp, REFp, S150p, etap = J.load()
g = (etap == 0.9 and TRp[("A", 10)]["post"] == -0.3 and EVp == {(10, "AB", "bad"): 0.3} and REFp == {(10, "base"): 0.0321, (10, "bad"): 0.0271}
     and RCp[("A", 10)] == (True, True, True) and RCp[("AB", 10)] == (True, True, False) and RCp[("A", 11)] is None and abs(S150p[10] - 2250.0) < 1e-9)
ok &= g
print("%-28s → %s" % ("줄 파싱(eta 줄 하나라도 다르면 실패)", "✓" if g else "✗ %s %s %s" % (etap, RCp, REFp)))
print("전체: %s" % ("통과" if ok else "실패"))
sys.exit(0 if ok else 1)
