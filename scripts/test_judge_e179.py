#!/usr/bin/env python3
"""judge_e179.py 합성 시험(조건 1) — 대칭 맥락 A·B: 획득·경계·부분(B만 = 치우침 쪽, A만)·요소식·차 0.10 정확, 조작검증 실패 27종
(두 맥락 집단 발화 교차 4·맥락 입력 2 줄·맥락 B 수 2·추적 3·반사·적재·A·B 평가 표지 4·동작점 4·진단·A·B 평가 포화 2·A·B 결정 표현 3·kcctx 대칭 꼬리 없음),
경계 통과(포화 하나 0.6599, 결정 0.50), 결측, bicond_stats(B = 열 37 1), 줄·원 로그·추적 파싱.
실행: python3 scripts/test_judge_e179.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e179 as J  # noqa: E402

KB = ("side=l off=60 on=80 ctx=3 keep=50 lost=10 conj=19 jac=0.5000 | side=r off=61 on=82 ctx=4 keep=50 lost=11 conj=18 jac=0.5100 | ctx_i=0.00 | KC 발화 합 good 끔 700 켬 700 · 맥락 단독 400 · 기준선 300 | n_pres=40"
      " | ctx_inh_i=0.00 억제 부분집합 발화 끔 0 켬 0 | ctx_ab_i=0.00 연합 결합 발화 끔 0 켬 0 | ctx_n=200 w=4.00 p=0.100 level=0.90 맥락 집단 발화 끔 0 켬 2400")
KL = KB + " | 대칭(E179 — 끔 자리 = 맥락 A) ctx2_n=200 w2=4.00 p2=0.100 맥락2 집단 발화 끔 2400 켬 0"
X = -4000.0
R_A = (0.52, 0.48, 0.47, 0.51)
R_B = (0.50, 0.53, 0.54, 0.49)


def build(eA=-0.30, eB=0.30, per=None, bad=None):
    K, TRN, EV, S, RAW, EL = {}, {}, {}, {}, {}, {}
    for b in J.BRAINS:
        a_, b_ = (per or {}).get(b, (eA, eB))
        K[b] = KL
        TRN[b] = {"pre": 0.01, "post": -0.2, "rew": 2000, "nctx": 1500, "load": 2}
        S[b] = {"n": 3000, "has_ctx": True, "frac_ctx": 0.5, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0, "dec_on": 0.10, "dec_off": 0.12}
        RAW[b] = (1500, 3000, True, 2, [X, X], 1)
        bA, bB = 0.0150, 0.0050
        EV[(b, "none", "A")] = bA; EV[(b, "none", "B")] = bB
        EV[(b, "learn", "A")] = round(bA + a_, 4); EV[(b, "learn", "B")] = round(bB + b_, 4)
        for (w, c) in J.EVS:
            EL[(b, w, c)] = (1, c == "B", [X, X], R_B if c == "B" else R_A, c == "A")
    if bad:
        kind, b = bad
        rep = {"kc_B_off": ("맥락 집단 발화 끔 0 켬 2400", "맥락 집단 발화 끔 9 켬 2400"),
               "kc_B_on0": ("맥락 집단 발화 끔 0 켬 2400", "맥락 집단 발화 끔 0 켬 0"),
               "kc_A_off0": ("맥락2 집단 발화 끔 2400 켬 0", "맥락2 집단 발화 끔 0 켬 0"),
               "kc_A_on": ("맥락2 집단 발화 끔 2400 켬 0", "맥락2 집단 발화 끔 2400 켬 5")}
        if kind in rep:
            K[b] = K[b].replace(*rep[kind])
        elif kind == "kc_nosym":
            K[b] = KB
        elif kind == "ctx2_line":
            RAW[b] = (1500, 3000, True, 2, [X, X], 0)
        elif kind == "nctx_range":
            RAW[b] = (1700, 3000, True, 2, [X, X], 1); TRN[b]["nctx"] = 1700
        elif kind == "nctx_mismatch":
            TRN[b]["nctx"] = 1499
        elif kind == "agree":
            S[b]["agree"] = 0.999
        elif kind == "no_col":
            S[b]["has_ctx"] = False
        elif kind == "res":
            S[b]["res"] = 0.01
        elif kind == "refl":
            RAW[b] = (1500, 3000, False, 2, [X, X], 1)
        elif kind == "load":
            RAW[b] = (1500, 3000, True, 1, [X, X], 1)
        elif kind == "ev_B_noB":
            EL[(b, "learn", "B")] = (1, False, [X, X], R_B, False)
        elif kind == "ev_B_hasA":
            EL[(b, "none", "B")] = (1, True, [X, X], R_B, True)
        elif kind == "ev_A_noA":
            EL[(b, "learn", "A")] = (1, False, [X, X], R_A, False)
        elif kind == "ev_A_hasB":
            EL[(b, "none", "A")] = (1, True, [X, X], R_A, True)
        elif kind == "tonic_train_none":
            RAW[b] = (1500, 3000, True, 2, [], 1)
        elif kind == "tonic_train_val":
            RAW[b] = (1500, 3000, True, 2, [X, -1000.0], 1)
        elif kind == "tonic_eval_none":
            EL[(b, "none", "A")] = (1, False, [], R_A, True)
        elif kind == "tonic_eval_val":
            EL[(b, "learn", "B")] = (1, True, [-2000.0], R_B, False)
        elif kind == "diag_none":
            EL[(b, "learn", "A")] = (1, False, [X, X], None, True)
        elif kind == "sat_learn_A":
            EL[(b, "learn", "A")] = (1, False, [X, X], (0.6664, 0.6662, 0.6663, 0.6600), True)
        elif kind == "sat_none_B":
            EL[(b, "none", "B")] = (1, True, [X, X], (0.6664, 0.6662, 0.6663, 0.6661), False)
        elif kind == "dec_B_051":
            S[b]["dec_on"] = 0.51
        elif kind == "dec_A_051":
            S[b]["dec_off"] = 0.51
        elif kind == "dec_nan":
            S[b]["dec_off"] = float("nan")
    return K, TRN, EV, S, RAW, EL


ok_all = True


def chk(name, data, want):
    global ok_all
    c, r = J.judge(*data)
    got = r["verdict"] if r else c[0]
    good = (got.startswith(want) and (want != "보류" or got == "보류")) if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-34s 기대 %-24s → %-34s %s" % (name, want[:24], got[:34], "✓" if good else "✗ %s" % c))


chk("획득(e_A −0.30·e_B +0.30)", build(), "획득(H100)")
chk("경계 e_A −0.10·e_B +0.10 정확 → 획득", build(eA=-0.10, eB=0.10), "획득(H100)")
chk("e_B +0.0999 → 부분 A만", build(eA=-0.30, eB=0.0999), "부분 — 한 맥락만(B(같은 쪽)만 0·A(교차)만 5)")
chk("부분 B만(e_A −0.05·e_B +0.30) → 치우침 쪽", build(eA=-0.05, eB=0.30), "부분 — 한 맥락만(B(같은 쪽)만 5·A(교차)만 0) — H100-bias 쪽")
chk("요소식(e_A +0.20·e_B +0.25)", build(eA=0.20, eB=0.25), "요소식(H100-null)")
chk("요소식 경계 차 0.0999", build(eA=-0.05, eB=0.0499), "요소식(H100-null)")
chk("차 0.10 정확 → 요소식 아님 → 보류", build(eA=-0.05, eB=0.05), "보류")
chk("획득 4/5", build(per={14: (-0.05, 0.30)}), "획득(H100)")
chk("획득 3/5·요소식 2 → 보류", build(per={13: (0.2, 0.2), 14: (0.2, 0.2)}), "보류")
for kind, b in (("kc_B_off", 10), ("kc_B_on0", 11), ("kc_A_off0", 12), ("kc_A_on", 13), ("kc_nosym", 14), ("ctx2_line", 10),
                ("nctx_range", 11), ("nctx_mismatch", 12), ("agree", 13), ("no_col", 14), ("res", 10), ("refl", 11), ("load", 12),
                ("ev_B_noB", 13), ("ev_B_hasA", 14), ("ev_A_noA", 10), ("ev_A_hasB", 11),
                ("tonic_train_none", 12), ("tonic_train_val", 13), ("tonic_eval_none", 14), ("tonic_eval_val", 10), ("diag_none", 11),
                ("sat_learn_A", 12), ("sat_none_B", 13), ("dec_B_051", 14), ("dec_A_051", 10), ("dec_nan", 11)):
    chk("조작: %s" % kind, build(bad=(kind, b)), "보류(조작검증 실패)")
d = build(); d[5][(12, "learn", "B")] = (1, True, [X, X], (0.6664, 0.6662, 0.6663, 0.6599), False)
chk("평가 하나 0.6599 → 비포화 통과 → 획득", d, "획득(H100)")
d = build(); d[3][13]["dec_off"] = 0.50
chk("A 결정 0.50 정확 → 통과 → 획득", d, "획득(H100)")
d = build(); del d[2][(12, "none", "B")]
chk("결측(평가 하나)", d, "결측")
d = build(); del d[3][11]
chk("결측(추적)", d, "결측")
# bicond_stats: 열 37 = 1 → B(같은 쪽), 0 → A(교차); dec_on = B, dec_off = A
n = 3000
R = np.zeros((n, 38)); R[:, 2] = np.arange(n) % 2
ctx = (np.arange(n) // 2) % 2
R[:, 37] = ctx
R[:, 6] = np.where(ctx == 1, R[:, 2], 1 - R[:, 2]); R[:, 7] = 1
R[:, 13] = 1.0; R[:, 21] = J.R20
R[:, 3] = 0.0
R[:, 5] = np.where(ctx == 1, 0.2, np.where(np.arange(n) % 8 == 0, 0.01, 0.2))   # A 시행(i mod 4 ∈ {0,1}) 중 i mod 8 = 0 인 1/4 이 작은 v
st = J.bicond_stats(R)
g = st["agree"] == 1.0 and st["dec_on"] == 0.0 and abs(st["dec_off"] - 0.25) < 1e-12 and st["n"] == 3000
ok_all &= g
print("%-34s → %s" % ("bicond_stats(B = 열37 1, A 결정 1/4)", "✓" if g else "✗ %s" % st))
# 파싱
with tempfile.TemporaryDirectory() as td:
    for d_ in (("logs", "E179"), ("traces", "E179")):
        os.makedirs(os.path.join(td, *d_))
    open(os.path.join(td, "E179.log"), "w", encoding="utf-8").write(
        "  e179 kcctx b10: => KCCTX %s\n  e179 train b10: => 사전 +0.0080 사후 -0.1000 보상 2000 || 맥락 켬 1490 || 적재 2 || x\n"
        "  e179 b10 learn A: => mod -0.3000\n" % KL)
    open(os.path.join(td, "logs", "E179", "train_b10.log"), "w", encoding="utf-8").write(
        "  [motor 동작점] motor 좌·우 Ioffset 기저값 -4000.0(긴장성 억제)\n  [맥락 입력] 맥락 전용 집단 zz_ctx 200 → KC\n  [맥락 입력 2] 맥락 전용 집단 zz_ctx2 200 → KC 좌·우\n"
        "[E153 종류 입력 적재] k 검증 일치 — x\n[맥락 과제] bicond2 frac=0.50\n[맥락 과제] 시행 3000 중 맥락 켬 1490\n"
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n"
        "  [motor 동작점] motor 좌·우 Ioffset 기저값 -4000.0(긴장성 억제)\n[E153 종류 입력 적재] k 검증 일치 — y\n")
    open(os.path.join(td, "logs", "E179", "ev_b10_learn_A.log"), "w", encoding="utf-8").write(
        "  [motor 동작점] motor 좌·우 Ioffset 기저값 -4000.0(긴장성 억제)\n[E153 종류 입력 적재] k 검증 일치 — x\n[맥락 평가 A] ctx2=켬 ctx2_n=200\n"
        "[E171 평가 진단] variant=base side=left n=250 motor L/R 0.520000/0.480000 KC L/R 0.006926/0.003237\n"
        "[E171 평가 진단] variant=base side=right n=250 motor L/R 0.470000/0.510000 KC L/R 0.003227/0.007001\n=> DECOMP mode=all mod=-0.3000\n")
    np.savez_compressed(os.path.join(td, "traces", "E179", "tr_bc_b10.npz"), rows=R)
    K, TRN, EV, S, RAW, EL = J.load(td)
g = (K[10].startswith("side=l off=60") and TRN[10] == {"pre": 0.008, "post": -0.1, "rew": 2000, "nctx": 1490, "load": 2}
     and EV == {(10, "learn", "A"): -0.3} and RAW[10] == (1490, 3000, True, 2, [-4000.0, -4000.0], 1)
     and EL[(10, "learn", "A")] == (1, False, [-4000.0], (0.52, 0.48, 0.47, 0.51), True)
     and EL[(10, "none", "B")] is None and S[10]["agree"] == 1.0 and RAW[11] is None)
ok_all &= g
print("%-34s → %s" % ("줄·원 로그·추적 파싱", "✓" if g else "✗ %s %s %s %s" % (TRN, RAW, EL, K)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
