#!/usr/bin/env python3
"""judge_e173.py 합성 시험(조건 1): 획득·요소식·부분(켬만·끔만)·보류, 경계(e ±0.10 정확, 차 0.10 정확), 조작검증 실패 9종, 결측, bicond_stats(맥락별 규칙), 줄·원 로그 파싱.
실행: python3 scripts/test_judge_e173.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e173 as J

KL = "side=l off=60 on=80 ctx=3 keep=58 lost=2 conj=19 jac=0.7073 | side=r off=61 on=82 ctx=4 keep=60 lost=1 conj=18 jac=0.7229 | ctx_n=200 w=3.00 p=0.100 level=0.90 연결 l=60000 r=60000 | 맥락 발화 끔 0 켬 2400 | n_pres=40"


def build(eoff=-0.30, eon=0.30, per=None, bad=None):
    K, TRN, EV, S, RAW, EL = {}, {}, {}, {}, {}, {}
    for b in J.BRAINS:
        o, n_ = (per or {}).get(b, (eoff, eon))
        K[b] = KL
        TRN[b] = {"pre": 0.01, "post": -0.2, "rew": 2000, "nctx": 1500, "load": 2}
        S[b] = {"n": 3000, "has_ctx": True, "frac_ctx": 0.5, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0}
        RAW[b] = (1500, 3000, True, 2)
        base_off, base_on = 0.0150, 0.0050
        EV[(b, "none", "off")] = base_off; EV[(b, "none", "on")] = base_on
        EV[(b, "learn", "off")] = round(base_off + o, 4); EV[(b, "learn", "on")] = round(base_on + n_, 4)
        for (w, c) in J.EVS:
            EL[(b, w, c)] = (1, c == "on")
    if bad:
        kind, b = bad
        if kind == "kc_off":
            K[b] = KL.replace("맥락 발화 끔 0", "맥락 발화 끔 7")
        elif kind == "kc_on0":
            K[b] = KL.replace("켬 2400", "켬 0")
        elif kind == "nctx_range":
            RAW[b] = (1700, 3000, True, 2); TRN[b]["nctx"] = 1700
        elif kind == "nctx_mismatch":
            TRN[b]["nctx"] = 1499
        elif kind == "agree":
            S[b]["agree"] = 0.999
        elif kind == "no_col":
            S[b]["has_ctx"] = False
        elif kind == "res":
            S[b]["res"] = 0.01
        elif kind == "refl":
            RAW[b] = (1500, 3000, False, 2)
        elif kind == "load":
            RAW[b] = (1500, 3000, True, 1)
        elif kind == "ev_noctx":
            EL[(b, "learn", "on")] = (1, False)
        elif kind == "ev_ctx_in_off":
            EL[(b, "none", "off")] = (1, True)
    return K, TRN, EV, S, RAW, EL


ok_all = True


def chk(name, data, want):
    global ok_all
    c, r = J.judge(*data)
    got = r["verdict"] if r else c[0]
    good = (got.startswith(want) and (want != "보류" or got == "보류")) if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-34s 기대 %-22s → %-30s %s" % (name, want[:22], got[:30], "✓" if good else "✗ %s" % c))


chk("획득(e_off −0.30·e_on +0.30)", build(), "획득(H097)")
chk("경계 e_off −0.10·e_on +0.10 정확 → 획득", build(eoff=-0.10, eon=0.10), "획득(H097)")
chk("e_on +0.0999 → 획득 아님 → 부분(끔만)", build(eoff=-0.30, eon=0.0999), "부분(H097-partial)")
chk("요소식(둘 다 −0.20)", build(eoff=-0.20, eon=-0.20), "요소식(H097-null)")
chk("요소식 경계 차 0.0999", build(eoff=-0.05, eon=0.0499), "요소식(H097-null)")
chk("차 0.10 정확 → 요소식 아님 → 보류", build(eoff=-0.05, eon=0.05), "보류")
chk("부분 켬만(e_on +0.30·e_off −0.05)", build(eoff=-0.05, eon=0.30), "부분(H097-partial) — 한 맥락만(켬만 5")
chk("부분 끔만(e_off −0.30·e_on 0.00)", build(eoff=-0.30, eon=0.0), "부분(H097-partial) — 한 맥락만(켬만 0")
chk("획득 4/5", build(per={14: (-0.05, 0.30)}), "획득(H097)")
chk("획득 3/5·요소식 2 → 부분 2 + 보류", build(per={13: (-0.2, -0.2), 14: (-0.2, -0.2)}), "보류")
for kind, b in (("kc_off", 10), ("kc_on0", 11), ("nctx_range", 12), ("nctx_mismatch", 13), ("agree", 14), ("no_col", 10),
                ("res", 11), ("refl", 12), ("load", 13), ("ev_noctx", 14), ("ev_ctx_in_off", 10)):
    chk("조작: %s" % kind, build(bad=(kind, b)), "보류(조작검증 실패)")
d = build(); del d[2][(12, "none", "on")]
chk("결측(평가 하나)", d, "결측")
d = build(); del d[3][11]
chk("결측(추적)", d, "결측")
# bicond_stats: 맥락별 규칙
n = 3000
R = np.zeros((n, 38)); R[:, 2] = np.arange(n) % 2
ctx = (np.arange(n) // 2) % 2
R[:, 37] = ctx
R[:, 6] = np.where(ctx == 1, R[:, 2], 1 - R[:, 2]); R[:, 7] = 1
R[:, 13] = 1.0; R[:, 21] = J.R20
st = J.bicond_stats(R)
g = st["n"] == 3000 and st["has_ctx"] and st["agree"] == 1.0 and st["res"] < 1e-12 and abs(st["frac_ctx"] - 0.5) < 1e-12
R2 = R.copy(); R2[5, 7] = 0
g2 = J.bicond_stats(R2)["agree"] < 1.0
R3 = R.copy(); R3[:, 6] = 1 - R3[:, 2]       # 맥락 무시(늘 교차) 인데 보상 1 → 켬 시행 불일치
g3 = abs(J.bicond_stats(R3)["agree"] - 0.5) < 1e-12
g4 = not J.bicond_stats(np.zeros((10, 37)))["has_ctx"]
ok_all &= g and g2 and g3 and g4
print("%-34s → %s" % ("bicond_stats(맥락별 규칙·불일치·열 없음)", "✓" if (g and g2 and g3 and g4) else "✗ %s" % st))
# 파싱(load)
with tempfile.TemporaryDirectory() as td:
    for d_ in (("logs", "E173"), ("traces", "E173")):
        os.makedirs(os.path.join(td, *d_))
    open(os.path.join(td, "E173.log"), "w", encoding="utf-8").write(
        "  e173 kcctx b10: => KCCTX %s\n  e173 train b10: => 사전 +0.0080 사후 -0.1000 보상 2000 || 맥락 켬 1490 || 적재 2 || x\n"
        "  e173 b10 learn on: => mod +0.3000\n" % KL)
    open(os.path.join(td, "logs", "E173", "train_b10.log"), "w", encoding="utf-8").write(
        "[E153 종류 입력 적재] k 검증 일치 — x\n[맥락 과제] bicond frac=0.50\n[맥락 과제] 시행 3000 중 맥락 켬 1490\n"
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n[E153 종류 입력 적재] k 검증 일치 — y\n")
    open(os.path.join(td, "logs", "E173", "ev_b10_learn_on.log"), "w", encoding="utf-8").write(
        "[E153 종류 입력 적재] k 검증 일치 — x\n=> DECOMP mode=all mod=+0.3000\n[맥락 평가] ctx=켬 level=0.90\n")
    np.savez_compressed(os.path.join(td, "traces", "E173", "tr_bc_b10.npz"), rows=R)
    J.EXP = td
    K, TRN, EV, S, RAW, EL = J.load()
g = (K[10].startswith("side=l off=60") and TRN[10] == {"pre": 0.008, "post": -0.1, "rew": 2000, "nctx": 1490, "load": 2}
     and EV == {(10, "learn", "on"): 0.3} and RAW[10] == (1490, 3000, True, 2) and EL[(10, "learn", "on")] == (1, True)
     and EL[(10, "none", "off")] is None and S[10]["agree"] == 1.0 and RAW[11] is None)
ok_all &= g
print("%-34s → %s" % ("줄·원 로그·추적 파싱", "✓" if g else "✗ %s %s %s %s" % (TRN, RAW, EL, K)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
