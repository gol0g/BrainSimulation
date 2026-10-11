#!/usr/bin/env python3
"""judge_e178.py 합성 시험(조건 1) — test_judge_e177.py 를 옮기고 새 조작검증을 더함:
획득·요소식·부분(켬만·끔만)·보류, 경계(e ±0.10 정확, 차 0.10 정확), E177 조작검증 실패 11종 + 동작점(줄 없음·값 다름, 학습·평가)·진단 줄 없음·
켬 평가 포화(4개 모두 ≥ 0.66 실패, 하나 0.6599 통과, 학습 켬·무학습 켬 각각)·켬 결정 표현(0.50 통과·0.51 실패·nan 실패), pick 결측, 평가·추적 결측,
bicond_stats(맥락별 규칙·탐욕 결정 비율), 줄·원 로그·추적 파싱(동작점·진단 줄).
실행: python3 scripts/test_judge_e178.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e178 as J  # noqa: E402

KL = "side=l off=60 on=80 ctx=3 keep=58 lost=2 conj=19 jac=0.7073 | side=r off=61 on=82 ctx=4 keep=60 lost=1 conj=18 jac=0.7229 | ctx_i=0.00 | KC 발화 합 good 끔 500 켬 700 · 맥락 단독 40 · 기준선 300 | n_pres=40 | ctx_inh_i=0.00 억제 부분집합 발화 끔 0 켬 0 | ctx_ab_i=0.00 연합 결합 발화 끔 0 켬 0 | ctx_n=200 w=4.00 p=0.100 level=0.90 맥락 집단 발화 끔 0 켬 2400"
X = -1000.0
OFF_R = (0.40, 0.48, 0.48, 0.39)
ON_R = (0.55, 0.58, 0.58, 0.54)


def build(eoff=-0.30, eon=0.30, per=None, bad=None):
    K, TRN, EV, S, RAW, EL = {}, {}, {}, {}, {}, {}
    for b in J.BRAINS:
        o, n_ = (per or {}).get(b, (eoff, eon))
        K[b] = KL
        TRN[b] = {"pre": 0.01, "post": -0.2, "rew": 2000, "nctx": 1500, "load": 2}
        S[b] = {"n": 3000, "has_ctx": True, "frac_ctx": 0.5, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0, "dec_on": 0.10, "dec_off": 0.05}
        RAW[b] = (1500, 3000, True, 2, [X, X])
        base_off, base_on = 0.0150, 0.0050
        EV[(b, "none", "off")] = base_off; EV[(b, "none", "on")] = base_on
        EV[(b, "learn", "off")] = round(base_off + o, 4); EV[(b, "learn", "on")] = round(base_on + n_, 4)
        for (w, c) in J.EVS:
            EL[(b, w, c)] = (1, c == "on", [X, X], ON_R if c == "on" else OFF_R)
    if bad:
        kind, b = bad
        if kind == "kc_off":
            K[b] = KL.replace("맥락 집단 발화 끔 0 켬 2400", "맥락 집단 발화 끔 7 켬 2400")
        elif kind == "kc_on0":
            K[b] = KL.replace("맥락 집단 발화 끔 0 켬 2400", "맥락 집단 발화 끔 0 켬 0")
        elif kind == "nctx_range":
            RAW[b] = (1700, 3000, True, 2, [X, X]); TRN[b]["nctx"] = 1700
        elif kind == "nctx_mismatch":
            TRN[b]["nctx"] = 1499
        elif kind == "agree":
            S[b]["agree"] = 0.999
        elif kind == "no_col":
            S[b]["has_ctx"] = False
        elif kind == "res":
            S[b]["res"] = 0.01
        elif kind == "refl":
            RAW[b] = (1500, 3000, False, 2, [X, X])
        elif kind == "load":
            RAW[b] = (1500, 3000, True, 1, [X, X])
        elif kind == "ev_noctx":
            EL[(b, "learn", "on")] = (1, False, [X, X], ON_R)
        elif kind == "ev_ctx_in_off":
            EL[(b, "none", "off")] = (1, True, [X, X], OFF_R)
        elif kind == "tonic_train_none":
            RAW[b] = (1500, 3000, True, 2, [])
        elif kind == "tonic_train_val":
            RAW[b] = (1500, 3000, True, 2, [X, -500.0])
        elif kind == "tonic_eval_none":
            EL[(b, "none", "off")] = (1, False, [], OFF_R)
        elif kind == "tonic_eval_val":
            EL[(b, "learn", "on")] = (1, True, [-2000.0], ON_R)
        elif kind == "diag_none":
            EL[(b, "learn", "off")] = (1, False, [X, X], None)
        elif kind == "sat_learn_on":
            EL[(b, "learn", "on")] = (1, True, [X, X], (0.6664, 0.6662, 0.6663, 0.6600))
        elif kind == "sat_none_on":
            EL[(b, "none", "on")] = (1, True, [X, X], (0.6664, 0.6662, 0.6663, 0.6661))
        elif kind == "dec_051":
            S[b]["dec_on"] = 0.51
        elif kind == "dec_nan":
            S[b]["dec_on"] = float("nan")
    return K, TRN, EV, S, RAW, EL


ok_all = True


def chk(name, data, want, x=X):
    global ok_all
    c, r = J.judge(*data, x)
    got = r["verdict"] if r else c[0]
    good = (got.startswith(want) and (want != "보류" or got == "보류")) if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-34s 기대 %-22s → %-30s %s" % (name, want[:22], got[:30], "✓" if good else "✗ %s" % c))


chk("획득(e_off −0.30·e_on +0.30)", build(), "획득(H099)")
chk("경계 e_off −0.10·e_on +0.10 정확 → 획득", build(eoff=-0.10, eon=0.10), "획득(H099)")
chk("e_on +0.0999 → 획득 아님 → 부분(끔만)", build(eoff=-0.30, eon=0.0999), "부분(H099-partial)")
chk("요소식(둘 다 −0.20)", build(eoff=-0.20, eon=-0.20), "요소식(H099-null)")
chk("요소식 경계 차 0.0999", build(eoff=-0.05, eon=0.0499), "요소식(H099-null)")
chk("차 0.10 정확 → 요소식 아님 → 보류", build(eoff=-0.05, eon=0.05), "보류")
chk("부분 켬만(e_on +0.30·e_off −0.05)", build(eoff=-0.05, eon=0.30), "부분(H099-partial) — 한 맥락만(켬만 5")
chk("부분 끔만(e_off −0.30·e_on 0.00)", build(eoff=-0.30, eon=0.0), "부분(H099-partial) — 한 맥락만(켬만 0")
chk("획득 4/5", build(per={14: (-0.05, 0.30)}), "획득(H099)")
chk("획득 3/5·요소식 2 → 부분 2 + 보류", build(per={13: (-0.2, -0.2), 14: (-0.2, -0.2)}), "보류")
for kind, b in (("kc_off", 10), ("kc_on0", 11), ("nctx_range", 12), ("nctx_mismatch", 13), ("agree", 14), ("no_col", 10),
                ("res", 11), ("refl", 12), ("load", 13), ("ev_noctx", 14), ("ev_ctx_in_off", 10),
                ("tonic_train_none", 11), ("tonic_train_val", 12), ("tonic_eval_none", 13), ("tonic_eval_val", 14), ("diag_none", 10),
                ("sat_learn_on", 11), ("sat_none_on", 12), ("dec_051", 13), ("dec_nan", 14)):
    chk("조작: %s" % kind, build(bad=(kind, b)), "보류(조작검증 실패)")
d = build(); d[5][(12, "learn", "on")] = (1, True, [X, X], (0.6664, 0.6662, 0.6663, 0.6599))
chk("켬 평가 하나 0.6599 → 비포화 통과 → 획득", d, "획득(H099)")
d = build(); d[3][13]["dec_on"] = 0.50
chk("켬 결정 0.50 정확 → 통과 → 획득", d, "획득(H099)")
chk("pick X 결측 → 결측", build(), "결측", x=None)
d = build(); del d[2][(12, "none", "on")]
chk("결측(평가 하나)", d, "결측")
d = build(); del d[3][11]
chk("결측(추적)", d, "결측")
# bicond_stats: 맥락별 규칙 + 탐욕 결정 비율
n = 3000
R = np.zeros((n, 38)); R[:, 2] = np.arange(n) % 2
ctx = (np.arange(n) // 2) % 2
R[:, 37] = ctx
R[:, 6] = np.where(ctx == 1, R[:, 2], 1 - R[:, 2]); R[:, 7] = 1
R[:, 13] = 1.0; R[:, 21] = J.R20
R[:, 3] = (np.arange(n) % 5 == 0).astype(float)                # 탐색 1/5
# 켬 시행은 i mod 4 ∈ {2, 3} — 그중 i mod 8 = 2 인 1/4 에 |v| 0.01(첫 판은 i mod 4 = 0(끔 시행)에 넣어 켬 비율이 0 이 된 시험 착오)
R[:, 5] = np.where(ctx == 1, np.where(np.arange(n) % 8 == 2, 0.01, 0.2), 0.15)
R[R[:, 3] == 1, 5] = 0.001                                      # 탐색 시행의 작은 v 는 세지 않아야 한다
st = J.bicond_stats(R)
gon = (R[:, 3] == 0) & (R[:, 37] == 1)
want_on = float(np.mean(np.abs(R[gon, 5]) <= 0.02))
g = (st["n"] == 3000 and st["has_ctx"] and st["agree"] == 1.0 and st["res"] < 1e-12 and abs(st["frac_ctx"] - 0.5) < 1e-12
     and abs(st["dec_on"] - want_on) < 1e-12 and 0.2 < st["dec_on"] < 0.3 and st["dec_off"] == 0.0)
R2 = R.copy(); R2[5, 7] = 0
g2 = J.bicond_stats(R2)["agree"] < 1.0
R3 = R.copy(); R3[:, 6] = 1 - R3[:, 2]
g3 = abs(J.bicond_stats(R3)["agree"] - 0.5) < 1e-12
g4 = not J.bicond_stats(np.zeros((10, 37)))["has_ctx"]
ok_all &= g and g2 and g3 and g4
print("%-34s → %s" % ("bicond_stats(규칙·불일치·열 없음·탐욕 결정)", "✓" if (g and g2 and g3 and g4) else "✗ %s want_on %.4f" % (st, want_on)))
# 파싱(load·read_pick)
with tempfile.TemporaryDirectory() as td:
    for d_ in (("logs", "E178"), ("traces", "E178")):
        os.makedirs(os.path.join(td, *d_))
    open(os.path.join(td, "E178.log"), "w", encoding="utf-8").write(
        "  e178 kcctx b10: => KCCTX %s\n  e178 train b10: => 사전 +0.0080 사후 -0.1000 보상 2000 || 맥락 켬 1490 || 적재 2 || x\n"
        "  e178 b10 learn on: => mod +0.3000\n" % KL)
    open(os.path.join(td, "logs", "E178", "pick.txt"), "w", encoding="utf-8").write("X=-1000\n")
    open(os.path.join(td, "logs", "E178", "train_b10.log"), "w", encoding="utf-8").write(
        "  [motor 동작점] motor 좌·우 Ioffset 기저값 -1000.0(긴장성 억제)\n[E153 종류 입력 적재] k 검증 일치 — x\n[맥락 과제] bicond frac=0.50\n[맥락 과제] 시행 3000 중 맥락 켬 1490\n"
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n"
        "  [motor 동작점] motor 좌·우 Ioffset 기저값 -1000.0(긴장성 억제)\n[E153 종류 입력 적재] k 검증 일치 — y\n")
    open(os.path.join(td, "logs", "E178", "ev_b10_learn_on.log"), "w", encoding="utf-8").write(
        "  [motor 동작점] motor 좌·우 Ioffset 기저값 -1000.0(긴장성 억제)\n[E153 종류 입력 적재] k 검증 일치 — x\n[맥락 평가] ctx=켬 level=0.90\n"
        "[E171 평가 진단] variant=base side=left n=250 motor L/R 0.550000/0.580000 KC L/R 0.006926/0.003237\n"
        "[E171 평가 진단] variant=base side=right n=250 motor L/R 0.580000/0.540000 KC L/R 0.003227/0.007001\n=> DECOMP mode=all mod=+0.3000\n")
    np.savez_compressed(os.path.join(td, "traces", "E178", "tr_bc_b10.npz"), rows=R)
    K, TRN, EV, S, RAW, EL = J.load(td)
    Xp = J.read_pick(td)
g = (K[10].startswith("side=l off=60") and TRN[10] == {"pre": 0.008, "post": -0.1, "rew": 2000, "nctx": 1490, "load": 2}
     and EV == {(10, "learn", "on"): 0.3} and RAW[10] == (1490, 3000, True, 2, [-1000.0, -1000.0])
     and EL[(10, "learn", "on")] == (1, True, [-1000.0], (0.55, 0.58, 0.58, 0.54))
     and EL[(10, "none", "off")] is None and S[10]["agree"] == 1.0 and RAW[11] is None and Xp == -1000.0)
ok_all &= g
print("%-34s → %s" % ("줄·원 로그·추적·pick 파싱", "✓" if g else "✗ %s %s %s %s %s" % (TRN, RAW, EL, K, Xp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
