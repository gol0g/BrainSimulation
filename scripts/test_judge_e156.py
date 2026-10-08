#!/usr/bin/env python3
"""judge_e156.py 합성 시험(조건 1): L2 달성·거스름·효과 없음·보류, 경계(m −0.02·e −0.10·|e| 0.03 정확), 조작검증 M1·M1b·M2·M3·M4·M5 실패, 결측, 줄 파싱·추적 통계.
실행: python3 scripts/test_judge_e156.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e156 as J


def build(m1500=-0.10, e500=-0.40, per=None, s_bad=None, rf_bad=None, load_bad=None, pre_shift=None):
    T = {a: {} for a in J.ARMS}; S = {a: {} for a in J.ARMS}; RF = {a: {} for a in J.ARMS}
    for b in J.BRAINS:
        pre = 0.4000
        p = {"m1500": m1500, "e500": e500}
        p.update((per or {}).get(b, {}))
        T["F500"][b] = {"pre": pre, "post": round(pre + p["e500"], 4), "rew": 200, "load": 2}
        T["F1500"][b] = {"pre": pre + (pre_shift[1] if pre_shift and pre_shift[0] == b else 0.0), "post": p["m1500"], "rew": 600, "load": 2}
        for a, n in J.ARMS.items():
            st = {"n": n, "pre_ratio": 0.0, "res": 1e-8, "alive": 1.0, "BA": -1.0, "CP": -0.7, "blk": [0.0]}
            if s_bad and s_bad[0] == (a, b):
                st.update(s_bad[1])
            S[a][b] = st
            RF[a][b] = not (rf_bad == (a, b))
            if load_bad == (a, b):
                T[a][b]["load"] = 1
    return T, S, RF


ok_all = True


def chk(name, data, want):
    global ok_all
    c, r = J.judge(*data)
    got = r["verdict"] if r else c[0]
    good = got.startswith(want) and (want != "보류" or got == "보류")
    ok_all &= good
    print("%-30s 기대 %-22s → %s %s" % (name, want[:22], got[:28], "✓" if good else "✗"))


chk("L2 달성(m1500 −0.10)", build(), "L2 달성(H079)")
chk("경계 m −0.02 정확", build(m1500=-0.0200), "L2 달성(H079)")
chk("m −0.0199 → 거스름", build(m1500=-0.0199), "반사를 거스름(H079-partial)")
chk("거스름(m +0.10, e500 −0.30)", build(m1500=0.10, e500=-0.30), "반사를 거스름(H079-partial)")
chk("경계 e500 −0.10 정확", build(m1500=0.10, e500=-0.1000), "반사를 거스름(H079-partial)")
chk("효과 없음(e500 −0.02)", build(m1500=0.35, e500=-0.02), "효과 없음(H079-null)")
chk("경계 |e| 0.03 → 효과 없음 아님", build(m1500=0.35, e500=-0.0300), "보류")
chk("섞임 → 보류", build(per={13: {"m1500": 0.1}, 14: {"m1500": 0.1}}, e500=-0.05), "보류")
chk("M1 동결 실패", build(s_bad=(("F1500", 12), {"res": 0.01})), "보류(조작검증 실패)")
chk("M1b 되돌림 실패", build(s_bad=(("F500", 10), {"alive": 0.5})), "보류(조작검증 실패)")
chk("M2 두 팔 출발점 다름", build(pre_shift=(11, 0.0021)), "보류(조작검증 실패)")
chk("M3 추적 시행 수", build(s_bad=(("F1500", 13), {"n": 1400})), "보류(조작검증 실패)")
chk("M4 반사 변함", build(rf_bad=("F500", 14)), "보류(조작검증 실패)")
chk("M5 적재 1줄", build(load_bad=("F1500", 10)), "보류(조작검증 실패)")
T, S, RF = build(); del T["F1500"][12]
c, r = J.judge(T, S, RF); g = r is None and "결측" in c[0]; ok_all &= g; print("%-30s → %s" % ("결측", "✓" if g else "✗"))
# 추적 통계·줄 파싱
R = np.zeros((500, 37)); R[:, 7] = (np.arange(500) % 2).astype(float); R[:, 8] = 2.0; R[:, 9] = 1.0; R[:, 13] = 1.0; R[:, 14] = 1.0
R[:, 21] = J.R_STAR; R[:, 22] = J.R_STAR
st = J.stats(R)
g = st["n"] == 500 and st["res"] < 1e-12 and st["alive"] == 1.0 and abs(st["BA"] - 0.5) < 1e-12; ok_all &= g
print("%-30s → %s" % ("stats", "✓" if g else "✗ %s" % st))
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E156")); os.makedirs(os.path.join(td, "traces", "E156"))
    open(os.path.join(td, "E156.log"), "w", encoding="utf-8").write("  e156 F1500 b10: => 사전 +0.4000 사후 -0.1000 보상 600 || 적재 2 || x\n")
    open(os.path.join(td, "logs", "E156", "F1500_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 25.0000→25.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (x)\n")
    open(os.path.join(td, "logs", "E156", "F500_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 25.0000→24.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (x)\n")
    np.savez_compressed(os.path.join(td, "traces", "E156", "tr_F1500_b10.npz"), rows=R)
    J.EXP = td
    Tp, Sp, RFp, Wp = J.load()
g = (Tp["F1500"][10] == {"pre": 0.4, "post": -0.1, "rew": 600, "load": 2} and RFp["F1500"][10] is True and RFp["F500"][10] is False
     and RFp["F500"][11] is None and Sp["F1500"][10]["n"] == 500 and Wp is None and RFp["W1500"] == {})
ok_all &= g
print("%-30s → %s" % ("줄 파싱(학습·반사 25·추적)", "✓" if g else "✗ %s %s" % (Tp, RFp)))


# 수정 1 — 판정 2(맞춤)·종합
def build_w(m=-0.10, d=0.0, per=None, s_bad=None, rf_bad=None, load_bad=None, drop=None):
    T = {"W1500": {}}; S = {"W1500": {}}; RF = {"W1500": {}}
    for b in J.BRAINS:
        p = {"m": m, "d": d}
        p.update((per or {}).get(b, {}))
        T["W1500"][b] = {"pre": round(J.E142_PRE[b] + p["d"], 4), "post": p["m"], "rew": 600, "load": 1 if load_bad == b else 2}
        st = {"n": 1500, "pre_ratio": 0.0, "res": 1e-8, "alive": 1.0, "BA": -1.0, "CP": -0.7, "blk": [0.0]}
        if s_bad and s_bad[0] == b:
            st.update(s_bad[1])
        S["W1500"][b] = st
        RF["W1500"][b] = not (rf_bad == b)
    if drop is not None:
        del T["W1500"][drop]
    return T, S, RF


def chk2(name, data, want, wstar=90):
    global ok_all
    _, r2 = J.judge_w(*data, wstar)
    good = r2["verdict"] == want
    ok_all &= good
    print("%-30s 기대 %-22s → %s %s" % (name, want[:22], r2["verdict"][:28], "✓" if good else "✗"))


chk2("판정2 이김(m −0.10)", build_w(), "반사 발현을 맞춰도 학습이 이김")
chk2("판정2 경계 m −0.02 정확", build_w(m=-0.0200), "반사 발현을 맞춰도 학습이 이김")
chk2("판정2 m −0.0199 → L2 아님", build_w(m=-0.0199), "반사 발현을 맞추면 L2 아님")
chk2("판정2 3/5 → L2 아님", build_w(per={13: {"m": 0.05}, 14: {"m": 0.05}}), "반사 발현을 맞추면 L2 아님")
chk2("판정2 4/5 → 이김", build_w(per={14: {"m": 0.05}}), "반사 발현을 맞춰도 학습이 이김")
chk2("MW 경계 −0.05 정확 → 통과", build_w(d=-0.0500), "반사 발현을 맞춰도 학습이 이김")
chk2("MW −0.0501 → 보류", build_w(per={12: {"d": -0.0501}}), "보류(조작검증 실패)")
chk2("MW 경계 +0.10 정확 → 통과", build_w(d=0.1000), "반사 발현을 맞춰도 학습이 이김")
chk2("MW +0.1001 → 보류", build_w(per={10: {"d": 0.1001}}), "보류(조작검증 실패)")
chk2("M1 동결 실패", build_w(s_bad=(11, {"res": 0.01})), "보류(조작검증 실패)")
chk2("M1b 되돌림 실패", build_w(s_bad=(11, {"alive": 0.5})), "보류(조작검증 실패)")
chk2("M3 시행 1400", build_w(s_bad=(13, {"n": 1400})), "보류(조작검증 실패)")
chk2("M4w 반사 변함", build_w(rf_bad=14), "보류(조작검증 실패)")
chk2("M5 적재 1줄", build_w(load_bad=10), "보류(조작검증 실패)")
chk2("결측", build_w(drop=12), "보류(결측)")
chk2("보정 실패", build_w(), "없음(보정 실패)", wstar=None)
V1L2 = "L2 달성(H079) — x"; V1P = "반사를 거스름(H079-partial) — x"
for name, v1, v2, want in (("종합 지지", V1L2, "반사 발현을 맞춰도 학습이 이김", "H079 지지"),
                           ("종합 부분", V1L2, "반사 발현을 맞추면 L2 아님", "H079 부분"),
                           ("종합 미해결", V1P, "반사 발현을 맞추면 L2 아님", "판정 1 그대로(반사를 거스름(H079-partial)) — L2 미해결"),
                           ("종합 판정1 보류·판정2 L2 아님", "보류", "반사 발현을 맞추면 L2 아님", "판정 1 그대로(보류) — L2 미해결"),
                           ("종합 예측 밖", V1P, "반사 발현을 맞춰도 학습이 이김", "보류(예측 밖"),
                           ("종합 판정2 보류", V1L2, "보류(조작검증 실패)", "판정 1(L2 달성(H079)) + 식별 불가"),
                           ("종합 보정 실패", V1L2, "없음(보정 실패)", "판정 1(L2 달성(H079)) + 식별 불가")):
    got = J.combine(v1, v2)
    g = got.startswith(want)
    ok_all &= g
    print("%-30s → %s %s" % (name, got[:40], "✓" if g else "✗"))
# W 팔 줄 파싱(반사 W*·wstar.txt)
with tempfile.TemporaryDirectory() as td:
    for d_ in (("logs", "E156"), ("traces", "E156")):
        os.makedirs(os.path.join(td, *d_))
    open(os.path.join(td, "logs", "E156", "wstar.txt"), "w", encoding="utf-8").write("W*=90 사전 +0.4100 (목표 +0.4397 ± 0.05)\n")
    open(os.path.join(td, "E156.log"), "w", encoding="utf-8").write("  e156 W1500 b10: => 사전 +0.4100 사후 -0.0500 보상 600 || 적재 2 || x\n")
    open(os.path.join(td, "logs", "E156", "W1500_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 90.0000→90.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 90.0000→90.0000 (x)\n")
    open(os.path.join(td, "logs", "E156", "W1500_b11.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 25.0000→25.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (x)\n")
    np.savez_compressed(os.path.join(td, "traces", "E156", "tr_W1500_b10.npz"), rows=np.zeros((1500, 37)))
    J.EXP = td
    Tw, Sw, RFw, Ww = J.load()
g = (Ww == 90 and Tw["W1500"][10] == {"pre": 0.41, "post": -0.05, "rew": 600, "load": 2} and RFw["W1500"][10] is True
     and RFw["W1500"][11] is False and RFw["W1500"][12] is None and Sw["W1500"][10]["n"] == 1500)
ok_all &= g
print("%-30s → %s" % ("W 팔 줄 파싱(W*·반사 W*)", "✓" if g else "✗ %s %s %s" % (Ww, Tw, RFw)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
