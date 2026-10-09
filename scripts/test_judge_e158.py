#!/usr/bin/env python3
"""judge_e158.py 합성 시험(조건 1): 판정 1(L2·거스름·효과 없음·보류, 경계), 판정 2(엄격 L2 경계·4/5), 조작검증 M1~M7 실패, 결측, 줄 파싱.
실행: python3 scripts/test_judge_e158.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e158 as J


def build(m1500=-0.20, e500=-0.40, per=None, s_bad=None, rf_bad=None, ld_bad=None, sc_bad=None, pre_shift=None, pre_all=None):
    T = {a: {} for a in J.ARMS}; S = {a: {} for a in J.ARMS}; RF = {a: {} for a in J.ARMS}
    for b in J.BRAINS:
        pre = J.E157_FK_PRE[b] if pre_all is None else pre_all
        p = {"m1500": m1500, "e500": e500}
        p.update((per or {}).get(b, {}))
        T["F500"][b] = {"pre": pre, "post": round(pre + p["e500"], 4), "rew": 200, "load": 2, "scale": 2}
        T["F1500"][b] = {"pre": pre + (pre_shift[1] if pre_shift and pre_shift[0] == b else 0.0), "post": p["m1500"], "rew": 600, "load": 2, "scale": 2}
        for a, n in J.ARMS.items():
            st = {"n": n, "pre_ratio": 0.0, "res": 1e-8, "alive": 1.0, "BA": -1.0, "CP": -0.7}
            if s_bad and s_bad[0] == (a, b):
                st.update(s_bad[1])
            S[a][b] = st
            RF[a][b] = not (rf_bad == (a, b))
            if ld_bad == (a, b):
                T[a][b]["load"] = 1
            if sc_bad == (a, b):
                T[a][b]["scale"] = 1
    return T, S, RF


ok_all = True


def chk(name, data, w1, w2=None):
    global ok_all
    c, r = J.judge(*data)
    g1 = r["v1"] if r else c[0]
    g2 = r["v2"] if r else ""
    good = g1.startswith(w1) and (w1 != "보류" or g1 == "보류") and (w2 is None or g2.startswith(w2))
    ok_all &= good
    print("%-34s 기대 %-14s/%-10s → %-16s / %-10s %s" % (name, w1[:14], (w2 or "-")[:10], g1[:16], g2[:10], "✓" if good else "✗"))


# 판정 1
chk("L2(m1500 −0.20)", build(), "L2 달성(H081)")
chk("경계 m −0.02 정확 → L2", build(m1500=-0.0200), "L2 달성(H081)")
chk("m −0.0199 → 거스름", build(m1500=-0.0199), "반사를 거스름(H081-partial)")
chk("거스름(m +0.10, e500 −0.30)", build(m1500=0.10, e500=-0.30), "반사를 거스름(H081-partial)")
chk("경계 e500 −0.10 정확 → 거스름", build(m1500=0.10, e500=-0.1000), "반사를 거스름(H081-partial)")
chk("효과 없음(e500 −0.02)", build(m1500=0.35, e500=-0.02), "효과 없음(H081-null)")
chk("경계 |e| 0.03 → 효과 없음 아님", build(m1500=0.35, e500=-0.0300), "보류")
chk("섞임 → 보류", build(per={13: {"m1500": 0.1}, 14: {"m1500": 0.1}}, e500=-0.05), "보류")
# 판정 2 — e1500 = m1500 − pre(E157 Fk). b10 pre 0.3519, 기본 0.4148 → 엄격 경계 m = 0.3519 − 0.4148 = −0.0629
chk("엄격 달성(m −0.20 → e ≈ −0.55)", build(), "L2 달성(H081)", "엄격 L2 달성")
chk("엄격 아님(m −0.05)", build(m1500=-0.05), "L2 달성(H081)", "엄격 L2 아님")
T, S, RF = build(per={b: {"m1500": round(J.E157_FK_PRE[b] - J.E142_PRE[b], 4)} for b in J.BRAINS})
c, r = J.judge(T, S, RF); g = r["v2"].startswith("엄격 L2 달성") and r["v1"].startswith("L2 달성"); ok_all &= g
print("%-34s → %s %s" % ("엄격 경계 정확(e = −기본 [사전]) 5/5", r["v2"][:12], "✓" if g else "✗ %s" % r["strict"]))
T, S, RF = build(per={b: {"m1500": round(J.E157_FK_PRE[b] - J.E142_PRE[b] + 0.0001, 4)} for b in J.BRAINS})
c, r = J.judge(T, S, RF); g = r["v2"] == "엄격 L2 아님"; ok_all &= g
print("%-34s → %s %s" % ("엄격 경계 0.0001 모자람 → 아님", r["v2"][:12], "✓" if g else "✗ %s" % r["strict"]))
T, S, RF = build(per={14: {"m1500": -0.05}})
c, r = J.judge(T, S, RF); g = r["v2"].startswith("엄격 L2 달성") and len(r["strict"]) == 4; ok_all &= g
print("%-34s → %s %s" % ("엄격 4/5 → 달성", r["v2"][:12], "✓" if g else "✗ %s" % r["strict"]))
# 조작검증
chk("M1 동결 실패", build(s_bad=(("F1500", 12), {"res": 0.01})), "보류(조작검증 실패)")
chk("M1b 되돌림 실패", build(s_bad=(("F500", 10), {"alive": 0.5})), "보류(조작검증 실패)")
chk("M2 두 팔 출발점 다름", build(pre_shift=(11, 0.0021)), "보류(조작검증 실패)")
chk("M3 추적 시행 수", build(s_bad=(("F1500", 13), {"n": 1400})), "보류(조작검증 실패)")
chk("M4 반사 변함", build(rf_bad=("F500", 14)), "보류(조작검증 실패)")
chk("M5 적재 1줄", build(ld_bad=("F1500", 10)), "보류(조작검증 실패)")
chk("M6 배율 1줄", build(sc_bad=("F500", 12)), "보류(조작검증 실패)")
chk("M7 E157 재현 어긋남(0.0021)", build(pre_all=0.3540), "보류(조작검증 실패)")
T, S, RF = build(); del T["F1500"][12]
c, r = J.judge(T, S, RF); g = r is None and "결측" in c[0]; ok_all &= g; print("%-34s → %s" % ("결측", "✓" if g else "✗"))
# 줄 파싱·추적 통계
R = np.zeros((500, 37)); R[:, 7] = (np.arange(500) % 2).astype(float); R[:, 8] = 2.0; R[:, 9] = 1.0; R[:, 13] = 1.0; R[:, 14] = 1.0
R[:, 21] = J.R_STAR; R[:, 22] = J.R_STAR
st = J.stats(R)
g = st["n"] == 500 and st["res"] < 1e-12 and st["alive"] == 1.0 and abs(st["BA"] - 0.5) < 1e-12; ok_all &= g
print("%-34s → %s" % ("stats", "✓" if g else "✗ %s" % st))
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E158")); os.makedirs(os.path.join(td, "traces", "E158"))
    open(os.path.join(td, "E158.log"), "w", encoding="utf-8").write("  e158 F1500 b10: => 사전 +0.3519 사후 -0.1000 보상 600 || 적재 2 배율 2 || => KCTRACE x\n")
    open(os.path.join(td, "logs", "E158", "F1500_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 25.0000→25.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (x)\n")
    open(os.path.join(td, "logs", "E158", "F500_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 25.0000→24.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (x)\n")
    np.savez_compressed(os.path.join(td, "traces", "E158", "tr_F1500_b10.npz"), rows=R)
    J.EXP = td
    Tp, Sp, RFp = J.load()
g = (Tp["F1500"][10] == {"pre": 0.3519, "post": -0.1, "rew": 600, "load": 2, "scale": 2} and RFp["F1500"][10] is True and RFp["F500"][10] is False
     and RFp["F500"][11] is None and Sp["F1500"][10]["n"] == 500)
ok_all &= g
print("%-34s → %s" % ("줄 파싱(배율 포함)·반사·추적", "✓" if g else "✗ %s %s" % (Tp, RFp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
