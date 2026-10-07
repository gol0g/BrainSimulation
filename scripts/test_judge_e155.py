#!/usr/bin/env python3
"""judge_e155.py 합성 시험(조건 1): 판정 1(통과·일반화 실패·용량 포화·부분·측정 검증 V1~V3·적재 실패·결측), 판정 2(통과·부분·고착·조작검증 6종 실패),
경계(C3 몫 0.5 정확, C1 −0.05 정확, m +0.02 정확), 종합, 줄 파싱.
실행: python3 scripts/test_judge_e155.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e155 as J


def ev(ratio=None, e500=-0.48, e1500=-0.58, per=None, el_bad=None, shift=None):
    """ratio[v] = e_v/e_base(500). e1500 은 자극 공통 비율로. per = {(b,w,s): 값} 덮어쓰기."""
    ratio = ratio or {"base": 1.0, "int05": 0.98, "int07": 0.99, "occ": 0.98, "noise": 0.6}
    EV, EL = {}, {}
    for b in J.BRAINS:
        for s in J.STIMS:
            n = J.E153_PRE[b] if s == "base" else 0.0100
            EV[(b, "none", s)] = n
            EV[(b, "E153", s)] = round(n + (J.E153_POST[b] - J.E153_PRE[b] if s == "base" else e500 * ratio[s]), 4)
            EV[(b, "E154A", s)] = round(n + (J.E154A_POST[b] - J.E153_PRE[b] if s == "base" else e1500 * ratio[s]), 4)
            for w in J.WSETS:
                EL[(b, w, s)] = 0 if el_bad == (b, w, s) else 1
    for k, v in (per or {}).items():
        EV[k] = v
    if shift:
        EV[shift[0]] = round(EV[shift[0]] + shift[1], 4)
    return EV, EL


def rev(post=0.30, per=None, rc_bad=None, s_bad=None, load=2):
    T, S, RC = {}, {}, {}
    for b in J.BRAINS:
        p = (per or {}).get(b, post)
        T[b] = {"pre": J.E153_PRE[b], "post": p, "rew": 2000, "load": 1 if load == ("one", b) else 2}
        st = {"n": 3000, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0, "rew_blk": [70] * 30}
        if s_bad and s_bad[0] == b:
            st.update(s_bad[1])
        S[b] = st
        RC[b] = (rc_bad != ("rev", b), rc_bad != ("refl", b))
    return T, S, RC


ok_all = True


def chk(name, got, want):
    global ok_all
    good = got.startswith(want) and (want not in ("보류", "부분") or got.startswith(want))
    ok_all &= good
    print("%-34s 기대 %-24s → %s %s" % (name, want[:24], got[:30], "✓" if good else "✗"))


# 판정 1
chk("1: 통과", J.judge1(*ev())[1]["verdict"], "통과(K80 보존)")
chk("1: 일반화 실패(int05·noise 0.4)", J.judge1(*ev(ratio={"base": 1, "int05": 0.4, "int07": 0.99, "occ": 0.98, "noise": 0.4}))[1]["verdict"], "일반화 실패")
chk("1: 용량 포화(변형 e1500 = e500 × 0.9)", J.judge1(*ev(e1500=-0.48 * 0.9))[1]["verdict"], "용량 포화")   # base 는 상수로 C4 통과, 변형 4종 C4 실패
r = J.judge1(*ev(e1500=-0.48 * 0.9))[1]
chk("1: 용량 포화 판정 문구", r["verdict"] if r["c4"]["base"] == 5 and sum(r["c4"][s] < 4 for s in J.STIMS) >= 3 else "x", "용량 포화")
# C3 경계: 뇌마다 e_v = floor(e_base/2)(짝수면 정확히 절반) → 2·e_v ≤ e_base 성립. 한 칸 덜(+1e-4)이면 실패.
import math
def c3_case(delta):
    EVc, ELc = ev()
    for b in J.BRAINS:
        eb = int(round(J.E153_POST[b] * 1e4)) - int(round(J.E153_PRE[b] * 1e4))
        for v in J.VARS:
            EVc[(b, "E153", v)] = round(0.0100 + (math.floor(eb / 2) + delta) / 1e4, 4)
            EVc[(b, "E154A", v)] = round(EVc[(b, "E153", v)] - 0.0500, 4)
    return J.judge1(EVc, ELc)[1]
rc = c3_case(0)
chk("1: C3 경계 몫 0.5(floor) 통과", rc["verdict"] if all(rc["c3"][v] == 5 for v in J.VARS) else "x", "통과(K80 보존)")
rc = c3_case(1)
chk("1: C3 경계 바로 안쪽 실패", rc["verdict"] if all(rc["c3"][v] == 0 for v in J.VARS) else "x", "일반화 실패")
EV0, EL0 = ev(); r0 = J.judge1(EV0, EL0)[1]
# C1 경계는 c1 개수로 직접: noise 의 e_v(E153) = −0.0500 정확 → 5/5, −0.0499 → 0/5
EVc, ELc = ev(per={(b, "E153", "noise"): round(0.0100 - 0.0500, 4) for b in J.BRAINS})
g = J.judge1(EVc, ELc)[1]["c1"]["noise"] == 5
EVc, ELc = ev(per={(b, "E153", "noise"): round(0.0100 - 0.0499, 4) for b in J.BRAINS})
g = g and J.judge1(EVc, ELc)[1]["c1"]["noise"] == 0
ok_all &= g; print("%-34s → %s" % ("1: C1 경계 −0.05 정확 5/5·−0.0499 0/5", "✓" if g else "✗"))
chk("1: V1 실패(±0.0021)", J.judge1(*ev(shift=((12, "E153", "base"), 0.0021)))[1]["verdict"], "보류(측정 검증 실패)")
chk("1: V2 실패", J.judge1(*ev(per={(13, "none", "base"): 0.0300}))[1]["verdict"], "보류(측정 검증 실패)")
chk("1: V3 실패", J.judge1(*ev(shift=((10, "E154A", "base"), -0.0030)))[1]["verdict"], "보류(측정 검증 실패)")
chk("1: 평가 적재 없음", J.judge1(*ev(el_bad=(11, "E154A", "occ")))[1]["verdict"], "보류(측정 검증 실패)")
EVm, ELm = ev(); del EVm[(14, "none", "noise")]
l, r = J.judge1(EVm, ELm); g = r is None and "결측" in l; ok_all &= g; print("%-34s → %s" % ("1: 결측", "✓" if g else "✗"))
# 판정 2
chk("2: 통과(사후 +0.30)", J.judge2(*rev())[1]["verdict"], "통과(K81 보존")
chk("2: 경계 m +0.02 정확", J.judge2(*rev(post=0.02))[1]["verdict"], "통과(K81 보존")
chk("2: 부분(사후 −0.30)", J.judge2(*rev(post=-0.30))[1]["verdict"], "부분")
chk("2: 고착(Δ 0.04)", J.judge2(*rev(per={b: J.E154A_POST[b] + 0.04 for b in J.BRAINS}))[1]["verdict"], "고착")
chk("2: M1 반전 줄 없음", J.judge2(*rev(rc_bad=("rev", 12)))[1]["verdict"], "보류(조작검증 실패)")
chk("2: M2 규칙 불일치", J.judge2(*rev(s_bad=(11, {"agree": 0.999})))[1]["verdict"], "보류(조작검증 실패)")
chk("2: M3 동결 실패", J.judge2(*rev(s_bad=(10, {"res": 0.01})))[1]["verdict"], "보류(조작검증 실패)")
chk("2: M4 반사 변함", J.judge2(*rev(rc_bad=("refl", 13)))[1]["verdict"], "보류(조작검증 실패)")
T5, S5, RC5 = rev(); T5[14]["pre"] = 0.0200
chk("2: M5 출발점 다름", J.judge2(T5, S5, RC5)[1]["verdict"], "보류(조작검증 실패)")
chk("2: M6 적재 1줄", J.judge2(*rev(load=("one", 10)))[1]["verdict"], "보류(조작검증 실패)")
# 종합
r1 = J.judge1(*ev())[1]; r2 = J.judge2(*rev())[1]
chk("종합: 둘 다 통과", J.overall(r1, r2), "형성 표현 기준선 확립(H078)")
chk("종합: 반전 부분", J.overall(r1, J.judge2(*rev(post=-0.3))[1]), "기준선 미확립 — 통과 못 한 회귀 K81")
# 줄 파싱·추적
R = np.zeros((3000, 37)); rng = np.random.default_rng(1); R[:, 2] = rng.integers(0, 2, 3000); R[:, 6] = rng.integers(0, 2, 3000)
R[:, 7] = np.where(np.arange(3000) < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]).astype(float); R[:, 13:17] = 1.0; R[:, 21:25] = J.R20
st = J.rev_stats(R); g = st["agree"] == 1.0 and st["n"] == 3000 and len(st["rew_blk"]) == 30; ok_all &= g
print("%-34s → %s" % ("rev_stats", "✓" if g else "✗ %s" % st))
with tempfile.TemporaryDirectory() as td:
    for d in ("logs/E155", "traces/E155"):
        os.makedirs(os.path.join(td, d))
    open(os.path.join(td, "E155.log"), "w", encoding="utf-8").write(
        "  e155 b10 E153 int05: => mod -0.4800\n  e155 rev b10: => 사전 +0.0117 사후 +0.3000 보상 2000 || 적재 2 || x\n")
    open(os.path.join(td, "logs", "E155", "ev_b10_E153_int05.log"), "w", encoding="utf-8").write(
        "[E153 종류 입력 적재] k 검증 일치 — x\n[E146 변형] variant=int05 vseed=0\n=> DECOMP mode=all mod=-0.4800\n")
    open(os.path.join(td, "logs", "E155", "ev_b10_E153_noise.log"), "w", encoding="utf-8").write(
        "[E153 종류 입력 적재] k 검증 일치 — x\n[E146 변형] variant=base vseed=0\n")   # 변형 표시가 다르면 적재 0 으로 센다
    open(os.path.join(td, "logs", "E155", "rev_b10.log"), "w", encoding="utf-8").write(
        "[반전] 시행 1500 부터 정답 = 같은 쪽\n[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")
    np.savez_compressed(os.path.join(td, "traces", "E155", "tr_rev_b10.npz"), rows=R)
    J.EXP = td
    EVp, ELp, Tp, Sp, RCp = J.load()
g = (EVp == {(10, "E153", "int05"): -0.48} and ELp[(10, "E153", "int05")] == 1 and ELp[(10, "E153", "noise")] == 0 and ELp[(10, "none", "base")] is None
     and Tp[10] == {"pre": 0.0117, "post": 0.3, "rew": 2000, "load": 2} and RCp[10] == (True, True) and RCp[11] is None and Sp[10]["agree"] == 1.0)
ok_all &= g
print("%-34s → %s" % ("줄 파싱(평가·반전·적재·변형 표시)", "✓" if g else "✗ %s %s %s" % (EVp, ELp, Tp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
