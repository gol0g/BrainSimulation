#!/usr/bin/env python3
"""judge_e138.py 합성 시험(조건 1): 표현 상한·학습 상한·혼재, 공통 모드 억제·없음, 경계(ρ = 2.5, κ = 1.5 정확), e≈0 뇌, 조작검증 실패 5종, 결측, 줄 파싱.
실행: python3 scripts/test_judge_e138.py  (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e138 as J

B = J.BRAINS


def pop(nL, nR, sp=900):
    return {"nL": nL, "nR": nR, "nNS": 300, "n0": 100, "D": 0.6, "g1": (50, 40, 120, 90), "M": 100.0, "shL": 0.5, "shR": 0.4, "shNS": 0.1,
            "n03": nL + nR + 20, "n07": max(nL + nR - 20, 0), "sp": sp}


def build(e, Rsel, eso, R=0.55, mut=None):
    """e, Rsel, eso: 뇌별 리스트(또는 스칼라). none = E119 [사전], all = [사후], kc_only = none + e, kcpop = none − R, kcsel = none − Rsel, kcselonly = none + eso."""
    f = lambda x, i: x[i] if isinstance(x, (list, tuple)) else x
    RT = {b: {"l": pop(80, 30), "r": pop(25, 85)} for b in B}
    EV = {}
    for i, b in enumerate(B):
        n = J.PRE[b]
        vals = {"none": n, "all": J.POST[b], "kc_only": n + f(e, i), "kcpop": n - f(R, i), "kcsel": n - f(Rsel, i), "kcselonly": n + f(eso, i)}
        for m, v in vals.items():
            EV[(b, m)] = {"mode": m, "mod": round(v, 4)}
    if mut:
        mut(RT, EV)
    return RT, EV


cases = []
cases.append(("표현 상한 + 억제 없음", build(-0.08, 0.16, -0.08), ("표현 상한", "공통 모드 억제 없음")))
cases.append(("학습 상한 + 공통 모드 억제", build(-0.08, 0.40, -0.14), ("학습 상한", "공통 모드 억제 —")))
cases.append(("혼재(ρ 3) + 보류(κ 1.3)", build(-0.08, 0.24, -0.104), ("보류(Q1)", "보류(Q2)")))
# 경계: ρ = 2.5 정확(R_sel 0.2, e −0.08), κ = 1.5 정확(e_so −0.12) — mod 는 소수 4자리라 정확히 표현되는 값으로
cases.append(("경계 ρ 2.5·κ 1.5", build(-0.08, 0.20, -0.12), ("표현 상한", "공통 모드 억제 —")))
# 3/5 만 표현 상한 → 보류
cases.append(("ρ ≤ 2.5 3/5", build(-0.08, [0.16, 0.16, 0.16, 0.40, 0.40], -0.08), ("보류(Q1)", "공통 모드 억제 없음")))
# e ≈ 0 인 뇌 2개(유효 3) → 어느 쪽도 4/5 못 채움
cases.append(("e≈0 뇌 2개", build([-0.08, -0.08, -0.08, -0.002, 0.003], 0.16, -0.08), ("보류(Q1)", "보류(Q2)")))
def m_c1(RT, EV):
    EV[(12, "none")]["mod"] += 0.0102
cases.append(("C1 재현 실패", build(-0.08, 0.16, -0.08, mut=m_c1), ("보류(조작검증 실패)", "보류(조작검증 실패)")))
cases.append(("C2 권한 미달", build(-0.08, 0.16, -0.08, R=[0.55, 0.55, 0.35, 0.55, 0.55]), ("보류(조작검증 실패)", "보류(조작검증 실패)")))
def m_c4(RT, EV):
    for b in (10, 11):
        RT[b]["l"]["nL"], RT[b]["l"]["nR"] = 20, 60
cases.append(("C4 편측성 3/5", build(-0.08, 0.16, -0.08, mut=m_c4), ("보류(조작검증 실패)", "보류(조작검증 실패)")))
def m_c5(RT, EV):
    RT[13]["r"]["sp"] = 0
cases.append(("C5 스파이크 0", build(-0.08, 0.16, -0.08, mut=m_c5), ("보류(조작검증 실패)", "보류(조작검증 실패)")))
def m_mode(RT, EV):
    EV[(14, "kcsel")]["mode"] = "kcpop"
cases.append(("mode 표기 불일치", build(-0.08, 0.16, -0.08, mut=m_mode), ("보류(조작검증 실패)", "보류(조작검증 실패)")))

ok_all = True
for name, (RT, EV), (w1, w2) in cases:
    c, r = J.judge(RT, EV)
    good = r is not None and r["q1"].startswith(w1) and r["q2"].startswith(w2)
    ok_all &= good
    print("%-26s 기대 %s / %s → %s / %s %s" % (name, w1, w2, r["q1"] if r else None, r["q2"] if r else None, "✓" if good else "✗"))

RT, EV = build(-0.08, 0.16, -0.08); del EV[(11, "kcselonly")]
c, r = J.judge(RT, EV)
good = r is None and "결측 1/35" in c[0]
ok_all &= good
print("%-26s 기대 결측·수치 미출력 → %s %s" % ("결측", c[0][:30], "✓" if good else "✗"))

# 줄 파싱 — 러너 형식 그대로(KCRATE 두 집단 + DECOMP + [E138] 꼬리)
kl = ("KCRATE kc_l | 좌선택 81 우선택 12 비선택 310 무활동 597 | 희석 0.612 | ≥1스파이크 좌전용 60 우전용 9 공유 120 무반응 811 | 여유합 +1234.5 몫 좌선택 0.700 우선택 -0.050 비선택 0.350"
      " | θ0.3 선택 140 θ0.7 선택 60 | 제시 스파이크 4321 기준선(제시창) 평균 0.0123 | KC별 ΔS 좌선택(→L +1.0 →R +9.0)")
kr = kl.replace("kc_l", "kc_r").replace("좌선택 81 우선택 12", "좌선택 10 우선택 77").replace("희석 0.612", "희석 nan").replace("몫 좌선택 0.700", "몫 좌선택 nan")
lines = ["  e138 b10 kcrate: => " + kl + " || " + kr + "\n",
         "  e138 b10 kcsel: => DECOMP mode=kcsel mod=-0.1234 acc=60.0 off=+0.0100 pushed=4 kc_means[kc_l>l=150.0 kc_l>r=170.0 kc_r>l=160.0 kc_r>r=150.0]"
         " || [E138] kcsel: 선택 KC kc_l 좌 81 우 12 / kc_r 좌 10 우 77 (wmax 300, init 150)\n",
         "  e138 b10 none: => DECOMP mode=none mod=+0.0195 acc=7.0 off=+0.0217 pushed=0 kc_means[kc_l>l=nan kc_l>r=nan kc_r>l=nan kc_r>r=nan]\n"]
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E138.log"), "w", encoding="utf-8").writelines(lines)
    J.EXP = td
    RTp, EVp, _first = J.load()
import math
good = (RTp[10]["l"]["nL"] == 81 and RTp[10]["l"]["D"] == 0.612 and RTp[10]["l"]["sp"] == 4321 and RTp[10]["l"]["shR"] == -0.05
        and RTp[10]["r"]["nR"] == 77 and math.isnan(RTp[10]["r"]["D"]) and math.isnan(RTp[10]["r"]["shL"]) and RTp[10]["l"]["g1"] == (60, 9, 120, 811)
        and EVp[(10, "kcsel")] == {"mode": "kcsel", "mod": -0.1234} and EVp[(10, "none")]["mod"] == 0.0195 and len(EVp) == 2)
ok_all &= good
print("%-26s 기대 KCRATE 2집단·DECOMP 2줄 → %s" % ("줄 파싱", "✓" if good else "✗ %s %s" % (RTp, EVp)))

# 수리 재실행(e138f) 대체: 1차 줄 + e138f 15줄이 있으면 kcsel·kcselonly·kcrate 를 e138f 로, 나머지 모드는 1차로
lines2 = []
for b in J.BRAINS:
    lines2.append("  e138 b%d kcrate: => " % b + kl + " || " + kr.replace("희석 nan", "희석 0.100").replace("몫 좌선택 nan", "몫 좌선택 0.000") + "\n")
    for mm, v in (("none", J.PRE[b]), ("all", J.POST[b]), ("kc_only", J.PRE[b] - 0.08), ("kcpop", J.PRE[b] - 0.55), ("kcsel", J.PRE[b] - 0.16), ("kcselonly", J.PRE[b] - 0.08)):
        lines2.append("  e138 b%d %s: => DECOMP mode=%s mod=%+.4f acc=1 off=+0 pushed=4 kc_means[x]\n" % (b, mm, mm, v))
for b in J.BRAINS:
    lines2.append("  e138f b%d kcrate: => " % b + kl.replace("좌선택 81 우선택 12", "좌선택 80 우선택 11") + " || " + kr.replace("희석 nan", "희석 0.100").replace("몫 좌선택 nan", "몫 좌선택 0.000") + "\n")
    lines2.append("  e138f b%d kcsel: => DECOMP mode=kcsel mod=%+.4f acc=1 off=+0 pushed=4 kc_means[x]\n" % (b, J.PRE[b] - 0.40))
    lines2.append("  e138f b%d kcselonly: => DECOMP mode=kcselonly mod=%+.4f acc=1 off=+0 pushed=4 kc_means[x]\n" % (b, J.PRE[b] - 0.08))
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E138.log"), "w", encoding="utf-8").writelines(lines2)
    J.EXP = td
    RT2, EV2, first2 = J.load()
c2, r2 = J.judge(RT2, EV2); c1_, r1_ = J.judge(*first2)
good = (first2 is not None and RT2[10]["l"]["nL"] == 80 and r2["q1"].startswith("학습 상한") and r1_["q1"].startswith("표현 상한")
        and EV2[(10, "none")]["mod"] == J.PRE[10])
ok_all &= good
print("%-26s 기대 e138f 대체(1차 표현 상한 → 수리 학습 상한) → %s / %s %s" % ("수리 재실행 대체", r1_["q1"][:5] if r1_ else None, r2["q1"][:5] if r2 else None, "✓" if good else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
