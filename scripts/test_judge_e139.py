#!/usr/bin/env python3
"""judge_e139.py 합성 시험(조건 1, 수정 기준): H062-rw·H062-dec·보류, 경계(B/A·C/P·e_end 0.5, e_da 0 정확), motor 침묵,
조작검증 실패(V1' 다른 시냅스·보상·사후, V2, V3, V4, 요약 줄 불일치), 결측, 줄 파싱.
실행: python3 scripts/test_judge_e139.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e139 as J


def rows_for(rc, rs, pc, ps, eda=(5.0, -5.0), eend=(4.0, 3.0), motor=(0.2, 0.2), pre=0.0, n=500, nrew=250):
    r = np.zeros((n, 27))
    r[:, 1] = np.arange(n) % 100; r[:, 0] = np.arange(n) // 100
    r[:nrew, 7] = 1
    r[:nrew, 8] = rc; r[:nrew, 9] = rs; r[nrew:, 8] = pc; r[nrew:, 9] = ps
    r[:, 10] = 0.5
    r[:, 12] = r[:, 8:12].sum(1)
    r[:nrew, 13] = eda[0]; r[:nrew, 14] = eda[1]
    r[:, 17] = pre
    r[:nrew, 21] = eend[0]; r[:nrew, 22] = eend[1]
    r[:, 25] = motor[0]; r[:, 26] = motor[1]
    return r


def build(spec, mutS=None, mutT=None, W=None, **kw):
    T, S = {}, {}
    for b in J.BRAINS:
        rr = rows_for(*spec, **kw) if not callable(spec) else spec(b)
        st = J.stats(rr)
        S[b] = st
        T[b] = {"pre": 0.01, "post": J.POST[b], "n": st["n"], "nrew": J.REW[b], "A": round(st["A"], 1), "B": round(st["B"], 1),
                "C": round(st["C"], 1), "P": round(st["P"], 1), "cons": 0.0, "gap": 0.0}
    Wd = {b: (600, 0.3, 1_500_000) for b in J.BRAINS} if W is None else W
    if mutS:
        mutS(S)
    if mutT:
        mutT(T)
    return T, S, Wd


RW = (10.0, 8.0, -7.0, -8.0)         # B/A 0.8, C/P 0.875
cases = []
cases.append(("H062-rw", build(RW), "H062-rw"))
cases.append(("H062-dec", build((10.0, 1.0, -1.0, -10.0), eda=(5.0, -5.0), eend=(5.0, -5.0)), "H062-dec"))
cases.append(("공동 움직임이나 e_end 비 0.3", build(RW, eend=(10.0, 3.0)), "보류"))
cases.append(("경계 정확(0.5·0.5·0.5·0)", build((10.0, 5.0, -5.0, -10.0), eda=(5.0, 0.0), eend=(4.0, 2.0)), "H062-rw"))
cases.append(("보상 창 motor 침묵", build(RW, motor=(0.0, 0.2)), "보류"))
def s_mix(b):
    return rows_for(*RW) if b in (10, 11, 12) else rows_for(10.0, 1.0, -1.0, -10.0, eend=(5.0, -5.0))
cases.append(("rw 3/5·dec 2/5", build(s_mix), "보류"))
cases.append(("V1' 다른 시냅스 0.33%", build(RW, W={b: (5000 if b == 12 else 600, 0.3, 1_500_000) for b in J.BRAINS}), "보류(조작검증 실패)"))
cases.append(("V1' 최대|차| 1.5(수정 2: 판정에 안 씀)", build(RW, W={b: (600, 1.5 if b == 10 else 0.3, 1_500_000) for b in J.BRAINS}), "H062-rw"))
def mt_rew(T):
    T[13]["nrew"] = J.REW[13] + 4
cases.append(("V1' 보상 수 +4", build(RW, mutT=mt_rew), "보류(조작검증 실패)"))
def mt_post(T):
    T[11]["post"] = J.POST[11] + 0.003
cases.append(("V1' 사후 +0.003", build(RW, mutT=mt_post), "보류(조작검증 실패)"))
def ms_cons(S):
    S[11]["cons_ok"] = False
cases.append(("V2 합 불일치", build(RW, mutS=ms_cons), "보류(조작검증 실패)"))
def mt_gap(T):
    T[10]["gap"] = 1e-4
cases.append(("V3 g 연속 깨짐", build(RW, mutT=mt_gap), "보류(조작검증 실패)"))
cases.append(("V4 도파민 전 변화", build(RW, pre=0.5), "보류(조작검증 실패)"))
def mt_A(T):
    T[12]["A"] += 1.0
cases.append(("요약 줄 ≠ npz", build(RW, mutT=mt_A), "보류(조작검증 실패)"))
ok_all = True
for name, (T, S, W), want in cases:
    c, r = J.judge(T, S, W)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-26s 기대 %-16s → %s %s" % (name, want, r["verdict"][:40] if r else None, "✓" if good else "✗"))
T, S, W = build(RW); del W[13]
c, r = J.judge(T, S, W)
good = r is None and "결측" in c[0]
ok_all &= good
print("%-26s 기대 결측·수치 미출력 → %s %s" % ("결측", c[0][:24], "✓" if good else "✗"))
ln = ("  e139 b10: => 사전 +0.0195 사후 -0.0838 || => KCTRACE 시행 500 보상 258 | A +12345.6 B +2345.6 C -345.6 P -4567.8 | ΔD +14222.2 | share_B 0.871 B+/A 0.190 C-/A 0.028"
      " | 합일관성 최대 3.21e-12 | g연속 최대 0 | 블록 ΔD +1 +2 +3 +4 +5 | 보상시행 e_same/e_cross 0.123 || => KCTRACE2 탐색 A +1 B +1 C +1 P +1 | 탐욕 A +1 B +1 C +1 P +1 | 비선택 Δg 보상 +1 처벌 +1 → x\n")
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E139.log"), "w", encoding="utf-8").write(ln)
    J.EXP = td
    Tp, Sp, Wp = J.load(check_weights=False)
good = Tp[10] == {"pre": 0.0195, "post": -0.0838, "n": 500, "nrew": 258, "A": 12345.6, "B": 2345.6, "C": -345.6, "P": -4567.8, "cons": 3.21e-12, "gap": 0.0}
ok_all &= good
print("%-26s 기대 요약 줄 파싱 → %s" % ("줄 파싱", "✓" if good else "✗ %s" % Tp))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
