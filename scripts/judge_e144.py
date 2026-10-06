#!/usr/bin/env python3
"""E144 판정 — 기준 logs/E144/criteria_fixed.txt(실행 전 고정). 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
A1500 = E142 F1500 + 행동 창 반대쪽 음 전류 N_선택(--act-current-neg). 짝 = E142 F1500. e = [사후] − [사전], m = [사후], d = e − e_F1500.
R0 = 보정의 반사 0 · 반대쪽 5000 행동 창 반대 motor 발화율 합 평균(traces/E144/calib/tr_R0_5000_b15.npz 열 36) — 파일에서 직접 읽는다.
rows 열: 7 correct 8~11 dg 12 dg_전체 13~16 e_da(교차·같은쪽·비선택·무활동) 17~20 dg_도파민전 21~24 e_end 35 행동 창 실행 발화율 36 반대.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
E119_PRE = {10: 0.4148, 11: 0.4195, 12: 0.3954, 13: 0.3773, 14: 0.4248}
F1500_POST = {10: 0.1026, 11: 0.1141, 12: 0.0953, 13: 0.0833, 14: 0.1266}
F1500_EFF = {10: -0.3122, 11: -0.3054, 12: -0.3001, 13: -0.2940, 14: -0.2982}
R20 = (1.0 - 1.0 / 12.0) ** 20
N_EXP = 1500
TL = re.compile(r"^\s*e144 b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def stats(rows):
    rw = rows[:, 7] == 1
    A, B = rows[rw, 8].sum(), rows[rw, 9].sum()
    C, P = rows[~rw, 8].sum(), rows[~rw, 9].sum()
    eda = rows[:, 13:17]
    n = len(rows)
    return {"n": n, "ncol": rows.shape[1],
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "alive": float(np.mean((np.abs(rows[:, 13]) + np.abs(rows[:, 14])) > 1.0)),
            "aw_ex": float(np.nanmean(rows[:, 35])) if rows.shape[1] >= 37 else float("nan"),
            "aw_ot": float(np.nanmean(rows[:, 36])) if rows.shape[1] >= 37 else float("nan"),
            "same_rw": float(rows[rw, 14].sum()), "cross_rw": float(rows[rw, 13].sum()),
            "BA": (B / A) if A else float("nan"), "CP": (C / P) if P else float("nan"),
            "dD": float((A + C) - (B + P)),
            "blk": [float((rows[i:i + 100, 8] - rows[i:i + 100, 9]).sum()) for i in range(0, n, 100)]}


def reflex_ok(path):
    got = {}
    try:
        for ln in open(path, encoding="utf-8", errors="replace"):
            m = RW.match(ln)
            if m:
                got[m.group(1)] = (m.group(2), m.group(3))
    except FileNotFoundError:
        return None
    if set(got) != {"l", "r"}:
        return None
    return all(v == ("25.0000", "25.0000") for v in got.values())


def judge(T, S, RF, r0):
    miss = [b for b in BRAINS if b not in T or b not in S or RF.get(b) is None]
    if miss or r0 is None or r0 != r0:
        return ["[측정 확인] 결측 뇌 %s 또는 기준 R0 없음 — **판정 보류, 수치 미출력**" % miss], None
    thr = r0 + 0.01
    m1 = sum(S[b]["res"] <= 1e-3 for b in BRAINS)
    m1b = sum(S[b]["alive"] >= 0.9 for b in BRAINS)
    m2 = sum(abs(T[b]["pre"] - E119_PRE[b]) <= 0.002 + 1e-9 for b in BRAINS)
    m3 = sum(S[b]["n"] == N_EXP and S[b]["pre_ratio"] <= 1e-3 for b in BRAINS)
    m4 = sum(bool(RF[b]) for b in BRAINS)
    m5 = sum(S[b]["aw_ot"] <= thr + 1e-9 for b in BRAINS)          # nan 이면 False, 1e-9 = 평균의 부동소수 오차 허용
    m6 = sum(S[b]["same_rw"] < 0 for b in BRAINS)
    ok = (m1 == m1b == m2 == m3 == m4 == m5 == m6 == 5)
    checks = ["[조작검증] M1 동결 %d/5 · M1b 되돌림 %d/5 · M2 출발점 %d/5 · M3 추적 %d/5 · M4 반사 25 불변 %d/5 · M5 반대쪽 침묵(≤ R0 %.4f + 0.01) %d/5 · M6 보상 시행 같은 쪽 흔적 < 0 %d/5 %s"
              % (m1, m1b, m2, m3, m4, r0, m5, m6, "통과" if ok else "실패")]
    e = {b: round(T[b]["post"] - T[b]["pre"], 6) for b in BRAINS}
    d = {b: round(e[b] - F1500_EFF[b], 6) for b in BRAINS}
    m = {b: T[b]["post"] for b in BRAINS}
    l2 = [b for b in BRAINS if m[b] <= -0.02 + 1e-9]
    par = [b for b in BRAINS if d[b] <= -0.03 + 1e-9]
    nul = [b for b in BRAINS if abs(d[b]) < 0.03]
    rev = [b for b in BRAINS if d[b] >= 0.03 - 1e-9]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif len(l2) >= 4:
        verdict = "L2 달성(H067) — 행동 창 반대쪽을 침묵시키면 1,500시행 사후 변조폭이 교차 쪽(≤ −0.02, ≥4/5): 학습이 반사 25 를 이긴다"
    elif len(par) == 5:
        verdict = "반대쪽 발화가 정체 원인(H067-partial) — 침묵이 거스름을 0.03 이상 키움(5/5), 반사는 아직 우세"
    elif len(nul) >= 4:
        verdict = "무관(H067-null) — |d| < 0.03(≥4/5)"
    elif len(rev) >= 4:
        verdict = "반대(H067-rev) — 침묵이 거스름을 줄임(d ≥ +0.03, ≥4/5)"
    else:
        verdict = "보류"
    return checks, {"e": e, "d": d, "m": m, "l2": l2, "par": par, "nul": nul, "rev": rev, "ok": ok, "verdict": verdict}


def report(checks, res, T=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        s = S[b]
        print("b%d 사후 %+.4f (F1500 %+.4f) e %+.4f d %+.4f 보상 %d | 행동 창 실행 %.3f 반대 %.3f | 보상 시행 흔적 교차 %+.2e 같은쪽 %+.2e | B/A %+.2f C/P %+.2f | 블록 ΔD 첫·끝 %+.3g→%+.3g"
              % (b, res["m"][b], F1500_POST[b], res["e"][b], res["d"][b], T[b]["rew"], s["aw_ex"], s["aw_ot"], s["cross_rw"], s["same_rw"],
                 s["BA"], s["CP"], s["blk"][0], s["blk"][-1]))
    print("m ≤ −0.02 %d/5 | d ≤ −0.03 %d/5 | |d| < 0.03 %d/5 | d ≥ +0.03 %d/5 | e 평균 %+.4f (F1500 %+.4f)"
          % (len(res["l2"]), len(res["par"]), len(res["nul"]), len(res["rev"]), sum(res["e"].values()) / 5, sum(F1500_EFF.values()) / 5))
    print("판정: %s" % res["verdict"])


def load():
    T, S, RF = {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E144.log"), encoding="utf-8", errors="replace"):
            mm = TL.match(ln)
            if mm:
                T[int(mm.group(1))] = {"pre": float(mm.group(2)), "post": float(mm.group(3)), "rew": int(mm.group(4))}
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E144", "tr_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = stats(np.load(f)["rows"])
        RF[b] = reflex_ok(os.path.join(EXP, "logs", "E144", "b%d.log" % b))
    r0 = None
    f = os.path.join(EXP, "traces", "E144", "calib", "tr_R0_5000_b15.npz")
    if os.path.exists(f):
        r0 = float(np.nanmean(np.load(f)["rows"][:, 36]))
    return T, S, RF, r0


if __name__ == "__main__":
    T, S, RF, r0 = load()
    c, r = judge(T, S, RF, r0)
    report(c, r, T, S)
    sys.exit(0)
