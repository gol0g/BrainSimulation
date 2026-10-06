#!/usr/bin/env python3
"""E147 판정 — 기준 logs/E147/criteria_fixed.txt(실행 전 고정). 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
m = [사후] 이식 변조폭(양수 = 같은 쪽 = 반전 규칙), Δ = m − R(E146 R0F1500 사후). 1e-4 정수 비교.
추적 열: 2 side_r 3 explore 6 ex_r(행동 창 실행 오른쪽=1) 7 correct 13~16 e_da 17~20 dg_도파민전 21~24 e_end 12 dg_전체.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
REF = {10: -0.3194, 11: -0.3309, 12: -0.3215, 13: -0.3217, 14: -0.3486}
PRE0 = {10: 0.0195, 11: 0.0150, 12: 0.0262, 13: 0.0320, 14: 0.0165}
R20 = (1.0 - 1.0 / 12.0) ** 20
REV_AT, N_EXP = 1500, 3000
TL = re.compile(r"^\s*e147 b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def stats(rows):
    n = len(rows)
    idx = np.arange(n)
    act = rows[:, 6] >= 0
    rule = np.where(idx < REV_AT, rows[:, 6] != rows[:, 2], rows[:, 6] == rows[:, 2])
    agree = float(np.mean((rows[act, 7] == 1) == rule[act])) if act.any() else float("nan")
    eda = rows[:, 13:17]
    rew_blk = [int((rows[i:i + 100, 7] == 1).sum()) for i in range(0, n, 100)]
    dD_blk = [float((rows[i:i + 100, 8] - rows[i:i + 100, 9]).sum()) for i in range(0, n, 100)]
    return {"n": n, "agree": agree, "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "rew_blk": rew_blk, "dD_blk": dD_blk}


def rawcheck(path):
    """(반전 줄 있음, 반사 0→0) — 파일 없으면 None."""
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None
    got = {}
    for ln in txt.splitlines():
        m = RW.match(ln)
        if m:
            got[m.group(1)] = (m.group(2), m.group(3))
    return ("[반전] 시행 %d 부터" % REV_AT) in txt, (set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values()))


def judge(T, S, RC):
    miss = [b for b in BRAINS if b not in T or b not in S or RC.get(b) is None]
    if miss:
        return ["[측정 확인] 결측 뇌 %s — **판정 보류, 수치 미출력**" % miss], None
    m1 = sum(RC[b][0] for b in BRAINS)
    m2 = sum(S[b]["agree"] == 1.0 for b in BRAINS)
    m3 = sum(S[b]["res"] <= 1e-3 and S[b]["n"] == N_EXP and S[b]["pre_ratio"] <= 1e-3 for b in BRAINS)
    m4 = sum(RC[b][1] for b in BRAINS)
    m5 = sum(abs(T[b]["pre"] - PRE0[b]) <= 0.002 + 1e-9 for b in BRAINS)
    ok = m1 == m2 == m3 == m4 == m5 == 5
    checks = ["[조작검증] M1 반전 줄 %d/5 · M2 보상-규칙 일치 %d/5 · M3 동결·시행·도파민전 %d/5 · M4 반사 0 불변 %d/5 · M5 출발점 %d/5 %s"
              % (m1, m2, m3, m4, m5, "통과" if ok else "실패")]
    mi = {b: int(round(T[b]["post"] * 1e4)) for b in BRAINS}
    di = {b: mi[b] - int(round(REF[b] * 1e4)) for b in BRAINS}
    succ = [b for b in BRAINS if mi[b] >= 200]
    part = [b for b in BRAINS if di[b] >= 1000]
    stuck = [b for b in BRAINS if di[b] < 500]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif len(succ) >= 4:
        verdict = "반전 성공(H070) — 1,500시행 반전 뒤 사후 변조폭이 같은 쪽(≥ +0.02, ≥4/5): 익힌 규칙을 뒤집어 다시 배운다"
    elif len(part) == 5:
        verdict = "부분(H070-partial) — 반전 쪽으로 0.10 이상 움직였으나(5/5) 아직 교차 쪽이 남음"
    elif len(stuck) >= 4:
        verdict = "고착(H070-stuck) — 반전 쪽 이동 < 0.05(≥4/5)"
    else:
        verdict = "보류"
    return checks, {"m": {b: mi[b] / 1e4 for b in BRAINS}, "d": {b: di[b] / 1e4 for b in BRAINS}, "succ": succ, "part": part,
                    "stuck": stuck, "ok": ok, "verdict": verdict}


def report(checks, res, T=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        s = S[b]
        print("b%d 사후 %+.4f (반전 전 기준 %+.4f, Δ %+.4f) 보상 %d | 반전 구간 블록 보상 %s | ΔD 블록(반전 구간) %s"
              % (b, res["m"][b], REF[b], res["d"][b], T[b]["rew"], " ".join(str(x) for x in s["rew_blk"][15:]),
                 " ".join("%+.0e" % x for x in s["dD_blk"][15:])))
    print("m ≥ +0.02 %d/5 | Δ ≥ +0.10 %d/5 | Δ < +0.05 %d/5" % (len(res["succ"]), len(res["part"]), len(res["stuck"])))
    print("판정: %s" % res["verdict"])


def load():
    T, S, RC = {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E147.log"), encoding="utf-8", errors="replace"):
            mm = TL.match(ln)
            if mm:
                T[int(mm.group(1))] = {"pre": float(mm.group(2)), "post": float(mm.group(3)), "rew": int(mm.group(4))}
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E147", "tr_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = stats(np.load(f)["rows"])
        RC[b] = rawcheck(os.path.join(EXP, "logs", "E147", "b%d.log" % b))
    return T, S, RC


if __name__ == "__main__":
    T, S, RC = load()
    c, r = judge(T, S, RC)
    report(c, r, T, S)
    sys.exit(0)
