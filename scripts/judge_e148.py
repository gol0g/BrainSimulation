#!/usr/bin/env python3
"""E148 판정 — 기준 logs/E148/criteria_fixed.txt(실행 전 고정). 학습 5줄 + 평가 20줄이 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄: 학습 "  e148 train b10: => 사전 ... 사후 ... 보상 N || ...", 평가 "  e148 b10 learn bad: => mod +0.1000" (가중치 learn·none × 자극 base·bad).
eA = base(learn) − base(none), eA1 = E146 R0F1500 사후 − E119 사전, rA = eA/eA1, eB = bad(learn) − bad(none). 1e-4 정수.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
F1500_POST = {10: -0.3194, 11: -0.3309, 12: -0.3215, 13: -0.3217, 14: -0.3486}
PRE0 = {10: 0.0195, 11: 0.0150, 12: 0.0262, 13: 0.0320, 14: 0.0165}
R20 = (1.0 - 1.0 / 12.0) ** 20
SWITCH, N_EXP = 1500, 3000
TT = re.compile(r"^\s*e148 train b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")
TE = re.compile(r"^\s*e148 b(\d+) (learn|none) (base|bad): => mod ([-+0-9.]+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def stats(rows):
    n = len(rows)
    idx = np.arange(n)
    act = rows[:, 6] >= 0
    rule = np.where(idx < SWITCH, rows[:, 6] != rows[:, 2], rows[:, 6] == rows[:, 2])
    eda = rows[:, 13:17]
    return {"n": n, "agree": float(np.mean((rows[act, 7] == 1) == rule[act])) if act.any() else float("nan"),
            "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "rew_blk": [int((rows[i:i + 100, 7] == 1).sum()) for i in range(0, n, 100)]}


def rawcheck(path):
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None
    got = {}
    for ln in txt.splitlines():
        m = RW.match(ln)
        if m:
            got[m.group(1)] = (m.group(2), m.group(3))
    return ("[과제 B] 시행 %d 부터" % SWITCH) in txt, (set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values()))


def judge(TR, EV, S, RC):
    miss = [b for b in BRAINS if b not in TR or b not in S or RC.get(b) is None]
    miss += [(b, w, s) for b in BRAINS for w in ("learn", "none") for s in ("base", "bad") if (b, w, s) not in EV]
    if miss:
        return ["[측정 확인] 결측 %d — **판정 보류, 수치 미출력**" % len(miss)], None
    I = {k: int(round(v * 1e4)) for k, v in EV.items()}
    m1 = sum(RC[b][0] for b in BRAINS)
    m2 = sum(S[b]["agree"] == 1.0 for b in BRAINS)
    m3 = sum(S[b]["res"] <= 1e-3 and S[b]["n"] == N_EXP and S[b]["pre_ratio"] <= 1e-3 for b in BRAINS)
    m4 = sum(RC[b][1] for b in BRAINS)
    m5 = sum(abs(TR[b]["pre"] - PRE0[b]) <= 0.002 + 1e-9 for b in BRAINS)
    m6 = sum(abs(I[(b, "none", "base")] - int(round(PRE0[b] * 1e4))) <= 20 for b in BRAINS)
    ok = m1 == m2 == m3 == m4 == m5 == m6 == 5
    eA = {b: I[(b, "learn", "base")] - I[(b, "none", "base")] for b in BRAINS}
    eA1 = {b: int(round(F1500_POST[b] * 1e4)) - int(round(PRE0[b] * 1e4)) for b in BRAINS}
    eB = {b: I[(b, "learn", "bad")] - I[(b, "none", "bad")] for b in BRAINS}
    rA = {b: (eA[b] / eA1[b]) if eA1[b] else float("nan") for b in BRAINS}
    mb = sum(eB[b] >= 500 for b in BRAINS)
    keep = sum(eA1[b] < 0 and 2 * eA[b] <= eA1[b] for b in BRAINS)        # rA ≥ 0.5 ⇔ 2·eA ≤ eA1 (둘 다 음수)
    lost = 5 - keep
    checks = ["[조작검증] M1 과제 B 줄 %d/5 · M2 보상-규칙 일치 %d/5 · M3 동결·시행·도파민전 %d/5 · M4 반사 0 %d/5 · M5 출발점 %d/5 · M6 무학습 base 재현 %d/5 %s · MB 과제 B 학습(eB ≥ +0.05) %d/5"
              % (m1, m2, m3, m4, m5, m6, "통과" if ok else "실패", mb)]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif mb < 4:
        verdict = "보류(과제 B 미학습 — 간섭 시험 불가)"
    elif keep >= 4:
        verdict = "유지(H071) — 상충하는 과제 B 를 배운 뒤에도 과제 A 효과의 절반 이상 유지(≥4/5)"
    elif lost >= 4:
        verdict = "간섭(H071-int) — 과제 B 학습이 과제 A 효과를 절반 미만으로 줄임(≥4/5)"
    else:
        verdict = "보류"
    return checks, {"eA": {b: eA[b] / 1e4 for b in BRAINS}, "eA1": {b: eA1[b] / 1e4 for b in BRAINS}, "rA": rA,
                    "eB": {b: eB[b] / 1e4 for b in BRAINS}, "keep": keep, "mb": mb, "ok": ok, "verdict": verdict}


def report(checks, res, EV=None, TR=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d 과제 A: 사후 %+.4f eA %+.4f (과제 A 만 학습 %+.4f, 유지 몫 %.2f) | 과제 B: eB %+.4f | 학습 사후 %+.4f 보상 %d | 과제 B 구간 블록 보상 %s"
              % (b, EV[(b, "learn", "base")], res["eA"][b], res["eA1"][b], res["rA"][b], res["eB"][b], TR[b]["post"], TR[b]["rew"],
                 " ".join(str(x) for x in S[b]["rew_blk"][15:])))
    print("과제 B 학습 %d/5 | 유지(rA ≥ 0.5) %d/5" % (res["mb"], res["keep"]))
    print("판정: %s" % res["verdict"])


def load():
    TR, EV, S, RC = {}, {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E148.log"), encoding="utf-8", errors="replace"):
            m = TT.match(ln)
            if m:
                TR[int(m.group(1))] = {"pre": float(m.group(2)), "post": float(m.group(3)), "rew": int(m.group(4))}
                continue
            m = TE.match(ln)
            if m:
                EV[(int(m.group(1)), m.group(2), m.group(3))] = float(m.group(4))
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E148", "tr_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = stats(np.load(f)["rows"])
        RC[b] = rawcheck(os.path.join(EXP, "logs", "E148", "train_b%d.log" % b))
    return TR, EV, S, RC


if __name__ == "__main__":
    TR, EV, S, RC = load()
    c, r = judge(TR, EV, S, RC)
    report(c, r, EV, TR, S)
    sys.exit(0)
