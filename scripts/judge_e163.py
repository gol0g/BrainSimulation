#!/usr/bin/env python3
"""E163 판정 — 망 안 형성 표현의 반전(K81 회귀, E147·E155 판정 2 규칙). 뇌 5개 자료가 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄 "  e163 rev b10: => 사전 .. 사후 .. 보상 N || 적재 K". 반전 직전 기준 R = 같은 뇌 E162 A단독 사후(원 로그 logs/E162/train_A_b*.log —
같은 시드·같은 망 안 형성 가중치·같은 앞 1,500시행). 출발점 M5 = E162 A단독 [사전] ±0.002. m = [사후], Δ = m − R. 1e-4 정수.
실행: python3 scripts/judge_e163.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
R20 = (1.0 - 1.0 / 12.0) ** 20
REV_AT, N_REV = 1500, 3000
TR = re.compile(r"^\s*e163 rev b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+) \|\| 적재 (\d+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def i4(x):
    return int(round(x * 1e4))


def rev_stats(rows):
    n = len(rows)
    idx = np.arange(n)
    act = rows[:, 6] >= 0
    rule = np.where(idx < REV_AT, rows[:, 6] != rows[:, 2], rows[:, 6] == rows[:, 2])
    eda = rows[:, 13:17]
    return {"n": n, "agree": float(np.mean((rows[act, 7] == 1) == rule[act])) if act.any() else float("nan"),
            "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "rew_blk": [int((rows[i:i + 100, 7] == 1).sum()) for i in range(0, n, 100)]}


def ref(b):
    """E162 A단독 원 로그 → (사전, 사후) 1e-4 정수, 없으면 None."""
    f = os.path.join(EXP, "logs", "E162", "train_A_b%d.log" % b)
    if not os.path.exists(f):
        return None
    t = open(f, encoding="utf-8", errors="replace").read()
    a = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", t, re.M)
    p = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", t, re.M)
    return (i4(float(a.group(1))), i4(float(p.group(1)))) if (a and p) else None


def judge(T, S, RC, REF):
    miss = [b for b in BRAINS if b not in T or b not in S or RC.get(b) is None or REF.get(b) is None]
    if miss:
        return "[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss, None
    m1 = sum(RC[b][0] for b in BRAINS)
    m2 = sum(S[b]["agree"] == 1.0 for b in BRAINS)
    m3 = sum(S[b]["res"] <= 1e-3 and S[b]["n"] == N_REV and S[b]["pre_ratio"] <= 1e-3 for b in BRAINS)
    m4 = sum(RC[b][1] for b in BRAINS)
    m5 = sum(abs(i4(T[b]["pre"]) - REF[b][0]) <= 20 for b in BRAINS)
    m6 = sum(T[b]["load"] >= 2 for b in BRAINS)
    ok = m1 == m2 == m3 == m4 == m5 == m6 == 5
    mi = {b: i4(T[b]["post"]) for b in BRAINS}
    di = {b: mi[b] - REF[b][1] for b in BRAINS}
    succ = [b for b in BRAINS if mi[b] >= 200]
    part = [b for b in BRAINS if di[b] >= 1000]
    stuck = [b for b in BRAINS if di[b] < 500]
    if not ok:
        v = "보류(조작검증 실패)"
    elif len(succ) >= 4:
        v = "반전 성공(H086) — 망 안 형성 표현에서도 익힌 규칙을 뒤집어 다시 배운다(K81 보존)"
    elif len(part) == 5:
        v = "부분(H086-partial) — 반전 쪽으로 0.10 이상(5/5) 움직였으나 교차 쪽이 남음"
    elif len(stuck) >= 4:
        v = "고착(H086-null) — 반전 쪽 이동 < 0.05(≥4/5)"
    else:
        v = "보류"
    line = "[조작검증] M1 반전 줄 %d/5 · M2 규칙 일치 %d/5 · M3 동결·시행·도파민전 %d/5 · M4 반사 0 %d/5 · M5 출발점(E162 A [사전]) %d/5 · M6 적재 %d/5 %s" % (
        m1, m2, m3, m4, m5, m6, "통과" if ok else "실패")
    return line, {"m": mi, "d": di, "succ": succ, "part": part, "stuck": stuck, "ok": ok, "verdict": v}


def report(line, res, T=None, S=None, REF=None):
    print(line)
    if res is None:
        return
    for b in BRAINS:
        print("b%d 반전 사후 %+.4f (반전 전 E162 A %+.4f, Δ %+.4f) 보상 %d | 반전 구간 블록 보상 %s"
              % (b, res["m"][b] / 1e4, REF[b][1] / 1e4, res["d"][b] / 1e4, T[b]["rew"], " ".join(str(x) for x in S[b]["rew_blk"][15:])))
    print("m ≥ +0.02 %d/5 | Δ ≥ +0.10 %d/5 | Δ < +0.05 %d/5" % (len(res["succ"]), len(res["part"]), len(res["stuck"])))
    print("판정: %s" % res["verdict"])


def load():
    T, S, RC, REF = {}, {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E163.log"), encoding="utf-8", errors="replace"):
            m = TR.match(ln)
            if m:
                T[int(m.group(1))] = {"pre": float(m.group(2)), "post": float(m.group(3)), "rew": int(m.group(4)), "load": int(m.group(5))}
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E163", "tr_rev_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = rev_stats(np.load(f)["rows"])
        try:
            txt = open(os.path.join(EXP, "logs", "E163", "rev_b%d.log" % b), encoding="utf-8", errors="replace").read()
            got = {m.group(1): (m.group(2), m.group(3)) for m in (RW.match(ln) for ln in txt.splitlines()) if m}
            RC[b] = (("[반전] 시행 %d 부터" % REV_AT) in txt, set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values()))
        except FileNotFoundError:
            RC[b] = None
        r = ref(b)
        if r is not None:
            REF[b] = r
    return T, S, RC, REF


if __name__ == "__main__":
    T, S, RC, REF = load()
    l, r = judge(T, S, RC, REF)
    report(l, r, T, S, REF)
    sys.exit(0)
