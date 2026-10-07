#!/usr/bin/env python3
"""E155 판정 — 기준 logs/E155/criteria_fixed.txt(경로 검사·본실험 전 고정). 판정 1(K80 회귀, E146 규칙)·판정 2(K81 회귀, E147 규칙).
요약 줄: 평가 "  e155 b10 E153 int05: => mod -0.4800"(가중치 none·E153·E154A × 자극 5종), 반전 학습 "  e155 rev b10: => 사전 .. 사후 .. 보상 N || 적재 K || ...".
평가 적재(VL)는 원 평가 로그(logs/E155/ev_b*_*_*.log), 반전 조작검증은 추적(traces/E155/tr_rev_b*.npz)·원 로그(logs/E155/rev_b*.log). 1e-4 정수.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
WSETS = ("none", "E153", "E154A")
STIMS = ("base", "int05", "int07", "occ", "noise")
VARS = ("int05", "int07", "occ", "noise")
E153_POST = {10: -0.4893, 11: -0.4801, 12: -0.5024, 13: -0.4582, 14: -0.5143}
E153_PRE = {10: 0.0117, 11: 0.0015, 12: 0.0064, 13: 0.0231, 14: 0.0116}
E154A_POST = {10: -0.6015, 11: -0.5886, 12: -0.5881, 13: -0.5568, 14: -0.6153}
R20 = (1.0 - 1.0 / 12.0) ** 20
REV_AT, N_REV = 1500, 3000
TE = re.compile(r"^\s*e155 b(\d+) (none|E153|E154A) (base|int05|int07|occ|noise): => mod ([-+0-9.]+)")
TR = re.compile(r"^\s*e155 rev b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+) \|\| 적재 (\d+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")
LD = re.compile(r"^\[E153 종류 입력 적재\].*검증 일치")


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


def judge1(EV, EL):
    miss = [(b, w, s) for b in BRAINS for w in WSETS for s in STIMS if (b, w, s) not in EV or EL.get((b, w, s)) is None]
    if miss:
        return "[판정 1] 결측 %d — 보류, 수치 미출력" % len(miss), None
    I = {k: i4(v) for k, v in EV.items()}
    v1 = sum(abs(I[(b, "E153", "base")] - i4(E153_POST[b])) <= 20 for b in BRAINS)
    v2 = sum(abs(I[(b, "none", "base")] - i4(E153_PRE[b])) <= 20 for b in BRAINS)
    v3 = sum(abs(I[(b, "E154A", "base")] - i4(E154A_POST[b])) <= 20 for b in BRAINS)
    vl = sum(EL[k] >= 1 for k in EL)
    ok = v1 == v2 == v3 == 5 and vl == 75
    e = {(b, w, s): I[(b, w, s)] - I[(b, "none", s)] for b in BRAINS for w in ("E153", "E154A") for s in STIMS}
    c1 = {v: sum(e[(b, "E153", v)] <= -500 for b in BRAINS) for v in VARS}
    c3 = {v: sum(e[(b, "E153", "base")] < 0 and 2 * e[(b, "E153", v)] <= e[(b, "E153", "base")] for b in BRAINS) for v in VARS}
    c4 = {s: sum(e[(b, "E154A", s)] < e[(b, "E153", s)] for b in BRAINS) for s in STIMS}
    c1_ok = all(c1[v] == 5 for v in VARS); c3_ok = all(c3[v] >= 4 for v in VARS); c4_ok = all(c4[s] >= 4 for s in STIMS)
    c3_fail = [v for v in VARS if c3[v] < 4]; c4_fail = [s for s in STIMS if c4[s] < 4]
    if not ok:
        v = "보류(측정 검증 실패)"
    elif c1_ok and c3_ok and c4_ok:
        v = "통과(K80 보존) — 형성 표현에서도 무학습 초과·자극 변형 일반화·용량-반응"
    elif len(c3_fail) >= 2:
        v = "일반화 실패 — 훈련 자극 효과의 절반 이상을 못 지키는 변형 %s" % "·".join(c3_fail)
    elif c1_ok and c3_ok and len(c4_fail) >= 3:
        v = "용량 포화 — 1,500시행이 500시행보다 크지 않은 자극 %s" % "·".join(c4_fail)
    else:
        v = "부분(보류) — C1 %s · C3 실패 %s · C4 실패 %s" % ("통과" if c1_ok else "실패", c3_fail or "없음", c4_fail or "없음")
    line = "[판정 1 측정 검증] V1 %d/5 · V2 %d/5 · V3 %d/5 · 평가 적재 %d/75 %s" % (v1, v2, v3, vl, "통과" if ok else "실패")
    return line, {"e": e, "c1": c1, "c3": c3, "c4": c4, "ok": ok, "verdict": v, "pass": v.startswith("통과")}


def judge2(T, S, RC):
    miss = [b for b in BRAINS if b not in T or b not in S or RC.get(b) is None]
    if miss:
        return "[판정 2] 결측 %d — 보류, 수치 미출력" % len(miss), None
    m1 = sum(RC[b][0] for b in BRAINS)
    m2 = sum(S[b]["agree"] == 1.0 for b in BRAINS)
    m3 = sum(S[b]["res"] <= 1e-3 and S[b]["n"] == N_REV and S[b]["pre_ratio"] <= 1e-3 for b in BRAINS)
    m4 = sum(RC[b][1] for b in BRAINS)
    m5 = sum(abs(i4(T[b]["pre"]) - i4(E153_PRE[b])) <= 20 for b in BRAINS)
    m6 = sum(T[b]["load"] >= 2 for b in BRAINS)
    ok = m1 == m2 == m3 == m4 == m5 == m6 == 5
    mi = {b: i4(T[b]["post"]) for b in BRAINS}
    di = {b: mi[b] - i4(E154A_POST[b]) for b in BRAINS}
    succ = [b for b in BRAINS if mi[b] >= 200]; part = [b for b in BRAINS if di[b] >= 1000]; stuck = [b for b in BRAINS if di[b] < 500]
    if not ok:
        v = "보류(조작검증 실패)"
    elif len(succ) >= 4:
        v = "통과(K81 보존 — 반전 성공) — 형성 표현에서도 익힌 규칙을 뒤집어 다시 배운다"
    elif len(part) == 5:
        v = "부분 — 반전 쪽으로 0.10 이상(5/5) 움직였으나 교차 쪽이 남음"
    elif len(stuck) >= 4:
        v = "고착 — 반전 쪽 이동 < 0.05(≥4/5)"
    else:
        v = "보류"
    line = "[판정 2 조작검증] M1 반전 줄 %d/5 · M2 규칙 일치 %d/5 · M3 동결·시행·도파민전 %d/5 · M4 반사 0 %d/5 · M5 출발점 %d/5 · M6 적재 %d/5 %s" % (
        m1, m2, m3, m4, m5, m6, "통과" if ok else "실패")
    return line, {"m": mi, "d": di, "succ": succ, "part": part, "stuck": stuck, "ok": ok, "verdict": v, "pass": v.startswith("통과")}


def overall(r1, r2):
    if r1 is None or r2 is None:
        return "보류(결측)"
    if r1["pass"] and r2["pass"]:
        return "형성 표현 기준선 확립(H078) — 앞 능력(K80·K81) 보존"
    lost = [n for n, r in (("K80", r1), ("K81", r2)) if not r["pass"]]
    return "기준선 미확립 — 통과 못 한 회귀 %s" % "·".join(lost)


def report(l1, r1, l2, r2, EV=None, T=None, S=None):
    print(l1)
    if r1 is not None:
        for s in STIMS:
            print("%-5s none %s | e500(E153) %s | e1500(E154A) %s" % (s, " ".join("%+.4f" % EV[(b, "none", s)] for b in BRAINS),
                  " ".join("%+.4f" % (r1["e"][(b, "E153", s)] / 1e4) for b in BRAINS), " ".join("%+.4f" % (r1["e"][(b, "E154A", s)] / 1e4) for b in BRAINS)))
        print("C1 %s | C3 %s | C4 %s" % (" ".join("%s %d/5" % (v, r1["c1"][v]) for v in VARS), " ".join("%s %d/5" % (v, r1["c3"][v]) for v in VARS),
                                        " ".join("%s %d/5" % (s, r1["c4"][s]) for s in STIMS)))
        print("판정 1(K80 회귀): %s" % r1["verdict"])
    print(l2)
    if r2 is not None:
        for b in BRAINS:
            print("b%d 반전 사후 %+.4f (반전 전 E154A %+.4f, Δ %+.4f) 보상 %d | 반전 구간 블록 보상 %s"
                  % (b, r2["m"][b] / 1e4, E154A_POST[b], r2["d"][b] / 1e4, T[b]["rew"], " ".join(str(x) for x in S[b]["rew_blk"][15:])))
        print("m ≥ +0.02 %d/5 | Δ ≥ +0.10 %d/5 | Δ < +0.05 %d/5" % (len(r2["succ"]), len(r2["part"]), len(r2["stuck"])))
        print("판정 2(K81 회귀): %s" % r2["verdict"])
    print("종합: %s" % overall(r1, r2))


def load():
    EV, EL, T, S, RC = {}, {}, {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E155.log"), encoding="utf-8", errors="replace"):
            m = TE.match(ln)
            if m:
                EV[(int(m.group(1)), m.group(2), m.group(3))] = float(m.group(4))
                continue
            m = TR.match(ln)
            if m:
                T[int(m.group(1))] = {"pre": float(m.group(2)), "post": float(m.group(3)), "rew": int(m.group(4)), "load": int(m.group(5))}
    except FileNotFoundError:
        pass
    for b in BRAINS:
        for w in WSETS:
            for s in STIMS:
                try:
                    txt = open(os.path.join(EXP, "logs", "E155", "ev_b%d_%s_%s.log" % (b, w, s)), encoding="utf-8", errors="replace").read()
                    EL[(b, w, s)] = sum(bool(LD.match(ln)) for ln in txt.splitlines()) if ("[E146 변형] variant=%s " % s) in txt else 0
                except FileNotFoundError:
                    EL[(b, w, s)] = None
        f = os.path.join(EXP, "traces", "E155", "tr_rev_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = rev_stats(np.load(f)["rows"])
        try:
            txt = open(os.path.join(EXP, "logs", "E155", "rev_b%d.log" % b), encoding="utf-8", errors="replace").read()
            got = {m.group(1): (m.group(2), m.group(3)) for m in (RW.match(ln) for ln in txt.splitlines()) if m}
            RC[b] = (("[반전] 시행 %d 부터" % REV_AT) in txt, set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values()))
        except FileNotFoundError:
            RC[b] = None
    return EV, EL, T, S, RC


if __name__ == "__main__":
    EV, EL, T, S, RC = load()
    l1, r1 = judge1(EV, EL)
    l2, r2 = judge2(T, S, RC)
    report(l1, r1, l2, r2, EV, T, S)
    sys.exit(0)
