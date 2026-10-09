#!/usr/bin/env python3
"""E158 판정 — 기준 logs/E158/criteria_fixed.txt(2026-10-09 13:07:04 고정). 두 팔(F500·F1500) × 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
판정 1 = E142 규칙(L2·거스름·효과 없음·보류), 판정 2 = 엄격 L2(F1500 e ≤ −같은 뇌 기본 [사전]).
조작검증 M1 동결·M1b 되돌림·M2 두 팔 [사전] 일치·M3 추적·M4 반사 25 불변·M5 적재 ≥2·M6 배율 ≥2·M7 [사전] = E157 r25 Fk ±0.002.
e = [사후] − [사전](음수 = 교차 = 반사 반대), m = [사후]. 1e-4 정수 비교.
실행: python3 scripts/judge_e158.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = {"F500": 500, "F1500": 1500}
E142_PRE = {10: 0.4148, 11: 0.4195, 12: 0.3954, 13: 0.3773, 14: 0.4248}     # 같은 뇌 기본 표현 반사 25 [사전](E142 F500 원 로그)
E157_FK_PRE = {10: 0.3519, 11: 0.3617, 12: 0.3513, 13: 0.3338, 14: 0.3714}  # E157 r25 Fk [사전](같은 설정)
E142_F1500_M = {10: 0.1026, 11: 0.1141, 12: 0.0953, 13: 0.0833, 14: 0.1266}
R_STAR = (1.0 - 1.0 / 12.0) ** 20
TL = re.compile(r"^\s*e158 (F500|F1500) b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+) \|\| 적재 (\d+) 배율 (\d+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def stats(rows):
    rw = rows[:, 7] == 1
    A, B = rows[rw, 8].sum(), rows[rw, 9].sum()
    C, P = rows[~rw, 8].sum(), rows[~rw, 9].sum()
    eda, eend = rows[:, 13:17], rows[:, 21:25]
    return {"n": len(rows), "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(eend - R_STAR * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "alive": float(np.mean((np.abs(rows[:, 13]) + np.abs(rows[:, 14])) > 1.0)),
            "BA": (B / A) if A else float("nan"), "CP": (C / P) if P else float("nan")}


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


def i4(x):
    return int(round(x * 1e4))


def judge(T, S, RF):
    miss = [(a, b) for a in ARMS for b in BRAINS if b not in T.get(a, {}) or b not in S.get(a, {}) or RF.get(a, {}).get(b) is None]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss], None
    checks, ok = [], True
    m2 = sum(abs(i4(T["F500"][b]["pre"]) - i4(T["F1500"][b]["pre"])) <= 20 for b in BRAINS)
    for a, n_exp in ARMS.items():
        m1 = sum(S[a][b]["res"] <= 1e-3 for b in BRAINS)
        m1b = sum(S[a][b]["alive"] >= 0.9 for b in BRAINS)
        m3 = sum(S[a][b]["n"] == n_exp and S[a][b]["pre_ratio"] <= 1e-3 for b in BRAINS)
        m4 = sum(bool(RF[a][b]) for b in BRAINS)
        m5 = sum(T[a][b]["load"] >= 2 for b in BRAINS)
        m6 = sum(T[a][b]["scale"] >= 2 for b in BRAINS)
        m7 = sum(abs(i4(T[a][b]["pre"]) - i4(E157_FK_PRE[b])) <= 20 for b in BRAINS)
        good = (m1 == m1b == m3 == m4 == m5 == m6 == m7 == m2 == 5)
        checks.append("[%s 조작검증] M1 동결 %d/5 · M1b 되돌림 %d/5 · M2 두 팔 출발점 %d/5 · M3 추적 %d/5 · M4 반사 25 불변 %d/5 · M5 적재 %d/5 · M6 배율 %d/5 · M7 E157 재현 %d/5 %s"
                      % (a, m1, m1b, m2, m3, m4, m5, m6, m7, "통과" if good else "실패"))
        ok &= good
    e = {a: {b: i4(T[a][b]["post"]) - i4(T[a][b]["pre"]) for b in BRAINS} for a in ARMS}
    m = {a: {b: i4(T[a][b]["post"]) for b in BRAINS} for a in ARMS}
    l2 = [b for b in BRAINS if m["F1500"][b] <= -200]
    opp = [b for b in BRAINS if e["F500"][b] <= -1000]
    nul = [b for b in BRAINS if abs(e["F500"][b]) < 300]
    strict = [b for b in BRAINS if e["F1500"][b] <= -i4(E142_PRE[b])]
    acc = [b for b in BRAINS if e["F1500"][b] < e["F500"][b]]
    if not ok:
        v1, v2 = "보류(조작검증 실패)", "보류(조작검증 실패)"
    else:
        if len(l2) >= 4:
            v1 = "L2 달성(H081) — 이득 맞춘 형성 표현에서 1,500시행 사후 변조폭이 교차 쪽(≤ −0.02, ≥4/5): 학습된 매핑이 반사 25 를 이긴다"
        elif len(opp) == 5:
            v1 = "반사를 거스름(H081-partial) — 500시행 효과 ≤ −0.10 5/5, 1,500시행에도 반사 쪽이 남음"
        elif len(nul) >= 4:
            v1 = "효과 없음(H081-null) — 500시행 효과 |e| < 0.03(≥4/5)"
        else:
            v1 = "보류"
        v2 = "엄격 L2 달성 — 1,500시행 효과가 같은 뇌 기본 표현의 반사 발현 전체를 넘는다(≥4/5)" if len(strict) >= 4 else "엄격 L2 아님"
    return checks, {"e": e, "m": m, "l2": l2, "opp": opp, "nul": nul, "strict": strict, "acc": acc, "ok": ok, "v1": v1, "v2": v2}


def report(checks, res, T=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d [사전] %+.4f | F500 사후 %+.4f e %+.4f 보상 %d | F1500 사후 %+.4f e %+.4f 보상 %d | 기본 [사전] %+.4f (E142 F1500 사후 %+.4f) | B/A %+.2f/%+.2f C/P %+.2f/%+.2f"
              % (b, T["F500"][b]["pre"], res["m"]["F500"][b] / 1e4, res["e"]["F500"][b] / 1e4, T["F500"][b]["rew"],
                 res["m"]["F1500"][b] / 1e4, res["e"]["F1500"][b] / 1e4, T["F1500"][b]["rew"], E142_PRE[b], E142_F1500_M[b],
                 S["F500"][b]["BA"], S["F1500"][b]["BA"], S["F500"][b]["CP"], S["F1500"][b]["CP"]))
    print("F1500 m ≤ −0.02 %d/5 | F500 e ≤ −0.10 %d/5 | F500 |e| < 0.03 %d/5 | 엄격(F1500 e ≤ −기본 [사전]) %d/5 | 누적(e1500 < e500) %d/5"
          % (len(res["l2"]), len(res["opp"]), len(res["nul"]), len(res["strict"]), len(res["acc"])))
    print("판정 1: %s" % res["v1"])
    print("판정 2: %s" % res["v2"])


def load():
    T = {a: {} for a in ARMS}; S = {a: {} for a in ARMS}; RF = {a: {} for a in ARMS}
    try:
        for ln in open(os.path.join(EXP, "E158.log"), encoding="utf-8", errors="replace"):
            mm = TL.match(ln)
            if mm:
                T[mm.group(1)][int(mm.group(2))] = {"pre": float(mm.group(3)), "post": float(mm.group(4)), "rew": int(mm.group(5)),
                                                    "load": int(mm.group(6)), "scale": int(mm.group(7))}
    except FileNotFoundError:
        pass
    for a in ARMS:
        for b in BRAINS:
            f = os.path.join(EXP, "traces", "E158", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                S[a][b] = stats(np.load(f)["rows"])
            RF[a][b] = reflex_ok(os.path.join(EXP, "logs", "E158", "%s_b%d.log" % (a, b)))
    return T, S, RF


if __name__ == "__main__":
    T, S, RF = load()
    c, r = judge(T, S, RF)
    report(c, r, T, S)
    sys.exit(0)
