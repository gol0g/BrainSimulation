#!/usr/bin/env python3
"""E181 판정 — 반사 25 출력 효율 저하(K92 의 O 0.52~0.58)는 motor 동작점 포화 때문인가: E159 교차 평가(뇌 10~14, 이득 맞춘 형성 표현 ×0.70, 학습 없음)를
motor 기저값 −4000 에서 다시 잰다. 기준 logs/E181/criteria_fixed.txt. 6칸 × 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
원 로그(logs/E181/{칸}_b{B}.log)의 '=> DECOMP ... mod= ... pushed=' 줄, 적재·배율 검증 줄, '[motor 동작점]' 줄, '[E171 평가 진단]' 줄을 읽는다.
E(W,R) = v(W,R) − v(none,R). O = E(W0,R25)/E(W0,R0), C = E(W25,R0)/E(W0,R0). 1e-4 정수.
판정 1(출력 효율, 적용 순서): 회복(H102) O ≥ 0.85(⇔ 100·E(W0,R25) ≤ 85·E(W0,R0), E(W0,R0) < 0) 인 뇌 ≥ 4/5 → 손실 유지(H102-null) O ≤ 0.70(⇔ 10·E(W0,R25) ≥ 7·E(W0,R0)) ≥ 4/5 → 그 밖 부분.
판정 2(L2 지표): 반사 25 학습 가중치가 새 동작점에서 반사를 뒤집음 = v(W25,R25) ≤ −0.02(⇔ ≤ −200) 인 뇌 ≥ 4/5, 그 밖 역전 없음.
조작검증(하나라도 실패면 보류): R5 30칸 pushed(W 8·none 0)·적재 ≥ 2·배율 ≥ 2 / 동작점 30칸 '[motor 동작점]' ≥ 1·값 = −4000 / 진단 30칸 좌·우 / 비포화 30칸(진단 motor 4개가 모두 ≥ 0.66 아님) /
R6 E(W0,R0) ≤ −0.10(5/5) / R7 반사 발현 v(none,R25) − v(none,R0) ≥ +0.10(5/5).
실행: python3 scripts/judge_e181.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
CELLS = ("none_R0", "none_R25", "W0_R0", "W0_R25", "W25_R0", "W25_R25")
X181 = -4000.0
SAT = 6600
DL = re.compile(r"^=> DECOMP mode=(\w+) mod=([-+0-9.]+) .*pushed=(\d+)", re.M)
TONIC = re.compile(r"^\s*\[motor 동작점\] motor 좌·우 Ioffset 기저값 ([-0-9.]+)", re.M)
DG = re.compile(r"^\[E171 평가 진단\] variant=\w+ side=(left|right) n=\d+ motor L/R ([0-9.]+)/([0-9.]+)", re.M)


def i4(x):
    return int(round(float(x) * 1e4))


def parse(txt):
    m = DL.search(txt)
    if not m:
        return None
    dg = {d.group(1): (float(d.group(2)), float(d.group(3))) for d in DG.finditer(txt)}
    return {"mode": m.group(1), "v": i4(m.group(2)), "pushed": int(m.group(3)),
            "ld": len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", txt, re.M)),
            "sc": len(re.findall(r"^\[E157 종류 입력 배율\].*검증 일치", txt, re.M)),
            "tonic": [float(x) for x in TONIC.findall(txt)],
            "rates": (dg["left"] + dg["right"]) if set(dg) == {"left", "right"} else None}


def load(base=EXP):
    X = {}
    for b in BRAINS:
        for c in CELLS:
            f = os.path.join(base, "logs", "E181", "%s_b%d.log" % (c, b))
            if os.path.exists(f):
                p = parse(open(f, encoding="utf-8", errors="replace").read())
                if p is not None:
                    X[(c, b)] = p
    return X


def judge(X):
    miss = [(c, b) for b in BRAINS for c in CELLS if (c, b) not in X]
    if miss:
        return ["[측정 확인] 결측 %d칸(예: %s) — **판정 보류, 수치 미출력**" % (len(miss), miss[:4])], None
    v = {k: X[k]["v"] for k in X}
    keys = [(c, b) for b in BRAINS for c in CELLS]
    r5 = sum(X[k]["pushed"] == (0 if k[0].startswith("none") else 8) and X[k]["ld"] >= 2 and X[k]["sc"] >= 2 for k in keys)
    mt = sum(len(X[k]["tonic"]) >= 1 and all(abs(t - X181) <= 1e-6 for t in X[k]["tonic"]) for k in keys)
    md = sum(X[k]["rates"] is not None for k in keys)
    mu = sum(X[k]["rates"] is not None and not all(i4(x) >= SAT for x in X[k]["rates"]) for k in keys)
    E = {}
    for b in BRAINS:
        for w in ("W0", "W25"):
            for r in ("R0", "R25"):
                E[(w, r, b)] = v[("%s_%s" % (w, r), b)] - v[("none_%s" % r, b)]
    r6 = sum(E[("W0", "R0", b)] <= -1000 for b in BRAINS)
    r7 = sum(v[("none_R25", b)] - v[("none_R0", b)] >= 1000 for b in BRAINS)
    ok = r5 == 30 and mt == 30 and md == 30 and mu == 30 and r6 == 5 and r7 == 5
    checks = ["[조작검증] R5 이식·적재·배율 %d/30 · 동작점(−4000) %d/30 · 진단 %d/30 · 비포화 %d/30 · R6 기준 효과 %d/5 · R7 반사 발현 %d/5 %s"
              % (r5, mt, md, mu, r6, r7, "통과" if ok else "실패")]
    rec, loss, l2 = [], [], []
    for b in BRAINS:
        base = E[("W0", "R0", b)]
        if 100 * E[("W0", "R25", b)] <= 85 * base:
            rec.append(b)
        if 10 * E[("W0", "R25", b)] >= 7 * base:
            loss.append(b)
        if v[("W25_R25", b)] <= -200:
            l2.append(b)
    v1 = "회복(H102)" if len(rec) >= 4 else ("손실 유지(H102-null)" if len(loss) >= 4 else "부분")
    v2 = "반사 역전(기존 반사 25 가중치)" if len(l2) >= 4 else "역전 없음"
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    return checks, {"E": E, "rec": rec, "loss": loss, "l2": l2, "v1": v1, "v2": v2, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    E = res["E"]
    for b in BRAINS:
        base = E[("W0", "R0", b)]
        print("b%d v none R0 %+.4f R25 %+.4f(반사 발현 %+.4f) | E W0 R0 %+.4f R25 %+.4f | E W25 R0 %+.4f R25 %+.4f | O %.3f C %.3f | v W25·R25 %+.4f | motor W25·R25 %s"
              % (b, X[("none_R0", b)]["v"] / 1e4, X[("none_R25", b)]["v"] / 1e4, (X[("none_R25", b)]["v"] - X[("none_R0", b)]["v"]) / 1e4,
                 base / 1e4, E[("W0", "R25", b)] / 1e4, E[("W25", "R0", b)] / 1e4, E[("W25", "R25", b)] / 1e4,
                 E[("W0", "R25", b)] / base, E[("W25", "R0", b)] / base, X[("W25_R25", b)]["v"] / 1e4,
                 "/".join("%.4f" % x for x in X[("W25_R25", b)]["rates"])))
    print("O ≥ 0.85 %d/5 · O ≤ 0.70 %d/5 | v(W25,R25) ≤ −0.02 %d/5" % (len(res["rec"]), len(res["loss"]), len(res["l2"])))
    print("판정 1: %s" % res["v1"])
    print("판정 2: %s" % res["v2"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
