#!/usr/bin/env python3
"""E180 판정 — 조합 비가산(K103·K104)이 motor 동작점 포화 때문인가: E162 AB 가중치(두 연합 학습)를 motor 기저값 −4000 에서 다시 평가했을 때
처음 보는 일치 조합 자극에 두 단독 효과의 합으로 반응하는가. E170 판정을 옮기고 회귀 검사(E162 값과 같음)를 동작점·진단·비포화 검사로 바꿈. 기준 logs/E180/criteria_fixed.txt.
40 평가가 다 모이기 전에는 수치를 출력하지 않는다. 뇌 10~14, 가중치 = E162 AB / none(학습 없음), 형성 표현 E160 가중치, 기저값 −4000(모든 평가).
e_v = m_AB(v) − m_none(v)(1e-4 정수). v ∈ base(good 단독)·bad(bad 단독)·agree(good 한쪽 + bad 반대쪽 — 두 규칙 같은 방향)·conflict(good·bad 같은 쪽 — 반대 방향).
판정 1(일치 조합): 합성 성공(H101) = ρ = e_agree / (e_base − e_bad) ≥ 0.80(⇔ 5·e_agree ≤ 4·(e_base − e_bad), 분모 < 0) 인 뇌 ≥ 4/5.
          합성 없음(H101-null) = |e_agree| < 1.10 × max(|e_base|, |e_bad|)(⇔ 10·|e_agree| < 11·max) ≥ 4/5. 그 밖 부분(합성 성공을 먼저 본다).
판정 2(충돌 조합): 가산 상쇄 = |e_conflict − (e_base + e_bad)| ≤ 0.15 × (|e_base| + |e_bad|)(⇔ 20·|차| ≤ 3·합) ≥ 4/5, 그 밖 비가산.
조작검증(하나라도 실패면 보류): 자극 줄 '[E170 자극]' 조합 20 로그 × 2줄의 쪽별 광선이 설계와 같음, 동작점 — 40 로그 모두 '[motor 동작점]' 줄 ≥ 1·값 = −4000,
진단 — 40 로그 모두 '[E171 평가 진단]' 좌·우, 비포화 — 40 로그 모두 진단 motor 4개가 모두 ≥ 0.66 이 아님, DECOMP pushed AB 8·none 0(40/40), 전제 e_base ≤ −0.10·e_bad ≥ +0.10(5/5).
실행: python3 scripts/judge_e180.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
VARS = ("base", "bad", "agree", "conflict")
WS = ("AB", "none")
X180 = -4000.0
SAT = 6600
DESIGN = {  # (변형, 쪽) → good L/R, bad L/R, food L/R
    ("agree", "left"): (0.9, 0.0, 0.0, 0.9, 0.9, 0.9), ("agree", "right"): (0.0, 0.9, 0.9, 0.0, 0.9, 0.9),
    ("conflict", "left"): (0.9, 0.0, 0.9, 0.0, 0.9, 0.0), ("conflict", "right"): (0.0, 0.9, 0.0, 0.9, 0.0, 0.9)}
SL = re.compile(r"^\[E170 자극\] variant=(\w+) side=(\w+) good L/R ([0-9.]+)/([0-9.]+) bad L/R ([0-9.]+)/([0-9.]+) food L/R ([0-9.]+)/([0-9.]+)", re.M)
TONIC = re.compile(r"^\s*\[motor 동작점\] motor 좌·우 Ioffset 기저값 ([-0-9.]+)", re.M)
DG = re.compile(r"^\[E171 평가 진단\] variant=\w+ side=(left|right) n=\d+ motor L/R ([0-9.]+)/([0-9.]+)", re.M)


def i4(x):
    return int(round(float(x) * 1e4))


def rd(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def ev(t, var):
    m = re.search(r"^=> DECOMP mode=(\w+) mod=([-+]?\d+\.\d+) .*?pushed=(\d+)", t, re.M) if t else None
    v = re.search(r"^\[E146 변형\] variant=(\w+)", t, re.M) if t else None
    if not (m and v and v.group(1) == var):
        return None
    stim = {(g.group(1), g.group(2)): tuple(i4(g.group(k)) for k in range(3, 9)) for g in SL.finditer(t)}
    dg = {d.group(1): (float(d.group(2)), float(d.group(3))) for d in DG.finditer(t)}
    return {"mode": m.group(1), "mod": i4(m.group(2)), "pushed": int(m.group(3)), "stim": stim,
            "tonic": [float(x) for x in TONIC.findall(t)], "rates": (dg["left"] + dg["right"]) if set(dg) == {"left", "right"} else None}


def load():
    X = {}
    for b in BRAINS:
        for w in WS:
            for v in VARS:
                r = ev(rd("logs", "E180", "ev_b%d_%s_%s.log" % (b, w, v)), v)
                if r:
                    X[(w, v, b)] = r
    return X


def judge(X):
    miss = [(w, v, b) for b in BRAINS for w in WS for v in VARS if (w, v, b) not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    ms = 0
    for b in BRAINS:
        for w in WS:
            for v in ("agree", "conflict"):
                st = X[(w, v, b)]["stim"]
                ms += all(st.get((v, sd)) == tuple(i4(x) for x in DESIGN[(v, sd)]) for sd in ("left", "right"))
    keys = [(w, v, b) for b in BRAINS for w in WS for v in VARS]
    mt = sum(len(X[k]["tonic"]) >= 1 and all(abs(t - X180) <= 1e-6 for t in X[k]["tonic"]) for k in keys)
    md = sum(X[k]["rates"] is not None for k in keys)
    mu = sum(X[k]["rates"] is not None and not all(i4(x) >= SAT for x in X[k]["rates"]) for k in keys)
    mp = sum(X[(w, v, b)]["pushed"] == (8 if w == "AB" else 0) for b in BRAINS for w in WS for v in VARS)
    e = {(v, b): X[("AB", v, b)]["mod"] - X[("none", v, b)]["mod"] for v in VARS for b in BRAINS}
    pc = sum(e[("base", b)] <= -1000 and e[("bad", b)] >= 1000 for b in BRAINS)
    ok = ms == 20 and mt == 40 and md == 40 and mu == 40 and mp == 40 and pc == 5
    checks = ["[조작검증] 조합 자극 구성 %d/20 · 동작점(−4000) %d/40 · 진단 %d/40 · 비포화 %d/40 · 적재(pushed) %d/40 · 전제 단독 효과 %d/5 %s"
              % (ms, mt, md, mu, mp, pc, "통과" if ok else "실패")]
    succ, none_, add2 = [], [], []
    for b in BRAINS:
        eb, ed, ea, ec = e[("base", b)], e[("bad", b)], e[("agree", b)], e[("conflict", b)]
        if 5 * ea <= 4 * (eb - ed):
            succ.append(b)
        if 10 * abs(ea) < 11 * max(abs(eb), abs(ed)):
            none_.append(b)
        if 20 * abs(ec - (eb + ed)) <= 3 * (abs(eb) + abs(ed)):
            add2.append(b)
    v1 = "합성 성공(H101)" if len(succ) >= 4 else ("합성 없음(H101-null)" if len(none_) >= 4 else "부분")
    v2 = "가산 상쇄" if len(add2) >= 4 else "비가산"
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    return checks, {"e": e, "succ": succ, "none": none_, "add2": add2, "v1": v1, "v2": v2, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    e = res["e"]
    for b in BRAINS:
        eb, ed, ea, ec = (e[(v, b)] for v in VARS)
        print("b%d e base %+.4f bad %+.4f | agree %+.4f (가산 예측 %+.4f, ρ %.3f) | conflict %+.4f (가산 예측 %+.4f) | 학습 m: agree %+.4f conflict %+.4f · 무학습 agree %+.4f conflict %+.4f | AB agree motor %s"
              % (b, eb / 1e4, ed / 1e4, ea / 1e4, (eb - ed) / 1e4, ea / (eb - ed) if eb != ed else float("nan"), ec / 1e4, (eb + ed) / 1e4,
                 X[("AB", "agree", b)]["mod"] / 1e4, X[("AB", "conflict", b)]["mod"] / 1e4, X[("none", "agree", b)]["mod"] / 1e4, X[("none", "conflict", b)]["mod"] / 1e4,
                 "/".join("%.4f" % x for x in X[("AB", "agree", b)]["rates"])))
    print("ρ ≥ 0.80 %d/5 · |e_agree| < 1.1×max 단독 %d/5 | 충돌 가산 상쇄 %d/5" % (len(res["succ"]), len(res["none"]), len(res["add2"])))
    print("판정 1: %s" % res["v1"])
    print("판정 2: %s" % res["v2"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
