#!/usr/bin/env python3
"""E166 판정 — 보상 창 흔적 동결(호스트 가소성 제어) 제거의 능력 보존(외부 검토 권고 2, 전체 모델). 기준 logs/E166/criteria_fixed.txt.
10런이 다 모이기 전에는 수치를 출력하지 않는다.
팔 FNF = E141 인자에서 --rw-apm-scale 0 만 뺌 + 같은 뇌 E161 망 안 형성 가중치, 팔 DNF = 같은 인자·기본 표현. 뇌 16~20, 반사 0, 500시행.
기준 원 로그: 같은 뇌 E161 F(형성 + 동결) → e_F, E161 D(기본 + 동결) → e_D. q_F = e_FNF / e_F, q_D = e_DNF / e_D. 1e-4 정수 산술.
판정 1: 남음(H089) = q_F ≥ 0.80(⇔ 10·E_FNF ≤ 8·E_F, E_F < 0) 인 뇌 ≥ 4/5. 손실(H089-null) = q_F ≤ 0.50(⇔ 2·E_FNF ≥ E_F) ≥ 4/5. 그 밖 부분.
판정 2(부): 형성이 동결 의존을 줄임 = q_F − q_D ≥ 0.20(⇔ 5·(E_FNF·E_D − E_DNF·E_F) ≥ E_F·E_D) ≥ 4/5,
          줄이지 않음 = |q_F − q_D| < 0.10(⇔ 10·|E_FNF·E_D − E_DNF·E_F| < E_F·E_D) ≥ 4/5, 그 밖 중간.
조작검증(10/10 + 전제 5/5 — 하나라도 실패면 보류): 동결 꺼짐(보상 창 끝 흔적이 감쇠만으로 설명되지 않음: 잔차 ≥ 0.05), 추적 500시행·도파민 전 변화 ≤ 1e-3,
적재 줄(FNF ≥ 2, DNF 0), [사전] = 같은 뇌 E161 F(FNF)·D(DNF) [사전] ±0.002. 전제: e_F ≤ −0.10, e_D ≤ −0.10.
실행: python3 scripts/judge_e166.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
REF = {"FNF": "F", "DNF": "D"}          # 팔 → 같은 뇌 E161 기준 로그 접두
R_STAR = (1.0 - 1.0 / 12.0) ** 20


def i4(x):
    return int(round(x * 1e4))


def rd(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def lrn(t):
    a = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    b = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    if not (a and b):
        return None
    r = re.search(r"보상 (\d+)회", t)
    return {"pre": i4(float(a.group(1))), "post": i4(float(b.group(1))), "rew": int(r.group(1)) if r else None,
            "ld": len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M))}


def stats(rows):
    eda, eend = rows[:, 13:17], rows[:, 21:25]
    return {"n": len(rows), "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(eend - R_STAR * eda).sum() / max(np.abs(eda).sum(), 1e-12))}


def load():
    X = {}
    for b in BRAINS:
        for a, ref in REF.items():
            t = rd("logs", "E166", "%s_b%d.log" % (a, b))
            if lrn(t):
                X[("x", a, b)] = lrn(t)
            t = rd("logs", "E161", "%s_b%d.log" % (ref, b))
            if lrn(t):
                X[("b", a, b)] = lrn(t)
            f = os.path.join(EXP, "traces", "E166", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                X[("s", a, b)] = stats(np.load(f)["rows"])
    return X


def judge(X):
    miss = [(k, a, b) for a in REF for b in BRAINS for k in ("x", "b", "s") if (k, a, b) not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    E = {(a, b): X[("x", a, b)]["post"] - X[("x", a, b)]["pre"] for a in REF for b in BRAINS}
    B = {(a, b): X[("b", a, b)]["post"] - X[("b", a, b)]["pre"] for a in REF for b in BRAINS}
    mf = sum(X[("s", a, b)]["res"] >= 0.05 for a in REF for b in BRAINS)
    mt = sum(X[("s", a, b)]["n"] == 500 and X[("s", a, b)]["pre_ratio"] <= 1e-3 for a in REF for b in BRAINS)
    ml = sum((X[("x", a, b)]["ld"] >= 2) if a == "FNF" else (X[("x", a, b)]["ld"] == 0) for a in REF for b in BRAINS)
    mp = sum(abs(X[("x", a, b)]["pre"] - X[("b", a, b)]["pre"]) <= 20 for a in REF for b in BRAINS)
    pc = sum(B[("FNF", b)] <= -1000 and B[("DNF", b)] <= -1000 for b in BRAINS)
    ok = mf == mt == ml == mp == 10 and pc == 5
    checks = ["[조작검증] 동결 꺼짐(잔차 ≥ 0.05) %d/10 · 추적 500·도파민 전 %d/10 · 적재 %d/10 · [사전] 재현 %d/10 · 전제 e_F·e_D ≤ −0.10 %d/5 %s"
              % (mf, mt, ml, mp, pc, "통과" if ok else "실패")]
    eF = {b: B[("FNF", b)] for b in BRAINS}; eD = {b: B[("DNF", b)] for b in BRAINS}
    xF = {b: E[("FNF", b)] for b in BRAINS}; xD = {b: E[("DNF", b)] for b in BRAINS}
    keep = [b for b in BRAINS if 10 * xF[b] <= 8 * eF[b]]
    lose = [b for b in BRAINS if 2 * xF[b] >= eF[b]]
    dd = {b: xF[b] * eD[b] - xD[b] * eF[b] for b in BRAINS}       # (q_F − q_D)·e_F·e_D
    red = [b for b in BRAINS if 5 * dd[b] >= eF[b] * eD[b]]
    same = [b for b in BRAINS if 10 * abs(dd[b]) < eF[b] * eD[b]]
    v1 = "남음(H089)" if len(keep) >= 4 else ("손실(H089-null)" if len(lose) >= 4 else "부분")
    v2 = "형성이 동결 의존을 줄임" if len(red) >= 4 else ("줄이지 않음" if len(same) >= 4 else "중간")
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    return checks, {"E": E, "B": B, "keep": keep, "lose": lose, "red": red, "same": same, "v1": v1, "v2": v2, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        xF, xD = res["E"][("FNF", b)], res["E"][("DNF", b)]
        eF, eD = res["B"][("FNF", b)], res["B"][("DNF", b)]
        print("b%d FNF e %+.4f (동결 형성 %+.4f, q_F %.3f) 보상 %s 잔차 %.3f | DNF e %+.4f (동결 기본 %+.4f, q_D %.3f) 보상 %s 잔차 %.3f | q_F − q_D %+.3f"
              % (b, xF / 1e4, eF / 1e4, xF / eF, X[("x", "FNF", b)]["rew"], X[("s", "FNF", b)]["res"],
                 xD / 1e4, eD / 1e4, xD / eD, X[("x", "DNF", b)]["rew"], X[("s", "DNF", b)]["res"], xF / eF - xD / eD))
    print("q_F ≥ 0.80 %d/5 · q_F ≤ 0.50 %d/5 | q_F − q_D ≥ 0.20 %d/5 · |q_F − q_D| < 0.10 %d/5"
          % (len(res["keep"]), len(res["lose"]), len(res["red"]), len(res["same"])))
    print("판정 1: %s" % res["v1"])
    print("판정 2: %s" % res["v2"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
