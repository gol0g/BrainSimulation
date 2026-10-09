#!/usr/bin/env python3
"""E164 판정 — 구현 정합성 측정(외부 검토 2026-10-09 ①·③). 기준 logs/E164/criteria_fixed.txt. 10런이 다 모이기 전에는 수치를 출력하지 않는다.
FIX(--rw-da-reset --offset-steps 3) 학습 효과 e_X 와 같은 뇌 기본 코드 기준 e_B(반사 0 = E141 b*.log, 반사 25 = E142 F500_b*.log 원 로그)의 차 Δ = e_X − e_B.
수준별: 영향 큼 = Δ ≥ +0.05 인 뇌 ≥4/5 또는 Δ ≤ −0.05 인 뇌 ≥4/5(같은 부호), 무시 가능 = |Δ| < 0.03 인 뇌 ≥4/5. 종합: 한 수준이라도 영향 큼 → 영향 큼, 두 수준 모두 무시 가능 → 무시 가능, 그 밖 보류. 1e-4 정수.
조작검증(10/10): '[구현 점검]' 줄 3개(설정·오프셋·첫 보상 창 끝 I_input → 0.0), [사전] = 기준 [사전] ±0.002(사전 평가는 학습 전이라 옵션과 무관 — 재현), 추적 500시행·동결 잔차 ≤ 1e-3·도파민 전 ≤ 1e-3.
실행: python3 scripts/judge_e164.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = {"R0X": ("E141", "b%d.log"), "R25X": ("E142", "F500_b%d.log")}
R20 = (1.0 - 1.0 / 12.0) ** 20


def i4(x):
    return int(round(x * 1e4))


def rd(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def mods(t):
    a = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    b = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    return (i4(float(a.group(1))), i4(float(b.group(1)))) if (a and b) else None


def checks_of(t):
    lines = re.findall(r"^\[구현 점검\].*$", t, re.M)
    da = re.search(r"^\[구현 점검\] 첫 보상 창 끝 도파민 뉴런 I_input ([0-9.]+) → ([0-9.]+)", t, re.M)
    return {"n": len(lines), "set": any("rw_da_reset=True offset_steps=3" in ln for ln in lines),
            "off": any(ln.startswith("[구현 점검] 오프셋(조향 3처리 합)") for ln in lines),
            "da": (float(da.group(1)), float(da.group(2))) if da else None}


def stats(rows):
    eda = rows[:, 13:17]
    return {"n": len(rows), "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12))}


def load():
    X = {}
    for a, (be, pat) in ARMS.items():
        for b in BRAINS:
            t = rd("logs", "E164", "%s_b%d.log" % (a, b))
            if mods(t):
                X[("x", a, b)] = mods(t)
                X[("c", a, b)] = checks_of(t)
            tb = rd("logs", be, pat % b)
            if mods(tb):
                X[("b", a, b)] = mods(tb)
            f = os.path.join(EXP, "traces", "E164", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                X[("s", a, b)] = stats(np.load(f)["rows"])
    return X


def level(D):
    up = sum(D[b] >= 500 for b in BRAINS)
    dn = sum(D[b] <= -500 for b in BRAINS)
    small = sum(abs(D[b]) < 300 for b in BRAINS)
    if up >= 4 or dn >= 4:
        return "영향 큼", up, dn, small
    if small >= 4:
        return "무시 가능", up, dn, small
    return "중간", up, dn, small


def judge(X):
    miss = [(k, a, b) for a in ARMS for b in BRAINS for k in ("x", "c", "b", "s") if (k, a, b) not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    mc = sum(X[("c", a, b)]["n"] >= 3 and X[("c", a, b)]["set"] and X[("c", a, b)]["off"] and X[("c", a, b)]["da"] is not None
             and X[("c", a, b)]["da"][1] == 0.0 for a in ARMS for b in BRAINS)
    mp = sum(abs(X[("x", a, b)][0] - X[("b", a, b)][0]) <= 20 for a in ARMS for b in BRAINS)
    ms = sum(X[("s", a, b)]["n"] == 500 and X[("s", a, b)]["res"] <= 1e-3 and X[("s", a, b)]["pre_ratio"] <= 1e-3 for a in ARMS for b in BRAINS)
    ok = mc == mp == ms == 10
    checks = ["[조작검증] 구현 점검 줄(설정·오프셋·I_input → 0) %d/10 · [사전] 재현 %d/10 · 추적·동결 %d/10 %s" % (mc, mp, ms, "통과" if ok else "실패")]
    D, L = {}, {}
    for a in ARMS:
        D[a] = {b: (X[("x", a, b)][1] - X[("x", a, b)][0]) - (X[("b", a, b)][1] - X[("b", a, b)][0]) for b in BRAINS}
        L[a] = level(D[a])
    if not ok:
        v = "보류(조작검증 실패)"
    elif any(L[a][0] == "영향 큼" for a in ARMS):
        v = "영향 큼(H087) — %s 에서 수정이 학습 효과를 0.05 이상 같은 방향으로 바꾼다" % "·".join(a for a in ARMS if L[a][0] == "영향 큼")
    elif all(L[a][0] == "무시 가능" for a in ARMS):
        v = "무시 가능(H087-null) — 두 수준 모두 |Δ| < 0.03(≥4/5)"
    else:
        v = "보류(중간)"
    return checks, {"D": D, "L": L, "ok": ok, "verdict": v}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for a in ARMS:
        for b in BRAINS:
            ex = X[("x", a, b)][1] - X[("x", a, b)][0]; eb = X[("b", a, b)][1] - X[("b", a, b)][0]
            print("%s b%d e 수정 %+.4f 기준 %+.4f Δ %+.4f | 사후 수정 %+.4f 기준 %+.4f | I_input %s"
                  % (a, b, ex / 1e4, eb / 1e4, res["D"][a][b] / 1e4, X[("x", a, b)][1] / 1e4, X[("b", a, b)][1] / 1e4, X[("c", a, b)]["da"]))
        lv, up, dn, sm = res["L"][a]
        print("%s: %s (Δ ≥ +0.05 %d/5 · Δ ≤ −0.05 %d/5 · |Δ| < 0.03 %d/5)" % (a, lv, up, dn, sm))
    print("판정: %s" % res["verdict"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
