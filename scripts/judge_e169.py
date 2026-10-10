#!/usr/bin/env python3
"""E169 판정 — 보상 창 1처리(10 ms)로 호스트 동결 대체. 기준 logs/E169/criteria_fixed.txt. 10런이 다 모이기 전에는 수치를 출력하지 않는다.
팔 FW1 = E141 인자에서 보상 창 2 → 1, 동결 없음 + E161 형성 가중치. FFW1 = 같은 것 + 동결(--rw-apm-scale 0). 뇌 16~20, 반사 0, 500시행.
기준: e_F = 같은 뇌 E161 F(동결, 창 2). 부지표 기준: e_FNF = 같은 뇌 E166 FNF(동결 없음, 창 2). 1e-4 정수 산술.
판정 1: 대체 성공(H092) = q₁ = e_FW1/e_F ≥ 0.80(⇔ 10·E_FW1 ≤ 8·E_F) ≥ 4/5, 실패(H092-null) = q₁ ≤ 0.50(⇔ 2·E_FW1 ≥ E_F) ≥ 4/5, 그 밖 부분.
판정 2(부, 전제 e_FFW1 ≤ −0.10 5/5): 창 단축이 오염을 줄임 = c₁ − c₂ ≥ 0.20 ≥ 4/5(c₁ = e_FW1/e_FFW1, c₂ = e_FNF/e_F), 줄이지 않음 = |c₁ − c₂| < 0.10 ≥ 4/5, 그 밖 중간.
조작검증(하나라도 실패면 보류): 창 1 확인 — FFW1 보상 창 끝 흔적 잔차(r₁ = (11/12)^10) ≤ 1e-3, FW1 잔차 ≥ 0.05(동결 꺼짐), [사전] = E161 F ±0.002(10/10),
추적 500·도파민 전 ≤ 1e-3(10/10), 적재 줄 ≥ 2(10/10). 전제 e_F ≤ −0.10(5/5).
실행: python3 scripts/judge_e169.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
R1 = (1.0 - 1.0 / 12.0) ** 10


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
            "res1": float(np.abs(eend - R1 * eda).sum() / max(np.abs(eda).sum(), 1e-12))}


def load():
    X = {}
    for b in BRAINS:
        for a in ("FW1", "FFW1"):
            t = rd("logs", "E169", "%s_b%d.log" % (a, b))
            if lrn(t):
                X[(a, b)] = lrn(t)
            f = os.path.join(EXP, "traces", "E169", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                X[("s", a, b)] = stats(np.load(f)["rows"])
        for k, (e, nm) in (("F", ("E161", "F_b%d.log")), ("FNF", ("E166", "FNF_b%d.log"))):
            t = rd("logs", e, nm % b)
            if lrn(t):
                X[(k, b)] = lrn(t)
    return X


def judge(X):
    need = [(a, b) for a in ("FW1", "FFW1", "F", "FNF") for b in BRAINS] + [("s", a, b) for a in ("FW1", "FFW1") for b in BRAINS]
    miss = [k for k in need if k not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    e = {(a, b): X[(a, b)]["post"] - X[(a, b)]["pre"] for a in ("FW1", "FFW1", "F", "FNF") for b in BRAINS}
    m1 = sum(X[("s", "FFW1", b)]["res1"] <= 1e-3 for b in BRAINS) + sum(X[("s", "FW1", b)]["res1"] >= 0.05 for b in BRAINS)
    mp = sum(abs(X[(a, b)]["pre"] - X[("F", b)]["pre"]) <= 20 for a in ("FW1", "FFW1") for b in BRAINS)
    mt = sum(X[("s", a, b)]["n"] == 500 and X[("s", a, b)]["pre_ratio"] <= 1e-3 for a in ("FW1", "FFW1") for b in BRAINS)
    ml = sum(X[(a, b)]["ld"] >= 2 for a in ("FW1", "FFW1") for b in BRAINS)
    pc = sum(e[("F", b)] <= -1000 for b in BRAINS)
    ok = m1 == mp == mt == ml == 10 and pc == 5
    checks = ["[조작검증] 창 1·동결 확인(FFW1 잔차 ≤ 1e-3·FW1 ≥ 0.05) %d/10 · [사전] 재현 %d/10 · 추적 %d/10 · 적재 %d/10 · 전제 e_F ≤ −0.10 %d/5 %s"
              % (m1, mp, mt, ml, pc, "통과" if ok else "실패")]
    succ = [b for b in BRAINS if 10 * e[("FW1", b)] <= 8 * e[("F", b)]]
    fail = [b for b in BRAINS if 2 * e[("FW1", b)] >= e[("F", b)]]
    v1 = "대체 성공(H092)" if len(succ) >= 4 else ("실패(H092-null)" if len(fail) >= 4 else "부분")
    pre2 = sum(e[("FFW1", b)] <= -1000 for b in BRAINS) == 5
    red = none = []
    if pre2:
        # c1 − c2 = (E_FW1·E_F − E_FNF·E_FFW1) / (E_FFW1·E_F), 분모 > 0
        num = {b: e[("FW1", b)] * e[("F", b)] - e[("FNF", b)] * e[("FFW1", b)] for b in BRAINS}
        den = {b: e[("FFW1", b)] * e[("F", b)] for b in BRAINS}
        red = [b for b in BRAINS if 5 * num[b] >= den[b]]
        none = [b for b in BRAINS if 10 * abs(num[b]) < den[b]]
        v2 = "창 단축이 오염을 줄임" if len(red) >= 4 else ("줄이지 않음" if len(none) >= 4 else "중간")
    else:
        v2 = "판정 불가(전제 e_FFW1 ≤ −0.10 미충족)"
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    return checks, {"e": e, "succ": succ, "fail": fail, "red": red, "none": none, "v1": v1, "v2": v2, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    e = res["e"]
    for b in BRAINS:
        print("b%d FW1 e %+.4f q₁ %.3f 보상 %s 잔차 %.3f | FFW1 e %+.4f (동결 창 1 / 창 2 %.3f) 잔차 %.2e | 창 2: 동결 %+.4f 무동결 %+.4f | c₁ %.3f c₂ %.3f"
              % (b, e[("FW1", b)] / 1e4, e[("FW1", b)] / e[("F", b)], X[("FW1", b)]["rew"], X[("s", "FW1", b)]["res1"],
                 e[("FFW1", b)] / 1e4, e[("FFW1", b)] / e[("F", b)], X[("s", "FFW1", b)]["res1"], e[("F", b)] / 1e4, e[("FNF", b)] / 1e4,
                 (e[("FW1", b)] / e[("FFW1", b)]) if e[("FFW1", b)] else float("nan"), e[("FNF", b)] / e[("F", b)]))
    print("q₁ ≥ 0.80 %d/5 · q₁ ≤ 0.50 %d/5 | c₁ − c₂ ≥ 0.20 %d/5 · |c₁ − c₂| < 0.10 %d/5" % (len(res["succ"]), len(res["fail"]), len(res["red"]), len(res["none"])))
    print("판정 1: %s" % res["v1"])
    print("판정 2: %s" % res["v2"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
