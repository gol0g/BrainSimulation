#!/usr/bin/env python3
"""E104 판정 — research/experiments/E104.md 4절 구현. 24런 요약 + 회귀 전에는 수치 미출력. --selftest."""
import os
import re
import statistics as st
import sys

EXP = "research/experiments"
W, TS = [5, 6, 7], [300, 301, 302, 303]
TWIN = {"F400": "logs/E103/T2_w%d_t%d.log", "F800": "logs/E101/W2-L8_w%d_t%d.log"}
FIRST = re.compile(r"first=([0-9.]+)")
EV = re.compile(r"\*\*eval=([0-9.]+)\*\*")


def summ(c, s, t):
    p = os.path.join(EXP, "logs/E104/%s_w%d_t%d_summary.log" % (c, s, t))
    if not os.path.exists(p):
        return None
    for ln in open(p, encoding="utf-8"):
        if ln.startswith("E104SUM "):
            d = {}
            for x in ln.split()[1:]:
                k, v = x.split("=", 1)
                d[k] = v if k.startswith("traj_") else float(v)
            return d
    return None


def minc(p, rx):
    p = os.path.join(EXP, p)
    if not os.path.exists(p):
        return None
    m = [l for l in open(p, encoding="utf-8") if l.startswith("=> MINCIRC")]
    return float(rx.search(m[-1]).group(1)) if m else None


def decide(runs, evals):
    both = [r["cross_A"] >= 0 and r["cross_B"] >= 0 for r in runs]
    nb = sum(both)
    a = "H033 지지(느리지만 가능)" if nb >= 10 else ("불가" if nb <= 2 else "보류")
    ratios = []
    for r in runs:
        for tg in ("A", "B"):
            if r["cross_" + tg] < 0 and r["rate_early_" + tg] > 0:
                ratios.append(r["rate_late_" + tg] / r["rate_early_" + tg])
    med = st.median(ratios) if ratios else float("nan")
    b = "해당 없음(교차 못 한 자극 없음)" if not ratios else ("H033-brake 지지" if (med < 0.3 and not a.startswith("H033 지지")) else ("제동 아님(더 느림)" if med >= 0.3 else "제동 비율 낮음(단 (a) 지지)"))
    okb = [e >= 90.0 for e, bth in zip(evals, both) if bth]
    c = (sum(okb) / len(okb)) if okb else float("nan")
    return a, b, med, nb, c


def selftest():
    def mk(ncross, late):
        return [{"cross_A": (100 if i < ncross else -1), "cross_B": (200 if i < ncross else -1),
                 "rate_early_A": 10.0, "rate_late_A": late, "rate_early_B": 10.0, "rate_late_B": late} for i in range(12)]
    cases = [((mk(10, 5), [100] * 12), "H033 지지", None), ((mk(9, 5), [100] * 12), "보류", "제동 아님"),
             ((mk(2, 2.9), [0] * 12), "불가", "H033-brake"), ((mk(2, 3.0), [0] * 12), "불가", "제동 아님"),
             ((mk(3, 1), [0] * 12), "보류", "H033-brake")]
    ok = 0
    for (runs, ev), ea, eb in cases:
        a, b, _, _, _ = decide(runs, ev)
        g = a.startswith(ea) and (eb is None or b.startswith(eb))
        ok += g
        if not g:
            print("실패", ea, eb, "→", a, b)
    _, _, _, _, c = decide(mk(12, 5), [100] * 9 + [50] * 3); ok += abs(c - 0.75) < 1e-9
    print("자체 검증 %d/%d 통과" % (ok, len(cases) + 1))
    return ok == len(cases) + 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    log = open(os.path.join(EXP, "E104.log"), encoding="utf-8").read() if os.path.exists(os.path.join(EXP, "E104.log")) else ""
    reg = re.search(r"\[회귀\] 평가 100%: (\d+)/(\d+)", log)
    D = {(c, s, t): summ(c, s, t) for c in ("F400", "F800") for s in W for t in TS}
    have = sum(v is not None for v in D.values())
    if have < 24 or not reg:
        print("[E104] 요약 %d/24, 회귀 %s — **판정 보류. 다 모일 때까지 수치 미출력.**" % (have, "완료" if reg else "미완"))
        sys.exit(0)
    print("[E104] 요약 24/24, 회귀 %s/%s" % reg.groups())
    for c in ("F400", "F800"):
        same = sum(minc("logs/E104/%s_w%d_t%d.log" % (c, s, t), FIRST) == minc(TWIN[c] % (s, t), FIRST) for s in W for t in TS)
        early = sum(D[(c, s, t)]["rate_early_A"] > 0 and D[(c, s, t)]["rate_early_B"] > 0 for s in W for t in TS)
        print("조작검증 %s: 첫 구간 쌍둥이 동일 %d/12 | 반전 초반 두 자극 감소 %d/12%s" % (c, same, early, "" if same == 12 and early >= 10 else "  ← 확인 실패"))
    res = {}
    for c in ("F400", "F800"):
        runs = [D[(c, s, t)] for s in W for t in TS]
        evs = [minc("logs/E104/%s_w%d_t%d.log" % (c, s, t), EV) for s in W for t in TS]
        for (s, t), r, e in zip([(s, t) for s in W for t in TS], runs, evs):
            print("  %s w%d t%d: A %s (교차 %d) | B %s (교차 %d) | 속도 초/후 A %.1f/%.1f B %.1f/%.1f | 옛선호 %.2f→%.2f | 평가 %.0f" % (
                c, s, t, r["traj_A"], r["cross_A"], r["traj_B"], r["cross_B"], r["rate_early_A"], r["rate_late_A"],
                r["rate_early_B"], r["rate_late_B"], r["oldpref_early"], r["oldpref_late"], e))
        res[c] = decide(runs, evs)
        a, b, med, nb, cc = res[c]
        print("%s: (a) %s — 두 자극 교차 %d/12 | (b) %s (미교차 속도비 중앙값 %.2f) | (c) 교차 런 평가≥90%% 비율 %.2f%s" % (
            c, a, nb, b, med, cc, "  ← 가중치 교차가 행동 반전을 보장 안 함" if cc == cc and cc < 0.8 else ""))
    rs = [D[("F800", s, t)]["start_A"] / D[("F400", s, t)]["start_A"] for s in W for t in TS if D[("F400", s, t)]["start_A"]]
    print("(d) 반전 시작 강도 F800/F400 (A) 중앙값 %.2f" % st.median(rs))
    print("판정(F400 기준): %s / %s" % (res["F400"][0], res["F400"][1]))
