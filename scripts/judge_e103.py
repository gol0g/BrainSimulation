#!/usr/bin/env python3
"""E103 판정 — research/experiments/E103.md 4절 구현.
  (인자 없음): 24런 요약 + 회귀가 다 모이기 전에는 수치 미출력.  --selftest: 합성 경계 검증."""
import os
import re
import statistics as st
import sys

EXP = "research/experiments"
W, TS = [5, 6, 7], [300, 301, 302, 303]
TWIN = {"T20": "logs/E100/rev_w%d_t%d.log", "T2": "logs/E101/W2-rev8_w%d_t%d.log"}


def summ(c, s, t):
    p = os.path.join(EXP, "logs/E103/%s_w%d_t%d_summary.log" % (c, s, t))
    if not os.path.exists(p):
        return None
    for ln in open(p, encoding="utf-8"):
        if ln.startswith("E103SUM "):
            return {k: float(v) for k, v in (x.split("=") for x in ln.split()[1:])}
    return None


def minc(p):
    p = os.path.join(EXP, p)
    if not os.path.exists(p):
        return None
    m = [l.strip() for l in open(p, encoding="utf-8") if l.startswith("=> MINCIRC")]
    return m[-1] if m else None


def most(xs):
    return len(xs) > 0 and sum(xs) / len(xs) >= 10 / 12 - 1e-9


def judge_cond(runs):
    v = [r for r in runs if r["RN_n"] >= 5 and r["PO_n"] >= 5]
    n = len(v)
    if n == 0:
        return {"n": 0, "a": "판정 불가", "b": "해당 없음", "c": "판정 불가", "ratio": float("nan")}
    leak = [r["RN_spkL"] > 0 and r["RN_dOld"] > 0 for r in v]
    a = "누수 확인" if most(leak) else ("누수 없음" if most([r["RN_dOld"] <= 0 for r in v]) else "보류")
    ratios = [r["cOld_rewNew"] / abs(r["cOld_pun"]) for r in v if r["cOld_pun"] != 0]
    ratio = st.median(ratios) if ratios else float("nan")
    b = "해당 없음"
    if a == "누수 확인":
        b = "주 요인급(≥0.5)" if ratio >= 0.5 else ("작다(<0.2)" if ratio < 0.2 else "중간")
    c = "역전 실패" if most([r["end_A"] > 0 for r in v]) else ("역전" if most([r["end_A"] <= 0 for r in v]) else "보류")
    return {"n": n, "a": a, "b": b, "c": c, "ratio": ratio}


def overall(j2, readout_all_reversed):
    if j2["a"] == "누수 확인" and j2["b"].startswith("주 요인급") and j2["c"] == "역전 실패":
        return "H032 지지(누수가 주 요인)"
    if j2["c"] == "역전" and readout_all_reversed:
        return "H032-readout(가중치·판독 근사 모두 역전인데 행동 실패)"
    if j2["c"] == "역전" or j2["b"].startswith("작다"):
        return "H032-asym(비대칭 축소 시 가중치 역전 또는 누수 작음)"
    return "보류"


def selftest():
    def mk(k_leak, ratio, k_endpos, n=12):
        return [{"RN_n": 10, "PO_n": 10, "RN_spkL": 3.0, "RN_dOld": (0.1 if i < k_leak else -0.1),
                 "cOld_rewNew": ratio, "cOld_pun": -1.0, "end_A": (1.0 if i < k_endpos else -1.0)} for i in range(n)]
    cases = [
        (mk(10, 0.5, 10), False, "H032 지지"), (mk(9, 0.5, 10), False, "보류"), (mk(10, 0.49, 10), False, "보류"),
        (mk(10, 0.5, 2), False, "H032-asym"), (mk(10, 0.1, 10), False, "H032-asym"), (mk(10, 0.6, 1), True, "H032-readout"),
        (mk(0, 0.5, 10), False, "보류"),
    ]
    ok = 0
    for runs, ro, exp in cases:
        got = overall(judge_cond(runs), ro)
        ok += got.startswith(exp)
        if not got.startswith(exp):
            print("실패", exp, "→", got, judge_cond(runs))
    j = judge_cond(mk(0, 0.5, 10)); ok += (j["a"] == "누수 없음")
    print("자체 검증 %d/%d 통과" % (ok, len(cases) + 1))
    return ok == len(cases) + 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    log = open(os.path.join(EXP, "E103.log"), encoding="utf-8").read() if os.path.exists(os.path.join(EXP, "E103.log")) else ""
    reg = re.search(r"\[회귀\] 평가 100%: (\d+)/(\d+)", log)
    D = {(c, s, t): summ(c, s, t) for c in ("T20", "T2") for s in W for t in TS}
    have = sum(v is not None for v in D.values())
    if have < 24 or not reg:
        print("[E103] 요약 %d/24, 회귀 %s — **판정 보류. 다 모일 때까지 수치 미출력.**" % (have, "완료" if reg else "미완"))
        sys.exit(0)
    print("[E103] 요약 24/24, 회귀 %s/%s" % reg.groups())
    for c in ("T20", "T2"):
        same = sum(minc("logs/E103/%s_w%d_t%d.log" % (c, s, t)) == minc(TWIN[c] % (s, t)) for s in W for t in TS)
        print("조작검증 %s: 쌍둥이 요약 줄 동일 %d/12%s" % (c, same, "" if same == 12 else "  ← 추적이 동역학을 바꿈 — 해석 보류"))
    big = sum(abs(D[("T2", s, t)]["PO_dOld"]) > abs(D[("T20", s, t)]["PO_dOld"]) for s in W for t in TS)
    print("조작검증 w_max: |처벌 1회 옛 연합 Δg| T2 > T20 %d/12%s" % (big, "" if big >= 10 else "  ← w_max 조작 확인 실패"))
    J = {}
    for c in ("T20", "T2"):
        runs = [D[(c, s, t)] for s in W for t in TS]
        J[c] = judge_cond(runs)
        for s in W:
            for t in TS:
                r = D[(c, s, t)]
                print("  %s w%d t%d: 새행동보상 n=%d 옛출력발화 %.1f(선택 %.1f) 옛연합Δg %+.4g 새연합Δg %+.4g | 옛행동처벌 Δg %+.4g | 누수/처벌 %.2f | 끝 A전용 %+.4g (시작 %+.4g) 공유포함 %+.4g" % (
                    c, s, t, r["RN_n"], r["RN_spkL"], r["RN_spkR"], r["RN_dOld"], r["RN_dNew"], r["PO_dOld"],
                    (r["cOld_rewNew"] / abs(r["cOld_pun"])) if r["cOld_pun"] else float("nan"), r["end_A"], r["start_A"], r["end_Aall"]))
        print("%s: (a) %s | (b) 누수/처벌 중앙값 %.2f → %s | (c) %s  (유효 %d)" % (c, J[c]["a"], J[c]["ratio"], J[c]["b"], J[c]["c"], J[c]["n"]))
    ro = most([D[("T2", s, t)]["end_Aall"] <= 0 for s in W for t in TS])
    print("종합(T2 기준): %s" % overall(J["T2"], ro))
