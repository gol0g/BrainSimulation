#!/usr/bin/env python3
"""E145 판정 — 기준 logs/E145/criteria_fixed.txt(실행 전 고정). 25줄이 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄 "  e145 b10 A_all: => mod +0.1026". Δs = A_sameB − A_all, Δc = A_crossB − A_all, Δd = A_d1B − A_all, ΔB = B_all − A_all.
"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
MODES = ("A_all", "B_all", "A_sameB", "A_crossB", "A_d1B")
F1500_POST = {10: 0.1026, 11: 0.1141, 12: 0.0953, 13: 0.0833, 14: 0.1266}
E144_POST = {10: 0.1735, 11: 0.1612, 12: 0.1464, 13: 0.1333, 14: 0.1735}
TL = re.compile(r"^\s*e145 b(\d+) (A_all|B_all|A_sameB|A_crossB|A_d1B): => mod ([-+0-9.]+)")
GROUPS = (("same", "H068-same — 같은 쪽 KC→motor 집단이 차이를 낸다"), ("cross", "H068-cross — 교차 KC→motor 집단이 차이를 낸다"),
          ("d1", "H068-d1 — D1 경로가 차이를 낸다"))


def judge(M):
    miss = [(b, m) for b in BRAINS for m in MODES if (b, m) not in M]
    if miss:
        return ["[측정 확인] 결측 %d줄 — **판정 보류, 수치 미출력**" % len(miss)], None
    v1 = sum(abs(M[(b, "A_all")] - F1500_POST[b]) <= 0.002 + 1e-9 for b in BRAINS)
    v2 = sum(abs(M[(b, "B_all")] - E144_POST[b]) <= 0.002 + 1e-9 for b in BRAINS)
    dB = {b: round(M[(b, "B_all")] - M[(b, "A_all")], 6) for b in BRAINS}
    v3 = sum(dB[b] >= 0.03 - 1e-9 for b in BRAINS)
    ok = v1 == v2 == v3 == 5
    checks = ["[측정 검증] V1 A_all = F1500 사후 ±0.002 %d/5 · V2 B_all = E144 사후 ±0.002 %d/5 · V3 ΔB ≥ 0.03 %d/5 %s"
              % (v1, v2, v3, "통과" if ok else "실패")]
    D = {"same": {}, "cross": {}, "d1": {}}
    for b in BRAINS:
        D["same"][b] = round(M[(b, "A_sameB")] - M[(b, "A_all")], 6)
        D["cross"][b] = round(M[(b, "A_crossB")] - M[(b, "A_all")], 6)
        D["d1"][b] = round(M[(b, "A_d1B")] - M[(b, "A_all")], 6)
    frac = {g: {b: (D[g][b] / dB[b] if dB[b] else float("nan")) for b in BRAINS} for g in D}
    S = [g for g, _ in GROUPS if sum(frac[g][b] >= 0.6 - 1e-9 for b in BRAINS) >= 4]
    resid = {b: round(D["same"][b] + D["cross"][b] + D["d1"][b] - dB[b], 6) for b in BRAINS}
    nonadd = sum(abs(resid[b]) >= 0.4 * abs(dB[b]) - 1e-9 for b in BRAINS)
    if not ok:
        verdict = "보류(측정 검증 실패)"
    elif len(S) == 1:
        verdict = dict(GROUPS)[S[0]]
    elif len(S) >= 2:
        verdict = "H068-multi — 복수 집단 %s" % "·".join(S)
    elif nonadd >= 4:
        verdict = "H068-int — 비가산(집단 상호작용)"
    else:
        verdict = "보류"
    return checks, {"dB": dB, "D": D, "frac": frac, "S": S, "resid": resid, "nonadd": nonadd, "ok": ok, "verdict": verdict}


def report(checks, res, M=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d A_all %+.4f B_all %+.4f ΔB %+.4f | Δ같은쪽 %+.4f(%.2f) Δ교차 %+.4f(%.2f) ΔD1 %+.4f(%.2f) | 잔차 %+.4f"
              % (b, M[(b, "A_all")], M[(b, "B_all")], res["dB"][b], res["D"]["same"][b], res["frac"]["same"][b],
                 res["D"]["cross"][b], res["frac"]["cross"][b], res["D"]["d1"][b], res["frac"]["d1"][b], res["resid"][b]))
    print("몫 ≥ 0.6 인 뇌 수: 같은쪽 %d · 교차 %d · D1 %d | 비가산(|잔차| ≥ 0.4ΔB) %d/5"
          % tuple([sum(res["frac"][g][b] >= 0.6 - 1e-9 for b in BRAINS) for g in ("same", "cross", "d1")] + [res["nonadd"]]))
    print("판정: %s" % res["verdict"])


def load():
    M = {}
    try:
        for ln in open(os.path.join(EXP, "E145.log"), encoding="utf-8", errors="replace"):
            m = TL.match(ln)
            if m:
                M[(int(m.group(1)), m.group(2))] = float(m.group(3))
    except FileNotFoundError:
        pass
    return M


if __name__ == "__main__":
    M = load()
    c, r = judge(M)
    report(c, r, M)
    sys.exit(0)
