#!/usr/bin/env python3
"""E102 판정 — research/experiments/E102.md 4절 사전기준 구현.
  (인자 없음): E102.log + logs/E102/*_summary.log + E100 rev 로그. **12런 + 회귀가 다 모이기 전에는 수치 미출력.**
  --selftest : 합성 요약으로 (a)(b) 경계 검증."""
import os
import re
import sys

EXP = "research/experiments"
W, TS = [5, 6, 7], [300, 301, 302, 303]
MINC = re.compile(r"=> MINCIRC (.*)$")


def summary(s, t):
    p = os.path.join(EXP, "logs/E102/trace_w%d_t%d_summary.log" % (s, t))
    if not os.path.exists(p):
        return None
    for ln in open(p, encoding="utf-8"):
        if ln.startswith("E102SUM "):
            return {k: (float(v) if v not in ("nan",) else float("nan")) for k, v in (x.split("=") for x in ln.split()[1:])}
    return None


def minc(path):
    if not os.path.exists(path):
        return None
    m = [MINC.search(l) for l in open(path, encoding="utf-8") if l.startswith("=> MINCIRC")]
    return m[-1].group(1).strip() if m else None


def decide(runs):
    """runs: [{'PA_n','PA_dg','DA'}] → (a, b)"""
    v = [r for r in runs if r["PA_n"] >= 5 and r["PA_dg"] == r["PA_dg"]]
    n = len(v)
    if n == 0:
        return "판정 불가(유효 런 0)", "해당 없음", 0
    pos = sum(r["PA_dg"] >= 0 for r in v)
    if pos / n >= 10 / 12:
        a = "H031 지지(처벌이 옛 연합을 깎지 못함)"
    elif (n - pos) / n >= 10 / 12:
        a = "처벌 정상(옛 연합을 깎음)"
    else:
        a = "보류"
    b = "해당 없음"
    if a.startswith("처벌 정상"):
        neg = sum(r["DA"] < 0 for r in v)
        b = "H031-normal(가중치는 새 방향)" if neg / n >= 10 / 12 else ("H031-slow(순 균형이 옛 쪽)" if (n - neg) / n >= 10 / 12 else "보류")
    return a, b, n


def selftest():
    mk = lambda npos, nda_neg: [{"PA_n": 10, "PA_dg": (0.1 if i < npos else -0.1), "DA": (-1 if i < nda_neg else 1)} for i in range(12)]
    cases = [(mk(10, 0), "H031 지지", None), (mk(9, 0), "보류", None), (mk(2, 0), "처벌 정상", "H031-slow"),
             (mk(3, 0), "보류", None), (mk(0, 10), "처벌 정상", "H031-normal"), (mk(0, 9), "처벌 정상", "보류")]
    ok = 0
    for runs, ea, eb in cases:
        a, b, _ = decide(runs)
        good = a.startswith(ea) and (eb is None or b.startswith(eb))
        ok += good
        if not good:
            print("실패", ea, eb, "→", a, b)
    # 사건 부족 런 제외
    r = mk(10, 0); r[0]["PA_n"] = 4
    a, _, n = decide(r); ok += (n == 11)
    print("자체 검증 %d/%d 통과" % (ok, len(cases) + 1))
    return ok == len(cases) + 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    log = open(os.path.join(EXP, "E102.log"), encoding="utf-8").read() if os.path.exists(os.path.join(EXP, "E102.log")) else ""
    reg = re.search(r"\[회귀\] 평가 100%: (\d+)/(\d+)", log)
    S = {(s, t): summary(s, t) for s in W for t in TS}
    have = sum(v is not None for v in S.values())
    if have < 12 or not reg:
        print("[E102] 추적 %d/12, 회귀 %s — **판정 보류. 다 모일 때까지 수치 미출력.**" % (have, "완료" if reg else "미완"))
        sys.exit(0)
    print("[E102] 추적 12/12, 회귀 %s/%s" % reg.groups())
    same = [(s, t) for s in W for t in TS
            if minc(os.path.join(EXP, "logs/E102/trace_w%d_t%d.log" % (s, t))) == minc(os.path.join(EXP, "logs/E100/rev_w%d_t%d.log" % (s, t)))]
    print("조작검증: 추적 런 = E100 rev 요약 줄 동일 %d/12%s" % (len(same), "" if len(same) == 12 else "  ← 추적이 동역학을 바꿈 — 해석 보류"))
    acq = [S[k]["acqAL_dg"] for k in S]
    print("측정 도구 확인: 획득 구간 A·L·보상 Δg(A전용→L) 양수 %d/12 (평균 %.4g)" % (sum(x > 0 for x in acq), sum(acq) / 12))
    for k in sorted(S):
        r = S[k]
        print("  w%d t%d: 처벌 P_A n=%d Δg=%+.4g (증가KC %.0f%%) e=%+.4g | 보상 R_A n=%d Δg=%+.4g | 처벌 때 새연합 %+.4g | D_A=%+.4g (시작 %+.4g) | P_B Δg=%+.4g R_B Δg=%+.4g" % (
            k[0], k[1], r["PA_n"], r["PA_dg"], 100 * r["PA_fpos"], r["PA_e_mean"], r["RA_n"], r["RA_dg"], r["PA_new_dg"],
            r["DA"], r["DA_start"], r["PB_dg"], r["RB_dg"]))
    a, b, n = decide(list(S.values()))
    print("(a) 처벌 효과: %s (유효 런 %d)" % (a, n))
    print("(b) 순 균형: %s" % b)
    c = {k: sum(S[x]["cAL_" + k] for x in S) / 12 for k in ("all", "A_L_pun", "A_R_rew", "B_any")}
    print("(c) 옛 연합 A전용→L 반전 구간 기여 평균: 전체 %+.4g = A·L·처벌 %+.4g + A·R·보상 %+.4g + B시행 %+.4g + 기타" % (c["all"], c["A_L_pun"], c["A_R_rew"], c["B_any"]))
