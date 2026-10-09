#!/usr/bin/env python3
"""E160 보정 선택 — 규칙 logs/E160/criteria_fixed.txt(2026-10-09 15:01:00 고정).
격자 η ∈ {0.005, 0.02, 0.08} × β ∈ {0.1, 0.3, 1.0, 3.0}. 칸 조건: 양쪽 sel_med ≥ 0.80, 양쪽 sum_med ∈ [0.80, 1.25].
조건 칸 중 max(|ln sum_med_l|, |ln sum_med_r|) 최소(동점이면 작은 η, 그다음 작은 β). 없으면 보정 실패.
경로 확인: 칸마다 4집단 Σ|Δg| > 0(Oja 가 실제로 변함) — 아니면 그 칸 제외(사유 출력).
실행: python3 scripts/e160_pick.py (저장소 루트에서) → logs/E160/oja_pick.txt"""
import math
import os
import re
import sys

EXP = "research/experiments"
ETAS = ("0.005", "0.02", "0.08")
BETAS = ("0.1", "0.3", "1.0", "3.0")
SIDE = re.compile(r"side=([lr]) fired=(\d+) sel_med0=([0-9.na]+) sel_med=([0-9.na]+) frac09=([0-9.na]+) goodfrac=([0-9.na]+) "
                  r"sum_med=([0-9.na]+) sum_q10=([0-9.na]+) sum_q90=([0-9.na]+) dg_good=([0-9.na]+) dg_bad=([0-9.na]+)")


def parse(txt):
    line = next((ln for ln in txt.splitlines() if ln.startswith("=> KCDEVOJA ")), None)
    if line is None:
        return None
    d = {}
    for m in SIDE.finditer(line):
        d[m.group(1)] = {"fired": int(m.group(2)), "sel_med": float(m.group(4)), "sum_med": float(m.group(7)),
                         "dg_good": float(m.group(10)), "dg_bad": float(m.group(11))}
    return d if set(d) == {"l", "r"} else None


def select(rows):
    """rows: {(η, β): parse 결과 또는 None} → ((η, β) 또는 None, 사유 목록)"""
    ok, why = {}, []
    for e in ETAS:
        for b in BETAS:
            r = rows.get((e, b))
            if r is None:
                why.append("η%s β%s 결측·파싱 실패" % (e, b))
                continue
            if any(r[s]["dg_good"] <= 0 or r[s]["dg_bad"] <= 0 for s in "lr"):
                why.append("η%s β%s 제외: Oja 변화 없음" % (e, b))
                continue
            sel_ok = all(r[s]["sel_med"] >= 0.80 for s in "lr")
            sum_ok = all(0.80 <= r[s]["sum_med"] <= 1.25 for s in "lr")
            if sel_ok and sum_ok:
                ok[(e, b)] = max(abs(math.log(r[s]["sum_med"])) for s in "lr")
            else:
                why.append("η%s β%s 조건 밖(sel_med %.3f/%.3f, sum_med %.3f/%.3f)" % (e, b, r["l"]["sel_med"], r["r"]["sel_med"], r["l"]["sum_med"], r["r"]["sum_med"]))
    if not ok:
        why.append("조건 칸 없음: 보정 실패")
        return None, why
    best = min(ok, key=lambda k: (round(ok[k], 12), float(k[0]), float(k[1])))
    return best, why


def main():
    rows = {}
    for e in ETAS:
        for b in BETAS:
            f = os.path.join(EXP, "logs", "E160", "calib", "oja_e%s_b%s_b15.log" % (e, b))
            rows[(e, b)] = parse(open(f, encoding="utf-8", errors="replace").read()) if os.path.exists(f) else None
    for e in ETAS:
        for b in BETAS:
            r = rows[(e, b)]
            if r:
                print("η%-5s β%-4s sel_med %.3f/%.3f sum_med %.3f/%.3f fired %d/%d Σ|Δg| %.0f/%.0f/%.0f/%.0f" % (
                    e, b, r["l"]["sel_med"], r["r"]["sel_med"], r["l"]["sum_med"], r["r"]["sum_med"], r["l"]["fired"], r["r"]["fired"],
                    r["l"]["dg_good"], r["l"]["dg_bad"], r["r"]["dg_good"], r["r"]["dg_bad"]))
    best, why = select(rows)
    for y in why:
        print("  " + y)
    if best is None:
        line = "보정 실패"
    else:
        r = rows[best]
        line = "eta=%s beta=%s (sel_med %.3f/%.3f, sum_med %.3f/%.3f)" % (best[0], best[1], r["l"]["sel_med"], r["r"]["sel_med"], r["l"]["sum_med"], r["r"]["sum_med"])
    print("=> " + line)
    open(os.path.join(EXP, "logs", "E160", "oja_pick.txt"), "w", encoding="utf-8").write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
