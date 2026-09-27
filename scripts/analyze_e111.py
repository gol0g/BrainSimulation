#!/usr/bin/env python3
"""E111 분석: --trace-kc-motor CSV → 사건별 KC→motor 자격흔적 부호.
활성 KC 쪽 = good 쪽(좌 good → kc_l). 실행 행동 = v 부호(K52: v>0 = 오른쪽 motor). 정답 = good 반대.
사건: (탐색·정답) 새 연합 흔적 e(활성KC→실행motor), (탐욕·오답=반사) 옛 연합 흔적 e(활성KC→반사motor)."""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
out = {}
def grp(k, m, r): return float(r["e_%s%s" % (k, m)])
cats = {"expl_correct": [], "greedy_wrong": [], "expl_wrong": []}
gchg = {}
for r in rows:
    k = "l" if r["good_side"] == "left" else "r"
    v = float(r["v"]); m = "r" if v > 0 else "l"      # 실행 motor
    ex, co = r["explore"] == "1", r["correct"] == "1"
    e_exec = grp(k, m, r)
    if ex and co: cats["expl_correct"].append(e_exec)
    elif (not ex) and (not co): cats["greedy_wrong"].append(e_exec)
    elif ex and not co: cats["expl_wrong"].append(e_exec)
for c, xs in cats.items():
    n = len(xs); pos = sum(x > 0 for x in xs)
    out[c + "_n"] = n; out[c + "_pos"] = (pos / n) if n else float("nan")
    out[c + "_mean"] = (sum(xs) / n) if n else float("nan")
ex_v = [abs(float(r["v"])) for r in rows if r["explore"] == "1"]
out["explore_absv"] = (sum(ex_v) / len(ex_v)) if ex_v else float("nan")
first, last = rows[0], rows[-1]
for g in ("ll", "lr", "rl", "rr"):
    out["g_" + g] = "%s→%s" % (first["g_" + g], last["g_" + g])
print("E111SUM " + " ".join("%s=%s" % (k, ("%.4g" % v) if isinstance(v, float) else v) for k, v in out.items()))
