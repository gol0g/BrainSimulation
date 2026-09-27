#!/usr/bin/env python3
"""E111 판정 — E111.md 4절. 6런 전에는 수치 미출력."""
import re
import sys

L = re.compile(r"^\s*B(\d+) b(\d): => .*?변조폭 변화 ([-+0-9.]+).*?E111SUM (.*)$")
D = {}
try:
    for ln in open("research/experiments/E111.log", encoding="utf-8"):
        m = L.match(ln)
        if m:
            bi, b, d, rest = m.groups()
            kv = dict(x.split("=", 1) for x in rest.split())
            kv["dmod"] = d
            D[(int(bi), int(b))] = kv
except FileNotFoundError:
    pass
if len(D) < 6:
    print("[E111] %d/6런 — **판정 보류. 다 모일 때까지 수치 미출력.**" % len(D)); sys.exit(0)
f = lambda k, b, key: float(D[(k, b)][key])
# 측정 도구 확인: CSV 에서 e 가 실제로 읽혔는가(1차 실행은 전부 0·nan 이었다)
import csv, math, os
bad = []
for k in (25, 8):
    for b in range(3):
        p = "research/experiments/logs/E111/B%d_b%d.csv.log" % (k, b)
        rows = list(csv.DictReader(open(p, encoding="utf-8"))) if os.path.exists(p) else []
        zero = sum(all(float(r[c]) == 0.0 for c in ("e_ll", "e_lr", "e_rl", "e_rr")) for r in rows)
        nang = sum(any(math.isnan(float(r[c])) for c in ("g_ll", "g_lr", "g_rl", "g_rr")) for r in rows)
        if not rows or zero > 0.1 * len(rows) or nang > 0:
            bad.append("B%d b%d (행 %d, e 전부 0인 행 %d, g nan 행 %d)" % (k, b, len(rows), zero, nang))
if bad:
    print("**측정 도구 실패 — 판정 무효**: " + "; ".join(bad)); sys.exit(0)
print("측정 도구 확인: 6런 모두 e 판독·g 유한")
print("조작검증: 추적 불변 — B25 b0 변조폭 변화 %s (E110 eta0.15 학습 b0 = +0.2679): %s" % (D[(25, 0)]["dmod"], D[(25, 0)]["dmod"] == "+0.2679"))
print("조작검증: 탐색 |v| B8 < B25 %d/3" % sum(f(8, b, "explore_absv") < f(25, b, "explore_absv") for b in range(3)))
print("조작검증: 탐색·정답 사건 ≥20: %s" % all(f(k, b, "expl_correct_n") >= 20 for k in (25, 8) for b in range(3)))
for k in (25, 8):
    for b in range(3):
        d = D[(k, b)]
        print("  B%d b%d: 변조폭 변화 %s | 탐색·정답 n=%s e>0 %s 평균 %s | 탐욕·오답 n=%s e>0 %s | 탐색·오답 e>0 %s | |v|탐색 %s | g %s %s %s %s" % (
            k, b, d["dmod"], d["expl_correct_n"], d["expl_correct_pos"], d["expl_correct_mean"], d["greedy_wrong_n"], d["greedy_wrong_pos"],
            d["expl_wrong_pos"], d["explore_absv"], d["g_ll"], d["g_lr"], d["g_rl"], d["g_rr"]))
nb = lambda cond: sum(cond(b) for b in range(3)) >= 2
if nb(lambda b: f(25, b, "expl_correct_pos") <= 0.2):
    v = "H038 지지(탐색·정답 흔적 음수)"
elif nb(lambda b: f(25, b, "expl_correct_pos") >= 0.8 and f(25, b, "greedy_wrong_pos") >= 0.8):
    v = "H038-normal"
elif nb(lambda b: f(25, b, "expl_correct_pos") >= 0.8 and f(25, b, "greedy_wrong_pos") <= 0.2):
    v = "H038-greedy"
else:
    v = "보류"
dep = sum(f(8, b, "expl_correct_pos") > f(25, b, "expl_correct_pos") for b in range(3))
print("판정: %s | 편향 의존(B8 > B25 양수 비율) %d/3%s" % (v, dep, " → 구동 세기가 흔적 부호를 정함" if dep >= 2 else ""))
