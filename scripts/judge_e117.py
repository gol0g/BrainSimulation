#!/usr/bin/env python3
"""E117 판정 — E117.md 4절. 5런 전에는 수치 미출력."""
import re
import sys

L = re.compile(r"^\s*kcsets b(\d): => (.*)$")
SEG = re.compile(r"(좌전용|우전용|공유|무반응) n=(\d+) →L ([-+0-9.na]+) →R ([-+0-9.na]+)")
D = {}
try:
    for ln in open("research/experiments/E117.log", encoding="utf-8"):
        m = L.match(ln)
        if not m: continue
        b = int(m.group(1)); parts = m.group(2).split("KCSETS kc_")[1:]
        d = {}
        for p in parts:
            side = p[0]; d[side] = {c: (int(n), float(l), float(r)) for c, n, l, r in SEG.findall(p)}
        if len(d) == 2: D[b] = d
except FileNotFoundError:
    pass
if len(D) < 5:
    print("[E117] %d/5 — **판정 보류. 다 모일 때까지 수치 미출력.**" % len(D)); sys.exit(0)
lat = sum(D[b]["l"]["좌전용"][0] > D[b]["l"]["우전용"][0] and D[b]["r"]["우전용"][0] > D[b]["r"]["좌전용"][0] for b in D)
print("조작검증: 편측성(kc_l 좌전용>우전용, kc_r 반대) %d/5%s" % (lat, "" if lat >= 4 else "  ← 집합 무의미"))
print("조작검증: 무반응 KC |Δg| < 1: %d/5" % sum(all(abs(D[b][s]["무반응"][i]) < 1 for s in "lr" for i in (1, 2)) for b in D))
h = asym = 0
for b in sorted(D):
    d = D[b]
    # 같은 쪽: kc_l→L, kc_r→R ; 교차: kc_l→R, kc_r→L
    S_ex = (d["l"]["좌전용"][1] + d["r"]["우전용"][2]) / 2; C_ex = (d["l"]["좌전용"][2] + d["r"]["우전용"][1]) / 2
    S_sh = (d["l"]["공유"][1] + d["r"]["공유"][2]) / 2; C_sh = (d["l"]["공유"][2] + d["r"]["공유"][1]) / 2
    S_op = (d["l"]["우전용"][1] + d["r"]["좌전용"][2]) / 2; C_op = (d["l"]["우전용"][2] + d["r"]["좌전용"][1]) / 2
    h += (S_ex <= 5 and S_sh >= 20); asym += S_ex >= 15
    print("  b%d: 전용 같은쪽 %+.2f 교차 %+.2f | 공유 같은쪽 %+.2f 교차 %+.2f | 반대쪽전용 같은쪽 %+.2f 교차 %+.2f | n(kc_l 좌/우/공/무)=%s" % (
        b, S_ex, C_ex, S_sh, C_sh, S_op, C_op, "/".join(str(d["l"][c][0]) for c in ("좌전용", "우전용", "공유", "무반응"))))
v = "H043 지지(공유 KC)" if h >= 4 else ("H043-asym 지지(경계 비대칭)" if asym >= 4 else "보류")
print("판정: %s (H043 조건 %d/5, asym 조건 %d/5)" % (v, h, asym))
