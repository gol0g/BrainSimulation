#!/usr/bin/env python3
"""E108 판정 — E108.md 4절. 30런 전에는 수치 미출력."""
import re
import sys

L = re.compile(r"^\s*(S\d) b(\d): => CALIB d1_left=([-+0-9.]+) d1_right=([-+0-9.]+) motor_left=([-+0-9.]+) motor_right=([-+0-9.]+) good_left=([-+0-9.]+)")
PRE = {0: (-0.0027, 0.0238), 1: (-0.0082, 0.0221), 2: (-0.0117, 0.0190), 3: (-0.0273, 0.0068), 4: (-0.0152, 0.0179)}
SET = {"S0": (20, -100), "S1": (40, -100), "S2": (80, -100), "S3": (20, -50), "S4": (40, -50), "S5": (80, -50)}
D = {}
try:
    for ln in open("research/experiments/E108.log", encoding="utf-8"):
        m = L.match(ln)
        if m:
            s, b, *v = m.groups(); D[(s, int(b))] = [float(x) for x in v]
except FileNotFoundError:
    pass
if len(D) < 30:
    print("[E108] %d/30런 — **판정 보류. 다 모일 때까지 수치 미출력.**" % len(D)); sys.exit(0)
ratio = lambda v: (v[1] - v[0]) / (2 * abs(v[4]))
rep = sum(abs(D[("S0", b)][0] - PRE[b][0]) < 1e-4 and abs(D[("S0", b)][1] - PRE[b][1]) < 1e-4 for b in range(5))
print("조작검증: S0 재현(사전 관측과 동일) %d/5" % rep)
for s in ("S1", "S2"):
    up = sum((D[(s, b)][1] - D[(s, b)][0]) > (D[("S0", b)][1] - D[("S0", b)][0]) for b in range(5))
    print("조작검증: %s 권한 > S0 %d/5%s" % (s, up, "" if up >= 4 else "  ← 설정 조작 무효 의심"))
tun = []
for s in SET:
    rs = [ratio(D[(s, b)]) for b in range(5)]
    refl = [D[(s, b)][4] / D[("S0", b)][4] for b in range(5)]
    mot = [(D[(s, b)][3] - D[(s, b)][2]) for b in range(5)]
    ok_refl = all(0.9 <= x <= 1.1 for x in refl)
    n1 = sum(r >= 1.0 for r in rs)
    print("  %s (d1→direct %d, direct억제 %d): 비 %s | 반사 S0대비 %s | motor편향차 %s" % (
        s, SET[s][0], SET[s][1], " ".join("%.3f" % r for r in rs), " ".join("%.2f" % x for x in refl), " ".join("%.3f" % x for x in mot)))
    if n1 >= 4 and ok_refl:
        tun.append(s)
s0 = sum(ratio(D[("S0", b)]) < 0.2 for b in range(5))
allbelow = all(sum(ratio(D[(s, b)]) < 1.0 for b in range(5)) >= 4 for s in SET)
print("H036(현 설정 권한 부족): S0 비<0.2 %d/5 → %s" % (s0, "지지" if s0 >= 4 else "미지지"))
if tun:
    print("판정: **H036-tunable 지지** — 비≥1(4/5 이상)·반사 불변 설정: %s" % ", ".join(tun))
elif allbelow:
    print("판정: **H036-none 지지** — 모든 설정에서 비<1 (뇌 ≥4/5) → 기저핵 경로로는 반사 역전 불가, 과제·배선 재설계")
else:
    print("판정: 보류 — 비 0.5~1.0 설정 존재 또는 반사 변화, 더 큰 스윕 필요")
