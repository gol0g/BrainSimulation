#!/usr/bin/env python3
"""E113 판정 — E113.md 4절. 5 학습 + 40 분해 전에는 수치 미출력. E112 검토 결함 반영:
허용오차 재현(±0.0005), pushed 확인, 성분 전부 보고(마지막 일치 채택 금지), 제외해도 기준 수 유지."""
import os
import re
import sys

LOGD = "research/experiments/logs/E113"
TR = re.compile(r"^\s*train b(\d): => ([-+0-9.]+) ([-+0-9.]+)")
DC = re.compile(r"^\s*dec b(\d) (\w+): => DECOMP mode=(\w+) mod=([-+0-9.]+) .*?pushed=(\d+)")
KP = re.compile(r"^=> KCPRE side=(\w) top5%: 반사\(\w+\) ([-+0-9.]+) 교차\(\w+\) ([-+0-9.]+) 차 ([-+0-9.]+) \| 하위50%: 반사 ([-+0-9.]+) 교차 ([-+0-9.]+)")
NR = re.compile(r"^=> NEURON group=(\w+) r=([-+0-9.na]+)")
BR = (5, 6, 7, 8, 9)
MODES = ("none", "all", "kc_only", "d1_only", "kc_shuffle", "kc_uniform", "kc_cm")
T, D = {}, {}
try:
    for ln in open("research/experiments/E113.log", encoding="utf-8"):
        m = TR.match(ln)
        if m: T[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        m = DC.match(ln)
        if m: D[(int(m.group(1)), m.group(2))] = (float(m.group(4)), int(m.group(5)))
except FileNotFoundError:
    pass
NE = {}
for b in BR:
    p = os.path.join(LOGD, "dec_b%d_neuron.log" % b)
    if os.path.exists(p):
        kp, nr = {}, {}
        for ln in open(p, encoding="utf-8"):
            m = KP.match(ln)
            if m: kp[m.group(1)] = tuple(float(x) for x in m.groups()[1:])
            m = NR.match(ln)
            if m: nr[m.group(1)] = float(m.group(2))
        if len(kp) == 2 and len(nr) == 4: NE[b] = (kp, nr)
nd = sum((b, M) in D for b in BR for M in MODES)
if len(T) < 5 or nd < 35 or len(NE) < 5:
    print("[E113] 학습 %d/5, 분해 %d/35, 뉴런 %d/5 — **판정 보류. 다 모일 때까지 수치 미출력.**" % (len(T), nd, len(NE))); sys.exit(0)
rep = sum(abs(D[(b, "all")][0] - T[b][1]) <= 0.0005 and abs(D[(b, "none")][0] - T[b][0]) <= 0.0005 for b in BR)
print("조작검증: 재현(±0.0005) %d/5%s" % (rep, "" if rep == 5 else "  ← 판정 무효"))
exp_push = {"none": 0, "kc_only": 4, "d1_only": 4}
pbad = [(b, M, D[(b, M)][1]) for b in BR for M in MODES if D[(b, M)][1] != exp_push.get(M, 8)]
print("조작검증: pushed 수 %s" % ("정상" if not pbad else "이상 %s" % pbad))
for M in ("kc_shuffle", "kc_uniform", "kc_cm"):
    same = sum(D[(b, M)][0] == D[(b, "all")][0] for b in BR)
    print("조작검증: %s = all 동일 %d/5%s" % (M, same, "  ← 변형 무효 의심" if same == 5 else ""))
valid = rep == 5 and not pbad
used, kcs, d1s, comp = [], [], [], {"S": [], "D": [], "CM": []}
for b in BR:
    g = lambda M: D[(b, M)][0]
    Tt = g("all") - g("none")
    line = "  b%d: " % b + " ".join("%s=%+.4f" % (M, g(M)) for M in MODES)
    if abs(Tt) < 0.05:
        print(line + " | 제외(|T|<0.05)"); continue
    used.append(b)
    kc = (g("kc_only") - g("none")) / Tt; d1 = (g("d1_only") - g("none")) / Tt
    Ek = g("all") - g("d1_only")
    S = (g("all") - g("kc_shuffle")) / Ek; Dm = (g("kc_uniform") - g("kc_cm")) / Ek; CM = (g("kc_cm") - g("d1_only")) / Ek
    kcs.append(kc); d1s.append(d1); comp["S"].append(S); comp["D"].append(Dm); comp["CM"].append(CM)
    print(line + " | T=%+.4f KC=%.2f D1=%.2f 합=%.2f | E_kc=%+.4f S=%.2f D=%.2f CM=%.2f" % (Tt, kc, d1, kc + d1, Ek, S, Dm, CM))
if len(used) < 4:
    print("판정 불가: 유효 뇌 %d (<4)" % len(used)); sys.exit(0)
a = "KC 주도" if sum(x >= 0.7 for x in kcs) >= 4 else ("H039-d1" if sum(x >= 0.7 for x in d1s) >= 4 else "혼합")
print("(a) 경로: %s | 보조: D1 반대 방향(D1<0) %d/%d" % (a, sum(x < 0 for x in d1s), len(d1s)))
if a == "KC 주도":
    cnt = {k: sum(x >= 0.5 for x in v) for k, v in comp.items()}
    main = [k for k, c in cnt.items() if c >= 4]
    lab = "보류(≥0.5 성분 없음)" if not main else ("복합: %s" % ", ".join(main) if len(main) > 1 else
          ("H039-map" if main[0] == "D" else "H039 지지 — 주 출처 %s" % main[0]))
    print("(b) KC 성분(≥0.5 뇌 수): S %d, D %d, CM %d → %s" % (cnt["S"], cnt["D"], cnt["CM"], lab))
conc = nonsel = mapsp = crossw = postlow = posthigh = 0
for b in BR:
    kp, nr = NE[b]
    tops = [(kp[s][0] + kp[s][1]) / 2 for s in "lr"]; bots = [(abs(kp[s][3]) + abs(kp[s][4])) / 2 for s in "lr"]
    rel = [(kp[s][0] - kp[s][1]) / ((kp[s][0] + kp[s][1]) / 2) for s in "lr"]
    relm = sum(rel) / 2
    c_ = all(t >= 10 * max(bb, 1e-9) for t, bb in zip(tops, bots))
    conc += c_; nonsel += abs(relm) < 0.1; mapsp += relm >= 0.3; crossw += relm <= -0.1
    rmax = max(abs(v) for v in nr.values() if v == v)
    postlow += rmax < 0.2; posthigh += any(v >= 0.3 for v in nr.values() if v == v)
    print("  b%d KC쪽: 상위5%% Δg %s | 하위50%% |Δg| %s | (반사−교차)/평균 %s → %+.3f | motor쪽 r %s" % (
        b, " ".join("%.2f" % t for t in tops), " ".join("%.4f" % x for x in bots), " ".join("%+.3f" % x for x in rel), relm,
        " ".join("%s=%+.3f" % (k, v) for k, v in sorted(nr.items()))))
h40 = "H040 지지(비선택)" if (conc >= 4 and nonsel >= 4) else ("H040-map" if mapsp >= 4 else ("교차 쪽 우세(정답 방향)" if crossw >= 4 else "보류"))
print("(c) KC쪽: 집중 %d/5, 비선택 %d/5, 매핑특이 %d/5, 교차우세 %d/5 → %s | motor쪽: |r|<0.2 %d/5, r≥0.3 %d/5 → %s" % (
    conc, nonsel, mapsp, crossw, h40, postlow, posthigh, "H040-post 기각" if postlow >= 4 else ("H040-post" if posthigh >= 4 else "보류")))
if not valid:
    print("※ 조작검증 실패 — 위 판정은 무효")
