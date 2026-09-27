#!/usr/bin/env python3
"""E112 판정 — E112.md 4절. 5 학습 + 35 분해 전에는 수치 미출력."""
import re
import sys

TR = re.compile(r"^\s*train b(\d): => ([-+0-9.]+) ([-+0-9.]+)")
DC = re.compile(r"^\s*dec b(\d) (\w+): => DECOMP mode=\w+ mod=([-+0-9.]+) .*?pushed=(\d+)")
MODES = ("none", "all", "kc_only", "d1_only", "kc_shuffle", "kc_uniform", "kc_cm")
T, D = {}, {}
try:
    for ln in open("research/experiments/E112.log", encoding="utf-8"):
        m = TR.match(ln)
        if m:
            T[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        m = DC.match(ln)
        if m:
            D[(int(m.group(1)), m.group(2))] = (float(m.group(3)), int(m.group(4)))
except FileNotFoundError:
    pass
if len(T) < 5 or len(D) < 35:
    print("[E112] 학습 %d/5, 분해 %d/35 — **판정 보류. 다 모일 때까지 수치 미출력.**" % (len(T), len(D))); sys.exit(0)
E110 = {0: 0.6122}
rep = sum(abs(D[(b, "all")][0] - T[b][1]) < 5e-5 and abs(D[(b, "none")][0] - T[b][0]) < 5e-5 for b in range(5))
print("조작검증: all=사후·none=사전 재현 %d/5%s" % (rep, "" if rep == 5 else "  ← 분해 경로≠이식 경로 — 판정 무효"))
print("조작검증: pushed none=%s kc_only=%s" % ({D[(b, 'none')][1] for b in range(5)}, {D[(b, 'kc_only')][1] for b in range(5)}))
for M in ("kc_shuffle", "kc_uniform", "kc_cm"):
    same = sum(D[(b, M)][0] == D[(b, "all")][0] for b in range(5))
    print("조작검증: %s = all 동일 %d/5%s" % (M, same, "  ← 변형 무효 의심" if same == 5 else ""))
kcd, d1d, Sd, Dd, CMd = [], [], [], [], []
used = 0
for b in range(5):
    g = lambda M: D[(b, M)][0]
    Tt = g("all") - g("none")
    if abs(Tt) < 0.05:
        print("  b%d 제외(|T|<0.05)" % b); continue
    used += 1
    kc = (g("kc_only") - g("none")) / Tt; d1 = (g("d1_only") - g("none")) / Tt
    Ek = g("all") - g("d1_only")
    S = (g("all") - g("kc_shuffle")) / Ek if Ek else float("nan")
    Dm = (g("kc_uniform") - g("kc_cm")) / Ek if Ek else float("nan")
    CM = (g("kc_cm") - g("d1_only")) / Ek if Ek else float("nan")
    kcd.append(kc); d1d.append(d1); Sd.append(S); Dd.append(Dm); CMd.append(CM)
    print("  b%d: " % b + " ".join("%s=%+.4f" % (M, g(M)) for M in MODES) +
          " | T=%+.4f KC분율=%.2f D1분율=%.2f 합=%.2f | E_kc=%+.4f S=%.2f D=%.2f CM=%.2f" % (Tt, kc, d1, kc + d1, Ek, S, Dm, CM))
need = 4 if used >= 5 else max(1, used - 1)
a = "KC 주도" if sum(x >= 0.7 for x in kcd) >= need else ("H039-d1(D1 주도)" if sum(x >= 0.7 for x in d1d) >= need else "혼합")
print("(a) 경로: %s (유효 뇌 %d)" % (a, used))
if a == "KC 주도":
    best = None
    for name, xs in (("S(구조)", Sd), ("D(매핑)", Dd), ("CM(공통)", CMd)):
        if sum(x >= 0.5 for x in xs) >= need:
            best = name
    b_ = "보류(≥0.5 성분 없음)" if best is None else ("H039-map" if best.startswith("D") else "H039 지지 — 주 출처 %s" % best)
    print("(b) KC 성분: %s" % b_)
