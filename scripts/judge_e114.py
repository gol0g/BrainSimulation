#!/usr/bin/env python3
"""E114 판정 — E114.md 4절. 10 학습/무학습 + 5 뉴런 전에는 수치 미출력."""
import os
import re
import sys

EXP = "research/experiments"
TR = re.compile(r"^\s*(학습|무학습) b(\d): => ([-+0-9.]+) ([-+0-9.]+) \|.*?변조폭 변화 ([-+0-9.]+)")
KP = re.compile(r"^=> KCPRE side=(\w) top5%: 반사\(\w+\) ([-+0-9.]+) 교차\(\w+\) ([-+0-9.]+)")
POST = re.compile(r"\[사후\] .*?변조폭 ([-+0-9.]+)")
BR = range(5)
R, NE = {}, {}
try:
    for ln in open(os.path.join(EXP, "E114.log"), encoding="utf-8"):
        m = TR.match(ln)
        if m: R[(m.group(1), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)), float(m.group(5)))
except FileNotFoundError:
    pass
for b in BR:
    p = os.path.join(EXP, "logs/E114/neuron_b%d.log" % b)
    if os.path.exists(p):
        kp = {}
        for ln in open(p, encoding="utf-8"):
            m = KP.match(ln)
            if m: kp[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        if len(kp) == 2: NE[b] = kp
if len(R) < 10 or len(NE) < 5:
    print("[E114] 학습/무학습 %d/10, 뉴런 %d/5 — **판정 보류. 다 모일 때까지 수치 미출력.**" % (len(R), len(NE))); sys.exit(0)
E110 = {}
for b in BR:
    t = open(os.path.join(EXP, "logs/E110/eta0.15_학습_b%d.log" % b), encoding="utf-8").read()
    E110[b] = float(POST.search(t).group(1))
zero = all(R[("무학습", b)][2] == 0.0 for b in BR)
diff = sum(abs(R[("학습", b)][1] - E110[b]) > 0.0005 for b in BR)
print("조작검증: 무학습 변화 0.0000 %s | 학습 사후 ≠ E110 같은 뇌 %d/5%s" % (zero, diff, "" if diff == 5 else "  ← 행동 창 무효 의심"))
eff = [R[("학습", b)][1] - R[("무학습", b)][1] for b in BR]
m = sum(eff) / 5
below = sum(R[("학습", b)][1] < E110[b] for b in BR)
for b in BR:
    kp = NE[b]
    rel = [(kp[s][0] - kp[s][1]) / ((kp[s][0] + kp[s][1]) / 2) for s in "lr"]
    print("  b%d: 학습 사후 %+.4f (E110 %+.4f) 무학습 %+.4f 효과 %+.4f | KC쪽 상위5%% 반사/교차 l %.2f/%.2f r %.2f/%.2f (반사−교차)/평균 %+.3f" % (
        b, R[("학습", b)][1], E110[b], R[("무학습", b)][1], eff[b], kp["l"][0], kp["l"][1], kp["r"][0], kp["r"][1], sum(rel) / 2))
if m <= -0.10 and sum(x < 0 for x in eff) >= 4:
    a = "역전 지지"
elif m >= 0.13:
    a = "수리 실패(여전히 반사 강화)"
elif m < 0.13 and below >= 4:
    a = "감소"
else:
    a = "보류"
rels = [sum((NE[b][s][0] - NE[b][s][1]) / ((NE[b][s][0] + NE[b][s][1]) / 2) for s in "lr") / 2 for b in BR]
if sum(x <= -0.10 for x in rels) >= 4:
    bsel = "교차(정답) 쪽 선택성 생김"
elif sum(abs(x) < 0.10 for x in rels) >= 4:
    bsel = "선택성 안 생김(H041-drive)"
elif sum(x >= 0.10 for x in rels) >= 4:
    bsel = "반사 쪽 선택성(반대)"
else:
    bsel = "보류"
if a == "역전 지지" and bsel.startswith("교차"):
    tot = "H041 지지"
elif bsel.startswith("선택성 안"):
    tot = "H041-drive"
elif bsel.startswith("교차") and a.startswith("수리 실패"):
    tot = "H041-other"
else:
    tot = "보류"
print("(a) 효과: 평균 %+.4f, 음수 %d/5, E110보다 작음 %d/5 → %s" % (m, sum(x < 0 for x in eff), below, a))
print("(b) 선택성: %s → %s" % (" ".join("%+.3f" % x for x in rels), bsel))
print("종합: %s" % tot)
