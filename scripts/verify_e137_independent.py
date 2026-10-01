#!/usr/bin/env python3
"""E137 독립 대조 — judge_e137.py 와 다른 입력·다른 코드로 다시 계산한다(기록 전 독립 대조, 규약).
입력: 요약 줄(E137.log)이 아니라 런별 원 로그 logs/E137/{learn,frozen}_w*_T*_t*.log 의 SDGEN/가중치 변화/블록 줄,
발달 원 로그 logs/E137/dev_corr_w*.log, 발달 npz(같은 위치 비율을 배열에서 직접), 보상 계열 traces/E137/rw_*.txt.
SDGEN 의 novel_bal 은 SDLAB 의 novel_lbal 과 같은 양(같음·다름 정답률 평균)이다 — 다른 줄에서 읽어 교차 확인한다.
부호검정은 이항 분포를 직접 합산(판정 코드의 comb 합과 다른 구현: 파스칼 삼각형)."""
import glob
import os
import re
import sys

import numpy as np

E = "research/experiments"
L = os.path.join(E, "logs", "E137")
TRD = os.path.join(E, "traces", "E137")
W = list(range(94, 110))
DOSES = [100, 200, 400, 800]


def pascal_two_sided(k, n):
    row = [1]
    for _ in range(n):
        row = [a + b for a, b in zip([0] + row, row + [0])]
    tail = sum(row[max(k, n - k):]) / 2 ** n
    return min(1.0, 2 * tail)


def run(path):
    s = open(path, encoding="utf-8").read()
    g = re.search(r"=> SDGEN mode=(\w+) seed=(\d+) trialseed=(\d+) train_same=([0-9.]+) train_diff=([0-9.]+) train_bal=([0-9.]+) "
                  r"novel_same=([0-9.]+) novel_diff=([0-9.]+) novel_bal=([0-9.]+)", s)
    lab = re.search(r"=> SDLAB .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)", s)
    dl = re.search(r"kc_out_l: n=\d+ \|Δ\|평균 ([0-9.]+)", s); dr = re.search(r"kc_out_r: n=\d+ \|Δ\|평균 ([0-9.]+)", s)
    blocks = len(re.findall(r"^  시행 +\d+~ *\d+: 정답률", s, flags=re.M))
    if not (g and lab and dl and dr):
        return None
    return {"mode": g.group(1), "seed": int(g.group(2)), "ts": int(g.group(3)), "nb": float(g.group(9)), "tb": float(g.group(6)),
            "nlab": float(lab.group(2)), "tlab": float(lab.group(1)), "dg": (float(dl.group(1)) + float(dr.group(1))) / 2,
            "dl": float(dl.group(1)), "dr": float(dr.group(1)), "blocks": blocks}


R, bad = {}, []
for w in W:
    for d in DOSES:
        for t in (600, 601):
            p = os.path.join(L, "learn_w%d_T%d_t%d.log" % (w, d, t))
            r = run(p) if os.path.exists(p) else None
            (R.__setitem__(("learn", w, d, t), r) if r else bad.append(p))
    for d in (100, 800):
        p = os.path.join(L, "frozen_w%d_T%d_t600.log" % (w, d))
        r = run(p) if os.path.exists(p) else None
        (R.__setitem__(("frozen", w, d, 600), r) if r else bad.append(p))
print("런 로그 파싱 %d/160, 실패·결측 %d" % (len(R), len(bad)))
if bad:
    print("  예:", bad[:3]); sys.exit(1)
# 교차 확인: SDGEN novel_bal == SDLAB novel_lbal, mode/seed/ts 일치
mism = [k for k, r in R.items() if abs(r["nb"] - r["nlab"]) > 0.05 or abs(r["tb"] - r["tlab"]) > 0.05 or r["mode"] != k[0] or r["seed"] != k[1] or r["ts"] != k[3]]
print("SDGEN↔SDLAB·표기 불일치 %d" % len(mism))
# 발달: npz 에서 같은 위치 비율 직접
mm, mx = [], []
for w in W:
    z = np.load(os.path.join(TRD, "dev_corr_w%d.npz" % w))
    a, b, e, i = (z[k].astype(int) for k in ("a", "b", "e", "i"))
    mm.append(np.mean((b - 50) == a)); mx.append(np.mean((i % 50) == (e % 50)))
print("발달 npz 같은 위치: 일치형 중앙 %.3f(%.3f~%.3f) 불일치형 중앙 %.3f(%.3f~%.3f)" % (np.median(mm), min(mm), max(mm), np.median(mx), min(mx), max(mx)))
# 블록·보상 계열
bb = [k for k, r in R.items() if r["blocks"] != k[2] // 100]
rw = {}
for k in R:
    if k[0] == "learn":
        f = os.path.join(TRD, "rw_w%d_T%d_t%d.txt" % (k[1], k[2], k[3]))
        rw[k] = [x.strip() for x in open(f, encoding="utf-8") if x.strip()] if os.path.exists(f) else None
rbad = [k for k, v in rw.items() if v is None or len(v) != k[2]]
nest = sum(1 for w in W for t in (600, 601) for d in (200, 400, 800) if rw[("learn", w, d, t)][:100] == rw[("learn", w, 100, t)])
print("블록 줄 불일치 %d/160 | 보상 계열 길이 불일치 %d/128 | 중첩 %d/96" % (len(bb), len(rbad), nest))
dgm = {d: np.mean([R[("learn", w, d, t)]["dg"] for w in W for t in (600, 601)]) for d in DOSES}
fz = sum(1 for w in W for d in (100, 800) if R[("frozen", w, d, 600)]["dl"] != 0 or R[("frozen", w, d, 600)]["dr"] != 0)
print("|Δg| 평균 " + " ".join("T%d %.5f" % (d, dgm[d]) for d in DOSES) + " | 무학습 |Δg|≠0 %d/32" % fz)
fe = [R[("frozen", w, 100, 600)]["nb"] - R[("frozen", w, 800, 600)]["nb"] for w in W]
print("무학습 T100−T800 새 항목: 최대 |차| %.1f, 같음 %d/16" % (max(abs(x) for x in fe), sum(x == 0 for x in fe)))
# 주 지표 — SDGEN 줄 값으로
N = {d: np.array([np.mean([R[("learn", w, d, t)]["nb"] for t in (600, 601)]) for w in W]) for d in DOSES}
TRn = {d: np.array([np.mean([R[("learn", w, d, t)]["tb"] for t in (600, 601)]) for w in W]) for d in DOSES}
F8 = np.array([R[("frozen", w, 800, 600)]["nb"] for w in W])
for d in DOSES:
    print("T%d 새 항목 배선 평균 %.2f (중앙 %.1f, 최소 %.1f) 훈련 %.2f | 런 ≥75%% %d/32" % (
        d, N[d].mean(), np.median(N[d]), N[d].min(), TRn[d].mean(), sum(R[("learn", w, d, t)]["nb"] >= 75 for w in W for t in (600, 601))))
print("무학습 800 새 항목 평균 %.2f" % F8.mean())
for name, x in (("N800−N100", N[800] - N[100]), ("N800−F800", N[800] - F8)):
    x = np.round(x, 6); nz = int(np.sum(x != 0)); k = int(np.sum(x > 0))
    print("%s: 평균 %+.2f, 양수 %d/%d, 최소 %+.1f, 양측 p=%.5f" % (name, x.mean(), k, nz, x.min(), pascal_two_sided(k, nz)))
print("배선별 N100/N200/N400/N800/F800:")
for j, w in enumerate(W):
    print("  w%d %.1f %.1f %.1f %.1f | %.1f" % (w, N[100][j], N[200][j], N[400][j], N[800][j], F8[j]))
