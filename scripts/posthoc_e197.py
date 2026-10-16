#!/usr/bin/env python3
"""E197 사후 기술(판정 밖) — 실제 로그에서: 팔별 XOR 방향 칸 효과(노출·새), 비율(sgnR/sel·sgnA/sgnR·sgnR/cat(E196)), E196 사후 예측(posthoc_readout.out) 대 관측.
실행: 저장소 루트에서 python3 <이 파일>"""
import re
import numpy as np
E = "research/experiments"
def rc(path):
    d = {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("RCEVAL "):
            kv = dict(x.split("=", 1) for x in ln.split(" | ")[0].split()[1:])
            d[(kv["cat"], kv["set"])] = float(kv["mod"])
    return d
XOR = {("X", "A"): -1, ("X", "B"): 1, ("Y", "A"): 1, ("Y", "B"): -1}
CAT = {"A": -1, "B": 1}
cells = [("X", "A"), ("X", "B"), ("Y", "A"), ("Y", "B")]
B = (10, 11, 12, 13, 14)
print("[E197 사후 기술 — 판정 밖] 칸 효과 = 팔 − none(같은 뇌·맥락, E197), XOR 방향 a. cat 은 E196(같은 평가·none 정확 회귀) 범주 방향.")
V = {}
for b in B:
    for w in ("sel", "sgnR", "sgnA"):
        for st in ("train", "new"):
            V[(b, w, st)] = [XOR[(x, c)] * (rc("%s/logs/E197/ev_b%d_%s_%s.log" % (E, b, w, x))[(c, st)] - rc("%s/logs/E197/ev_b%d_none_%s.log" % (E, b, x))[(c, st)]) for x, c in cells]
    V[(b, "cat", "train")] = [CAT[c] * (rc("%s/logs/E196/ev_b%d_cat_%s.log" % (E, b, x))[(c, "train")] - rc("%s/logs/E196/ev_b%d_none_%s.log" % (E, b, x))[(c, "train")]) for x, c in cells]
for w in ("sel", "sgnR", "sgnA"):
    t = np.array([v for b in B for v in V[(b, w, "train")]]); n = np.array([v for b in B for v in V[(b, w, "new")]])
    print("%-4s 노출 평균 %+.4f 범위 %+.4f~%+.4f ≥0.10 %d/20 | 새 평균 %+.4f 범위 %+.4f~%+.4f 새/노출 %.2f" % (w, t.mean(), t.min(), t.max(), (t >= 0.10).sum(), n.mean(), n.min(), n.max(), n.mean() / t.mean()))
c = np.array([v for b in B for v in V[(b, "cat", "train")]]); print("cat(E196) 노출 평균 %+.4f" % c.mean())
for b in B:
    ms = {w: np.mean(V[(b, w, "train")]) for w in ("sel", "sgnR", "sgnA", "cat")}
    print("b%d sgnR/sel %.2f · sgnA/sgnR %.2f · sgnR/cat %.2f · sgnR 최소 %+.4f · 새 최소 %+.4f" % (b, ms["sgnR"] / ms["sel"], ms["sgnA"] / ms["sgnR"], ms["sgnR"] / ms["cat"], min(V[(b, "sgnR", "train")]), min(V[(b, "sgnR", "new")])))
PX, G, cur = {}, {}, None
for ln in open("%s/logs/E196/posthoc_readout.out" % E, encoding="utf-8"):
    m = re.match(r"^b(\d+) \|", ln)
    if m:
        cur = int(m.group(1)); G[cur] = (float(re.search(r"XOR팔 g=([0-9.]+)", ln).group(1)), float(re.search(r"세 팔 g=([0-9.]+)", ln).group(1)))
    mm = re.search(r"sgnR ([-+0-9. ]+) \(최소", ln)
    if mm and cur is not None:
        PX[cur] = [float(x) for x in mm.group(1).split()]
o = np.array([v for b in B for v in V[(b, "sgnR", "train")]]); x = np.array([v for b in B for v in PX[b]])
x3 = np.array([v * G[b][1] / G[b][0] for b in B for v in PX[b]])
print("예측 대 관측 20칸: r = %.2f · 평균 관측 %+.3f 예측(XOR팔 이득) %+.3f 예측(세 팔 이득) %+.3f · 뇌마다 가장 약한 칸 일치 %d/5" % (
    np.corrcoef(o, x)[0, 1], o.mean(), x.mean(), x3.mean(), sum(int(np.argmin(o[i*4:(i+1)*4]) == np.argmin(x[i*4:(i+1)*4])) for i in range(5))))
