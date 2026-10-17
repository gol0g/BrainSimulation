#!/usr/bin/env python3
"""E199 사후 탐색(판정 밖 — E199 결과를 본 뒤 2026-10-17 작성): 학습 부호(E193 연성 상한·E199 경쟁형)의 정확도 구조와 집단 치우침.
(1) 반응 KC 부호 정확도 sign(Δ) = sign(SXI) 를 |SXI| 구간(0~0.1·0.1~0.2·0.2~0.3·0.3~0.5·≥ 0.5)별로, 약한 KC(|SXI| < 0.5)는 활동량(tot) 3분위(쪽마다)별로 — 뇌 10~14 합.
(2) 300 을 받는 쪽으로 나눈 반응 KC 의 교차 몫(수·활동 가중) — 측정 부호(SXI)·E193·E199.
Δ 는 verify_e198_independent.learned(np.add.at — 판정기와 다른 구현). 지수 traces/E196/sxi_b{B}.npz.
실행: 저장소 루트에서 python3 scripts/explore_e199_bias.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e198_independent as V  # noqa: E402

E = "research/experiments"
BINS = [0, 0.1, 0.2, 0.3, 0.5, 1.01]


def main():
    acc = {"E193": {}, "E199": {}}
    print("[E199 사후 탐색 — 판정 밖, 결과 뒤 작성] 학습 부호 정확도 구조·교차 몫")
    for b in (10, 11, 12, 13, 14):
        Z = np.load(os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b))
        line = []
        for nm in ("SXI", "E193", "E199"):
            if nm != "SXI":
                dl, _ = V.learned(os.path.join(E, "traces", nm, "w_b%d.npz" % b), os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b))
            cr = sa = wcr = wsa = 0.0
            for sd in "lr":
                rp = Z["resp_" + sd] > 0; tot = Z["tot_" + sd].astype(float)
                s = np.round(Z["sxi_" + sd].astype(float), 9)
                d = s if nm == "SXI" else np.asarray(dl[sd], float)
                cr += ((d > 0) & rp).sum(); sa += ((d < 0) & rp).sum()
                wcr += tot[(d > 0) & rp].sum(); wsa += tot[(d < 0) & rp].sum()
                if nm == "SXI":
                    continue
                v = rp & (d != 0) & (s != 0); ok = np.sign(d) == np.sign(s)
                for i in range(len(BINS) - 1):
                    m = v & (np.abs(s) >= BINS[i]) & (np.abs(s) < BINS[i + 1])
                    a = acc[nm].setdefault(("sxi", i), [0, 0]); a[0] += int((m & ok).sum()); a[1] += int(m.sum())
                tq = np.quantile(tot[v], [0, 1 / 3, 2 / 3, 1]) if v.any() else [0, 0, 0, 0]
                wkm = v & (np.abs(s) < 0.5)
                for i in range(3):
                    m = wkm & (tot >= tq[i]) & ((tot < tq[i + 1]) if i < 2 else (tot <= tq[i + 1]))
                    a = acc[nm].setdefault(("tot", i), [0, 0]); a[0] += int((m & ok).sum()); a[1] += int(m.sum())
            line.append("%s 교차 몫 수 %.3f·활동 %.3f" % (nm, cr / (cr + sa), wcr / (wcr + wsa)))
        print("b%d | " % b + " | ".join(line))
    for nm in ("E193", "E199"):
        print("%s 부호 정확도 |SXI| 구간 " % nm + " · ".join("%.1f~%.1f %d/%d(%.3f)" % (BINS[i], min(BINS[i + 1], 1.0), acc[nm][("sxi", i)][0], acc[nm][("sxi", i)][1],
                                                                                acc[nm][("sxi", i)][0] / max(acc[nm][("sxi", i)][1], 1)) for i in range(len(BINS) - 1)))
        print("%s 약한 KC 정확도 활동량 3분위(쪽마다) " % nm + " · ".join("%d분위 %d/%d(%.3f)" % (i + 1, acc[nm][("tot", i)][0], acc[nm][("tot", i)][1],
                                                                                    acc[nm][("tot", i)][0] / max(acc[nm][("tot", i)][1], 1)) for i in range(3)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
