#!/usr/bin/env python3
"""E201 사후 탐색 2(판정 밖 — E201 결과를 본 뒤 2026-10-17 작성): 반응 KC 의 같은 쪽 우세는 (가) 모든 KC 에 같은 쪽으로 더해진
치우침(offset)인가, (나) 교차를 원하는 KC 가 덜 배운 것(크기 비대칭)인가.
반응 KC 를 SXI 부호(+ = 교차가 정답 쪽, − = 같은 쪽)와 |SXI| 구간으로 나눠 학습 Δ(교차 − 같은 쪽 평균 가중치) 평균을 본다.
구간마다 offset = (교차 KC 평균 + 같은 쪽 KC 평균)/2, gain = (교차 KC 평균 − 같은 쪽 KC 평균)/2.
offset < 0 이 모든 구간에 비슷하면 (가), 교차 KC 의 |평균| 이 같은 쪽 KC 보다 작고 offset 이 |SXI| 와 함께 변하면 (나) 쪽.
Δ 는 verify_e198_independent.learned(np.add.at). 실행: 저장소 루트에서 python3 scripts/explore_e201_sxi.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e198_independent as V  # noqa: E402

E = "research/experiments"
BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)]


def split(dl, Z):
    """반응 KC 의 (|SXI| 구간, 부호)별 Δ 값 목록 — 두 KC 쪽을 합친다."""
    out = {(i, s): [] for i in range(len(BINS)) for s in (1, -1)}
    for sd in ("l", "r"):
        rp = Z["resp_" + sd] > 0
        sx = np.round(Z["sxi_" + sd].astype(np.float64), 9)
        d = np.asarray(dl[sd], np.float64)
        for i, (lo, hi) in enumerate(BINS):
            m = rp & (np.abs(sx) >= lo) & (np.abs(sx) < hi) & (sx != 0)
            out[(i, 1)] += list(d[m & (sx > 0)])
            out[(i, -1)] += list(d[m & (sx < 0)])
    return out


def summarize(out):
    rows = []
    for i in range(len(BINS)):
        c, s = out[(i, 1)], out[(i, -1)]
        mc = float(np.mean(c)) if c else float("nan")
        ms = float(np.mean(s)) if s else float("nan")
        rows.append({"bin": i, "n_c": len(c), "n_s": len(s), "mean_c": mc, "mean_s": ms,
                     "offset": (mc + ms) / 2, "gain": (mc - ms) / 2})
    return rows


def main():
    print("[E201 사후 탐색 2 — 판정 밖, 결과 뒤 작성] 반응 KC 를 SXI 부호·|SXI| 구간으로 나눈 학습 Δ(교차 − 같은 쪽, 가중치 단위 0~300)")
    print("교차 KC = SXI > 0(정답이 교차 쪽 — Δ > 0 이어야), 같은 쪽 KC = SXI < 0(Δ < 0 이어야). offset < 0 = 같은 쪽으로 치우침.")
    allrows = {"E200": [], "E201": []}
    for b in (10, 11, 12, 13, 14):
        Z = np.load(os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b))
        for nm in ("E200", "E201"):
            dl, _ = V.learned(os.path.join(E, "traces", nm, "w_b%d.npz" % b), os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b))
            rows = summarize(split(dl, Z))
            allrows[nm].append(rows)
            print("b%d %s | " % (b, nm) + " · ".join(
                "|SXI| %.1f~%.1f 교차 %+.1f(%d) 같은 쪽 %+.1f(%d) offset %+.1f"
                % (BINS[r["bin"]][0], min(BINS[r["bin"]][1], 1.0), r["mean_c"], r["n_c"], r["mean_s"], r["n_s"], r["offset"]) for r in rows))
    for nm in ("E200", "E201"):
        med = [float(np.median([rows[i]["offset"] for rows in allrows[nm]])) for i in range(len(BINS))]
        mc = [float(np.median([rows[i]["mean_c"] for rows in allrows[nm]])) for i in range(len(BINS))]
        ms = [float(np.median([rows[i]["mean_s"] for rows in allrows[nm]])) for i in range(len(BINS))]
        print("%s 뇌 중앙값 | " % nm + " · ".join("구간%d 교차 %+.1f 같은 쪽 %+.1f offset %+.1f" % (i, mc[i], ms[i], med[i]) for i in range(len(BINS))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
