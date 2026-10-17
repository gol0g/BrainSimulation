#!/usr/bin/env python3
"""E199 사후 탐색(판정 밖 — E199 결과를 본 뒤 2026-10-17 작성): 선형 구동 모델(posthoc_e198_drive 와 같은 정의 — E196 판독 추정, E197 r 0.91·E198 r 0.88)로
E199 학습 가중치의 칸 값을 예측하고 반사실을 계산한다. d_k = Δ_k / 300(등급 — E199 학습 Δ 그대로, 반응 KC 만).
반사실: (n01) |SXI| < 0.1 반응 KC 를 중립(d = 0) / (n02) |SXI| < 0.2 중립 / (ctr) 쪽마다 반응 KC Δ 중앙값을 뺀 d(좌우 공통 표류 제거).
관측 = E199 판정 칸 효과(노출). 뇌마다 네 칸 예측 ≥ 0.10 인지 보고.
실행: 저장소 루트에서 python3 scripts/explore_e199_drive.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e198 as J8  # noqa: E402
import posthoc_e198_drive as PD  # noqa: E402
from judge_e193 import CELLS, read_eval  # noqa: E402

E = "research/experiments"


def main():
    G = PD.gains()
    obs_all, pred_all = [], []
    print("[E199 사후 탐색 — 판정 밖, 결과 뒤 작성] 선형 구동 예측(E196 이득) 대 관측 — 칸 X·A X·B Y·A Y·B, d = Δ/300")
    for b in (10, 11, 12, 13, 14):
        M = PD.cell_means(os.path.join(E, "logs", "E196", "meas_b%d_T4900.log" % b))
        Z = np.load(os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b))
        D, _ = J8.deltas(os.path.join(E, "traces", "E199", "w_b%d.npz" % b), os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b))
        rp = {sd: Z["resp_" + sd].astype(bool) for sd in "lr"}
        s = {sd: np.abs(np.round(Z["sxi_" + sd], 9)) for sd in "lr"}
        base = {sd: np.where(rp[sd], D[sd] / 300.0, 0.0) for sd in "lr"}
        arms = {"learn": base,
                "n01": {sd: np.where(s[sd] < 0.1, 0.0, base[sd]) for sd in "lr"},
                "n02": {sd: np.where(s[sd] < 0.2, 0.0, base[sd]) for sd in "lr"},
                "ctr": {sd: np.where(rp[sd], (D[sd] - np.median(D[sd][rp[sd]])) / 300.0, 0.0) for sd in "lr"}}
        ev = {(w, x): read_eval(os.path.join(E, "logs", "E199", "ev_b%d_%s_%s.log" % (b, w, x)), x)[0] for w in ("learn", "none") for x in "XY"}
        ob = [sg * (ev[("learn", x)][(c, "train")] - ev[("none", x)][(c, "train")]) / 10000.0 for x, c, sg in CELLS]
        g = G[b]
        pr = {k: [g * v for v in (lambda p: [p[(x, c)] for x, c, _ in CELLS])(PD.drive(M, d))] for k, d in arms.items()}
        obs_all += ob; pred_all += pr["learn"]
        print("b%d 관측 %s | 예측 learn %s | n01 %s (≥0.10 %d칸) | n02 %s (%d칸) | ctr %s (%d칸)"
              % (b, " ".join("%+.3f" % v for v in ob), " ".join("%+.3f" % v for v in pr["learn"]),
                 " ".join("%+.3f" % v for v in pr["n01"]), sum(v >= 0.10 for v in pr["n01"]),
                 " ".join("%+.3f" % v for v in pr["n02"]), sum(v >= 0.10 for v in pr["n02"]),
                 " ".join("%+.3f" % v for v in pr["ctr"]), sum(v >= 0.10 for v in pr["ctr"])))
    print("learn 20칸 예측 대 관측 r = %.2f · 평균 관측 %+.3f 예측 %+.3f" % (np.corrcoef(obs_all, pred_all)[0, 1], np.mean(obs_all), np.mean(pred_all)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
