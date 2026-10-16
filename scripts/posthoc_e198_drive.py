#!/usr/bin/env python3
"""E198 사후 탐색(판정 밖 — E198 결과를 본 뒤 2026-10-16 작성): 판독 부호별 XOR 선형 구동 추정 대 관측.
E196 사후 탐색(logs/E196/posthoc_readout.out)과 같은 정의 — D(쪽, 칸) = Σ_k 칸 평균 발화_k × d_k(d = (교차 − 같은 쪽)/300 = 부호 판독에서 ±1, 대상 밖 0),
예측 = 두 쪽 평균의 부호 맞춘 D(교차 칸 X·A·Y·B 는 +D, 같은 쪽 칸 X·B·Y·A 는 −D) × g, g = E196 의 XOR팔 맞춤 이득(posthoc_readout.out 에서 읽음).
발화 = E196 −4900 측정 로그(logs/E196/meas_b{B}_T4900.log RCKX, 노출 사례)의 칸 평균. 부호: sgnR = sign(SXI)(재현 검사), lsgnR = sign(Δ)(E193 학습, judge_e198.deltas).
관측 = E198 판정의 칸 효과(노출). 뇌마다 lsgnR 의 가장 약한 칸이 예측과 같은지, 20칸 상관.
실행: 저장소 루트에서 python3 scripts/posthoc_e198_drive.py"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcxor  # noqa: E402
import judge_e198 as J  # noqa: E402
from judge_e196 import CELLS, read_eval  # noqa: E402

E = "research/experiments"
CELL_SG = {("X", "A"): +1, ("X", "B"): -1, ("Y", "A"): -1, ("Y", "B"): +1}   # 교차 칸 +D, 같은 쪽 칸 −D


def cell_means(path):
    rows = rcxor.parse(path, 3000)
    out = {}
    for sd in ("l", "r"):
        acc = {}
        for st, c, x, i, rep, v, *_ in rows[sd]:
            if st != "train":
                continue
            acc.setdefault((x, c), []).append(np.asarray(v, dtype=float))
        out[sd] = {k: np.mean(np.stack(vs), axis=0) for k, vs in acc.items()}
    return out


def drive(M, d):
    return {(x, c): CELL_SG[(x, c)] * np.mean([float(M[sd][(x, c)] @ d[sd]) for sd in ("l", "r")]) for x, c, _ in CELLS}


def gains():
    g = {}
    for ln in open(os.path.join(E, "logs", "E196", "posthoc_readout.out"), encoding="utf-8"):
        m = re.match(r"b(\d+) \| 구동 평균 .* 맞춤 XOR팔 g=([0-9.]+)", ln)
        if m:
            g[int(m.group(1))] = float(m.group(2))
    return g


def main():
    G = gains()
    obs_all, pred_all = [], []
    weak_ok = 0
    print("[E198 사후 탐색 — 판정 밖, 결과 뒤 작성] 판독 부호별 XOR 선형 구동(E196 정의) 대 관측(노출). 칸 순서 X·A X·B Y·A Y·B.")
    for b in J.BRAINS:
        M = cell_means(os.path.join(E, "logs", "E196", "meas_b%d_T4900.log" % b))
        Z = np.load(os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b))
        D, _ = J.deltas(os.path.join(E, "traces", "E193", "w_b%d.npz" % b), os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b))
        rp = {sd: Z["resp_" + sd].astype(bool) for sd in "lr"}
        dS = {sd: np.where(rp[sd], np.sign(np.round(Z["sxi_" + sd], 9)), 0.0) for sd in "lr"}
        dL = {sd: np.where(rp[sd], np.sign(D[sd]), 0.0) for sd in "lr"}
        pS, pL = drive(M, dS), drive(M, dL)
        ev = {(w, x): read_eval(os.path.join(E, "logs", "E198", "ev_b%d_%s_%s.log" % (b, w, x)), x)["rc"] for w in ("none", "sgnR", "lsgnR") for x in "XY"}
        ob = {w: [sg * (ev[(w, x)][(c, "train")] - ev[("none", x)][(c, "train")]) / 10000.0 for x, c, sg in CELLS] for w in ("sgnR", "lsgnR")}
        g = G[b]
        prS = [g * pS[(x, c)] for x, c, _ in CELLS]; prL = [g * pL[(x, c)] for x, c, _ in CELLS]
        wl = int(np.argmin(ob["lsgnR"])) == int(np.argmin(prL))
        weak_ok += wl
        obs_all += ob["lsgnR"]; pred_all += prL
        print("b%d g=%.5f | 구동 평균 sgnR %.1f lsgnR %.1f | 예측 sgnR %s | 예측 lsgnR %s | 관측 lsgnR %s | 가장 약한 칸 일치 %s"
              % (b, g, np.mean(list(pS.values())), np.mean(list(pL.values())),
                 " ".join("%+.3f" % v for v in prS), " ".join("%+.3f" % v for v in prL), " ".join("%+.3f" % v for v in ob["lsgnR"]), "예" if wl else "아니오"))
    r = np.corrcoef(obs_all, pred_all)[0, 1]
    print("lsgnR 20칸: 예측 대 관측 r = %.2f · 평균 관측 %+.3f 예측 %+.3f · 가장 약한 칸 일치 %d/5" % (r, np.mean(obs_all), np.mean(pred_all), weak_ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
