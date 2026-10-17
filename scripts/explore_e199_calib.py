#!/usr/bin/env python3
"""E199 보정 가중치 탐색(판정 밖 — 보정 뒤·본실험 결과 전, 2026-10-16 작성): 뇌 15(표본 밖)의 η 0.15·0.6·2.4 학습 가중치에서
반응 KC 학습 부호 일치 sign(Δ) = sign(SXI)(전체·강·약·가중 — posthoc_e198.agreement), 반응 KC |Δ| 분포, 경계(0·300) 시냅스 비율,
300 을 받는 쪽 motor 별 반응 KC 수(좌우 치우침), 쪽마다 Δ 중앙값. 지수 = traces/E197/pathcheck/sxi_b15.npz.
실행: 저장소 루트에서 python3 scripts/explore_e199_calib.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e198 as J8  # noqa: E402
import posthoc_e198 as P8  # noqa: E402

E = "research/experiments"
POPS = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")


def main():
    npz = os.path.join(E, "traces", "E197", "pathcheck", "sxi_b15.npz")
    sn = os.path.join(E, "traces", "E176", "snap", "conn_b15.npz")
    Z = np.load(npz)
    print("[E199 보정 가중치 탐색 — 뇌 15(표본 밖), 판정 밖] η 별 학습 부호 일치(반응 KC, 쪽 합)·|Δ| 분포·좌우")
    for et in ("0.15", "0.6", "2.4"):
        wf = os.path.join(E, "traces", "E199", "calib", "w_b15_eta%s.npz" % et)
        D, _ = J8.deltas(wf, sn)
        tot, wr, dz, to, cen, med = P8.agreement(D, npz)
        ad = np.concatenate([np.abs(D[sd][Z["resp_" + sd].astype(bool)]) for sd in "lr"])
        W = np.load(wf)
        fb = sum(int(np.sum((W[p] == 0) | (W[p] == 300))) for p in POPS) / sum(W[p].size for p in POPS)
        f = lambda k: "%d/%d(%.3f)" % (tot[k][0], tot[k][1], tot[k][0] / max(tot[k][1], 1))  # noqa: E731
        print("η %s | 일치 전체 %s · 강 %s · 약 %s · 가중 %.3f | 반응 KC |Δ| 중앙값 %.1f · 90%% %.1f · 최대 %.1f | 경계 시냅스 %.3f | 300쪽 motor_l %d · motor_r %d | 중앙값 l %+.1f r %+.1f | 중앙값 보정 일치 %d/%d(%.3f)"
              % (et, f("all"), f("strong"), f("weak"), wr, np.median(ad), np.percentile(ad, 90), ad.max(), fb, to["l"], to["r"], med["l"], med["r"], cen[0], cen[1], cen[0] / max(cen[1], 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
