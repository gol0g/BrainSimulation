#!/usr/bin/env python3
"""E178 탐색(판정 밖) — explore_e177_curves.py 의 함수(curves·tail_mean·greedy_v — 합성 6/6)를 그대로 써서 E178 추적(traces/E178/tr_bc_b{B}.npz)에 적용:
맥락별 블록 정답률, 탐욕 결정 조향(마지막 1/3), 행동 창 실행·반대 motor 발화율 합(열 35·36) 맥락별 평균.
실행: python3 scripts/explore_e178_curves.py (저장소 루트에서)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explore_e177_curves import curves, tail_mean, greedy_v  # noqa: E402

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def main():
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E178", "tr_bc_b%d.npz" % b)
        if not os.path.exists(f):
            print("b%d: 추적 없음" % b)
            continue
        rows = np.load(f)["rows"]
        cv = curves(rows)
        for c, nm in ((0, "끔(교차)"), (1, "켬(같은 쪽)")):
            print("b%d %-10s %s | 마지막 3 블록 %.3f" % (b, nm, " ".join("%.2f" % r for _, r in cv[c]), tail_mean(cv[c])))
        g = greedy_v(rows)
        print("b%d 탐욕 결정 조향(시행 2000~): 끔 정답방향 %+.3f |v| %.3f 작음 %.2f 정답비율 %.2f (n %d) · 켬 정답방향 %+.3f |v| %.3f 작음 %.2f 정답비율 %.2f (n %d)"
              % (b, g[0][1], g[0][2], g[0][3], g[0][4], g[0][0], g[1][1], g[1][2], g[1][3], g[1][4], g[1][0]))
        ctx = rows[:, 37] == 1
        print("b%d 행동 창 motor 발화율 합 — 실행(열 35): 끔 %.3f · 켬 %.3f / 반대(열 36): 끔 %.3f · 켬 %.3f"
              % (b, rows[~ctx, 35].mean(), rows[ctx, 35].mean(), rows[~ctx, 36].mean(), rows[ctx, 36].mean()))


if __name__ == "__main__":
    main()
