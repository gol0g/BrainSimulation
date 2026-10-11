#!/usr/bin/env python3
"""E179 탐색(판정 밖) — explore_e177_curves.py 의 함수(curves·tail_mean·greedy_v — 합성 6/6)를 그대로 써서 E179 추적(traces/E179/tr_bc_b{B}.npz)에 적용.
열 37 = 0 → 맥락 A(정답 교차), 1 → 맥락 B(정답 같은 쪽) — E177·E178 의 끔·켬 자리와 같은 의미.
맥락별 블록 정답률, 탐욕 결정 조향(마지막 1/3), 행동 창 실행·반대 motor 발화율 합(열 35·36) 맥락별 평균.
실행: python3 scripts/explore_e179_curves.py (저장소 루트에서)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explore_e177_curves import curves, tail_mean, greedy_v  # noqa: E402

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def main():
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E179", "tr_bc_b%d.npz" % b)
        if not os.path.exists(f):
            print("b%d: 추적 없음" % b)
            continue
        rows = np.load(f)["rows"]
        cv = curves(rows)
        for c, nm in ((0, "A(교차)"), (1, "B(같은 쪽)")):
            print("b%d %-10s %s | 마지막 3 블록 %.3f" % (b, nm, " ".join("%.2f" % r for _, r in cv[c]), tail_mean(cv[c])))
        g = greedy_v(rows)
        print("b%d 탐욕 결정 조향(시행 2000~): A 정답방향 %+.3f |v| %.3f 작음 %.2f 정답비율 %.2f (n %d) · B 정답방향 %+.3f |v| %.3f 작음 %.2f 정답비율 %.2f (n %d)"
              % (b, g[0][1], g[0][2], g[0][3], g[0][4], g[0][0], g[1][1], g[1][2], g[1][3], g[1][4], g[1][0]))
        ctx = rows[:, 37] == 1
        print("b%d 행동 창 motor 발화율 합 — 실행(열 35): A %.3f · B %.3f / 반대(열 36): A %.3f · B %.3f"
              % (b, rows[~ctx, 35].mean(), rows[ctx, 35].mean(), rows[~ctx, 36].mean(), rows[ctx, 36].mean()))


if __name__ == "__main__":
    main()
