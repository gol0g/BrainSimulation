#!/usr/bin/env python3
"""E200 사후 탐색(판정 밖 — E200 결과를 본 뒤 2026-10-17 작성): E199(고정 보상) 대 E200(기준선)의 집단 치우침 —
반응 KC 를 300 받는 쪽으로 나눈 교차 몫(수·활동 가중)과 좌우 비(300 받는 반응 KC 의 큰 쪽/작은 쪽 motor), 뇌마다·중앙값.
Δ 는 verify_e198_independent.learned(np.add.at). 실행: 저장소 루트에서 python3 scripts/explore_e200_bias.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e198_independent as V  # noqa: E402

E = "research/experiments"


def main():
    rat = {"E199": [], "E200": []}
    print("[E200 사후 탐색 — 판정 밖, 결과 뒤 작성] 교차 몫·좌우 비(E199 대 E200)")
    for b in (10, 11, 12, 13, 14):
        Z = np.load(os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b))
        out = []
        for nm in ("E199", "E200"):
            dl, _ = V.learned(os.path.join(E, "traces", nm, "w_b%d.npz" % b), os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b))
            cr = sa = wcr = wsa = 0.0
            to = {"l": 0, "r": 0}
            for sd in "lr":
                rp = Z["resp_" + sd] > 0; tot = Z["tot_" + sd].astype(float); d = np.asarray(dl[sd], float)
                cr += ((d > 0) & rp).sum(); sa += ((d < 0) & rp).sum(); wcr += tot[(d > 0) & rp].sum(); wsa += tot[(d < 0) & rp].sum()
                o = "r" if sd == "l" else "l"; to[o] += int((rp & (d > 0)).sum()); to[sd] += int((rp & (d < 0)).sum())
            r = max(to.values()) / max(min(to.values()), 1)
            rat[nm].append(r)
            out.append("%s 교차 몫 수 %.3f·활동 %.3f · 좌우 비 %.2f" % (nm, cr / (cr + sa), wcr / (wcr + wsa), r))
        print("b%d | " % b + " | ".join(out))
    for nm in ("E199", "E200"):
        print("%s 좌우 비 중앙값 %.2f" % (nm, float(np.median(rat[nm]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
