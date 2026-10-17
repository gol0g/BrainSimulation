#!/usr/bin/env python3
"""E201 사후 탐색(판정 밖 — E201 결과를 본 뒤 2026-10-17 작성): 경로 항상성은 4경로의 '전체 시냅스' 평균을 150 으로 지킨다.
그런데 반응 KC 의 300 배정 좌우 비(사후 기술)는 E200 과 거의 같았다 — 비반응 KC 가 반대로 기울어 평균을 맞췄는가?
경로마다 시냅스 평균을 (전체·반응 KC 발 시냅스·비반응 KC 발 시냅스·활동 가중)으로 나눠 E200 과 비교한다.
활동 가중 = 시냅스마다 그 KC 의 평가 발화 수(SXI 파일 tot)로 가중한 평균 — motor 가 실제로 받는 구동에 가까운 값.
실행: 저장소 루트에서 python3 scripts/explore_e201_comp.py"""
import os
import sys

import numpy as np

E = "research/experiments"
PATHS = [("l", "l"), ("l", "r"), ("r", "l"), ("r", "r")]  # (KC 쪽, motor 쪽) — 교차 = 다른 쪽


def means(wf, sf, zf):
    W, S, Z = np.load(wf), np.load(sf), np.load(zf)
    out = {}
    for kn, m in PATHS:
        nm = "kc_%s_to_motor_%s" % (kn, m)
        pre = S[nm + "__pre"].astype(np.int64)
        w = W[nm].astype(np.float64)
        if len(pre) != len(w):
            raise ValueError(nm)
        rp = Z["resp_" + kn][pre] > 0
        tot = Z["tot_" + kn].astype(np.float64)[pre]
        out[(kn, m)] = {
            "all": w.mean(),
            "resp": w[rp].mean(),
            "non": w[~rp].mean(),
            "act": (w * tot).sum() / tot.sum(),
            "n_resp_syn": int(rp.sum()), "n_syn": len(w),
        }
    return out


def main():
    print("[E201 사후 탐색 — 판정 밖, 결과 뒤 작성] 경로 평균 분해(전체·반응 KC·비반응 KC·활동 가중), E201 대 E200")
    print("경로 표기 KC쪽→motor쪽, 교차 = l→r·r→l. 값은 시냅스 가중치 평균(0~300, 초기 150).")
    gap = {"E200": [], "E201": []}
    for b in (10, 11, 12, 13, 14):
        zf = os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b)
        sf = os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b)
        for nm in ("E200", "E201"):
            r = means(os.path.join(E, "traces", nm, "w_b%d.npz" % b), sf, zf)
            seg = []
            for k in PATHS:
                v = r[k]
                seg.append("%s→%s 전체 %.1f 반응 %.1f 비반응 %.1f 활동 %.1f" % (k[0], k[1], v["all"], v["resp"], v["non"], v["act"]))
            # motor 별 활동 가중 구동(두 KC 쪽 평균)과 교차−같은 쪽 차(반응 KC)
            cross_resp = (r[("l", "r")]["resp"] + r[("r", "l")]["resp"]) / 2
            same_resp = (r[("l", "l")]["resp"] + r[("r", "r")]["resp"]) / 2
            cross_non = (r[("l", "r")]["non"] + r[("r", "l")]["non"]) / 2
            same_non = (r[("l", "l")]["non"] + r[("r", "r")]["non"]) / 2
            gap[nm].append(cross_resp - same_resp)
            print("b%d %s | %s | 반응 교차−같은 쪽 %+.2f · 비반응 교차−같은 쪽 %+.2f | 반응 시냅스 몫 %.3f"
                  % (b, nm, " · ".join(seg), cross_resp - same_resp, cross_non - same_non,
                     sum(r[k]["n_resp_syn"] for k in PATHS) / sum(r[k]["n_syn"] for k in PATHS)))
    for nm in ("E200", "E201"):
        print("%s 반응 KC 교차−같은 쪽 평균 차 중앙값 %+.2f (뇌별 %s)" % (nm, float(np.median(gap[nm])), " ".join("%+.2f" % g for g in gap[nm])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
