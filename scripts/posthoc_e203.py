#!/usr/bin/env python3
"""E203 사후 기술(판정 밖 — 2026-10-17 본실험 결과 전에 작성) — E203(활동 가중 정규화) 대 E200(같은 규칙·기준선, 정규화 없음), 뇌마다:
학습 보상 수, 평균 DA, 300 을 받는 쪽 motor 별 반응 KC 수·좌우 비(큰 쪽/작은 쪽)·활동 가중 교차 몫(두 실험), 반응 KC 부호 일치(전체·강·약·가중),
|SXI| 구간별 부호 정확도(두 실험 합), 칸 효과(노출/새) E203·E200, 좌우 비 중앙값,
+ 약한 무리(반응 KC SXI 크기 < 0.1) 교차 KC·같은 쪽 KC 평균 Δ 와 offset(explore_e201_sxi — 합성 22/22), 저장 가중치 쪽별 활동 가중 M(judge_e203.actw_offsets).
실행: 저장소 루트에서 python3 scripts/posthoc_e203.py [루트]"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e198 as J8  # noqa: E402
import posthoc_e198 as P8  # noqa: E402
import posthoc_e199 as P9  # noqa: E402
import explore_e201_sxi as XS  # noqa: E402
import judge_e203 as J3  # noqa: E402

BRAINS = (10, 11, 12, 13, 14)
BINS = [0, 0.1, 0.2, 0.3, 0.5, 1.01]


def mean_da(path):
    m = re.search(r"\[E200 도파민 통계\] 시행 \d+ · 평균 DA ([+-]?[0-9.]+)", open(path, encoding="utf-8", errors="replace").read())
    return float(m.group(1)) if m else float("nan")


def main(root="."):
    E = os.path.join(root, "research", "experiments")
    print("[E203 사후 기술 — 판정 밖, 결과 전 작성] E203(활동 가중 정규화) 대 E200(기준선만). 칸 X·A X·B Y·A Y·B, XOR 방향 a(노출).")
    acc = {"E203": [[0, 0] for _ in BINS[:-1]], "E200": [[0, 0] for _ in BINS[:-1]]}
    allc = {"E203": [], "E200": []}
    rat = {"E203": [], "E200": []}
    for b in BRAINS:
        sn = os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b)
        npz = os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b)
        Z = np.load(npz)
        res = {}
        csh = {}
        for ex in ("E203", "E200"):
            D, _ = J8.deltas(os.path.join(E, "traces", ex, "w_b%d.npz" % b), sn)
            res[ex] = P8.agreement(D, npz)
            wc_ = ws_ = 0.0
            for sd in "lr":
                rp_ = Z["resp_" + sd].astype(bool); tt_ = Z["tot_" + sd].astype(float)
                wc_ += float(tt_[rp_ & (D[sd] > 0)].sum()); ws_ += float(tt_[rp_ & (D[sd] < 0)].sum())
            csh[ex] = wc_ / max(wc_ + ws_, 1e-12)
            for sd in "lr":
                s = np.round(Z["sxi_" + sd], 9); rp = Z["resp_" + sd].astype(bool)
                v = rp & (D[sd] != 0) & (s != 0); ok = np.sign(D[sd]) == np.sign(s)
                for i in range(len(BINS) - 1):
                    m = v & (np.abs(s) >= BINS[i]) & (np.abs(s) < BINS[i + 1])
                    acc[ex][i][0] += int(np.sum(m & ok)); acc[ex][i][1] += int(np.sum(m))
        c0 = P9.cells(E, "E203", b); c9 = P9.cells(E, "E200", b)
        allc["E203"] += c0["train"]; allc["E200"] += c9["train"]
        t0, w0, _, to0, _, _ = res["E203"]; t9, w9, _, to9, _, _ = res["E200"]
        for ex, to_ in (("E203", to0), ("E200", to9)):
            rat[ex].append(max(to_.values()) / max(min(to_.values()), 1))
        f = lambda t, k: "%d/%d(%.3f)" % (t[k][0], t[k][1], t[k][0] / max(t[k][1], 1))  # noqa: E731
        r0 = P9.reward(os.path.join(E, "logs", "E203", "train_b%d.log" % b)); r9 = P9.reward(os.path.join(E, "logs", "E200", "train_b%d.log" % b))
        print("b%d 보상 E203 %d · E200 %d | 평균 DA E203 %+.4f | 300 받는 반응 KC motor_l·motor_r E203 %d·%d · E200 %d·%d · 좌우 비 E203 %.2f · E200 %.2f · 교차 몫(활동) E203 %.3f · E200 %.3f | 부호 일치 E203 전체 %s · 강 %s · 약 %s · 가중 %.3f | E200 약 %s"
              % (b, r0, r9, mean_da(os.path.join(E, "logs", "E203", "train_b%d.log" % b)), to0["l"], to0["r"], to9["l"], to9["r"], rat["E203"][-1], rat["E200"][-1], csh["E203"], csh["E200"],
                 f(t0, "all"), f(t0, "strong"), f(t0, "weak"), w0, f(t9, "weak")))
        print("    칸 노출 E203 %s · 새 %s | E200 %s" % (" ".join("%+.4f" % v for v in c0["train"]), " ".join("%+.4f" % v for v in c0["new"]), " ".join("%+.4f" % v for v in c9["train"])))
        wk = {}
        for ex in ("E203", "E200"):
            dl_, _ = J8.deltas(os.path.join(E, "traces", ex, "w_b%d.npz" % b), sn)
            wk[ex] = XS.summarize(XS.split(dl_, Z))[0]
        am_ = J3.actw_offsets(os.path.join(E, "traces", "E203", "w_b%d.npz" % b), sn, npz)
        print("    약한 무리(SXI 크기 < 0.1) E203 교차 KC %+.1f(%d)·같은 쪽 KC %+.1f(%d)·offset %+.1f | E200 %+.1f·%+.1f·offset %+.1f | 저장 가중치 활동 가중 M l %+.3f·r %+.3f"
              % (wk["E203"]["mean_c"], wk["E203"]["n_c"], wk["E203"]["mean_s"], wk["E203"]["n_s"], wk["E203"]["offset"],
                 wk["E200"]["mean_c"], wk["E200"]["mean_s"], wk["E200"]["offset"], am_["l"], am_["r"]))
    print("좌우 비 중앙값 E203 %.2f · E200 %.2f" % (float(np.median(rat["E203"])), float(np.median(rat["E200"]))))
    for ex in ("E203", "E200"):
        print("%s 부호 정확도 |SXI| 구간 " % ex + " · ".join("%.1f~%.1f %d/%d(%.3f)" % (BINS[i], min(BINS[i + 1], 1.0), acc[ex][i][0], acc[ex][i][1], acc[ex][i][0] / max(acc[ex][i][1], 1))
                                                           for i in range(len(BINS) - 1)))
        a = np.array(allc[ex])
        print("%s 노출 평균 %+.4f 범위 %+.4f~%+.4f ≥0.10 %d/%d 정답 방향 %d/%d" % (ex, a.mean(), a.min(), a.max(), int((a >= 0.10).sum()), a.size, int((a > 0).sum()), a.size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
