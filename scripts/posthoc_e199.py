#!/usr/bin/env python3
"""E199 사후 기술(판정 밖 — 2026-10-16 본실험 결과 전에 작성) — 실제 로그·가중치에서 뇌마다:
학습 보상 수(E199 대 E193), KC→motor 경계(0·300) 시냅스 비율, 반응 KC(E196 지수) |Δ| ≥ 250 비율,
반응 KC 부호 일치 sign(Δ_E199) = sign(SXI) — 전체·강한 선택(|SXI| ≥ 0.5)·약한 선택(posthoc_e198.agreement 와 같은 정의),
E193 학습 부호와 같은 비율(반응 KC, 둘 다 Δ ≠ 0), 300 을 받는 쪽 motor 별 반응 KC 수(좌우 치우침),
칸 효과(노출/새) E199 대 E193 대 E198 lsgnR(E193 부호 열린 고리 판독).
실행: 저장소 루트에서 python3 scripts/posthoc_e199.py [루트]"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_e193 import CELLS, read_eval  # noqa: E402
import judge_e198 as J8  # noqa: E402
import posthoc_e198 as P8  # noqa: E402

BRAINS = (10, 11, 12, 13, 14)


def reward(path):
    m = re.search(r"보상 (\d+)회", open(path, encoding="utf-8", errors="replace").read())
    return int(m.group(1)) if m else -1


def cells(E, exp, b, arm="learn", base="none"):
    ev = {(w, x): read_eval(os.path.join(E, "logs", exp, "ev_b%d_%s_%s.log" % (b, w, x)), x)[0] for w in (arm, base) for x in "XY"}
    return {s: [sg * (ev[(arm, x)][(c, s)] - ev[(base, x)][(c, s)]) / 10000.0 for x, c, sg in CELLS] for s in ("train", "new")}


def main(root="."):
    E = os.path.join(root, "research", "experiments")
    print("[E199 사후 기술 — 판정 밖, 결과 전 작성] 칸 순서 X·A X·B Y·A Y·B, XOR 방향 a(노출). 부호 일치 = 반응 KC 중 Δ ≠ 0·SXI ≠ 0 에서 sign(Δ) = sign(SXI)(쪽 l·r 합).")
    allc = {"e199": [], "e193": [], "lsgnR": []}
    for b in BRAINS:
        sn = os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b)
        w9, w3 = os.path.join(E, "traces", "E199", "w_b%d.npz" % b), os.path.join(E, "traces", "E193", "w_b%d.npz" % b)
        npz = os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b)
        D9, _ = J8.deltas(w9, sn); D3, _ = J8.deltas(w3, sn)
        Z = np.load(npz)
        tot, wr, dz, to, cen, med = P8.agreement(D9, npz)
        W = np.load(w9)
        pops = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")
        fb = sum(int(np.sum((W[p] == 0) | (W[p] == 300))) for p in pops) / sum(W[p].size for p in pops)
        nsat = nresp = same3 = both3 = 0
        for sd in "lr":
            rp = Z["resp_" + sd].astype(bool)
            nresp += int(rp.sum()); nsat += int(np.sum(rp & (np.abs(D9[sd]) >= 250)))
            m = rp & (D9[sd] != 0) & (D3[sd] != 0)
            both3 += int(m.sum()); same3 += int(np.sum(m & (np.sign(D9[sd]) == np.sign(D3[sd]))))
        c9 = cells(E, "E199", b); c3 = cells(E, "E193", b)
        try:
            c8 = cells(E, "E198", b, "lsgnR", "none")
        except (OSError, KeyError):
            c8 = None
        allc["e199"] += c9["train"]; allc["e193"] += c3["train"]
        if c8:
            allc["lsgnR"] += c8["train"]
        f = lambda k: "%d/%d(%.3f)" % (tot[k][0], tot[k][1], tot[k][0] / max(tot[k][1], 1))  # noqa: E731
        r9 = reward(os.path.join(E, "logs", "E199", "train_b%d.log" % b)); r3 = reward(os.path.join(E, "logs", "E193", "train_b%d.log" % b))
        print("b%d 보상 E199 %d · E193 %d | 경계 시냅스 %.3f · 반응 KC |Δ|≥250 %d/%d | 부호 일치 전체 %s · 강 %s · 약 %s · |SXI| 가중 %.3f | E193 부호와 같음 %d/%d(%.3f) | 300 받는 반응 KC motor_l %d · motor_r %d"
              % (b, r9, r3, fb, nsat, nresp, f("all"), f("strong"), f("weak"), wr, same3, both3, same3 / max(both3, 1), to["l"], to["r"]))
        print("    칸 노출 E199 %s · 새 %s | E193 %s | E198 lsgnR %s"
              % (" ".join("%+.4f" % v for v in c9["train"]), " ".join("%+.4f" % v for v in c9["new"]), " ".join("%+.4f" % v for v in c3["train"]),
                 " ".join("%+.4f" % v for v in c8["train"]) if c8 else "없음"))
    for k in ("e199", "e193", "lsgnR"):
        if allc[k]:
            a = np.array(allc[k])
            print("%-6s 노출 평균 %+.4f 범위 %+.4f~%+.4f ≥0.10 %d/%d 정답 방향 %d/%d" % (k, a.mean(), a.min(), a.max(), int((a >= 0.10).sum()), a.size, int((a > 0).sum()), a.size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
