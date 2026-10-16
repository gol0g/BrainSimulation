#!/usr/bin/env python3
"""E198 사후 기술(판정 밖 — 2026-10-16 본실험 결과 전에 작성) — 실제 로그·가중치에서:
팔별 XOR 방향 칸 효과(노출·새), 뇌마다 비율(lsgnR/sgnR·lsgnA/sgnR)·최소 칸,
반응 KC 의 부호 일치 sign(Δ) = sign(SXI) — 전체·|SXI| ≥ 0.5(강한 선택)·< 0.5(약한 선택)·|SXI| 가중, 반응 KC 중 Δ = 0 수.
Δ 는 judge_e198.deltas(가중치 파일 + 스냅숏 pre), 평가 값은 judge_e196.read_eval(1e-4 정수).
실행: 저장소 루트에서 python3 scripts/posthoc_e198.py [루트]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_e196 import CELLS, THETA, read_eval  # noqa: E402
import judge_e198 as J  # noqa: E402

ARMS = ("sgnR", "lsgnR", "lsgnA")


def motor_max(path):
    """평가 로그 RCEVAL 줄의 motor L/R 값 중 최대(한쪽 motor 상승 기술용)."""
    mx = 0.0
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("RCEVAL "):
            for part in ln.split("motor L/R ")[1:]:
                mx = max([mx] + [float(v) for v in part.split()[0].split("/")])
    return mx


def agreement(D, npz):
    """쪽 l·r 을 합친 반응 KC 부호 일치: (전체 일치, 분모), (강 일치, 분모), (약 일치, 분모), |SXI| 가중 일치율, 반응 중 Δ = 0 수,
    lsgnR 설계 KC 의 300 을 받는 motor 별 수 {l, r}(쪽 l: Δ > 0 → motor_r·< 0 → motor_l, 쪽 r: Δ > 0 → motor_l·< 0 → motor_r)."""
    Z = np.load(npz)
    tot = {"all": [0, 0], "strong": [0, 0], "weak": [0, 0]}
    wn, wd, dz = 0.0, 0.0, 0
    to = {"l": 0, "r": 0}
    cen, med = [0, 0], {}
    for sd in "lr":
        s = np.round(Z["sxi_" + sd], 9)
        d = D[sd]
        rp = Z["resp_" + sd].astype(bool)
        dz += int(np.sum(rp & (d == 0)))
        # (탐색) 쪽마다 반응 KC(Δ ≠ 0) Δ 중앙값을 뺀 부호 — 좌우 공통 표류(한쪽 motor 로 치우친 Δ)를 뺐을 때의 일치
        nz = rp & (d != 0)
        med[sd] = float(np.median(d[nz])) if nz.any() else 0.0
        dc = np.round(d - med[sd], 9)
        okc = nz & (dc != 0) & (s != 0)
        cen[0] += int(np.sum(okc & (np.sign(dc) == np.sign(s))))
        cen[1] += int(np.sum(okc))
        oc = "r" if sd == "l" else "l"
        to[oc] += int(np.sum(rp & (d > 0)))
        to[sd] += int(np.sum(rp & (d < 0)))
        ok = rp & (d != 0) & (s != 0)
        ag = ok & (np.sign(d) == np.sign(s))
        st = np.abs(s) >= THETA
        for k, m in (("all", ok), ("strong", ok & st), ("weak", ok & ~st)):
            tot[k][0] += int(np.sum(ag & m))
            tot[k][1] += int(np.sum(m))
        wn += float(np.sum(np.abs(s)[ag]))
        wd += float(np.sum(np.abs(s)[ok]))
    return tot, (wn / wd if wd > 0 else float("nan")), dz, to, cen, med


def main(root="."):
    E = os.path.join(root, "research", "experiments")
    V, AG = {}, {}
    for b in J.BRAINS:
        ev = {(w, x): read_eval(os.path.join(E, "logs", "E198", "ev_b%d_%s_%s.log" % (b, w, x)), x)["rc"]
              for w in ("none",) + ARMS for x in "XY"}
        for w in ARMS:
            for s in ("train", "new"):
                V[(b, w, s)] = np.array([sg * (ev[(w, x)][(c, s)] - ev[("none", x)][(c, s)]) for x, c, sg in CELLS], dtype=float) / 10000.0
        D, _ = J.deltas(os.path.join(E, "traces", "E193", "w_b%d.npz" % b), os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b))
        AG[b] = agreement(D, os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b)) + (
            max(motor_max(os.path.join(E, "logs", "E198", "ev_b%d_lsgnR_%s.log" % (b, x))) for x in "XY"),
            max(motor_max(os.path.join(E, "logs", "E198", "ev_b%d_sgnR_%s.log" % (b, x))) for x in "XY"))
    print("[E198 사후 기술 — 판정 밖] 칸 효과 = 팔 − none(같은 뇌·맥락, E198), XOR 방향 a. 부호 일치 = 반응 KC 중 Δ ≠ 0·SXI ≠ 0 에서 sign(Δ) = sign(SXI)(쪽 l·r 합).")
    for w in ARMS:
        t = np.concatenate([V[(b, w, "train")] for b in J.BRAINS]); n = np.concatenate([V[(b, w, "new")] for b in J.BRAINS])
        print("%-5s 노출 평균 %+.4f 범위 %+.4f~%+.4f ≥0.10 %d/20 정답 방향 %d/20 | 새 평균 %+.4f 범위 %+.4f~%+.4f 새/노출 %.2f"
              % (w, t.mean(), t.min(), t.max(), int((t >= 0.10).sum()), int((t > 0).sum()), n.mean(), n.min(), n.max(), n.mean() / t.mean() if t.mean() != 0 else float("nan")))
    for b in J.BRAINS:
        ms = {w: V[(b, w, "train")].mean() for w in ARMS}
        tot, wr, dz, to, cen, med, mxl, mxs = AG[b]
        f = lambda k: "%d/%d(%.3f)" % (tot[k][0], tot[k][1], tot[k][0] / max(tot[k][1], 1))  # noqa: E731
        print("b%d lsgnR/sgnR %.2f · lsgnA/sgnR %.2f · lsgnR 최소 %+.4f · 새 최소 %+.4f | 부호 일치 전체 %s · |SXI|≥0.5 %s · <0.5 %s · |SXI| 가중 %.3f · 반응 중 Δ=0 %d"
              " | lsgnR 300 받는 KC motor_l %d · motor_r %d · 평가 motor 최대 lsgnR %.3f · sgnR %.3f | (탐색) 중앙값 보정 일치 %d/%d(%.3f) — 중앙값 l %+.2f · r %+.2f"
              % (b, ms["lsgnR"] / ms["sgnR"], ms["lsgnA"] / ms["sgnR"], V[(b, "lsgnR", "train")].min(), V[(b, "lsgnR", "new")].min(),
                 f("all"), f("strong"), f("weak"), wr, dz, to["l"], to["r"], mxl, mxs, cen[0], cen[1], cen[0] / max(cen[1], 1), med["l"], med["r"]))
    x = np.array([AG[b][0]["all"][0] / max(AG[b][0]["all"][1], 1) for b in J.BRAINS]); y = np.array([V[(b, "lsgnR", "train")].mean() for b in J.BRAINS])
    print("뇌 5 점(기술만): 부호 일치율 %s 대 lsgnR 노출 평균 %s" % (" ".join("%.3f" % v for v in x), " ".join("%+.4f" % v for v in y)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
