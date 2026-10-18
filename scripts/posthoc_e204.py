#!/usr/bin/env python3
"""E204 사후 기술(판정 밖 — 2026-10-18 본실험 결과 전에 작성) — 게이팅이 '무리마다 맥락 전용 범주 판독' 을 만들었는지(G18, 세계 18300 — E196 지수가 있는 세계).
뇌마다: 무리 배정(ctx_gate.build_all(19300 + genn_seed) 재현, genn_seed = 12345 + 뇌), 반응 KC 의 KC 별 학습 차 Δ = 교차 평균 − 같은 쪽 평균(judge_e198.deltas),
  무리 규칙 일치: X 무리 sign(Δ) = sign(CSI)(맥락 X: A 교차·B 같은 쪽), Y 무리 sign(Δ) = −sign(CSI)(맥락 Y: 반대) — |CSI| 구간(< 0.1·0.1~0.5·≥ 0.5)별,
  비교: 같은 뇌 E200 의 XOR 규칙 일치 sign(Δ) = sign(SXI) 를 |SXI| 구간별(E198~E203 의 기존 지표).
  (보고) 팔마다 칸 값·가장 약한 칸, 300 받는 반응 KC 좌우 비, 학습 중 보상.
실행: python3 scripts/posthoc_e204.py [저장소 루트]"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backend", "genesis"))
import ctx_gate as CG  # noqa: E402
import judge_e198 as J8  # noqa: E402

BRAINS = (10, 11, 12, 13, 14)
BINS = (0.0, 0.1, 0.5, 1.0000001)


def subsets(b, n_kc=3000, n_x=200, n_y=200, frac=0.5, p=0.10):
    m = CG.build_all(19300 + 12345 + b, n_kc, n_x, n_y, frac, p)
    return {sd: m[sd][0] for sd in "lr"}


def rule_agree(D, Z, isx):
    """반환 {구간 i: [맞음, 수]} — 반응 KC, Δ ≠ 0, CSI ≠ 0. X 무리 sign(Δ)=sign(CSI), Y 무리 sign(Δ)=−sign(CSI)."""
    acc = {i: [0, 0] for i in range(len(BINS) - 1)}
    for sd in "lr":
        c = np.round(np.asarray(Z["csi_" + sd], dtype=np.float64), 9)
        rp = np.asarray(Z["resp_" + sd]) > 0
        d = np.asarray(D[sd], dtype=np.float64)
        want = np.where(isx[sd], np.sign(c), -np.sign(c))
        v = rp & (d != 0) & (c != 0)
        ok = np.sign(d) == want
        for i in range(len(BINS) - 1):
            m = v & (np.abs(c) >= BINS[i]) & (np.abs(c) < BINS[i + 1])
            acc[i][0] += int(np.sum(m & ok)); acc[i][1] += int(np.sum(m))
    return acc


def xor_agree(D, Z):
    acc = {i: [0, 0] for i in range(len(BINS) - 1)}
    for sd in "lr":
        s = np.round(np.asarray(Z["sxi_" + sd], dtype=np.float64), 9)
        rp = np.asarray(Z["resp_" + sd]) > 0
        d = np.asarray(D[sd], dtype=np.float64)
        v = rp & (d != 0) & (s != 0)
        ok = np.sign(d) == np.sign(s)
        for i in range(len(BINS) - 1):
            m = v & (np.abs(s) >= BINS[i]) & (np.abs(s) < BINS[i + 1])
            acc[i][0] += int(np.sum(m & ok)); acc[i][1] += int(np.sum(m))
    return acc


def fmt_acc(acc):
    return " · ".join("%.1f~%.1f %d/%d(%.3f)" % (BINS[i], min(BINS[i + 1], 1.0), a[0], a[1], a[0] / max(a[1], 1)) for i, a in sorted(acc.items()))


def reward(path):
    m = re.search(r"보상 (\d+)회", open(path, encoding="utf-8", errors="replace").read())
    return int(m.group(1)) if m else -1


def main(root="."):
    E = os.path.join(root, "research", "experiments")
    print("[E204 사후 기술 — 판정 밖, 결과 전 작성] G18 무리별 범주 규칙 일치 대 E200 XOR 규칙 일치(반응 KC, |지수| 구간 < 0.1·0.1~0.5·≥ 0.5)")
    totG = {i: [0, 0] for i in range(3)}
    totE = {i: [0, 0] for i in range(3)}
    for b in BRAINS:
        sn = os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b)
        Z = np.load(os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b))
        isx = subsets(b)
        DG, _ = J8.deltas(os.path.join(E, "traces", "E204", "w_G18_b%d.npz" % b), sn)
        D0, _ = J8.deltas(os.path.join(E, "traces", "E200", "w_b%d.npz" % b), sn)
        ag, ae = rule_agree(DG, Z, isx), xor_agree(D0, Z)
        for i in range(3):
            totG[i][0] += ag[i][0]; totG[i][1] += ag[i][1]; totE[i][0] += ae[i][0]; totE[i][1] += ae[i][1]
        rg = reward(os.path.join(E, "logs", "E204", "train_G18_b%d.log" % b))
        print("b%d 무리 X 좌 %d·우 %d | G18 무리 규칙(CSI) %s | E200 XOR 규칙(SXI) %s | 보상 G18 %d" % (b, int(isx["l"].sum()), int(isx["r"].sum()), fmt_acc(ag), fmt_acc(ae), rg))
    print("합 G18 무리 규칙(CSI) %s" % fmt_acc(totG))
    print("합 E200 XOR 규칙(SXI) %s" % fmt_acc(totE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
