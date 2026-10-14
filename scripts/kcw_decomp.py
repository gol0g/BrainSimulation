#!/usr/bin/env python3
"""(탐색 분석) KC 부류별 학습된 KC→motor 가중치 차 — 스냅숏 연결(전시냅스 번호) + 학습 가중치 + KC 반응 부류.
쪽 s(l·r)의 KC 마다 같은 쪽 motor(s)·반대쪽 motor 로 가는 시냅스 평균 가중치 w_same·w_cross(같은 쪽 = 자극 쪽으로 도는 행동 — 왼쪽 자극이면 motor_l).
부류: (범주) E183 무가소성 측정(rc_b{B}_Z.log, tag=pre) 노출 사례 반응 집합 — A 전용(A ≥ 3/6·B ≤ 1/6)·B 전용·둘 다;
      (XOR) E192 측정(meas_b{B}_C2.log) 노출 제시 평균 스파이크 — 결합(|(AX+BY)−(AY+BX)|/2 ≥ 0.5·평균 ≥ 0.25, 부호 + = 교차 칸 선호)·그 밖 반응·무반응.
보고: 부류별 정렬 차 — 범주: A 전용은 w_cross − w_same, B 전용은 w_same − w_cross(E184 catrule: A 교차·B 같은 쪽이면 둘 다 > 0 이어야 — 양성 대조);
      XOR: 결합 KC 는 교차 선호면 w_cross − w_same, 같은 쪽 선호면 w_same − w_cross(> 0 이어야 정답 방향).
실행: python3 scripts/kcw_decomp.py selftest | cat <B> <가중치 npz> | xor <B> <가중치 npz> [<가중치 npz> ...]"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
N_KC = 3000


def per_kc(snap, w, side):
    """쪽 side KC 마다 (w_same 평균, w_cross 평균) — 같은 쪽 motor = side."""
    other = "r" if side == "l" else "l"
    out = {}
    for tgt, key in ((side, "same"), (other, "cross")):
        nm = "kc_%s_to_motor_%s" % (side, tgt)
        pre = snap[nm + "__pre"]
        g = w[nm]
        if pre.size != g.size:
            raise RuntimeError("%s 연결 %d ≠ 가중치 %d" % (nm, pre.size, g.size))
        s = np.bincount(pre, weights=g, minlength=N_KC)[:N_KC]
        c = np.bincount(pre, minlength=N_KC)[:N_KC]
        out[key] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out["same"], out["cross"]


def cat_classes(b, side):
    """E183 무가소성(Z) 측정 tag=pre, 노출 사례 반응 집합 → A 전용·B 전용·둘 다(bool 배열)."""
    t = open(os.path.join(EXP, "logs", "E183", "rc_b%d_Z.log" % b), encoding="utf-8", errors="replace").read()
    cnt = {"A": np.zeros(N_KC), "B": np.zeros(N_KC)}
    n = {"A": 0, "B": 0}
    for m in re.finditer(r"^RCKC tag=pre side=%s set=train cat=([AB]) i=\d+ n=\d+ ids=(.*)$" % side, t, re.M):
        ids = [int(x) for x in m.group(2).split(",") if x]
        cnt[m.group(1)][ids] += 1
        n[m.group(1)] += 1
    if n != {"A": 6, "B": 6}:
        raise RuntimeError("E183 b%d %s 노출 사례 집합 수 %s" % (b, side, n))
    a_only = (cnt["A"] >= 3) & (cnt["B"] <= 1)
    b_only = (cnt["B"] >= 3) & (cnt["A"] <= 1)
    both = (cnt["A"] >= 3) & (cnt["B"] >= 3)
    return a_only, b_only, both


def xor_classes(b, side, cfg="C2"):
    """E192 측정 RCKX(노출 제시) 칸 평균 → 결합(부호: + 교차 칸 선호)·그 밖 반응·무반응."""
    t = open(os.path.join(EXP, "logs", "E192", "meas_b%d_%s.log" % (b, cfg)), encoding="utf-8", errors="replace").read()
    acc = {k: np.zeros(N_KC) for k in ("AX", "AY", "BX", "BY")}
    n = {k: 0 for k in acc}
    for m in re.finditer(r"^RCKX rep=\d+ side=%s(?: k=\d+)? set=train cat=([AB]) ctx=([XY]) i=\d+ n=\d+ ids=(.*)$" % side, t, re.M):
        key = m.group(1) + m.group(2)
        for tok in m.group(3).split(","):
            if tok:
                k, c = tok.split(":")
                acc[key][int(k)] += float(c)
        n[key] += 1
    if set(n.values()) != {18}:
        raise RuntimeError("E192 b%d %s 칸 제시 수 %s" % (b, side, n))
    m_ = {k: acc[k] / n[k] for k in acc}
    inter = ((m_["AX"] + m_["BY"]) - (m_["AY"] + m_["BX"])) / 2.0      # + = 교차 칸(AX·BY) 선호
    mean4 = (m_["AX"] + m_["AY"] + m_["BX"] + m_["BY"]) / 4.0
    conj = (np.abs(inter) >= 0.5) & (mean4 >= 0.25)
    resp = (mean4 >= 0.25) & ~conj
    return conj, np.sign(inter), resp


def report_cat(b, wpath):
    snap = np.load(os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b))
    w = np.load(wpath)
    out = []
    for side in "lr":
        ws, wc = per_kc(snap, w, side)
        a_only, b_only, both = cat_classes(b, side)
        dA = np.nanmean((wc - ws)[a_only]); dB = np.nanmean((ws - wc)[b_only]); dO = np.nanmean((wc - ws)[both])
        out.append((side, int(a_only.sum()), dA, int(b_only.sum()), dB, int(both.sum()), dO))
    return out


def report_xor(b, wpath):
    snap = np.load(os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b))
    w = np.load(wpath)
    out = []
    for side in "lr":
        ws, wc = per_kc(snap, w, side)
        conj, sg, resp = xor_classes(b, side)
        aligned = np.where(sg > 0, wc - ws, ws - wc)        # 결합 KC: 선호 칸의 정답 방향으로 기운 정도
        out.append((side, int(conj.sum()), float(np.nanmean(aligned[conj])), float(np.nanmax(np.abs((wc - ws)[conj]))) if conj.any() else float("nan"),
                    int(resp.sum()), float(np.nanmean(np.abs(wc - ws)[resp])), float(np.nanmean((wc - ws)[resp]))))
    return out


def _selftest():
    """합성: 알려진 KC 부류에 가중치 차를 심은 연결·가중치로 per_kc 정렬 확인."""
    rs = np.random.RandomState(1)
    snap, w = {}, {}
    for s in "lr":
        for t in "lr":
            nm = "kc_%s_to_motor_%s" % (s, t)
            pre = np.sort(rs.randint(0, N_KC, 20000)); snap[nm + "__pre"] = pre
            g = np.full(pre.size, 150.0)
            if t != s:          # 교차 motor: KC 0~99 는 +40
                g[pre < 100] += 40.0
            w[nm] = g
    ws, wc = per_kc(snap, w, "l")
    ok = abs(np.nanmean((wc - ws)[:100]) - 40.0) < 1e-9 and abs(np.nanmean((wc - ws)[100:])) < 1e-9
    print("  per_kc 정렬(심은 차 +40 회수) %s" % ("✓" if ok else "✗"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "selftest":
        r = _selftest()
        print("kcw_decomp 자체 시험: %s" % ("통과" if r else "실패"))
        sys.exit(0 if r else 1)
    mode, b = sys.argv[1], int(sys.argv[2])
    for wp in sys.argv[3:]:
        if mode == "cat":
            for side, nA, dA, nB, dB, nO, dO in report_cat(b, wp):
                print("%s b%d %s | A 전용 %d: w교차−w같은 %+.2f | B 전용 %d: w같은−w교차 %+.2f | 둘 다 %d: w교차−w같은 %+.2f" % (os.path.basename(wp), b, side, nA, dA, nB, dB, nO, dO))
        else:
            for side, nC, dC, mx, nR, absR, dR in report_xor(b, wp):
                print("%s b%d %s | 결합 %d: 정답 방향 기울기 %+.2f (최대 |차| %.1f) | 그 밖 반응 %d: |차| %.2f·평균 차 %+.2f" % (os.path.basename(wp), b, side, nC, dC, mx, nR, absR, dR))
