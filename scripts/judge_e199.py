#!/usr/bin/env python3
"""E199 판정 — 경쟁형 학습 규칙(가산형 경성 상한 R-STDP + KC 별 두 motor 평균 합 보존 감산 정규화)으로 맥락 의존 범주(헌장 L4 — 범주 × 맥락 XOR)를 배우는가.
규칙은 research/experiments/logs/E199/criteria_fixed.txt. 판정 범주는 E193(judge_e193)과 같다.
입력(뇌 10~14): 학습 logs/E199/train_b{B}.log, 평가 logs/E199/ev_b{B}_{learn|none}_{X|Y}.log, 가중치 traces/E199/w_b{B}.npz,
  초기 연결 스냅숏 traces/E176/snap/conn_b{B}.npz(KC→motor __pre), 보정 학습률 logs/E199/calib_eta.txt, 회귀 기준 logs/E193/ev_b{B}_none_{X|Y}.log, 비교 traces/E193/w_b{B}.npz.
지표(1e-4 정수): e = m(learn) − m(none); a = −e(교차 칸 X·A·Y·B), +e(같은 쪽 칸 X·B·Y·A). 뇌 단위: 획득 칸 n = #{a_train ≥ 1000}; 획득 = n 4;
  일반화 = 네 칸 a_new ≥ 500 이면서 2·a_new ≥ a_train. 맥락 의존 범주 = 획득·일반화, 사례 한정 = 획득·일반화 아님, 부분 = 2~3 칸, 실패 = ≤ 1 칸.
조작검증(하나라도 어기면 보류): 학습 줄(E193 과 같은 검사: catctx·3,000시행·맥락/범주 수 범위·특징·맥락 점검 3줄·[학습] 줄·종류 입력 적재),
  '[E199 가산형(경성 상한)' 줄, '[E199 경쟁 정규화] 적용 3000회 … 정규화 뒤 최대 ≤ 1e-3 · hard=1', KC→motor 줄의 eta = 보정 학습률,
  가중치 파일: KC→motor 4집단 값 [0, 300], KC 별 mean(g→motor_l) + mean(g→motor_r) = 300(|차| ≤ 0.01, 두 집단에 시냅스 있는 KC 전부), E193 가중치와 다름,
  평가 줄(E193 과 같음: RCEVAL 4·trials 96·ctx·포화 없음·종류 입력·특징 가중치 없음·pushed 8/0), none 회귀(E193 none 과 뇌마다 8값 |차| ≤ 2).
판정(우선순위): 조작검증 실패 → 보류 / 맥락 의존 범주 ≥ 4/5 → 경쟁형 규칙으로 L4 학습(H120) / 사례 한정 ≥ 4/5 / 부분 ≥ 4/5 / 실패 ≥ 4/5 → H120-null / 그 밖 보류.
(보고) 뇌마다 같은 뇌 E193(연성 상한) 칸 값·차, KC→motor 시냅스 중 경계(0·300) 비율, 반응 KC(E196 지수) |Δ| ≥ 250 비율.
실행: python3 scripts/judge_e199.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backend", "genesis"))
from judge_e193 import CELLS, read_eval, read_train  # noqa: E402 — 검증된 E193 파서(같은 판정)
import judge_e198 as J8  # noqa: E402 — Δ(가중치 + 스냅숏 pre)
import kc_comp  # noqa: E402 — KC 별 합(합성 시험 12/12)

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
NT = 3000
POPS = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")


def read_eta(path):
    with open(path, encoding="utf-8") as f:
        m = re.search(r"eta=([0-9.]+)", f.read())
    if not m:
        raise ValueError("보정 학습률 없음")
    return float(m.group(1))


def read_comp(path):
    r = {"hard": False, "comp": None, "eta": None}
    for ln in open(path, encoding="utf-8", errors="replace"):
        s = ln.strip()
        if s.startswith("KC→motor [E199 가산형(경성 상한)"):
            r["hard"] = True
        elif s.startswith("KC→motor [E109 R-STDP 4방향]"):
            m = re.search(r"eta=([0-9.eE+-]+)", s)
            r["eta"] = float(m.group(1)) if m else None
        elif s.startswith("[E199 경쟁 정규화] 적용 "):
            m = re.search(r"적용 (\d+)회 .*정규화 뒤 최대 ([0-9.eE+-]+|nan) · hard=(\d)", s)
            r["comp"] = (int(m.group(1)), float(m.group(2)), int(m.group(3))) if m else ("?",)
    return r


def weight_checks(wfile, snapfile, e193file, n_kc=3000):
    """반환 (범위 ok, 합 최대 |S−300|, E193 과 다름, 경계 비율, Δ)."""
    W = np.load(wfile); S = np.load(snapfile); W3 = np.load(e193file)
    lo_hi = all(float(W[p].min()) >= 0.0 and float(W[p].max()) <= 300.0 for p in POPS)
    dmax = 0.0
    for kn in "lr":
        a, b = "kc_%s_to_motor_l" % kn, "kc_%s_to_motor_r" % kn
        s, both = kc_comp.kc_sums(S[a + "__pre"], W[a], S[b + "__pre"], W[b], n_kc)
        dmax = max(dmax, float(np.max(np.abs(s[both] - 300.0))))
    diff = any(not np.array_equal(W[p], W3[p]) for p in POPS)
    nb = sum(int(np.sum((W[p] == 0.0) | (W[p] == 300.0))) for p in POPS); nt = sum(int(W[p].size) for p in POPS)
    D, _ = J8.deltas(wfile, snapfile, n_kc)
    return lo_hi, dmax, diff, nb / max(nt, 1), D


def main():
    ok, why, E, A3, REP = True, [], {}, {}, {}
    try:
        eta = read_eta(os.path.join(EXP, "logs", "E199", "calib_eta.txt"))
    except (OSError, ValueError) as ex:
        print("판정: 보류(결측 — 보정 학습률 %s)" % type(ex).__name__)
        return 0
    for b in BRAINS:
        tp = os.path.join(EXP, "logs", "E199", "train_b%d.log" % b)
        try:
            tr = read_train(tp)
            cp = read_comp(tp)
            ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E199", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
            ref = {x: read_eval(os.path.join(EXP, "logs", "E193", "ev_b%d_none_%s.log" % (b, x)), x)[0] for x in "XY"}
            ev3 = {(w, x): read_eval(os.path.join(EXP, "logs", "E193", "ev_b%d_%s_%s.log" % (b, w, x)), x)[0] for w in ("learn", "none") for x in "XY"}
            wc = weight_checks(os.path.join(EXP, "traces", "E199", "w_b%d.npz" % b), os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b),
                               os.path.join(EXP, "traces", "E193", "w_b%d.npz" % b))
            Z = np.load(os.path.join(EXP, "traces", "E196", "sxi_b%d.npz" % b))
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        c, k = tr["ctx"], tr["cat"]
        if not (tr["catctx"] and tr["done"] and tr["kt"] >= 1 and c and k and c[0] == NT and c[1] + c[2] == NT and c[3] + c[4] == NT
                and 1200 <= c[1] <= 1800 and 1200 <= c[3] <= 1800 and k[0] == NT and 1200 <= k[1] <= 1800):
            ok = False; why.append("b%d 학습 줄(%s %s)" % (b, c, k))
        if len(tr["fchk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["fchk"]):
            ok = False; why.append("b%d 특징 점검 %s" % (b, tr["fchk"]))
        if len(tr["cchk"]) != 3 or not all(x > 0 and y == 0 and z == 0 for x, y, z in tr["cchk"]):
            ok = False; why.append("b%d 맥락 점검 %s" % (b, tr["cchk"]))
        cm = cp["comp"]
        if not (cp["hard"] and cm is not None and len(cm) == 3 and cm[0] == NT and cm[1] <= 1e-3 and cm[2] == 1
                and cp["eta"] is not None and abs(cp["eta"] - eta) <= 1e-9):
            ok = False; why.append("b%d 규칙 줄(hard %s·정규화 %s·eta %s 대 보정 %s)" % (b, cp["hard"], cm, cp["eta"], eta))
        lo_hi, dmax, diff, fb, D = wc
        if not (lo_hi and dmax <= 0.01 and diff):
            ok = False; why.append("b%d 가중치(범위 %s·KC 합 최대 |S−300| %.4g·E193 과 다름 %s)" % (b, lo_hi, dmax, diff))
        m = {}
        for (w, x), (rc, pushed, kt, fw, sat, bad) in ev.items():
            if set(rc) != {(cc, s) for cc in "AB" for s in ("train", "new")} or bad or sat or kt < 1 or fw != 0 or pushed != (8 if w == "learn" else 0):
                ok = False; why.append("b%d %s %s 평가(줄 %d·나쁨 %s·포화 %s·종류 %d·특징 %d·pushed %s)" % (b, w, x, len(rc), bad, sat, kt, fw, pushed))
                continue
            for (cc, s), v in rc.items():
                m[(w, x, cc, s)] = v
        for x in "XY":
            for cc in "AB":
                for s in ("train", "new"):
                    if ("none", x, cc, s) in m and abs(m[("none", x, cc, s)] - ref[x][(cc, s)]) > 2:
                        ok = False; why.append("b%d none 회귀 %s %s %s %d vs E193 %d" % (b, x, cc, s, m[("none", x, cc, s)], ref[x][(cc, s)]))
        if len(m) == 16:
            E[b] = {(x, cc, s): m[("learn", x, cc, s)] - m[("none", x, cc, s)] for x in "XY" for cc in "AB" for s in ("train", "new")}
            A3[b] = {(x, cc): sg * (ev3[("learn", x)][(cc, "train")] - ev3[("none", x)][(cc, "train")]) for x, cc, sg in CELLS}
        rp = {sd: Z["resp_" + sd].astype(bool) for sd in "lr"}
        nsat = sum(int(np.sum(rp[sd] & (np.abs(D[sd]) >= 250))) for sd in "lr"); nresp = sum(int(rp[sd].sum()) for sd in "lr")
        REP[b] = (fb, nsat, nresp)
    if not ok or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    n = {"conc": 0, "memo": 0, "part": 0, "fail": 0}
    for b in BRAINS:
        e = E[b]
        a = {(x, cc, s): sg * e[(x, cc, s)] for x, cc, sg in CELLS for s in ("train", "new")}
        acq_cells = sum(a[(x, cc, "train")] >= 1000 for x, cc, _ in CELLS)
        acq = acq_cells == 4
        gen = all(a[(x, cc, "new")] >= 500 and 2 * a[(x, cc, "new")] >= a[(x, cc, "train")] for x, cc, _ in CELLS)
        n["conc"] += acq and gen
        n["memo"] += acq and not gen
        n["part"] += 2 <= acq_cells <= 3
        n["fail"] += acq_cells <= 1
        r3 = A3[b]
        fb, nsat, nresp = REP[b]
        print("b%d 규칙 방향 효과(1e-4, 노출/새) X·A %+d/%+d X·B %+d/%+d Y·A %+d/%+d Y·B %+d/%+d | 획득 칸 %d | (보고) E193 X·A %+d X·B %+d Y·A %+d Y·B %+d · 경계 시냅스 %.3f · 반응 KC |Δ|≥250 %d/%d"
              % (b, a[("X", "A", "train")], a[("X", "A", "new")], a[("X", "B", "train")], a[("X", "B", "new")],
                 a[("Y", "A", "train")], a[("Y", "A", "new")], a[("Y", "B", "train")], a[("Y", "B", "new")], acq_cells,
                 r3[("X", "A")], r3[("X", "B")], r3[("Y", "A")], r3[("Y", "B")], fb, nsat, nresp))
    print("조작검증 통과(규칙 줄·KC 별 합 300·E193 none 회귀 포함, 보정 eta=%g) | 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · 부분 %d/5 · 획득 실패 %d/5"
          % (eta, n["conc"], n["memo"], n["part"], n["fail"]))
    v = ("경쟁형 규칙으로 L4 학습 — 맥락 의존 범주(H120)" if n["conc"] >= 4 else "맥락 의존(사례 한정)" if n["memo"] >= 4
         else "부분" if n["part"] >= 4 else "획득 실패(H120-null)" if n["fail"] >= 4 else "보류")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
