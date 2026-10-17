#!/usr/bin/env python3
"""E203 판정 — 활동 가중 정규화(반응 KC 무리의 교차−같은 쪽 평균을 매 시행 0 으로)를 더한 E200 학습(경쟁형 규칙·η 2.4·도파민 기준선)으로 맥락 의존 범주(L4)가 서는가.
규칙은 research/experiments/logs/E203/criteria_fixed.txt. 판정 범주는 E193·E199·E200·E201 과 같다.
입력(뇌 10~14): 학습 logs/E203/train_b{B}.log, 평가 logs/E203/ev_b{B}_{learn|none}_{X|Y}.log, 가중치 traces/E203/w_b{B}.npz, 스냅숏 traces/E176/snap/conn_b{B}.npz,
  회귀 기준 logs/E193/ev_b{B}_none_{X|Y}.log, 비교 traces/E200/w_b{B}.npz·logs/E200/ev_b{B}_{learn,none}_{X|Y}.log, 지수 traces/E196/sxi_b{B}.npz.
조작검증(하나라도 어기면 보류): E200 과 같은 학습 줄·규칙 줄(가산형·정규화 3000회·뒤 ≤ 1e-3·hard=1·eta 2.4)·기준선 줄(시행 3000·|평균 DA| ≤ 0.05·기준선 1·점검 3)·가중치(범위·KC 별 합 300 ≤ 0.01·E200 가중치와 다름)·평가 줄·none 회귀(E193, 8값 |차| ≤ 2),
  + '[E203 활동 가중 정규화]' 점검 3줄·'적용 3000회 … 뒤 최대 ≤ 1e-3', 저장 가중치 KC 쪽별 활동 가중 |M| ≤ 1.0(반응 KC — SXI 파일 resp·tot).
판정(우선순위): 조작검증 실패 → 보류 / 맥락 의존 범주 ≥ 4/5 → 활동 가중 정규화로 L4 학습(H124) / 사례 한정 ≥ 4/5 / 부분 ≥ 4/5 / 실패 ≥ 4/5 → H124-null / 그 밖 보류.
(보고) 같은 뇌 E200 칸 값, 300 받는 반응 KC motor_l·motor_r(좌우 치우침), 경계 시냅스 비율, 반응 KC |Δ| ≥ 250 수, 평균 DA, 저장 가중치 활동 가중 M(쪽별).
실행: python3 scripts/judge_e203.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from judge_e193 import CELLS, read_eval, read_train  # noqa: E402
import judge_e199 as J9  # noqa: E402 — read_comp·weight_checks(합성 24/24)

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
NT = 3000
ETA = 2.4


def read_da(path):
    r = {"stat": None, "chk": 0}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[E200 도파민 통계] "):
            m = re.search(r"시행 (\d+) · 평균 DA ([+-]?[0-9.]+|nan) · 평균 \|DA\| ([0-9.]+|nan) · 기준선 (\d)", ln)
            r["stat"] = (int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))) if m else ("?",)
        elif ln.startswith("[E200 도파민 기준선] 시행 "):
            r["chk"] += 1
    return r


def read_actw(path):
    """'[E203 활동 가중 정규화] 적용 N회 … 뒤 최대 x · 반복 최대 k' 와 점검 줄 수."""
    m, chk = None, 0
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[E203 활동 가중 정규화] 적용 "):
            mm = re.search(r"적용 (\d+)회 .*뒤 최대 (\S+) · 반복 최대 (\d+)", ln.strip())
            m = (int(mm.group(1)), float(mm.group(2)), int(mm.group(3))) if mm else ("?",)
        elif ln.startswith("[E203 활동 가중 정규화] 시행 "):
            chk += 1
    return m, chk


def actw_offsets(wfile, sfile, npz):
    """저장 가중치의 KC 쪽별 활동 가중 M = Σ tot·Δ / Σ tot(반응 KC, 두 경로 모두 시냅스가 있는 KC). Δ = 교차 평균 − 같은 쪽 평균."""
    W, S, Z = np.load(wfile), np.load(sfile), np.load(npz)
    out = {}
    for kn in "lr":
        oc = "r" if kn == "l" else "l"
        mean, cnt = {}, {}
        for m in (oc, kn):
            nm = "kc_%s_to_motor_%s" % (kn, m)
            pre = S[nm + "__pre"].astype(np.int64)
            cnt[m] = np.bincount(pre, minlength=3000).astype(np.float64)
            mean[m] = np.bincount(pre, weights=np.asarray(W[nm], dtype=np.float64), minlength=3000) / np.maximum(cnt[m], 1.0)
        sel = (np.asarray(Z["resp_" + kn]) > 0) & (cnt[oc] > 0) & (cnt[kn] > 0)
        w = np.where(sel, np.asarray(Z["tot_" + kn], dtype=np.float64), 0.0)
        out[kn] = float(np.sum(w * (mean[oc] - mean[kn])) / w.sum()) if w.sum() > 0 else 0.0
    return out


def side_counts(D, npz):
    Z = np.load(npz)
    to = {"l": 0, "r": 0}
    for sd in "lr":
        rp = Z["resp_" + sd].astype(bool)
        oc = "r" if sd == "l" else "l"
        to[oc] += int(np.sum(rp & (D[sd] > 0)))
        to[sd] += int(np.sum(rp & (D[sd] < 0)))
    return to


def main():
    ok, why, E, A9, REP = True, [], {}, {}, {}
    for b in BRAINS:
        tp = os.path.join(EXP, "logs", "E203", "train_b%d.log" % b)
        npz = os.path.join(EXP, "traces", "E196", "sxi_b%d.npz" % b)
        try:
            tr = read_train(tp)
            cp = J9.read_comp(tp)
            da = read_da(tp)
            aw, achk = read_actw(tp)
            am = actw_offsets(os.path.join(EXP, "traces", "E203", "w_b%d.npz" % b), os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b), npz)
            ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E203", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
            ref = {x: read_eval(os.path.join(EXP, "logs", "E193", "ev_b%d_none_%s.log" % (b, x)), x)[0] for x in "XY"}
            ev9 = {(w, x): read_eval(os.path.join(EXP, "logs", "E200", "ev_b%d_%s_%s.log" % (b, w, x)), x)[0] for w in ("learn", "none") for x in "XY"}
            wc = J9.weight_checks(os.path.join(EXP, "traces", "E203", "w_b%d.npz" % b), os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b),
                                  os.path.join(EXP, "traces", "E200", "w_b%d.npz" % b))
            Z = np.load(npz)
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
                and cp["eta"] is not None and abs(cp["eta"] - ETA) <= 1e-9):
            ok = False; why.append("b%d 규칙 줄(hard %s·정규화 %s·eta %s)" % (b, cp["hard"], cm, cp["eta"]))
        st = da["stat"]
        if not (st is not None and len(st) == 4 and st[0] == NT and abs(st[1]) <= 0.05 and st[3] == 1 and da["chk"] == 3):
            ok = False; why.append("b%d 기준선 줄(%s·점검 %d)" % (b, st, da["chk"]))
        if not (aw is not None and len(aw) == 3 and aw[0] == NT and aw[1] <= 1e-3 and achk == 3):
            ok = False; why.append("b%d 활동 가중 정규화 줄 %s·점검 %d" % (b, aw, achk))
        pdev = max(abs(v) for v in am.values())
        if pdev > 1.0:
            ok = False; why.append("b%d 저장 가중치 활동 가중 |M| %.3f" % (b, pdev))
        lo_hi, dmax, diff, fb, D = wc
        if not (lo_hi and dmax <= 0.01 and diff):
            ok = False; why.append("b%d 가중치(범위 %s·KC 합 최대 |S−300| %.4g·E200 와 다름 %s)" % (b, lo_hi, dmax, diff))
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
            A9[b] = {(x, cc): sg * (ev9[("learn", x)][(cc, "train")] - ev9[("none", x)][(cc, "train")]) for x, cc, sg in CELLS}
        rp = {sd: Z["resp_" + sd].astype(bool) for sd in "lr"}
        nsat = sum(int(np.sum(rp[sd] & (np.abs(D[sd]) >= 250))) for sd in "lr"); nresp = sum(int(rp[sd].sum()) for sd in "lr")
        REP[b] = (fb, nsat, nresp, side_counts(D, npz), st[1] if st and len(st) == 4 else float("nan"), pdev)
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
        r9 = A9[b]
        fb, nsat, nresp, to, mda, pdev = REP[b]
        print("b%d 규칙 방향 효과(1e-4, 노출/새) X·A %+d/%+d X·B %+d/%+d Y·A %+d/%+d Y·B %+d/%+d | 획득 칸 %d | (보고) E200 X·A %+d X·B %+d Y·A %+d Y·B %+d · 300 받는 반응 KC motor_l %d · motor_r %d · 경계 시냅스 %.3f · |Δ|≥250 %d/%d · 평균 DA %+.4f · 활동 가중 최대 |M| %.3f"
              % (b, a[("X", "A", "train")], a[("X", "A", "new")], a[("X", "B", "train")], a[("X", "B", "new")],
                 a[("Y", "A", "train")], a[("Y", "A", "new")], a[("Y", "B", "train")], a[("Y", "B", "new")], acq_cells,
                 r9[("X", "A")], r9[("X", "B")], r9[("Y", "A")], r9[("Y", "B")], to["l"], to["r"], fb, nsat, nresp, mda, pdev))
    print("조작검증 통과(규칙 줄·기준선 줄·활동 가중 정규화 줄·저장 가중치 활동 가중 M·KC 별 합 300·E193 none 회귀 포함) | 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · 부분 %d/5 · 획득 실패 %d/5"
          % (n["conc"], n["memo"], n["part"], n["fail"]))
    v = ("활동 가중 정규화로 L4 학습 — 맥락 의존 범주(H124)" if n["conc"] >= 4 else "맥락 의존(사례 한정)" if n["memo"] >= 4
         else "부분" if n["part"] >= 4 else "획득 실패(H124-null)" if n["fail"] >= 4 else "보류")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
