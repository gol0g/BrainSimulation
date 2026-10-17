#!/usr/bin/env python3
"""E202 독립 대조 — judge_e202 와 따로 구현한다(공유 코드 없음).
- 학습 Δ: 정렬 + np.add.reduceat 으로 KC 별 경로 평균(judge 는 verify_e198_independent.learned 의 np.add.at).
- 무리: 반복문으로 고른 KC 번호 목록(judge 는 불리언 마스크).
- 분해: D_obs = 범주 합, D_mag = Σ_r (W_Sr + W_Cr)·(m_Sr + m_Cr)/2, D_freq = D_obs − D_mag(잔차 — judge 는 직접 계산).
- 기전·종합 규칙은 criteria_fixed.txt 그대로(같은 문장을 따로 구현).
실행: 저장소 루트에서 python3 scripts/verify_e202_independent.py [기준 디렉터리]"""
import math
import os
import sys

import numpy as np

EXP = os.path.join("research", "experiments")
CL = ["S+", "S-", "C+", "C-"]
TAG = {"S+": "Sp", "S-": "Sm", "C+": "Cp", "C-": "Cm"}


def kc_delta(wf, sf, n=3000):
    W, S = np.load(wf), np.load(sf)
    out = {}
    for kn in "lr":
        oc = "r" if kn == "l" else "l"
        mean = {}
        for m in (oc, kn):
            nm = "kc_%s_to_motor_%s" % (kn, m)
            p = S[nm + "__pre"].astype(np.int64)
            w = W[nm].astype(np.float64)
            o = np.argsort(p, kind="stable")
            ps, ws = p[o], w[o]
            starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
            sums = np.add.reduceat(ws, starts)
            cnts = np.diff(np.r_[starts, ps.size])
            mv = np.full(n, np.nan)
            mv[ps[starts]] = sums / cnts
            mean[m] = mv
        d = mean[oc] - mean[kn]
        d[np.isnan(d)] = 0.0
        out[kn] = d
    return out


def members(Z, lo, hi):
    """(쪽, 번호) 목록: 교차(SXI > 0)·같은 쪽(SXI < 0) — 반응 KC, lo ≤ |SXI| < hi."""
    c, s = [], []
    for sd in "lr":
        r, x = Z["resp_" + sd], Z["sxi_" + sd]
        for k in range(len(r)):
            if r[k] <= 0:
                continue
            v = round(float(x[k]), 9)
            if v == 0 or not (lo <= abs(v) < hi):
                continue
            (c if v > 0 else s).append((sd, k))
    return c, s


def gmean(arr, mem):
    if not mem:
        return float("nan")
    return sum(float(arr[sd][k]) for sd, k in mem) / len(mem)


def balanced(arr, c, s):
    return (gmean(arr, c) + gmean(arr, s)) / 2


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def verify(base="."):
    ex = os.path.join(base, EXP)
    tally = {"M1": 0, "M2": 0, "혼합": 0}
    fails = []
    for b in (10, 11, 12, 13, 14):
        try:
            D = np.load(os.path.join(ex, "traces", "E202", "dg_b%d.npz" % b))
            Z = np.load(os.path.join(ex, "traces", "E196", "sxi_b%d.npz" % b))
            sf = os.path.join(ex, "traces", "E176", "snap", "conn_b%d.npz" % b)
            d2 = kc_delta(os.path.join(ex, "traces", "E202", "w_b%d.npz" % b), sf)
            d0 = kc_delta(os.path.join(ex, "traces", "E200", "w_b%d.npz" % b), sf)
            log = open(os.path.join(ex, "logs", "E202", "train_b%d.log" % b), encoding="utf-8", errors="replace").read().splitlines()
        except Exception as e_:  # noqa: BLE001
            print("독립 판정: 보류(결측 — b%d %s)" % (b, type(e_).__name__))
            return 1
        why = []
        # 로그 — 줄 머리와 숫자를 따로 읽는다
        def has(prefix, *needles):
            return any(ln.startswith(prefix) and all(nd in ln for nd in needles) for ln in log)
        if not has("[맥락 범주 과제] 시행 3000 "):
            why.append("과제")
        if not has("[E199 경쟁 정규화] 적용 3000회", "hard=1"):
            why.append("규칙")
        da_ok = False
        for ln in log:
            if ln.startswith("[E200 도파민 통계] 시행 3000 ") and "기준선 1" in ln:
                v = float(ln.split("평균 DA ")[1].split()[0])
                da_ok = abs(v) <= 0.05
        if not da_ok:
            why.append("기준선")
        if not has("[E202 계측] 시행 3000 · "):
            why.append("계측 요약")
        if sum(1 for ln in log if ln.startswith(("[E202 계측] 시행 1 범주", "[E202 계측] 시행 2 범주", "[E202 계측] 시행 3 범주"))) != 3:
            why.append("계측 점검")
        if np.asarray(D["rows"]).shape[0] != 3000 or int(sum(int(v) for v in D["N"])) != 3000:
            why.append("계측 행")
        for sd in "lr":
            acc = np.zeros(3000)
            for c in CL:
                acc = acc + D["x_%s_%s" % (TAG[c], sd)]
            if np.max(np.abs(acc - D["tot_x_" + sd])) > 1e-6:
                why.append("범주 합")
        cw, swk = members(Z, 0.0, 0.1)
        # (1) 재구성
        meas = np.r_[D["tot_x_l"], D["tot_x_r"]].astype(np.float64)
        lern = np.r_[d2["l"], d2["r"]]
        r1 = pearson(meas, lern)
        sl = float(np.dot(meas, lern) / np.dot(meas, meas))
        tot = {"l": D["tot_x_l"], "r": D["tot_x_r"]}
        om, ol = balanced(tot, cw, swk), balanced(d2, cw, swk)
        rel = abs(om - ol) / abs(ol) if ol != 0 else float("inf")
        if not (r1 >= 0.95 and 0.9 <= sl <= 1.1 and rel <= 0.2):
            why.append("재구성")
        # (2) E200 회귀 — 반응 KC
        ra = np.r_[d2["l"][Z["resp_l"] > 0], d2["r"][Z["resp_r"] > 0]]
        rb = np.r_[d0["l"][Z["resp_l"] > 0], d0["r"][Z["resp_r"] > 0]]
        keep = (ra != 0) & (rb != 0)
        ag = float(np.sum(np.sign(ra[keep]) == np.sign(rb[keep])) / max(int(keep.sum()), 1))
        r2 = pearson(ra, rb)
        if not (ag >= 0.97 and r2 >= 0.99):
            why.append("E200 회귀")
        # (3) 정정 1 — 자기 쪽 자극 몫: KC 마다 자극 쪽 = 그 KC 쪽 인 시행의 x 합을 직접 더해 균형 평균
        def own_val(sd, k):
            return sum(float(D["xs_%s_%s_%s" % (TAG[c], sd, sd)][k]) for c in CL)
        def tot_val(sd, k):
            return float(D["tot_x_" + sd][k])
        def bal_fn(fn):
            ms = [sum(fn(sd, k) for sd, k in grp) / len(grp) if grp else float("nan") for grp in (cw, swk)]
            return (ms[0] + ms[1]) / 2
        o_all = bal_fn(tot_val)
        share = bal_fn(own_val) / o_all if o_all != 0 else float("nan")
        if not (share == share and share >= 0.7):
            why.append("분류 오염")
        # 분해(약한 무리)
        Wc = {c: float(v) for c, v in zip(CL, D["W"])}
        Xc = {c: balanced({"l": D["x_%s_l" % TAG[c]], "r": D["x_%s_r" % TAG[c]]}, cw, swk) for c in CL}
        dobs = sum(Xc.values())
        mc = {c: Xc[c] / Wc[c] for c in CL}
        dmag = sum((Wc["S" + r] + Wc["C" + r]) * (mc["S" + r] + mc["C" + r]) / 2 for r in "+-")
        dfreq = dobs - dmag
        if dobs != 0 and abs(dmag) >= 2 * abs(dfreq) and (dmag > 0) == (dobs > 0):
            mk = "M2"
        elif dobs != 0 and abs(dfreq) >= 2 * abs(dmag) and (dfreq > 0) == (dobs > 0):
            mk = "M1"
        else:
            mk = "혼합"
        tally[mk] += 1
        if why:
            fails.append(b)
        print("b%d | D_obs %+.3f D_freq %+.3f D_mag %+.3f %s | 재구성 r %.4f 기울기 %.3f offset %+.3f/%+.3f | 회귀 부호 %.4f r %.4f | 자기 쪽 몫 %.3f | %s"
              % (b, dobs, dfreq, dmag, mk, r1, sl, om, ol, ag, r2, share, "통과" if not why else "실패 " + ",".join(why)))
    print("M2 %d M1 %d 혼합 %d · 계측 검증 실패 %s" % (tally["M2"], tally["M1"], tally["혼합"], fails if fails else "없음"))
    if fails:
        v = "보류(계측 검증 실패)"
    elif tally["M2"] >= 4:
        v = "흔적 크기 비대칭(H123)"
    elif tally["M1"] >= 4:
        v = "결과 빈도 비대칭(H123-alt)"
    else:
        v = "혼합"
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(verify(sys.argv[1] if len(sys.argv) > 1 else "."))
