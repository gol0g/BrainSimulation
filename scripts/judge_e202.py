#!/usr/bin/env python3
"""E202 판정 — 선택성 약한 반응 KC 의 같은 쪽 끌림은 결과 빈도(M1)와 흔적 크기(M2) 중 어디서 오는가(측정).
규칙은 research/experiments/logs/E202/criteria_fixed.txt. 설계는 research/abcd-2026-10-17-ctxtrace.md C 절.

입력(뇌 10~14): 계측 traces/E202/dg_b{B}.npz(backend/genesis/dg_trace), 학습 가중치 traces/E202/w_b{B}.npz, E200 가중치 traces/E200/w_b{B}.npz,
스냅숏 traces/E176/snap/conn_b{B}.npz(pre), SXI traces/E196/sxi_b{B}.npz, 학습 로그 logs/E202/train_b{B}.log.

무리: 반응 KC 중 SXI 크기 < 0.1(약)·≥ 0.5(강), 부호로 교차(SXI > 0)·같은 쪽(SXI < 0). 무리 값은 균형 평균 = (교차 KC 평균 + 같은 쪽 KC 평균)/2(offset).
분해(범주 c ∈ S+·S−·C+·C−, S = 실행 motor 가 자극 쪽, + = 보상): X_c = 무리 균형 평균 Σ_{t∈c} x, W_c = Σ_{t∈c} DA, m_c = X_c / W_c,
짝 r ∈ {+, −} 마다 μ_r = (m_Cr − m_Sr)/2, d_r = (m_Sr + m_Cr)/2 → D_obs = Σ X_c = D_freq + D_mag,
D_freq = Σ μ_r·(W_Cr − W_Sr)(같은 크기에서 도파민 합 비대칭), D_mag = Σ d_r·(W_Sr + W_Cr)(같은 도파민에서 크기 비대칭).
뇌마다: |D_mag| ≥ 2|D_freq| 이고 D_mag 와 D_obs 부호 같음 → M2 / |D_freq| ≥ 2|D_mag| 이고 부호 같음 → M1 / 그 밖 혼합.
조작검증(계측 검증, 뇌마다): 학습 로그(맥락 범주 과제 3000·경쟁 정규화 3000회 hard=1·도파민 통계 3000 |평균| ≤ 0.05 기준선 1·계측 줄 3000·점검 3줄),
계측 파일(행 3000·N 합 3000·범주 합 = 총 x), (1) KC 별 총 x 와 저장 가중치 학습 Δ: 상관 ≥ 0.95·원점 기울기 0.9~1.1·약한 무리 offset 상대 오차 ≤ 0.2,
(2) 반응 KC Δ 가 E200 과 부호 일치 ≥ 0.97·상관 ≥ 0.99, (3) 정정 1(결과 전): 약한 무리 D_obs 중 자기 쪽 자극 시행 몫 ≥ 0.7(같은 부호) — 아니면 분류 오염.
종합: 계측 검증 실패 → 보류 / M2 ≥ 4/5 → 흔적 크기 비대칭(H123) / M1 ≥ 4/5 → 결과 빈도 비대칭(H123-alt) / 그 밖 혼합.
실행: 저장소 루트에서 python3 scripts/judge_e202.py [기준 디렉터리 — 시험용]"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e198_independent as V  # noqa: E402

E = os.path.join("research", "experiments")
CLASSES = ("S+", "S-", "C+", "C-")
KEY = {"S+": "Sp", "S-": "Sm", "C+": "Cp", "C-": "Cm"}
BRAINS = (10, 11, 12, 13, 14)
WEAK, STRONG = 0.1, 0.5
NT = 3000


def groups(Z):
    g = {}
    for sd in "lr":
        rp = Z["resp_" + sd] > 0
        sx = np.round(Z["sxi_" + sd].astype(np.float64), 9)
        a = np.abs(sx)
        g[("weak", "c", sd)] = rp & (a < WEAK) & (sx > 0)
        g[("weak", "s", sd)] = rp & (a < WEAK) & (sx < 0)
        g[("strong", "c", sd)] = rp & (a >= STRONG) & (sx > 0)
        g[("strong", "s", sd)] = rp & (a >= STRONG) & (sx < 0)
    return g


def bal(vals, g, grp):
    """무리 grp 의 균형 평균 (교차 KC 평균 + 같은 쪽 KC 평균)/2 — 두 KC 쪽을 합친다. 빈 무리면 nan."""
    out = []
    for k in ("c", "s"):
        v = np.concatenate([np.asarray(vals[sd], dtype=np.float64)[g[(grp, k, sd)]] for sd in "lr"])
        if v.size == 0:
            return float("nan")
        out.append(float(v.mean()))
    return (out[0] + out[1]) / 2


def decompose(D, g, grp):
    X = {c: bal({sd: D["x_%s_%s" % (KEY[c], sd)] for sd in "lr"}, g, grp) for c in CLASSES}
    W = dict(zip(CLASSES, [float(v) for v in np.asarray(D["W"], dtype=np.float64)]))
    if any(W[c] == 0.0 for c in CLASSES) or any(np.isnan(X[c]) for c in CLASSES):
        return None
    m = {c: X[c] / W[c] for c in CLASSES}
    dfreq = dmag = 0.0
    for r in "+-":
        ms, mc = m["S" + r], m["C" + r]
        dfreq += (mc - ms) / 2 * (W["C" + r] - W["S" + r])
        dmag += (ms + mc) / 2 * (W["S" + r] + W["C" + r])
    return {"X": X, "W": W, "m": m, "Dobs": float(sum(X.values())), "Dfreq": dfreq, "Dmag": dmag}


def mech(dc):
    do, df, dm = dc["Dobs"], dc["Dfreq"], dc["Dmag"]
    if do != 0 and abs(dm) >= 2 * abs(df) and np.sign(dm) == np.sign(do):
        return "M2"
    if do != 0 and abs(df) >= 2 * abs(dm) and np.sign(df) == np.sign(do):
        return "M1"
    return "혼합"


def log_ok(path):
    s = open(path, encoding="utf-8", errors="replace").read()
    why = []
    if not re.search(r"^\[맥락 범주 과제\] 시행 %d " % NT, s, re.M):
        why.append("과제 줄")
    if not re.search(r"^\[E199 경쟁 정규화\] 적용 %d회 .*hard=1" % NT, s, re.M):
        why.append("규칙 줄")
    m = re.search(r"^\[E200 도파민 통계\] 시행 %d · 평균 DA ([-+0-9.]+) .*기준선 1" % NT, s, re.M)
    if not m or abs(float(m.group(1))) > 0.05:
        why.append("기준선 줄")
    if not re.search(r"^\[E202 계측\] 시행 %d · " % NT, s, re.M):
        why.append("계측 요약 줄")
    if len(re.findall(r"^\[E202 계측\] 시행 [123] 범주 ", s, re.M)) != 3:
        why.append("계측 점검 3줄")
    return why, (float(m.group(1)) if m else float("nan"))


def file_ok(D):
    why = []
    rows = np.asarray(D["rows"])
    if rows.shape[0] != NT:
        why.append("행 %d" % rows.shape[0])
    if int(np.asarray(D["N"]).sum()) != NT:
        why.append("N 합")
    for sd in "lr":
        tot = sum(np.asarray(D["x_%s_%s" % (KEY[c], sd)], dtype=np.float64) for c in CLASSES)
        if not np.allclose(tot, np.asarray(D["tot_x_" + sd], dtype=np.float64), atol=1e-6):
            why.append("범주 합≠총 x(%s)" % sd)
    return why


def validate(D, dl, g):
    meas = np.concatenate([np.asarray(D["tot_x_" + sd], dtype=np.float64) for sd in "lr"])
    lern = np.concatenate([np.asarray(dl[sd], dtype=np.float64) for sd in "lr"])
    r = float(np.corrcoef(meas, lern)[0, 1])
    slope = float((meas * lern).sum() / max((meas * meas).sum(), 1e-300))
    om = bal({sd: D["tot_x_" + sd] for sd in "lr"}, g, "weak")
    ol = bal(dl, g, "weak")
    rel = abs(om - ol) / abs(ol) if ol != 0 else float("inf")
    return (r >= 0.95 and 0.9 <= slope <= 1.1 and rel <= 0.2), r, slope, om, ol, rel


def own_share(D, g):
    """정정 1: 약한 무리 균형 평균에서 자기 쪽 자극 시행(자극 쪽 = KC 쪽)만의 합 / 전체 합."""
    own = {sd: sum(np.asarray(D["xs_%s_%s_%s" % (KEY[c], sd, sd)], dtype=np.float64) for c in CLASSES) for sd in "lr"}
    d_own = bal(own, g, "weak")
    d_all = bal({sd: D["tot_x_" + sd] for sd in "lr"}, g, "weak")
    share = d_own / d_all if d_all != 0 else float("nan")
    return share, d_own, d_all


def decompose_own(D, g, grp):
    """(보고, 근사) 자기 쪽 자극 시행만의 분해 — W 는 두 자극 쪽 평균."""
    X = {c: bal({sd: D["xs_%s_%s_%s" % (KEY[c], sd, sd)] for sd in "lr"}, g, grp) for c in CLASSES}
    Ws = np.asarray(D["Ws"], dtype=np.float64)
    W = {c: float(Ws[i].mean()) for i, c in enumerate(CLASSES)}
    if any(W[c] == 0.0 for c in CLASSES) or any(np.isnan(X[c]) for c in CLASSES):
        return None
    m = {c: X[c] / W[c] for c in CLASSES}
    dfreq = sum((m["C" + r] - m["S" + r]) / 2 * (W["C" + r] - W["S" + r]) for r in "+-")
    dmag = sum((m["S" + r] + m["C" + r]) / 2 * (W["S" + r] + W["C" + r]) for r in "+-")
    return {"Dobs": float(sum(X.values())), "Dfreq": dfreq, "Dmag": dmag}


def regress(d2, d0, Z):
    a = np.concatenate([np.asarray(d2[sd], dtype=np.float64)[Z["resp_" + sd] > 0] for sd in "lr"])
    b = np.concatenate([np.asarray(d0[sd], dtype=np.float64)[Z["resp_" + sd] > 0] for sd in "lr"])
    nz = (a != 0) & (b != 0)
    agree = float(np.mean(np.sign(a[nz]) == np.sign(b[nz]))) if nz.any() else float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    return (agree >= 0.97 and r >= 0.99), agree, r


def judge(base="."):
    ex = os.path.join(base, E)
    res = {}
    for b in BRAINS:
        try:
            D = np.load(os.path.join(ex, "traces", "E202", "dg_b%d.npz" % b))
            Z = np.load(os.path.join(ex, "traces", "E196", "sxi_b%d.npz" % b))
            snap = os.path.join(ex, "traces", "E176", "snap", "conn_b%d.npz" % b)
            d2, _ = V.learned(os.path.join(ex, "traces", "E202", "w_b%d.npz" % b), snap)
            d0, _ = V.learned(os.path.join(ex, "traces", "E200", "w_b%d.npz" % b), snap)
            lw, mda = log_ok(os.path.join(ex, "logs", "E202", "train_b%d.log" % b))
        except Exception as e_:  # noqa: BLE001
            print("판정: 보류(결측 — b%d %s)" % (b, type(e_).__name__))
            return 1
        g = groups(Z)
        fw = file_ok(D)
        vok, vr, vs, om, ol, rel = validate(D, d2, g)
        gok, agree, gr = regress(d2, d0, Z)
        osh, down, dall = own_share(D, g)
        sok = (not np.isnan(osh)) and osh >= 0.7
        dw, dsg = decompose(D, g, "weak"), decompose(D, g, "strong")
        dwo = decompose_own(D, g, "weak")
        res[b] = {"why": lw + fw + ([] if vok else ["재구성"]) + ([] if gok else ["E200 회귀"]) + ([] if sok else ["분류 오염"]),
                  "mech": mech(dw) if dw else "결측", "dw": dw, "ds": dsg}
        N = dict(zip(CLASSES, [int(v) for v in np.asarray(D["N"])]))
        pS = N["S+"] / max(N["S+"] + N["S-"], 1)
        pC = N["C+"] / max(N["C+"] + N["C-"], 1)
        line = ("b%d 약한 무리 D_obs %+.3f = D_freq %+.3f + D_mag %+.3f → %s" % (b, dw["Dobs"], dw["Dfreq"], dw["Dmag"], res[b]["mech"]) if dw
                else "b%d 약한 무리 결측" % b)
        print(line + " | 범주 N %s · W %s · m(약) %s | 보상률 실행 자극 쪽 %.3f · 반대 %.3f · 평균 DA %+.4f"
              % (" ".join("%s %d" % (c, N[c]) for c in CLASSES), " ".join("%s %+.1f" % (c, dw["W"][c]) for c in CLASSES) if dw else "-",
                 " ".join("%s %+.4f" % (c, dw["m"][c]) for c in CLASSES) if dw else "-", pS, pC, mda))
        print("    계측 검증: 상관 %.4f · 기울기 %.3f · 약한 무리 offset 계측 %+.3f 학습 %+.3f(상대 %.3f) | E200 회귀 반응 KC 부호 %.4f · 상관 %.4f | 자기 쪽 자극 몫 %.3f | %s"
              % (vr, vs, om, ol, rel, agree, gr, osh, "통과" if not res[b]["why"] else "실패: " + ", ".join(res[b]["why"])))
        if dwo:
            print("    (보고) 자기 쪽 자극 시행만(근사) D_obs %+.3f = D_freq %+.3f + D_mag %+.3f → %s" % (dwo["Dobs"], dwo["Dfreq"], dwo["Dmag"], mech(dwo)))
        # 정정 2 (보고): 도파민 성분 xd(정규화 전) — 약한 무리 offset 과 그 분해
        dxd = bal({sd: D["tot_xd_" + sd] for sd in "lr"}, g, "weak")
        Dd = {k: D[k] for k in D.files}
        for c in CLASSES:
            for sd in "lr":
                Dd["x_%s_%s" % (KEY[c], sd)] = D["xd_%s_%s" % (KEY[c], sd)]
        dwd = decompose(Dd, g, "weak")
        if dwd:
            print("    (보고) 도파민 성분 xd: 약한 무리 offset %+.3f(전체 x 의 %.2f배) · D_obs %+.3f = D_freq %+.3f + D_mag %+.3f → %s"
                  % (dxd, dxd / dw["Dobs"] if dw and dw["Dobs"] else float("nan"), dwd["Dobs"], dwd["Dfreq"], dwd["Dmag"], mech(dwd)))
        if dsg:
            print("    (보고) 강한 무리 D_obs %+.3f = D_freq %+.3f + D_mag %+.3f → %s" % (dsg["Dobs"], dsg["Dfreq"], dsg["Dmag"], mech(dsg)))
    bad = [b for b in BRAINS if res[b]["why"]]
    n = {k: sum(res[b]["mech"] == k for b in BRAINS) for k in ("M1", "M2", "혼합")}
    print("계측 검증 %s | M2(흔적 크기) %d/5 · M1(결과 빈도) %d/5 · 혼합 %d/5"
          % ("통과" if not bad else "실패(b%s)" % ",".join(str(b) for b in bad), n["M2"], n["M1"], n["혼합"]))
    if bad:
        v = "보류(계측 검증 실패)"
    elif n["M2"] >= 4:
        v = "흔적 크기 비대칭(H123)"
    elif n["M1"] >= 4:
        v = "결과 빈도 비대칭(H123-alt)"
    else:
        v = "혼합"
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(judge(sys.argv[1] if len(sys.argv) > 1 else "."))
