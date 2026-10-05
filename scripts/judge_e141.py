#!/usr/bin/env python3
"""E141 판정 — 기준 logs/E141/criteria_fixed.txt(실행 전 고정). 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
효과 e = [사후] − [사전] 이식 변조폭(음수 = 정답 교차), E119 효과와 뇌별 짝(d = e − e_E119).
조작검증·기전 수치는 추적 npz(traces/E141/tr_b*.npz, 비교 traces/E139/tr_b*.npz)에서.
rows 열(E139 와 같음): 7 correct 8~11 dg(교차·같은쪽·비선택·무활동) 12 dg_전체 13~16 e_da 17~20 dg_도파민전 21~24 e_end 25·26 보상 창 motor 발화율.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
E119_EFF = {10: -0.1032, 11: -0.0724, 12: -0.0689, 13: -0.0809, 14: -0.0607}
E119_PRE = {10: 0.0195, 11: 0.0150, 12: 0.0262, 13: 0.0320, 14: 0.0165}
R_STAR = (1.0 - 1.0 / 12.0) ** 20     # 보상 창 20스텝 동안 흔적 감쇠만 할 때 e_end/e_da (dt 1, tau_e 12, Euler)
TL = re.compile(r"^\s*e141 b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")


def stats(rows):
    rw = rows[:, 7] == 1
    A, B = rows[rw, 8].sum(), rows[rw, 9].sum()
    C, P = rows[~rw, 8].sum(), rows[~rw, 9].sum()
    eda, eend = rows[:, 13:17], rows[:, 21:25]
    return {"n": len(rows),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(eend - R_STAR * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "eda_abs": float((np.abs(rows[:, 13]) + np.abs(rows[:, 14])).mean()),
            "BA": (B / A) if A else float("nan"), "CP": (C / P) if P else float("nan"),
            "dD": float((A + C) - (B + P))}


def judge(T, S, S139):
    miss = [b for b in BRAINS if b not in T or b not in S or b not in S139]
    if miss:
        return ["[측정 확인] 결측 뇌 %s — **판정 보류, 수치 미출력**" % miss], None
    checks, ok = [], True
    m1 = [b for b in BRAINS if S[b]["res"] <= 1e-3]
    checks.append("[M1 동결] Σ|e_end − r*·e_da|/Σ|e_da| ≤ 1e-3 (r* %.6f): %d/5 (%s) %s"
                  % (R_STAR, len(m1), " ".join("%.1e" % S[b]["res"] for b in BRAINS), "통과" if len(m1) == 5 else "실패")); ok &= len(m1) == 5
    m1b = [b for b in BRAINS if S[b]["eda_abs"] >= 0.25 * S139[b]["eda_abs"]]
    checks.append("[M1b 되돌림] 결정 단계 흔적 ≥ 0.25 × E139: %d/5 (%s) %s"
                  % (len(m1b), " ".join("%.2f" % (S[b]["eda_abs"] / S139[b]["eda_abs"]) for b in BRAINS), "통과" if len(m1b) == 5 else "실패")); ok &= len(m1b) == 5
    m2 = [b for b in BRAINS if abs(T[b]["pre"] - E119_PRE[b]) <= 0.002 + 1e-9]
    checks.append("[M2 출발점] [사전] = E119 ±0.002: %d/5 %s" % (len(m2), "통과" if len(m2) == 5 else "실패")); ok &= len(m2) == 5
    m3 = [b for b in BRAINS if S[b]["n"] == 500 and S[b]["pre_ratio"] <= 1e-3]
    checks.append("[M3 추적 완결] 500시행·도파민 전 변화 0: %d/5 %s" % (len(m3), "통과" if len(m3) == 5 else "실패")); ok &= len(m3) == 5
    e = {b: round(T[b]["post"] - T[b]["pre"], 6) for b in BRAINS}
    d = {b: round(e[b] - E119_EFF[b], 6) for b in BRAINS}
    mean_e = round(sum(e.values()) / 5, 6)
    better = [b for b in BRAINS if d[b] < 0]
    worse = [b for b in BRAINS if d[b] >= 0.01]
    same = [b for b in BRAINS if abs(d[b]) < 0.01]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif mean_e <= -0.10 and len(better) == 5:
        verdict = "지지(H064) — 보상 창 흔적 생성을 멈추면 학습 효과가 커진다(평균 ≤ −0.10, E119 보다 5/5 큼)"
    elif len(worse) >= 4:
        verdict = "반대(H064-rev) — 동결하면 효과가 작아진다(d ≥ +0.01, ≥4/5): 보상 창 흔적이 학습에 보탠다"
    elif len(same) >= 4:
        verdict = "기각(H064-null) — 동결해도 효과가 E119 와 같다(|d| < 0.01, ≥4/5)"
    else:
        verdict = "보류"
    mech = [b for b in BRAINS if S[b]["BA"] < 0 and S[b]["CP"] < 0]
    dmore = [b for b in BRAINS if S[b]["dD"] > S139[b]["dD"]]
    return checks, {"e": e, "d": d, "mean_e": mean_e, "better": better, "worse": worse, "same": same,
                    "mech": mech, "dmore": dmore, "ok": ok, "verdict": verdict}


def report(checks, res, T=None, S=None, S139=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        s = S[b]
        print("b%d e %+.4f (E119 %+.4f, d %+.4f) 보상 %d | B/A %+.3f C/P %+.3f | ΔD %+.4g (E139 %+.4g, ×%.2f)"
              % (b, res["e"][b], E119_EFF[b], res["d"][b], T[b]["rew"], s["BA"], s["CP"], s["dD"], S139[b]["dD"],
                 s["dD"] / S139[b]["dD"] if S139[b]["dD"] else float("nan")))
    print("효과 평균 %+.4f (E119 평균 %+.4f) | d<0 %d/5 | d≥+0.01 %d/5 | |d|<0.01 %d/5 | 기전(B/A<0·C/P<0) %d/5 | ΔD>E139 %d/5"
          % (res["mean_e"], sum(E119_EFF.values()) / 5, len(res["better"]), len(res["worse"]), len(res["same"]),
             len(res["mech"]), len(res["dmore"])))
    print("판정: %s" % res["verdict"])


def load():
    T, S, S139 = {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E141.log"), encoding="utf-8"):
            m = TL.match(ln)
            if m:
                T[int(m.group(1))] = {"pre": float(m.group(2)), "post": float(m.group(3)), "rew": int(m.group(4))}
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E141", "tr_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = stats(np.load(f)["rows"])
        f = os.path.join(EXP, "traces", "E139", "tr_b%d.npz" % b)
        if os.path.exists(f):
            S139[b] = stats(np.load(f)["rows"])
    return T, S, S139


if __name__ == "__main__":
    T, S, S139 = load()
    c, r = judge(T, S, S139)
    report(c, r, T, S, S139)
    sys.exit(0)
