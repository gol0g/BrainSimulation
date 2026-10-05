#!/usr/bin/env python3
"""E140 판정 — 기준 logs/E140/criteria_fixed.txt(실행 전 고정). 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
효과 e = [사후] − [사전] 이식 변조폭(음수 = 정답 교차), E119 효과와 뇌별 짝. 조작검증·기전 수치는 추적 npz(traces/E140/tr_b*.npz)에서.
rows 열(E139 와 같음): 7 correct 8~11 dg(교차·같은쪽·비선택·무활동) 12 dg_전체 13~16 e_da 17~20 dg_도파민전 21~24 e_end 25·26 보상 창 motor 좌·우 발화율 합.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
E119_EFF = {10: -0.1032, 11: -0.0724, 12: -0.0689, 13: -0.0809, 14: -0.0607}
E119_PRE = {10: 0.0195, 11: 0.0150, 12: 0.0262, 13: 0.0320, 14: 0.0165}
TL = re.compile(r"^\s*e140 b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")


def stats(rows):
    rw = rows[:, 7] == 1
    A, B = rows[rw, 8].sum(), rows[rw, 9].sum()
    C, P = rows[~rw, 8].sum(), rows[~rw, 9].sum()
    pre = float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12))
    return {"n": len(rows), "ml": float(rows[:, 25].mean()), "mr": float(rows[:, 26].mean()), "pre_ratio": pre,
            "BA": (B / A) if A else float("nan"), "CP": (C / P) if P else float("nan"),
            "eda": rows[rw, 14].sum() / rows[rw, 13].sum() if rows[rw, 13].sum() else float("nan"),
            "eend": rows[rw, 22].sum() / rows[rw, 21].sum() if rows[rw, 21].sum() else float("nan")}


def judge(T, S):
    miss = [b for b in BRAINS if b not in T or b not in S]
    if miss:
        return ["[측정 확인] 결측 뇌 %s — **판정 보류, 수치 미출력**" % miss], None
    checks, ok = [], True
    m1 = [b for b in BRAINS if S[b]["ml"] <= 0.05 and S[b]["mr"] <= 0.05]
    checks.append("[M1 침묵] 보상 창 motor 발화율 좌·우 ≤0.05: %d/5 (%s) %s"
                  % (len(m1), " ".join("%.3f/%.3f" % (S[b]["ml"], S[b]["mr"]) for b in BRAINS), "통과" if len(m1) == 5 else "실패")); ok &= len(m1) == 5
    m2 = [b for b in BRAINS if abs(T[b]["pre"] - E119_PRE[b]) <= 0.002 + 1e-9]
    checks.append("[M2 출발점] [사전] = E119 ±0.002: %d/5 %s" % (len(m2), "통과" if len(m2) == 5 else "실패")); ok &= len(m2) == 5
    m3 = [b for b in BRAINS if S[b]["n"] == 500 and S[b]["pre_ratio"] <= 1e-3]
    checks.append("[M3 추적 완결] 500시행·도파민 전 변화 0: %d/5 %s" % (len(m3), "통과" if len(m3) == 5 else "실패")); ok &= len(m3) == 5
    e = {b: round(T[b]["post"] - T[b]["pre"], 6) for b in BRAINS}
    mean_e = round(sum(e.values()) / 5, 6)
    better = [b for b in BRAINS if e[b] < E119_EFF[b]]
    same = [b for b in BRAINS if abs(e[b] - E119_EFF[b]) < 0.01]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif mean_e <= -0.10 and len(better) == 5:
        verdict = "지지(H063) — 보상 창 motor 침묵이 학습 효과를 키운다(평균 ≤ −0.10, E119 보다 5/5 큼)"
    elif len(same) >= 4:
        verdict = "기각(H063-null) — 침묵해도 효과가 E119 와 같다(|차| < 0.01, ≥4/5)"
    else:
        verdict = "보류"
    mech = sum(1 for b in BRAINS if S[b]["BA"] <= 0.5 and S[b]["CP"] <= 0.5)
    return checks, {"e": e, "mean_e": mean_e, "better": better, "same": same, "mech": mech, "ok": ok, "verdict": verdict}


def report(checks, res, T=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        s = S[b]
        print("b%d e %+.4f (E119 %+.4f, 차 %+.4f) 보상 %d | B/A %.3f C/P %.3f | e_da %+.3f e_end %+.3f"
              % (b, res["e"][b], E119_EFF[b], res["e"][b] - E119_EFF[b], T[b]["rew"], s["BA"], s["CP"], s["eda"], s["eend"]))
    print("효과 평균 %+.4f (E119 평균 %+.4f) | E119 보다 큼 %d/5 | |차|<0.01 %d/5 | 기전(B/A·C/P ≤0.5) %d/5"
          % (res["mean_e"], sum(E119_EFF.values()) / 5, len(res["better"]), len(res["same"]), res["mech"]))
    print("판정: %s" % res["verdict"])


def load():
    T, S = {}, {}
    try:
        for ln in open(os.path.join(EXP, "E140.log"), encoding="utf-8"):
            m = TL.match(ln)
            if m:
                T[int(m.group(1))] = {"pre": float(m.group(2)), "post": float(m.group(3)), "rew": int(m.group(4))}
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E140", "tr_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = stats(np.load(f)["rows"])
    return T, S


if __name__ == "__main__":
    T, S = load()
    c, r = judge(T, S)
    report(c, r, T, S)
    sys.exit(0)
