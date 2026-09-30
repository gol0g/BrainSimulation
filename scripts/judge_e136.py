#!/usr/bin/env python3
"""E136 판정 — 기준 logs/E136/oja_rule_fixed.txt(E133 과 같은 판정, 배선 78~93). 발달 32 + 과제 96 이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): novel_lbal = 처음 보는 항목(4~7) 라벨 균형 정답률(%). 독립 단위 = 배선(두 난수열 평균).
sf = 가지치기 후 같은 위치 짝 비율((일치형 + 불일치형)/2), 발달 DEVHEBB 줄.
"""
import os
import re
import sys
from math import comb

EXP = "research/experiments"
WIRES = tuple(range(78, 94))
TSEEDS = (600, 601)
TD = re.compile(r"^\s*oj dev (corr|indep) w(\d+): => DEVHEBB seed=(\d+) env=(\w+) .*?발화율\(KC·노출당\) 일치형 ([0-9.]+) 불일치형 ([0-9.]+) \| "
                r"가지치기 후 같은 위치: 일치형 ([0-9.]+) 불일치형 ([0-9.]+)")
TT = re.compile(r"^\s*oj (corr|indep) (learn|frozen) w(\d+) t(\d+): => \[KC불러옴\] (\S+) \| 일치형 같은 위치 (\d+)/(\d+) \| 불일치형 같은 위치 (\d+)/(\d+)"
                r" \|\| => SDLAB diff=cyclic rule=samediff .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)")


def sign_p(d):
    n = sum(x > 0 for x in d); nz = sum(x != 0 for x in d)
    if nz == 0:
        return 1.0, n, nz
    k = max(n, nz - n)
    return min(1.0, 2 * sum(comb(nz, i) for i in range(k, nz + 1)) / 2 ** nz), n, nz


def judge(D, T):
    needD = [(e, w) for e in ("corr", "indep") for w in WIRES]
    needT = [(e, m, w, t) for (e, m) in (("corr", "learn"), ("indep", "learn"), ("corr", "frozen")) for w in WIRES for t in TSEEDS]
    miss = [k for k in needD if k not in D] + [k for k in needT if k not in T]
    if miss:
        return ["[측정 확인] 결측 %d/128: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    checks, ok = [], True
    sf = {k: (D[k]["mm"] + D[k]["mx"]) / 2 for k in needD}
    cs = sorted(sf[("corr", w)] for w in WIRES); med = (cs[7] + cs[8]) / 2
    r5 = sum(sf[("corr", w)] >= 5 * sf[("indep", w)] for w in WIRES)
    checks.append("[측정 확인] corr 같은 위치 비율 중앙값 %.3f (≥0.40), corr ≥ 5×indep: %d/16" % (med, r5))
    ok &= med >= 0.40 and r5 == 16
    envbad = [k for k in needD if D[k]["env"] != k[0] or D[k]["seed"] != k[1]]
    # DEVHEBB 비율은 소수 3자리 출력 → 개수로 되돌리면 ±0.2 오차(400 × 0.0005) — 1 미만 차이만 허용
    fbad = [k for k in needT if abs(D[(k[0], k[2])]["mm"] * T[k]["nm"] - T[k]["lm"]) >= 1.0 or abs(D[(k[0], k[2])]["mx"] * T[k]["nx"] - T[k]["lx"]) >= 1.0
            or ("dev_%s_w%d.npz" % (k[0], k[2])) not in T[k]["file"]]
    checks.append("[측정 확인] 발달 env·seed 표기 %d/32, 과제 연결 = 발달 결과 %d/96" % (32 - len(envbad), 96 - len(fbad)))
    ok &= not envbad and not fbad
    same = [(w, t) for w in WIRES for t in TSEEDS if (T[("corr", "learn", w, t)]["tl"], T[("corr", "learn", w, t)]["nl"]) == (T[("corr", "frozen", w, t)]["tl"], T[("corr", "frozen", w, t)]["nl"])]
    checks.append("[측정 확인] corr learn ≠ frozen: %d/32" % (32 - len(same)))
    ok &= not same
    wm = lambda e, m, w: sum(T[(e, m, w, t)]["nl"] for t in TSEEDS) / 2
    d_ci = [wm("corr", "learn", w) - wm("indep", "learn", w) for w in WIRES]
    d_cf = [wm("corr", "learn", w) - wm("corr", "frozen", w) for w in WIRES]
    p_ci, n_ci, nz_ci = sign_p(d_ci); p_cf, n_cf, nz_cf = sign_p(d_cf)
    m_ci = sum(d_ci) / 16; m_cf = sum(d_cf) / 16
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif m_ci >= 10 and p_ci < 0.01 and m_cf >= 10 and p_cf < 0.01:
        verdict = "지지 — 망 안 Oja 형 경쟁 발달로 형성된 비교 특징이 학습 후 새 항목 같음/다름 전이를 만든다"
    elif m_ci < 5 or p_ci >= 0.05:
        verdict = "효과 없음 — 망 안 Oja 발달의 corr/indep 전이 차이 검출 안 됨"
    else:
        verdict = "보류(혼재)"
    return checks, {"sf": sf, "m_ci": m_ci, "p_ci": p_ci, "n_ci": n_ci, "nz_ci": nz_ci, "m_cf": m_cf, "p_cf": p_cf, "n_cf": n_cf,
                    "nz_cf": nz_cf, "d_ci": d_ci, "ok": ok, "verdict": verdict}


def report(checks, res, D=None, T=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        print("w%d 발달 corr sf %.2f(발화 %.2f/%.2f) indep sf %.2f | 새 항목 corr %s indep %s frozen %s" % (
            w, res["sf"][("corr", w)], D[("corr", w)]["fm"], D[("corr", w)]["fx"], res["sf"][("indep", w)],
            "/".join("%.0f" % T[("corr", "learn", w, t)]["nl"] for t in TSEEDS), "/".join("%.0f" % T[("indep", "learn", w, t)]["nl"] for t in TSEEDS),
            "/".join("%.0f" % T[("corr", "frozen", w, t)]["nl"] for t in TSEEDS)))
    for e, m in (("corr", "learn"), ("indep", "learn"), ("corr", "frozen")):
        ks = [(e, m, w, t) for w in WIRES for t in TSEEDS]
        print("%s %s 평균 새 항목 %.1f%% 훈련 %.1f%% | 새 항목 ≥75%%: %d/32" % (e, m, sum(T[k]["nl"] for k in ks) / 32, sum(T[k]["tl"] for k in ks) / 32, sum(T[k]["nl"] >= 75 for k in ks)))
    print("배선 단위 corr−indep: 평균 %+.1f%%p, 양수 %d/%d, 부호검정 양측 p=%.5f" % (res["m_ci"], res["n_ci"], res["nz_ci"], res["p_ci"]))
    print("배선 단위 corr learn−frozen: 평균 %+.1f%%p, 양수 %d/%d, 부호검정 양측 p=%.5f" % (res["m_cf"], res["n_cf"], res["nz_cf"], res["p_cf"]))
    print("판정: %s" % res["verdict"])


def load():
    D, T = {}, {}
    try:
        for ln in open(os.path.join(EXP, "E136.log"), encoding="utf-8"):
            m = TD.match(ln)
            if m:
                g = m.groups()
                D[(g[0], int(g[1]))] = {"seed": int(g[2]), "env": g[3], "fm": float(g[4]), "fx": float(g[5]), "mm": float(g[6]), "mx": float(g[7])}
                continue
            m = TT.match(ln)
            if m:
                g = m.groups()
                T[(g[0], g[1], int(g[2]), int(g[3]))] = {"file": g[4], "lm": int(g[5]), "nm": int(g[6]), "lx": int(g[7]), "nx": int(g[8]),
                                                           "tl": float(g[9]), "nl": float(g[10])}
    except FileNotFoundError:
        pass
    return D, T


if __name__ == "__main__":
    D, T = load()
    c, r = judge(D, T)
    report(c, r, D, T)
    sys.exit(0)
