#!/usr/bin/env python3
"""E132 판정 — 기준 고정 logs/E132/criteria_fixed.txt. 96런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): novel_lbal = 처음 보는 항목(4~7) 라벨 균형 정답률(%). 짝 = 같은 배선·같은 난수열.
mf = 장치 연결의 일치형 KC 중 같은 위치 짝 비율.
"""
import os
import re
import sys
from math import comb

EXP = "research/experiments"
WIRES = tuple(range(30, 46))
TSEEDS = (600, 601)
TR = re.compile(r"^\s*dv (corr|indep) (learn|frozen) w(\d+) t(\d+): => \[KC발달\] env=(\w+) .*?일치형 같은 위치 (\d+)/(\d+) \| 불일치형 같은 위치 (\d+)/(\d+)"
                r".*?\|\| => SDLAB diff=cyclic rule=samediff .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)")


def sign_p(d):
    n = sum(x > 0 for x in d); nz = sum(x != 0 for x in d)
    if nz == 0:
        return 1.0, n, nz
    k = max(n, nz - n)
    return min(1.0, 2 * sum(comb(nz, i) for i in range(k, nz + 1)) / 2 ** nz), n, nz


def judge(R):
    need = [(e, m, w, t) for (e, m) in (("corr", "learn"), ("indep", "learn"), ("corr", "frozen")) for w in WIRES for t in TSEEDS]
    miss = [k for k in need if k not in R]
    if miss:
        return ["[측정 확인] 결측 %d/96: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    checks, ok = [], True
    cm = sorted(R[("corr", "learn", w, 600)]["mf"] for w in WIRES)
    med = (cm[7] + cm[8]) / 2
    ratio_ok = sum(R[("corr", "learn", w, 600)]["mf"] >= 5 * R[("indep", "learn", w, 600)]["mf"] for w in WIRES)
    checks.append("[측정 확인] corr 같은 위치 비율 중앙값 %.3f (≥0.40), corr ≥ 5×indep: %d/16" % (med, ratio_ok))
    ok &= med >= 0.40 and ratio_ok == 16
    envbad = [k for k in need if R[k]["env"] != k[0]]
    checks.append("[측정 확인] env 표기 = 조건: %d/96" % (96 - len(envbad)))
    ok &= not envbad
    same = [(w, t) for w in WIRES for t in TSEEDS if (R[("corr", "learn", w, t)]["tl"], R[("corr", "learn", w, t)]["nl"]) == (R[("corr", "frozen", w, t)]["tl"], R[("corr", "frozen", w, t)]["nl"])]
    checks.append("[측정 확인] corr learn ≠ frozen: %d/32" % (32 - len(same)))
    ok &= not same
    pairs = [(w, t) for w in WIRES for t in TSEEDS]
    d_ci = [R[("corr", "learn", w, t)]["nl"] - R[("indep", "learn", w, t)]["nl"] for w, t in pairs]
    d_cf = [R[("corr", "learn", w, t)]["nl"] - R[("corr", "frozen", w, t)]["nl"] for w, t in pairs]
    p_ci, n_ci, nz_ci = sign_p(d_ci)
    p_cf, n_cf, nz_cf = sign_p(d_cf)
    m_ci = sum(d_ci) / 32; m_cf = sum(d_cf) / 32
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif m_ci >= 10 and p_ci < 0.01 and m_cf >= 10 and p_cf < 0.01:
        verdict = "인과 효과 지지 — 상관 경험(형성된 비교 특징)이 학습 후 새 항목 같음/다름 전이를 높인다"
    elif m_ci < 5 or p_ci >= 0.05:
        verdict = "효과 없음 — 상관 경험과 독립 경험의 전이 차이가 검출 안 됨"
    else:
        verdict = "보류(혼재)"
    return checks, {"m_ci": m_ci, "p_ci": p_ci, "n_ci": n_ci, "nz_ci": nz_ci, "m_cf": m_cf, "p_cf": p_cf, "n_cf": n_cf, "nz_cf": nz_cf,
                    "ok": ok, "verdict": verdict}


def report(checks, res, R=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        print("w%d corr mf %.2f novel %s | indep mf %.2f novel %s | frozen novel %s" % (
            w, R[("corr", "learn", w, 600)]["mf"], "/".join("%.0f" % R[("corr", "learn", w, t)]["nl"] for t in TSEEDS),
            R[("indep", "learn", w, 600)]["mf"], "/".join("%.0f" % R[("indep", "learn", w, t)]["nl"] for t in TSEEDS),
            "/".join("%.0f" % R[("corr", "frozen", w, t)]["nl"] for t in TSEEDS)))
    for e, m in (("corr", "learn"), ("indep", "learn"), ("corr", "frozen")):
        ks = [(e, m, w, t) for w in WIRES for t in TSEEDS]
        print("%s %s 평균 새 항목 %.1f%% 훈련 %.1f%% | 새 항목 ≥75%%: %d/32" % (e, m, sum(R[k]["nl"] for k in ks) / 32, sum(R[k]["tl"] for k in ks) / 32, sum(R[k]["nl"] >= 75 for k in ks)))
    print("짝 corr−indep: 평균 %+.1f%%p, 양수 %d/%d, 부호검정 양측 p=%.5f" % (res["m_ci"], res["n_ci"], res["nz_ci"], res["p_ci"]))
    print("짝 corr learn−frozen: 평균 %+.1f%%p, 양수 %d/%d, 부호검정 양측 p=%.5f" % (res["m_cf"], res["n_cf"], res["nz_cf"], res["p_cf"]))
    print("판정: %s" % res["verdict"])


def parse(ln):
    m = TR.match(ln)
    if not m:
        return None
    g = m.groups()
    return (g[0], g[1], int(g[2]), int(g[3])), {"env": g[4], "mf": int(g[5]) / int(g[6]), "tl": float(g[9]), "nl": float(g[10])}


def load():
    R = {}
    try:
        for ln in open(os.path.join(EXP, "E132.log"), encoding="utf-8"):
            p = parse(ln)
            if p:
                R[p[0]] = p[1]
    except FileNotFoundError:
        pass
    return R


if __name__ == "__main__":
    R = load()
    c, r = judge(R)
    report(c, r, R)
    sys.exit(0)
