#!/usr/bin/env python3
"""E131 판정 — E130 기준 그대로(logs/E130/criteria_fixed.txt), 배선 20~27. 40런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): novel_lbal = 처음 보는 항목(4~7) 모든 쌍 라벨 균형 정답률(%), train_lbal = 훈련 8자극(%).
match_frac = 장치 연결에서 읽은 일치형 KC 중 같은 위치(반쪽1 p·반쪽2 p) 짝 비율.
"""
import os
import re
import sys

EXP = "research/experiments"
WIRES = tuple(range(20, 28))
TSEEDS = (600, 601)
TR = re.compile(r"^\s*dv (corr|indep) (learn|frozen) w(\d+) t(\d+): => \[KC발달\] env=(\w+) .*?일치형 같은 위치 (\d+)/(\d+) \| 불일치형 같은 위치 (\d+)/(\d+)"
                r".*?\|\| => SDLAB diff=cyclic rule=samediff .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)")


def judge(R):
    need = [("corr", "learn", w, t) for w in WIRES for t in TSEEDS] + [("indep", "learn", w, t) for w in WIRES for t in TSEEDS] \
        + [("corr", "frozen", w, 600) for w in WIRES]
    miss = [k for k in need if k not in R]
    if miss:
        return ["[측정 확인] 결측 %d/40: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    checks, ok = [], True
    cm = [R[("corr", "learn", w, 600)]["mf"] for w in WIRES]
    im = [R[("indep", "learn", w, 600)]["mf"] for w in WIRES]
    c_ok = sum(x >= 0.40 for x in cm); i_ok = sum(x <= 0.05 for x in im)
    checks.append("[측정 확인] 일치형 같은 위치 비율 corr ≥0.40: %d/8, indep ≤0.05: %d/8" % (c_ok, i_ok))
    ok &= c_ok == 8 and i_ok == 8
    envs = [w for w in WIRES if R[("corr", "learn", w, 600)]["env"] != "corr" or R[("indep", "learn", w, 600)]["env"] != "indep"]
    checks.append("[측정 확인] 로그 env 표기 = 조건: %d/8" % (8 - len(envs)))
    ok &= not envs
    same = [w for w in WIRES if (R[("corr", "learn", w, 600)]["tl"], R[("corr", "learn", w, 600)]["nl"]) == (R[("corr", "frozen", w, 600)]["tl"], R[("corr", "frozen", w, 600)]["nl"])]
    checks.append("[측정 확인] corr learn ≠ frozen(배선별 t600): %d/8" % (8 - len(same)))
    ok &= not same
    nc = sum(R[("corr", "learn", w, t)]["nl"] >= 75 for w in WIRES for t in TSEEDS)
    ni = sum(R[("indep", "learn", w, t)]["nl"] >= 75 for w in WIRES for t in TSEEDS)
    nf = sum(R[("corr", "frozen", w, 600)]["nl"] >= 75 for w in WIRES)
    if not ok:
        verdict = "보류(조작검증 실패 — 발달이 비교 특징을 만들지 않았거나 측정 문제)"
    elif nc >= 12 and ni <= 4 and nf <= 2:
        verdict = "경험 형성 지지 — 상관 환경 경험으로 형성된 비교 특징이 처음 보는 항목 전이를 만든다"
    elif nc <= 4:
        verdict = "형성 부족 — 상관 환경 발달로도 전이 없음"
    else:
        verdict = "보류(혼재)"
    return checks, {"nc": nc, "ni": ni, "nf": nf, "ok": ok, "verdict": verdict}


def report(checks, res, R=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        c = [R[("corr", "learn", w, t)] for t in TSEEDS]; i = [R[("indep", "learn", w, t)] for t in TSEEDS]; f = R[("corr", "frozen", w, 600)]
        print("w%d corr mf %.2f train %s novel %s | indep mf %.2f train %s novel %s || corr frozen novel %.0f"
              % (w, c[0]["mf"], "/".join("%.0f" % x["tl"] for x in c), "/".join("%.0f" % x["nl"] for x in c),
                 i[0]["mf"], "/".join("%.0f" % x["tl"] for x in i), "/".join("%.0f" % x["nl"] for x in i), f["nl"]))
    for env in ("corr", "indep"):
        ks = [(env, "learn", w, t) for w in WIRES for t in TSEEDS]
        print("%s learn 평균: train %.1f%% novel %.1f%%" % (env, sum(R[k]["tl"] for k in ks) / 16, sum(R[k]["nl"] for k in ks) / 16))
    print("새 항목 ≥75%%: corr learn %d/16, indep learn %d/16, corr frozen %d/8" % (res["nc"], res["ni"], res["nf"]))
    print("판정: %s" % res["verdict"])


def parse(ln):
    m = TR.match(ln)
    if not m:
        return None
    g = m.groups()
    return (g[0], g[1], int(g[2]), int(g[3])), {"env": g[4], "mf": int(g[5]) / int(g[6]), "xf": int(g[7]) / int(g[8]),
                                                 "tl": float(g[9]), "nl": float(g[10])}


def load():
    R = {}
    try:
        for ln in open(os.path.join(EXP, "E131.log"), encoding="utf-8"):
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
