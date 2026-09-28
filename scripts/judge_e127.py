#!/usr/bin/env python3
"""E127 판정 — 기준 고정 logs/E127/criteria_fixed.txt. 24런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): dSg = KC 별 Σg(→out_L) − Σg(→out_R)(가중치 합, 초기 0.5×연결 수 포함). 부류 평균.
margin_sign_ok = 훈련 8자극 중 Σ_{활성 KC} dSg 의 부호가 정답 쪽인 수.
"""
import os
import re
import sys

EXP = "research/experiments"
WIRES = tuple(range(10, 18))
TSEEDS = (600, 601)
NUM = r"([-+]?[0-9.]+|nan)"
TR = re.compile(r"^\s*cr (learn|frozen) w(\d+) t(\d+): => SDCREDIT .*?L전용 n=(\d+) dSg=" + NUM + r" \| R전용 n=(\d+) dSg=" + NUM +
                r" \| 양쪽 n=(\d+) dSg=" + NUM + r" \| margin_sign_ok=(\d+)/(\d+) \| spec_share\(.*?\) 평균 " + NUM +
                r" \|\| => SDLAB diff=cyclic rule=samediff .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)")
T26 = re.compile(r"^\s*cy (learn|frozen) w(\d+) t(\d+): => SDLAB diff=cyclic rule=samediff .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)")


def fnum(x):
    return float("nan") if x == "nan" else float(x)


def judge(R, E26):
    """R: {(mode,w,t): dict(nL,dL,nR,dR,nB,dB,mok,mtot,share,tl,nl)}, E26: {(mode,w,t): (train_lbal, novel_lbal)}"""
    need = [("learn", w, t) for w in WIRES for t in TSEEDS] + [("frozen", w, 600) for w in WIRES]
    miss = [k for k in need if k not in R] + [("E126",) + k for k in need if k not in E26]
    if miss:
        return ["[측정 확인] 결측 %d: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    checks, ok = [], True
    reg = [k for k in need if (R[k]["tl"], R[k]["nl"]) != E26[k]]
    checks.append("[측정 확인] 평가 값 = E126 같은 런(분석이 평가를 안 바꿈): %d/24%s" % (24 - len(reg), "" if not reg else " ← 다름 %s" % reg))
    ok &= not reg
    same = [w for w in WIRES if R[("learn", w, 600)] == R[("frozen", w, 600)]]
    checks.append("[측정 확인] learn ≠ frozen(배선별 t600): %d/8" % (8 - len(same)))
    ok &= not same
    L = [R[("learn", w, t)] for w in WIRES for t in TSEEDS]

    def right(x):
        return x["dL"] == x["dL"] and x["dR"] == x["dR"] and x["dL"] > 0 and x["dR"] < 0
    n_right = sum(right(x) for x in L)
    n_lowm = sum(x["mok"] <= 6 for x in L)
    if not ok:
        verdict = "보류(조작검증 실패 — 측정부터)"
    elif n_right >= 12 and n_lowm >= 12:
        verdict = "묻힘(표현 희석) — 전용 KC 신용은 맞으나 자극별 구동 여유가 정답 쪽이 아님"
    elif n_right <= 4:
        verdict = "신용 틀림(학습 규칙) — 전용 KC 의 가중치 차 방향부터 틀림"
    else:
        verdict = "보류(혼재)"
    return checks, {"n_right": n_right, "n_lowm": n_lowm, "ok": ok, "verdict": verdict}


def report(checks, res, R=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        xs = [R[("learn", w, t)] for t in TSEEDS]; f = R[("frozen", w, 600)]
        print("w%d learn L전용 %s | R전용 %s | margin %s | share %s || frozen L %+.3f R %+.3f margin %d/%d"
              % (w, " ".join("n%d %+.3f" % (x["nL"], x["dL"]) for x in xs), " ".join("n%d %+.3f" % (x["nR"], x["dR"]) for x in xs),
                 " ".join("%d/%d" % (x["mok"], x["mtot"]) for x in xs), " ".join("%.2f" % x["share"] for x in xs),
                 f["dL"], f["dR"], f["mok"], f["mtot"]))
    print("신용 방향 맞음(L전용 dSg>0 & R전용 dSg<0): %d/16 | margin_sign_ok ≤6/8: %d/16" % (res["n_right"], res["n_lowm"]))
    print("판정: %s" % res["verdict"])


def parse(ln):
    m = TR.match(ln)
    if not m:
        return None
    g = m.groups()
    return (g[0], int(g[1]), int(g[2])), {"nL": int(g[3]), "dL": fnum(g[4]), "nR": int(g[5]), "dR": fnum(g[6]),
                                          "nB": int(g[7]), "dB": fnum(g[8]), "mok": int(g[9]), "mtot": int(g[10]),
                                          "share": fnum(g[11]), "tl": float(g[12]), "nl": float(g[13])}


def load():
    R, E26 = {}, {}
    try:
        for ln in open(os.path.join(EXP, "E127.log"), encoding="utf-8"):
            p = parse(ln)
            if p:
                R[p[0]] = p[1]
    except FileNotFoundError:
        pass
    try:
        for ln in open(os.path.join(EXP, "E126.log"), encoding="utf-8"):
            m = T26.match(ln)
            if m:
                E26[(m.group(1), int(m.group(2)), int(m.group(3)))] = (float(m.group(4)), float(m.group(5)))
    except FileNotFoundError:
        pass
    return R, E26


if __name__ == "__main__":
    R, E26 = load()
    c, r = judge(R, E26)
    report(c, r, R)
    sys.exit(0)
