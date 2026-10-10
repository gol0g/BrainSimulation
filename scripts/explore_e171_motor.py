#!/usr/bin/env python3
"""E171 탐색적 사후 분석(판정 밖, 판정 2026-10-10 17:10 뒤 작성) — 평가 진단 줄(쪽별 평균 motor·KC 발화율)에서
motor 차이 D = mR - mL 의 학습분(AB - none) 가산성 ρ_D = 일치 / (good 단독 + bad 단독), 비우세·우세 motor 학습분, 무학습 motor 발화율.
실행: python3 scripts/explore_e171_motor.py research/experiments (저장소 루트에서)"""
import re, os, sys
EXP = sys.argv[1]
DL = re.compile(r"^\[E171 평가 진단\] variant=(\w+) side=(\w+) n=(\d+) motor L/R ([0-9.]+)/([0-9.]+) KC L/R ([0-9.]+)/([0-9.]+)", re.M)
def rd(b, w, v):
    t = open(os.path.join(EXP, "logs", "E171", "ev_b%d_%s_%s.log" % (b, w, v)), encoding="utf-8").read()
    d = {g.group(2): tuple(float(g.group(i)) for i in (4, 5, 6, 7)) for g in DL.finditer(t) if g.group(1) == v}
    mod = float(re.search(r"^=> DECOMP mode=\w+ mod=([-+0-9.]+)", t, re.M).group(1))
    return d, mod
for b in (10, 11, 12, 13, 14):
    X = {(w, v): rd(b, w, v) for w in ("AB", "none") for v in ("base", "bad", "agree")}
    D = lambda w, v, s: X[(w, v)][0][s][1] - X[(w, v)][0][s][0]          # mR - mL
    dD = lambda v, s: D("AB", v, s) - D("none", v, s)
    # 일치 side left = good L + bad R ; 일치 side right = good R + bad L
    rL = dD("agree", "left") / (dD("base", "left") + dD("bad", "right"))
    rR = dD("agree", "right") / (dD("base", "right") + dD("bad", "left"))
    m = lambda w, v, s, k: X[(w, v)][0][s][k]
    # 비우세 motor: side left 에서 mL(0), side right 에서 mR(1) — 학습분
    nd = lambda v, s, k: m("AB", v, s, k) - m("none", v, s, k)
    print("b%d D학습분 L: 일치 %+.4f 단독합 %+.4f(%+.4f %+.4f) ρ_D %.2f | R: 일치 %+.4f 단독합 %+.4f ρ_D %.2f | 비우세 학습분 L쪽 mL: 일치 %+.4f 단독 good %+.4f bad %+.4f | 우세 mR: 일치 %+.4f 단독 good %+.4f bad %+.4f | 무학습 motor 일치 L %.3f/%.3f 단독 %.3f/%.3f"
          % (b, dD("agree", "left"), dD("base", "left") + dD("bad", "right"), dD("base", "left"), dD("bad", "right"), rL,
             dD("agree", "right"), dD("base", "right") + dD("bad", "left"), rR,
             nd("agree", "left", 0), nd("base", "left", 0), nd("bad", "right", 0), nd("agree", "left", 1), nd("base", "left", 1), nd("bad", "right", 1),
             m("none", "agree", "left", 0), m("none", "agree", "left", 1), m("none", "base", "left", 0), m("none", "base", "left", 1)))
