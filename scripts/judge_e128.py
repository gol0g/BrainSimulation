#!/usr/bin/env python3
"""E128 판정 — E128.md 4절. crosshalf 24런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): train_lbal = 훈련 8자극(균형 같음/다름 cyclic) 라벨 균형 정답률(%), 평가 100시행.
shared_frac = 양쪽 라벨 반응 KC 수 / (양쪽 + L전용 + R전용) — 희석 지표(조작검증). K50 값은 E127 같은 런.
"""
import os
import re
import sys

EXP = "research/experiments"
WIRES = tuple(range(10, 18))
TSEEDS = (600, 601)
OK = 80.0
CNT = re.compile(r"L전용 n=(\d+) dSg=\S+ \| R전용 n=(\d+) dSg=\S+ \| 양쪽 n=(\d+)")
T28 = re.compile(r"^\s*xh (learn|frozen) w(\d+) t(\d+): => SDCREDIT (.*?) \|\| => SDRATE .*?both_spike_share\(.*?\) ([0-9.]+) .*?\|\| => SDLAB diff=cyclic rule=samediff .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)")
T27 = re.compile(r"^\s*cr (learn|frozen) w(\d+) t(\d+): => SDCREDIT (.*?) \|\| => SDLAB")


def sfrac(txt):
    m = CNT.search(txt)
    L, R, B = (int(x) for x in m.groups())
    return B / (B + L + R) if (B + L + R) else float("nan")


def judge(X, K):
    """X: {(mode,w,t): dict(sf, bss, tl, nl)} crosshalf, K: {(mode,w,t): sf} K50(E127)"""
    need = [("learn", w, t) for w in WIRES for t in TSEEDS] + [("frozen", w, 600) for w in WIRES]
    miss = [k for k in need if k not in X] + [("E127",) + k for k in need if k not in K]
    if miss:
        return ["[측정 확인] 결측 %d: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    checks, ok = [], True
    red = sum(X[("frozen", w, 600)]["sf"] < K[("frozen", w, 600)] for w in WIRES)
    checks.append("[측정 확인] 희석 감소(crosshalf shared_frac < K50, frozen 배선별): %d/8 (≥7 필요)" % red)
    ok &= red >= 7
    same = [w for w in WIRES if X[("learn", w, 600)] == X[("frozen", w, 600)]]
    checks.append("[측정 확인] learn ≠ frozen(배선별 t600): %d/8" % (8 - len(same)))
    ok &= not same
    L = [X[("learn", w, t)] for w in WIRES for t in TSEEDS]
    n_acq = sum(x["tl"] >= OK for x in L)
    n_fr = sum(X[("frozen", w, 600)]["tl"] >= OK for w in WIRES)
    if not ok:
        verdict = "보류(조작검증 실패 — 희석이 안 줄었거나 학습 무효)"
    elif n_acq >= 12 and n_fr <= 2:
        verdict = "획득 — 교차 결합 배선(희석 감소)에서 같음/다름 훈련 자극을 학습한다"
    elif n_acq <= 4:
        verdict = "불획득 — 희석을 줄여도 획득 못 함(희석 외 원인)"
    else:
        verdict = "보류(혼재 또는 frozen 성공 과다)"
    return checks, {"n_acq": n_acq, "n_fr": n_fr, "red": red, "ok": ok, "verdict": verdict}


def report(checks, res, X=None, K=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        xs = [X[("learn", w, t)] for t in TSEEDS]; f = X[("frozen", w, 600)]
        print("w%d crosshalf learn train %s novel %s | shared_frac %.2f (K50 %.2f) || frozen train %.0f"
              % (w, "/".join("%.0f" % x["tl"] for x in xs), "/".join("%.0f" % x["nl"] for x in xs), f["sf"], K[("frozen", w, 600)], f["tl"]))
    Lk = [("learn", w, t) for w in WIRES for t in TSEEDS]
    print("평균: crosshalf learn train %.1f%% novel %.1f%% | frozen train %.1f%% | shared_frac crosshalf %.2f vs K50 %.2f"
          % (sum(X[k]["tl"] for k in Lk) / 16, sum(X[k]["nl"] for k in Lk) / 16, sum(X[("frozen", w, 600)]["tl"] for w in WIRES) / 8,
             sum(X[("frozen", w, 600)]["sf"] for w in WIRES) / 8, sum(K[("frozen", w, 600)] for w in WIRES) / 8))
    print("획득(train ≥80%%): learn %d/16, frozen %d/8" % (res["n_acq"], res["n_fr"]))
    print("판정: %s" % res["verdict"])


def load():
    X, K = {}, {}
    for fn, rx in (("E128.log", T28), ("E127.log", T27)):
        try:
            for ln in open(os.path.join(EXP, fn), encoding="utf-8"):
                m = rx.match(ln)
                if not m:
                    continue
                k = (m.group(1), int(m.group(2)), int(m.group(3)))
                if rx is T28:
                    X[k] = {"sf": sfrac(m.group(4)), "bss": float(m.group(5)), "tl": float(m.group(6)), "nl": float(m.group(7))}
                else:
                    K[k] = sfrac(m.group(4))
        except FileNotFoundError:
            pass
    return X, K


if __name__ == "__main__":
    X, K = load()
    c, r = judge(X, K)
    report(c, r, X, K)
    sys.exit(0)
