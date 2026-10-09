#!/usr/bin/env python3
"""E167 판정 — 최소 회로 관계 전이에서 호스트 승자 선택·가중치 재부여 제거(외부 검토 권고 2). 기준 logs/E167/criteria_fixed.txt.
발달 16 + 과제 64 가 다 모이기 전에는 수치를 출력하지 않는다.
N = 발달이 끝난 망 그대로(--kc-wiring ojafull: 가지치기·재부여 없음), H = 같은 배선 E136 corr(호스트 가지치기 + 설계 가중치, loaded).
단위(규약 P19): novel_lbal = 처음 보는 항목(4~7) 라벨 균형 정답률(%). 독립 단위 = 배선(난수열 600·601 평균). 0.1%p 정수(×10).
Δ_N(w) = N learn − N frozen, Δ_H(w) = E136 corr learn − corr frozen.
판정: 남음(H090) = ΣΔ_N ≥ 0.8·ΣΔ_H 이고 Δ_N > 0 인 배선 ≥ 14/16. 손실(H090-null) = 평균 Δ_N < +5%p. 그 밖 부분.
조작검증(하나라도 실패면 보류): MDV 발달 재현 — E167 발달 가지치기(a,b,e,i)가 E136 발달 저장본과 정확히 같음 16/16(발달 경로 불변),
MLD '[KC망안]' 줄 64/64(후보 수 = 발달 저장본, 장치 대조 최대 |차| ≤ 1e-6), MH E136 기준 로그 64/64.
실행: python3 scripts/judge_e167.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
WIRES = tuple(range(78, 94))
TSEEDS = (600, 601)
SDL = re.compile(r"^=> SDLAB diff=cyclic rule=samediff mode=(learn|frozen) seed=(\d+) trialseed=(\d+) .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)", re.M)
KML = re.compile(r"^\[KC망안\] .*?흥분 후보 (\d+)개 평균 ([0-9.]+) 같은 위치 몫 ([0-9.]+) \| 억제 후보 (\d+)개 크기 평균 ([0-9.]+) 같은 위치 몫 ([0-9.]+) "
                 r"\| 균등 ([0-9.]+) \| 장치 대조 최대 \|차\| ([0-9.e+-]+)", re.M)


def i1(x):
    return int(round(float(x) * 10))


def rd(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def sd(t, mode, w, ts):
    m = SDL.search(t) if t else None
    if not m or m.group(1) != mode or int(m.group(2)) != w or int(m.group(3)) != ts:
        return None
    return {"train": i1(m.group(4)), "novel": i1(m.group(5))}


def load():
    X = {}
    for w in WIRES:
        f7, f6 = (os.path.join(EXP, "traces", e, "dev_corr_w%d.npz" % w) for e in ("E167", "E136"))
        if os.path.exists(f7) and os.path.exists(f6):
            z7, z6 = np.load(f7), np.load(f6)
            X[("dv", w)] = {"same": all(np.array_equal(z7[k], z6[k]) for k in ("a", "b", "e", "i")),
                            "nc": int(z7["cpre"].size) if "cpre" in z7.files else -1, "ni": int(z7["ipre"].size) if "ipre" in z7.files else -1}
        for ts in TSEEDS:
            for mode in ("learn", "frozen"):
                t = rd("logs", "E167", "N_%s_w%d_t%d.log" % (mode, w, ts))
                s = sd(t, mode, w, ts)
                k = KML.search(t) if t else None
                if s and k:
                    X[("N", mode, w, ts)] = dict(s, nc=int(k.group(1)), ni=int(k.group(4)), dmax=float(k.group(8)),
                                                 share_c=float(k.group(3)), share_i=float(k.group(6)), uni=float(k.group(7)))
                t = rd("logs", "E136", "corr_%s_w%d_t%d.log" % (mode, w, ts))
                s = sd(t, mode, w, ts)
                if s:
                    X[("H", mode, w, ts)] = s
    return X


def judge(X):
    need = [("dv", w) for w in WIRES] + [(g, mode, w, ts) for g in ("N", "H") for mode in ("learn", "frozen") for w in WIRES for ts in TSEEDS]
    miss = [k for k in need if k not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    mdv = sum(X[("dv", w)]["same"] for w in WIRES)
    mld = sum(X[("N", mode, w, ts)]["dmax"] <= 1e-6 and X[("N", mode, w, ts)]["nc"] == X[("dv", w)]["nc"] and X[("N", mode, w, ts)]["ni"] == X[("dv", w)]["ni"]
              and X[("dv", w)]["nc"] > 0 for mode in ("learn", "frozen") for w in WIRES for ts in TSEEDS)
    ok = mdv == 16 and mld == 64
    checks = ["[조작검증] MDV 발달 재현(가지치기 = E136) %d/16 · MLD 망 안 배선 적재·대조 %d/64 · MH 기준 로그 64/64 %s" % (mdv, mld, "통과" if ok else "실패")]
    nov = lambda g, mode, w: X[(g, mode, w, 600)]["novel"] + X[(g, mode, w, 601)]["novel"]      # 두 난수열 합(0.1%p) = 평균 × 2
    dN = {w: nov("N", "learn", w) - nov("N", "frozen", w) for w in WIRES}
    dH = {w: nov("H", "learn", w) - nov("H", "frozen", w) for w in WIRES}
    pos = sum(dN[w] > 0 for w in WIRES)
    keep = 10 * sum(dN.values()) >= 8 * sum(dH.values()) and pos >= 14
    lose = sum(dN.values()) < 16 * 2 * 50          # 평균 Δ_N < +5.0%p ⇔ Σ(두 난수열 합 차) < 16 배선 × 2 × 50(0.1%p)
    v = "보류(조작검증 실패)" if not ok else ("남음(H090)" if keep else ("손실(H090-null)" if lose else "부분"))
    return checks, {"dN": dN, "dH": dH, "pos": pos, "verdict": v, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        n = X[("N", "learn", w, 600)]
        print("w%d N 새 항목 learn %.1f/%.1f frozen %.1f/%.1f (훈련 learn %.1f/%.1f) Δ_N %+.1f | H learn %.1f/%.1f frozen %.1f/%.1f Δ_H %+.1f | 같은 위치 몫 %.4f/%.4f(균등 %.4f)"
              % (w, X[("N", "learn", w, 600)]["novel"] / 10, X[("N", "learn", w, 601)]["novel"] / 10, X[("N", "frozen", w, 600)]["novel"] / 10,
                 X[("N", "frozen", w, 601)]["novel"] / 10, X[("N", "learn", w, 600)]["train"] / 10, X[("N", "learn", w, 601)]["train"] / 10,
                 res["dN"][w] / 20, X[("H", "learn", w, 600)]["novel"] / 10, X[("H", "learn", w, 601)]["novel"] / 10,
                 X[("H", "frozen", w, 600)]["novel"] / 10, X[("H", "frozen", w, 601)]["novel"] / 10, res["dH"][w] / 20, n["share_c"], n["share_i"], n["uni"]))
    mean = lambda g, mode: sum(X[(g, mode, w, ts)]["novel"] for w in WIRES for ts in TSEEDS) / 320.0
    print("N learn 평균 새 항목 %.1f%% · N frozen %.1f%% · H learn %.1f%% · H frozen %.1f%% | 평균 Δ_N %+.1f%%p · Δ_H %+.1f%%p (비 %.2f) · Δ_N > 0 %d/16"
          % (mean("N", "learn"), mean("N", "frozen"), mean("H", "learn"), mean("H", "frozen"), sum(res["dN"].values()) / 320.0,
             sum(res["dH"].values()) / 320.0, sum(res["dN"].values()) / max(sum(res["dH"].values()), 1), res["pos"]))
    print("판정: %s" % res["verdict"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
