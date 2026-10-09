#!/usr/bin/env python3
"""E160 판정 — 기준 logs/E160/criteria_fixed.txt(2026-10-09 15:01:00 고정). 뇌 5개 자료가 다 모이기 전에는 수치를 출력하지 않는다.
원 로그: logs/E160/dev_b*.log(KCDEVOJA), ov_b*.log(KCOVERLAP), learn_b*.log([사전]/[사후]), kcrate_b*.log(부지표), 추적 traces/E160/tr_b*.npz,
기본 학습 logs/E141/b*.log(e_D), 기본 발화 logs/E157/kcrate_D_b*.log(부지표). r ≥ 1.30 ⇔ 10·E ≤ 13·E_D. 1e-4 정수.
실행: python3 scripts/judge_e160.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
R_STAR = (1.0 - 1.0 / 12.0) ** 20
SIDE = re.compile(r"side=([lr]) fired=(\d+) sel_med0=([0-9.na]+) sel_med=([0-9.na]+) frac09=([0-9.na]+) goodfrac=([0-9.na]+) "
                  r"sum_med=([0-9.na]+) sum_q10=([0-9.na]+) sum_q90=([0-9.na]+) dg_good=([0-9.na]+) dg_bad=([0-9.na]+)")


def i4(x):
    return int(round(x * 1e4))


def rd(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def dev(t):
    ln = next((x for x in t.splitlines() if x.startswith("=> KCDEVOJA ")), None) if t else None
    if ln is None:
        return None
    d = {m.group(1): {"fired": int(m.group(2)), "sel_med": float(m.group(4)), "frac09": float(m.group(5)), "sum_med": float(m.group(7)),
                      "dg_good": float(m.group(10)), "dg_bad": float(m.group(11))} for m in SIDE.finditer(ln)}
    return d if set(d) == {"l", "r"} else None


def jac(t):
    m = re.search(r"^=> KCOVERLAP side=l good=(\d+) bad=(\d+) jac=([0-9.]+) .*\| side=r good=(\d+) bad=(\d+) jac=([0-9.]+) ", t, re.M) if t else None
    return {"jl": float(m.group(3)), "jr": float(m.group(6)), "n": (int(m.group(1)), int(m.group(2)), int(m.group(4)), int(m.group(5)))} if m else None


def mods(t):
    a = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    b = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    return (i4(float(a.group(1))), i4(float(b.group(1)))) if (a and b) else None


def spikes(t):
    sp = {m.group(1): int(m.group(2)) for m in re.finditer(r"^=> KCRATE kc_([lr]) .*?제시 스파이크 (\d+) ", t, re.M)} if t else {}
    return sp["l"] + sp["r"] if set(sp) == {"l", "r"} else None


def nload(t):
    return len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M)) if t else 0


def stats(rows):
    eda, eend = rows[:, 13:17], rows[:, 21:25]
    return {"n": len(rows), "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(eend - R_STAR * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "alive": float(np.mean((np.abs(rows[:, 13]) + np.abs(rows[:, 14])) > 1.0))}


def load():
    X = {}
    for b in BRAINS:
        d = dev(rd("logs", "E160", "dev_b%d.log" % b))
        if d:
            X[("dev", b)] = d
        t = rd("logs", "E160", "ov_b%d.log" % b)
        if jac(t):
            X[("ov", b)] = dict(jac(t), ld=nload(t))
        t = rd("logs", "E160", "learn_b%d.log" % b)
        if mods(t):
            X[("learn", b)] = {"pre": mods(t)[0], "post": mods(t)[1], "ld": nload(t),
                               "rew": int(re.search(r"보상 (\d+)회", t).group(1)) if re.search(r"보상 (\d+)회", t) else None}
        t = rd("logs", "E141", "b%d.log" % b)
        if mods(t):
            X[("D", b)] = mods(t)[1] - mods(t)[0]
        f = os.path.join(EXP, "traces", "E160", "tr_b%d.npz" % b)
        if os.path.exists(f):
            X[("st", b)] = stats(np.load(f)["rows"])
        s1, s0 = spikes(rd("logs", "E160", "kcrate_b%d.log" % b)), spikes(rd("logs", "E157", "kcrate_D_b%d.log" % b))
        if s1 is not None and s0 is not None:
            X[("sp", b)] = (s1, s0)
    return X


def judge(X):
    miss = [(k, b) for b in BRAINS for k in ("dev", "ov", "learn", "D", "st") if (k, b) not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    mo = sum(all(X[("dev", b)][s][g] > 0 for s in "lr" for g in ("dg_good", "dg_bad")) for b in BRAINS)
    ms = sum(all(X[("dev", b)][s]["sel_med"] >= 0.80 for s in "lr") for b in BRAINS)
    ml = sum(X[("learn", b)]["ld"] >= 2 and X[("ov", b)]["ld"] >= 1 for b in BRAINS)
    m1 = sum(X[("st", b)]["res"] <= 1e-3 for b in BRAINS)
    m1b = sum(X[("st", b)]["alive"] >= 0.9 for b in BRAINS)
    m3 = sum(X[("st", b)]["n"] == 500 and X[("st", b)]["pre_ratio"] <= 1e-3 for b in BRAINS)
    md = sum(X[("D", b)] <= -1000 for b in BRAINS)
    ok = mo == ms == ml == m1 == m1b == m3 == md == 5
    checks = ["[조작검증] MO Oja 변화 %d/5 · MS 종류 선택성 %d/5 · ML 적재 %d/5 · M1 동결 %d/5 · M1b 되돌림 %d/5 · M3 추적 %d/5 · MD 기준 효과 %d/5 %s"
              % (mo, ms, ml, m1, m1b, m3, md, "통과" if ok else "실패")]
    sep = [b for b in BRAINS if i4(X[("ov", b)]["jl"]) <= 1000 and i4(X[("ov", b)]["jr"]) <= 1000]
    succ = [b for b in sep if 10 * (X[("learn", b)]["post"] - X[("learn", b)]["pre"]) <= 13 * X[("D", b)]]
    null = [b for b in BRAINS if i4(X[("ov", b)]["jl"]) > 2500 or i4(X[("ov", b)]["jr"]) > 2500]
    if not ok:
        v = "보류(조작검증 실패)"
    elif len(succ) >= 4:
        v = "형성 성공(H083) — 망 안 Oja 형성 가중치에서 겹침 ≤ 0.10·학습 효과 ≥ 기본의 1.30배(≥4/5)"
    elif len(sep) >= 4:
        v = "분리만(H083-sep-only) — 겹침은 사라지나 학습 이점 1.30배 미만"
    elif len(null) >= 4:
        v = "형성 실패(H083-null) — 겹침 > 0.25"
    else:
        v = "보류"
    return checks, {"sep": sep, "succ": succ, "null": null, "ok": ok, "verdict": v}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        L, d = X[("learn", b)], X[("dev", b)]
        e = L["post"] - L["pre"]
        sp = X.get(("sp", b))
        print("b%d 형성 sel_med %.3f/%.3f sum_med %.3f/%.3f frac09 %.2f/%.2f | 자카드 %.4f/%.4f | e %+.4f (기본 %+.4f, r %.2f) 보상 %s | 발화 비 %s"
              % (b, d["l"]["sel_med"], d["r"]["sel_med"], d["l"]["sum_med"], d["r"]["sum_med"], d["l"]["frac09"], d["r"]["frac09"],
                 X[("ov", b)]["jl"], X[("ov", b)]["jr"], e / 1e4, X[("D", b)] / 1e4, e / X[("D", b)], L["rew"],
                 ("%.3f" % (sp[0] / sp[1])) if sp else "-"))
    print("분리(자카드 ≤ 0.10 양쪽) %d/5 | 성공(+ r ≥ 1.30) %d/5 | 실패(> 0.25) %d/5" % (len(res["sep"]), len(res["succ"]), len(res["null"])))
    print("판정: %s" % res["verdict"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
