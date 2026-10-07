#!/usr/bin/env python3
"""E153 판정 — 기준 logs/E153/criteria_fixed.txt(경로 검사·본실험 전 고정). 뇌마다 형성·겹침·학습 3줄이 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄(E153.log):
  "  e153 b10 dev: => KCDEV side=l fired=.. sel_med0=.. sel_med=.. frac09=.. goodfrac=.. relerr=.. | side=r ... | n=100 eta=0.1 ..."
  "  e153 b10 ov: => KCOVERLAP side=l good=.. bad=.. jac=.. ... | side=r ... | ... || 적재 N"
  "  e153 b10 train: => 사전 +0.0195 사후 -0.2189 보상 273 || 적재 N || => KCTRACE ..."
K3 은 추적(traces/E153/tr_b*.npz)·원 학습 로그(반사 줄)로. 정수 비교(1e-4).
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
E141 = {10: -2384, 11: -2583, 12: -2733, 13: -2514, 14: -2715}
R20 = (1.0 - 1.0 / 12.0) ** 20
SIDE_DEV = r"side=%s fired=(\d+) sel_med0=([0-9.]+|nan) sel_med=([0-9.]+|nan) frac09=([0-9.]+|nan) goodfrac=([0-9.]+|nan) relerr=(\S+)"
LDEV = re.compile(r"^\s*e153 b(\d+) dev: => KCDEV " + SIDE_DEV % "l" + r" \| " + SIDE_DEV % "r")
LOV = re.compile(r"^\s*e153 b(\d+) ov: => KCOVERLAP side=l good=(\d+) bad=(\d+) jac=([0-9.]+|nan) .*\| side=r good=(\d+) bad=(\d+) jac=([0-9.]+|nan) .*\|\| 적재 (\d+)")
LTR = re.compile(r"^\s*e153 b(\d+) train: => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+) \|\| 적재 (\d+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def i4(x):
    return None if x == "nan" else int(round(float(x) * 1e4))


def stats(rows):
    act = rows[:, 6] >= 0
    rule = rows[:, 6] != rows[:, 2]
    eda = rows[:, 13:17]
    return {"n": len(rows), "agree": float(np.mean((rows[act, 7] == 1) == rule[act])) if act.any() else float("nan"),
            "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12))}


def judge(DEV, OV, TR, S, RF):
    miss = [b for b in BRAINS if b not in DEV or b not in OV or b not in TR or b not in S or RF.get(b) is None]
    if miss:
        return ["[측정 확인] 결측 %d — **판정 보류, 수치 미출력**" % len(miss)], None
    rows, k_ok = {}, 0
    sep_n = auth_n = both = nosep = sep_noauth = 0
    for b in BRAINS:
        d, o, t, s = DEV[b], OV[b], TR[b], S[b]
        k1 = all(d[k]["sel_med"] is not None and d[k]["sel_med"] >= 8000 and d[k]["relerr"] <= 1e-6 for k in "lr")
        k2 = o["load"] >= 1 and t["load"] >= 2
        k3 = s["res"] <= 1e-3 and s["n"] == 500 and s["agree"] == 1.0 and s["pre_ratio"] <= 1e-3 and RF[b]
        ok = k1 and k2 and k3
        k_ok += ok
        e = i4(t["post"]) - i4(t["pre"])
        sep = all(o[k] is not None and o[k] <= 2500 for k in ("Jl", "Jr"))
        auth = 3 * e <= 2 * E141[b]
        sep_n += sep; auth_n += auth
        both += sep and auth; nosep += not sep; sep_noauth += sep and not auth
        rows[b] = {"k1": k1, "k2": k2, "k3": k3, "e": e, "sep": sep, "auth": auth, "J": (o["Jl"], o["Jr"]), "G": o["Gl"] + o["Gr"], "B": o["Bl"] + o["Br"],
                   "sel": (d["l"]["sel_med"], d["r"]["sel_med"]), "sel0": (d["l"]["sel_med0"], d["r"]["sel_med0"]), "rew": t["rew"], "pre": t["pre"]}
    checks = ["[조작검증] K1 형성·K2 적재·K3 학습 모두 통과 %d/5" % k_ok]
    if k_ok < 5:
        v = "보류(조작검증 실패)"
    elif both >= 4:
        v = "형성 성공(H076) — 경험으로 형성한 종류 입력 재분배가 권한을 잃지 않고 good·bad KC 를 가른다"
    elif nosep >= 4:
        v = "분리 실패(H076-null) — 종류 입력 재분배로도 겹침이 0.25 아래로 내려가지 않는다"
    elif sep_noauth >= 4:
        v = "권한 상실(H076-auth) — 갈리지만 학습 효과가 기본의 2/3 미만"
    else:
        v = "보류"
    return checks, {"rows": rows, "sep": sep_n, "auth": auth_n, "both": both, "nosep": nosep, "sep_noauth": sep_noauth, "k_ok": k_ok, "verdict": v}


def report(checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        r = res["rows"][b]
        print("b%d: 형성 몫 중앙 %.3f/%.3f(전 %.3f/%.3f) | J %.4f/%.4f %s | 반응 G %d B %d | 효과 %+.4f (기본 E141 %+.4f, 2/3 = %+.4f) %s | 사전 %s 보상 %d | K1 %s K2 %s K3 %s"
              % (b, r["sel"][0] / 1e4, r["sel"][1] / 1e4, r["sel0"][0] / 1e4, r["sel0"][1] / 1e4, (r["J"][0] or 0) / 1e4, (r["J"][1] or 0) / 1e4,
                 "분리" if r["sep"] else "-", r["G"], r["B"], r["e"] / 1e4, E141[b] / 1e4, 2 * E141[b] / 3e4, "권한" if r["auth"] else "-",
                 r["pre"], r["rew"], "✓" if r["k1"] else "✗", "✓" if r["k2"] else "✗", "✓" if r["k3"] else "✗"))
    print("분리 %d/5 · 권한 %d/5 · 둘 다 %d/5 · 분리 아님 %d/5 · 분리·권한 아님 %d/5" % (res["sep"], res["auth"], res["both"], res["nosep"], res["sep_noauth"]))
    print("판정: %s" % res["verdict"])


def load():
    DEV, OV, TR, S, RF = {}, {}, {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E153.log"), encoding="utf-8", errors="replace"):
            m = LDEV.match(ln)
            if m:
                g = m.groups()
                b = int(g[0])
                DEV[b] = {k: {"fired": int(g[off]), "sel_med0": i4(g[off + 1]), "sel_med": i4(g[off + 2]), "frac09": g[off + 3], "goodfrac": g[off + 4],
                              "relerr": float(g[off + 5])} for k, off in (("l", 1), ("r", 7))}
                continue
            m = LOV.match(ln)
            if m:
                g = m.groups()
                OV[int(g[0])] = {"Gl": int(g[1]), "Bl": int(g[2]), "Jl": i4(g[3]), "Gr": int(g[4]), "Br": int(g[5]), "Jr": i4(g[6]), "load": int(g[7])}
                continue
            m = LTR.match(ln)
            if m:
                TR[int(m.group(1))] = {"pre": m.group(2), "post": m.group(3), "rew": int(m.group(4)), "load": int(m.group(5))}
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E153", "tr_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = stats(np.load(f)["rows"])
        try:
            txt = open(os.path.join(EXP, "logs", "E153", "train_b%d.log" % b), encoding="utf-8", errors="replace").read()
            got = {m.group(1): (m.group(2), m.group(3)) for m in (RW.match(ln) for ln in txt.splitlines()) if m}
            RF[b] = set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values())
        except FileNotFoundError:
            RF[b] = None
    return DEV, OV, TR, S, RF


if __name__ == "__main__":
    c, r = judge(*load())
    report(c, r)
    sys.exit(0)
