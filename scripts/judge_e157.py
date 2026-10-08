#!/usr/bin/env python3
"""E157 판정 — 기준 logs/E157/criteria_fixed.txt(2026-10-09 01:06:04 고정). 모든 팔·뇌 자료가 모이기 전에는 수치를 출력하지 않는다.
원 로그를 직접 읽는다: 학습 e(D = logs/E157/learn_D 가 있으면 그것, 없으면 E141 b{B}; F = learn_F 또는 E153 train_b{B}; Fk·Dk = logs/E157/learn_*),
kcrate 총수(logs/E157/kcrate_*), 겹침(ov_*), 반사 25 [사전](r25_*, 부지표), 추적(traces/E157/tr_*), 적재·배율 줄.
e = [사후] − [사전](1e-4 정수). S ≤ 0.30 ⇔ 10·E_Fk ≥ 13·E_D, G > 0.30 ⇔ 10·E_Dk < 13·E_D (e_D < 0).
실행: python3 scripts/judge_e157.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = ("D", "F", "Fk", "Dk")
R_STAR = (1.0 - 1.0 / 12.0) ** 20
CATS = ("이득(H080)", "분리(H080-sep)", "둘 다(H080-both)", "둘 다 아님(H080-int)")


def i4(x):
    return int(round(x * 1e4))


def rd(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None


def mods(txt):
    a = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", txt, re.M)
    b = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", txt, re.M)
    return (float(a.group(1)) if a else None), (float(b.group(1)) if b else None)


def n_load(txt):
    return len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", txt, re.M))


def n_scale(txt):
    return len(re.findall(r"^\[E157 종류 입력 배율\].*검증 일치", txt, re.M))


def spikes(txt):
    sp = {m.group(1): int(m.group(2)) for m in re.finditer(r"^=> KCRATE kc_([lr]) .*?제시 스파이크 (\d+) ", txt, re.M)}
    return sp["l"] + sp["r"] if set(sp) == {"l", "r"} else None


def jacs(txt):
    m = re.search(r"^=> KCOVERLAP side=l good=\d+ bad=\d+ jac=([0-9.]+) .*\| side=r good=\d+ bad=\d+ jac=([0-9.]+) ", txt, re.M)
    return (float(m.group(1)), float(m.group(2))) if m else None


def rewards(txt):
    m = re.search(r"보상 (\d+)회", txt)
    return int(m.group(1)) if m else None


def stats(rows):
    eda, eend = rows[:, 13:17], rows[:, 21:25]
    return {"n": len(rows), "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(eend - R_STAR * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "alive": float(np.mean((np.abs(rows[:, 13]) + np.abs(rows[:, 14])) > 1.0))}


def learn_path(a, b):
    own = os.path.join(EXP, "logs", "E157", "learn_%s_b%d.log" % (a, b))
    if a in ("Fk", "Dk") or os.path.exists(own):
        return own
    return os.path.join(EXP, "logs", "E141", "b%d.log" % b) if a == "D" else os.path.join(EXP, "logs", "E153", "train_b%d.log" % b)


def load():
    """→ {(종류, 팔, 뇌): 값} 자료. 없는 값은 넣지 않는다."""
    X = {}
    for b in BRAINS:
        for a in ARMS:
            t = rd(learn_path(a, b))
            if t is not None:
                pre, post = mods(t)
                if pre is not None and post is not None:
                    X[("e", a, b)] = i4(post) - i4(pre)
                    X[("rew", a, b)] = rewards(t)
                    X[("ld", a, b)] = (n_load(t), n_scale(t))
            t = rd(os.path.join(EXP, "logs", "E157", "kcrate_%s_b%d.log" % (a, b)))
            if t is not None and spikes(t) is not None:
                X[("sp", a, b)] = spikes(t)
            t = rd(os.path.join(EXP, "logs", "E157", "r25_%s_b%d.log" % (a, b)))
            if t is not None and mods(t)[0] is not None:
                X[("r25", a, b)] = mods(t)[0]
            if a in ("Fk", "Dk"):
                t = rd(os.path.join(EXP, "logs", "E157", "ov_%s_b%d.log" % (a, b)))
                if t is not None and jacs(t) is not None:
                    X[("J", a, b)] = jacs(t)
                f = os.path.join(EXP, "traces", "E157", "tr_%s_b%d.npz" % (a, b))
                if os.path.exists(f):
                    X[("st", a, b)] = stats(np.load(f)["rows"])
    return X


NEED = [("e", a) for a in ARMS] + [("sp", a) for a in ARMS] + [("r25", a) for a in ARMS] + [("J", "Fk"), ("J", "Dk"), ("st", "Fk"), ("st", "Dk")]


def judge(X):
    miss = [(k, a, b) for (k, a) in NEED for b in BRAINS if (k, a, b) not in X]
    if miss:
        return ["[측정 확인] 결측 %d건(예: %s) — **판정 보류, 수치 미출력**" % (len(miss), miss[:4])], None
    ms = sum(85 * X[("sp", "D", b)] <= 100 * X[("sp", "Fk", b)] <= 115 * X[("sp", "D", b)]
             and 85 * X[("sp", "F", b)] <= 100 * X[("sp", "Dk", b)] <= 115 * X[("sp", "F", b)] for b in BRAINS)
    mj = sum(all(i4(j) <= 500 for j in X[("J", "Fk", b)]) and all(i4(j) >= 2500 for j in X[("J", "Dk", b)]) for b in BRAINS)
    ml = sum(X[("ld", "Fk", b)][0] >= 2 and X[("ld", "Fk", b)][1] >= 2 and X[("ld", "Dk", b)] [1] >= 2 and X[("ld", "Dk", b)][0] == 0 for b in BRAINS)
    m1 = sum(all(X[("st", a, b)]["res"] <= 1e-3 for a in ("Fk", "Dk")) for b in BRAINS)
    m1b = sum(all(X[("st", a, b)]["alive"] >= 0.9 for a in ("Fk", "Dk")) for b in BRAINS)
    m3 = sum(all(X[("st", a, b)]["n"] == 500 and X[("st", a, b)]["pre_ratio"] <= 1e-3 for a in ("Fk", "Dk")) for b in BRAINS)
    md = sum(X[("e", "D", b)] <= -1000 for b in BRAINS)
    ok = ms == mj == ml == m1 == m1b == m3 == md == 5
    checks = ["[조작검증] MS 발화 맞춤 %d/5 · MJ 겹침 %d/5 · ML 적재·배율 %d/5 · M1 동결 %d/5 · M1b 되돌림 %d/5 · M3 추적 %d/5 · MD 기준 효과 %d/5 %s"
              % (ms, mj, ml, m1, m1b, m3, md, "통과" if ok else "실패")]
    cat = {}
    for b in BRAINS:
        eD, eFk, eDk = X[("e", "D", b)], X[("e", "Fk", b)], X[("e", "Dk", b)]
        s_small = 10 * eFk >= 13 * eD          # S ≤ 0.30
        g_big = 10 * eDk < 13 * eD             # G > 0.30
        cat[b] = CATS[0] if (s_small and g_big) else CATS[1] if (not s_small and not g_big) else CATS[2] if (not s_small and g_big) else CATS[3]
    cnt = {c: sum(cat[b] == c for b in BRAINS) for c in CATS}
    top = [c for c in CATS if cnt[c] >= 4]
    verdict = "보류(조작검증 실패)" if not ok else (top[0] if top else "보류")
    return checks, {"cat": cat, "cnt": cnt, "verdict": verdict, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        eD = X[("e", "D", b)] / 1e4
        r = {a: (X[("e", a, b)] / 1e4) / eD for a in ARMS}
        T = r["F"] - 1
        print("b%d e D %+.4f F %+.4f Fk %+.4f Dk %+.4f | r F %.2f Fk %.2f Dk %.2f | S %+.2f G %+.2f T %+.2f (S/T %.2f G/T %.2f) | 발화 D %d F %d Fk %d Dk %d"
              " (Fk/D %.3f Dk/F %.3f) | J Fk %.4f/%.4f Dk %.4f/%.4f | 반사25 사전 D %+.4f F %+.4f Fk %+.4f Dk %+.4f | 보상 %s | %s"
              % (b, eD, X[("e", "F", b)] / 1e4, X[("e", "Fk", b)] / 1e4, X[("e", "Dk", b)] / 1e4, r["F"], r["Fk"], r["Dk"],
                 r["Fk"] - 1, r["Dk"] - 1, T, (r["Fk"] - 1) / T if T else float("nan"), (r["Dk"] - 1) / T if T else float("nan"),
                 X[("sp", "D", b)], X[("sp", "F", b)], X[("sp", "Fk", b)], X[("sp", "Dk", b)],
                 X[("sp", "Fk", b)] / X[("sp", "D", b)], X[("sp", "Dk", b)] / X[("sp", "F", b)],
                 X[("J", "Fk", b)][0], X[("J", "Fk", b)][1], X[("J", "Dk", b)][0], X[("J", "Dk", b)][1],
                 X[("r25", "D", b)], X[("r25", "F", b)], X[("r25", "Fk", b)], X[("r25", "Dk", b)],
                 "/".join(str(X.get(("rew", a, b))) for a in ARMS), res["cat"][b]))
    print("범주: " + " · ".join("%s %d/5" % (c, res["cnt"][c]) for c in CATS))
    print("판정: %s" % res["verdict"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
