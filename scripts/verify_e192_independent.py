#!/usr/bin/env python3
"""E192 독립 대조 — judge_e192.py·rcxor.py 를 쓰지 않고 원 로그를 정규식으로 읽어 능형 회귀(직접 계산 형식)를 따로 구현해 다시 판정.
규칙은 logs/E192/criteria_fixed.txt. 실행: python3 scripts/verify_e192_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
CONFIGS = (("C0", 0.10, 4.0), ("C1", 0.05, 8.0), ("C2", 0.02, 20.0))
LAMS = (1.0, 10.0, 100.0, 1000.0)
NK = 3000


def load(path, p, w):
    t = open(path, encoding="utf-8", errors="replace").read()
    rows = {"l": [], "r": []}
    for m in re.finditer(r"^RCKX rep=(\d+) side=([lr])(?: k=\d+)? set=(train|new) cat=([AB]) ctx=([XY]) i=(\d+) n=(\d+) ids=(.*)$", t, re.M):
        v = np.zeros(NK)
        for k, c in re.findall(r"(\d+):(\d+)", m.group(8)):
            v[int(k)] = float(c)
        rows[m.group(2)].append((m.group(3), m.group(4), m.group(5), int(m.group(6)), v))
    ck = re.search(r"^RCKXCHK npres=(\d+) ctx_on_spikes=(\d+) ctx_off_spikes=(\d+) feat_inactive_spikes=(\d+) \S+ \S+ g_same=(\d) ctx_w=([0-9.]+) ctx_p=([0-9.]+) ctx2_w=([0-9.]+) ctx2_p=([0-9.]+)", t, re.M)
    cl = re.findall(r"^\s*\[맥락 입력(?: 2)?\] .* w=([0-9.]+), p=([0-9.]+), 연결 좌 (\d+)·우 (\d+)", t, re.M)
    bad = not ck or ck.group(1) != "288" or int(ck.group(2)) <= 0 or ck.group(3) != "0" or ck.group(4) != "0" or ck.group(5) != "1"
    bad = bad or any(abs(float(ck.group(g)) - (w if g in (6, 8) else p)) > 1e-6 for g in (6, 7, 8, 9)) if ck else True
    bad = bad or len(cl) != 2 or any(abs(float(a) - w) > 1e-6 or abs(float(b) - p) > 1e-9 or not (0.85 <= int(n1) / (600000 * p) <= 1.15 and 0.85 <= int(n2) / (600000 * p) <= 1.15)
                                     for a, b, n1, n2 in cl)
    bad = bad or not re.search(r"^\s*\[원시 특징\] .*w=24\.00, p=0\.100", t, re.M) or not re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M)
    for sd in "lr":
        bad = bad or sum(r[0] == "train" for r in rows[sd]) != 72 or sum(r[0] == "new" for r in rows[sd]) != 72
    return rows, bad


def lab(r, kind):
    _, c, x, _, _ = r
    return (1.0 if (c == "B") == (x == "X") else -1.0) if kind == "xor" else (1.0 if c == "B" else -1.0) if kind == "cat" else (1.0 if x == "Y" else -1.0)


def fitpred(Xtr, ytr, Xte, lam):
    mu, my = Xtr.mean(0), ytr.mean()
    A = Xtr - mu
    alpha = np.linalg.solve(A @ A.T + lam * np.eye(len(ytr)), ytr - my)
    return (Xte - mu) @ (A.T @ alpha) + my


def side_acc(rows, kind):
    tr = [r for r in rows if r[0] == "train"]; te = [r for r in rows if r[0] == "new"]
    Xtr = np.array([r[4] for r in tr]); ytr = np.array([lab(r, kind) for r in tr])
    Xte = np.array([r[4] for r in te]); yte = np.array([lab(r, kind) for r in te])
    g = [(r[1], r[3]) for r in tr]
    best = (-1.0, None)
    for lam in LAMS:
        hit = 0
        for gg in sorted(set(g)):
            m = np.array([x == gg for x in g])
            hit += int(np.sum(np.sign(fitpred(Xtr[~m], ytr[~m], Xtr[m], lam)) == ytr[m]))
        if hit / float(len(ytr)) >= best[0]:
            best = (hit / float(len(ytr)), lam)
    return float(np.mean(np.sign(fitpred(Xtr, ytr, Xte, best[1])) == yte))


def main():
    res, ok = {}, True
    try:
        for b in BRAINS:
            for c, p, w in CONFIGS:
                rows, bad = load(os.path.join(EXP, "logs", "E192", "meas_b%d_%s.log" % (b, c)), p, w)
                ok &= not bad
                if ok:
                    res[(b, c)] = {k: (side_acc(rows["l"], k) + side_acc(rows["r"], k)) / 2.0 for k in ("xor", "cat", "ctx")}
    except (OSError, ValueError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    suf, ins, mean = {}, {}, {}
    for c, _, _ in CONFIGS:
        vals = [res[(b, c)] for b in BRAINS]
        suf[c] = sum(v["cat"] >= 0.9 and v["ctx"] >= 0.9 and v["xor"] >= 0.75 for v in vals)
        ins[c] = sum(v["cat"] >= 0.9 and v["ctx"] >= 0.9 and v["xor"] <= 0.60 for v in vals)
        mean[c] = sum(v["xor"] for v in vals) / len(vals)
        print("%s XOR %s" % (c, " ".join("%.3f" % v["xor"] for v in vals)))
    if suf["C0"] >= 4:
        v = "표현 충분(H113-null)"
    elif ins["C0"] >= 4 and (suf["C1"] >= 4 or suf["C2"] >= 4):
        v = "설정 의존(H113) — %s" % max([c for c in ("C1", "C2") if suf[c] >= 4], key=lambda c: mean[c])
    elif all(ins[c] >= 4 for c in ("C0", "C1", "C2")):
        v = "결합 부호 부족(설정 무관)"
    else:
        v = "보류"
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
