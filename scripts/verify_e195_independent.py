#!/usr/bin/env python3
"""E195 독립 대조 — judge_e195.py·rcxor.py 를 쓰지 않고 원 로그를 정규식으로 읽어 순도를 따로 계산하고, 해독은 verify_e192_independent(직접 계산 능형 회귀)로 다시 판정.
규칙은 logs/E195/criteria_fixed.txt. 실행: python3 scripts/verify_e195_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e192_independent as V  # noqa: E402

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
LEVELS = (("S1", 1.0), ("SP", 1.5), ("SR", 2.0))
NK = 3000


def load(path, t):
    s = open(path, encoding="utf-8", errors="replace").read()
    rows = {"l": [], "r": []}
    for m in re.finditer(r"^RCKX rep=\d+ side=([lr])(?: k=\d+)? set=(train|new) cat=([AB]) ctx=([XY]) i=(\d+) n=\d+ ids=(.*)$", s, re.M):
        v = np.zeros(NK)
        for k, c in re.findall(r"(\d+):(\d+)", m.group(6)):
            v[int(k)] = float(c)
        rows[m.group(1)].append((m.group(2), m.group(3), m.group(4), int(m.group(5)), v))
    ck = re.search(r"^RCKXCHK npres=(\d+) ctx_on_spikes=(\d+) ctx_off_spikes=(\d+) feat_inactive_spikes=(\d+) \S+ \S+ g_same=(\d) ctx_w=([0-9.]+) ctx_p=([0-9.]+) ctx2_w=([0-9.]+) ctx2_p=([0-9.]+) kc_target=[0-9.]+ cfg_kc_target=([0-9.]+) kc_inh_scale=([0-9.]+) cfg_kc_inh_w=([-0-9.]+) inh_g_l=([-0-9.]+) inh_g_r=([-0-9.]+)", s, re.M)
    cl = re.findall(r"^\s*\[맥락 입력(?: 2)?\] .* w=([0-9.]+), p=([0-9.]+), 연결 좌 (\d+)·우 (\d+)", s, re.M)
    kinds = set(re.findall(r"^\s*\[(맥락 입력(?: 2)?)\] ", s, re.M))
    bad = (not ck or ck.group(1) != "288" or int(ck.group(2)) <= 0 or ck.group(3) != "0" or ck.group(4) != "0" or ck.group(5) != "1"
           or abs(float(ck.group(6)) - 20) > 1e-6 or abs(float(ck.group(7)) - 0.02) > 1e-9 or abs(float(ck.group(8)) - 20) > 1e-6 or abs(float(ck.group(9)) - 0.02) > 1e-9
           or abs(float(ck.group(10)) - 0.05) > 1e-9 or abs(float(ck.group(11)) - t) > 1e-9 or abs(float(ck.group(12)) + 15.0 * t) > 1e-6
           or abs(float(ck.group(13)) + 15.0 * t) > 1e-3 or abs(float(ck.group(14)) + 15.0 * t) > 1e-3)
    bad = bad or kinds != {"맥락 입력", "맥락 입력 2"} or any(abs(float(a) - 20) > 1e-6 or abs(float(b) - 0.02) > 1e-9 or not (0.85 <= int(n1) / 12000.0 <= 1.15 and 0.85 <= int(n2) / 12000.0 <= 1.15) for a, b, n1, n2 in cl)
    bad = bad or not re.search(r"^\s*\[원시 특징\] .*w=24\.00, p=0\.100", s, re.M) or not re.search(r"^\[E153 종류 입력 적재\].*검증 일치", s, re.M)
    if abs(t - 1.0) > 1e-9:
        m195 = re.search(r"^\[E195 KC 되먹임 억제\] 배율 ([0-9.]+)", s, re.M)
        bad = bad or not m195 or abs(float(m195.group(1)) - t) > 1e-9
    for sd in "lr":
        bad = bad or sum(r[0] == "train" for r in rows[sd]) != 72 or sum(r[0] == "new" for r in rows[sd]) != 72
    return rows, bad


def purity(rows):
    tr = [r for r in rows if r[0] == "train"]
    m = {}
    for key in ("AX", "AY", "BX", "BY"):
        vs = [r[4] for r in tr if r[1] + r[2] == key]
        m[key] = np.mean(vs, axis=0)
    tot = m["AX"] + m["AY"] + m["BX"] + m["BY"]
    inter = np.abs(m["AX"] + m["BY"] - m["AY"] - m["BX"])
    resp = tot >= 1.0                    # 칸 평균 ≥ 0.25 ⇔ 네 칸 합 ≥ 1
    p = inter[resp] / tot[resp]
    return float((p * tot[resp]).sum() / tot[resp].sum()), float(np.mean([np.count_nonzero(r[4]) for r in tr]))


def metrics(path, t):
    rows, bad = load(path, t)
    if bad:
        return None
    p = [purity(rows[sd]) for sd in "lr"]
    acc = {k: (V.side_acc([r for r in rows["l"]], k) + V.side_acc([r for r in rows["r"]], k)) / 2.0 for k in ("xor", "cat", "ctx")}
    return {"p": (p[0][0] + p[1][0]) / 2.0, "n": (p[0][1] + p[1][1]) / 2.0, **acc}


def main():
    M = {}
    try:
        for b in BRAINS:
            for lv, t in LEVELS:
                r = metrics(os.path.join(EXP, "logs", "E195", "meas_b%d_%s.log" % (b, lv)), t)
                if r is None:
                    print("독립 판정: 보류(조작검증 실패)")
                    return 0
                M[(b, lv)] = r
    except (OSError, ValueError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    i4 = lambda x: int(round(x * 10000))
    sp = sum(i4(M[(b, "SP")]["n"] * 10) <= i4(M[(b, "S1")]["n"] * 9) for b in BRAINS)
    rise = loss = flat = 0
    for b in BRAINS:
        a, z = M[(b, "S1")], M[(b, "SP")]
        dq = i4(z["p"]) - i4(a["p"])
        val = i4(z["cat"]) >= 9000 and i4(z["ctx"]) >= 9000
        rise += val and dq >= 500 and i4(z["xor"]) >= 7500
        loss += val and dq >= 500 and i4(z["xor"]) < 7500
        flat += val and abs(dq) < 200
        print("b%d p %.4f → %.4f · n %.0f → %.0f · XOR %.3f" % (b, a["p"], z["p"], a["n"], z["n"], z["xor"]))
    if sp < 4:
        print("독립 판정: 보류(조작 무효 — 희소화 %d/5)" % sp)
        return 0
    v = "순도 상승(H116)" if rise >= 4 else "결합 소실" if loss >= 4 else "변화 없음(H116-null)" if flat >= 4 else "보류"
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
