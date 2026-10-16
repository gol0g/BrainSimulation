#!/usr/bin/env python3
"""E199 독립 대조 — judge_e199·judge_e193·judge_e198·kc_comp 를 쓰지 않고 다시 구현(정규식 파싱·np.add.at 누적·실수 산술).
규칙은 research/experiments/logs/E199/criteria_fixed.txt(판정 범주 = E193).
실행: python3 scripts/verify_e199_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

R = "research/experiments"
BR = (10, 11, 12, 13, 14)
SIGN = {("X", "A"): -1, ("X", "B"): 1, ("Y", "A"): 1, ("Y", "B"): -1}
PO = ["kc_%s_to_motor_%s" % (k, m) for k in "lr" for m in "lr"]
RC = re.compile(r"^RCEVAL cat=([AB]) set=(train|new) mod=([+-]?\d+\.\d+) .*?trials=(\d+) feat_on=(\d+) ctx=(\w+) ctx_on=(\d+)")
MLR = re.compile(r"motor L/R ([0-9.]+)/([0-9.]+)")


def ev(path, ctx):
    rows, push, kt, fw, sat, bad = {}, None, 0, 0, False, False
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = RC.match(ln)
        if m:
            rows[(m.group(1), m.group(2))] = float(m.group(3))
            if (int(m.group(4)), int(m.group(5)), m.group(6), int(m.group(7))) != (96, 96, ctx, 96):
                bad = True
            for a, b in MLR.findall(ln):
                if float(a) >= 0.66 and float(b) >= 0.66:
                    sat = True
        elif ln.startswith("=> DECOMP ") and "rceval" in ln:
            push = int(re.search(r"pushed=(\d+)", ln).group(1))
        elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
            kt += 1
        elif ln.startswith("[E184 특징 가중치 적재]"):
            fw += 1
    return rows, push, kt, fw, sat, bad


def train(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    g = {}
    g["catctx"] = re.search(r"^\[맥락 범주 과제\] catctx", t, re.M) is not None
    m = re.search(r"^\[맥락 범주 과제\] 시행 (\d+) 중 맥락 X (\d+) · Y (\d+) · 정답 같은 쪽 (\d+) · 교차 (\d+)", t, re.M)
    g["ctx"] = tuple(int(x) for x in m.groups()) if m else None
    m = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+) · A\(교차\) (\d+)", t, re.M)
    g["cat"] = tuple(int(x) for x in m.groups()) if m else None
    g["f"] = [(int(a), int(b)) for a, b in re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)]
    g["c"] = [(int(a), int(b), int(c)) for a, b, c in re.findall(r"^\[맥락 범주 점검\].*켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · 시행 끝 두 맥락 발화 (\d+)", t, re.M)]
    g["done"] = re.search(r"^\[학습\] ", t, re.M) is not None
    g["kt"] = len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M))
    g["hard"] = re.search(r"KC→motor \[E199 가산형\(경성 상한\)", t) is not None
    etas = re.findall(r"KC→motor \[E109 R-STDP 4방향\]: .*?eta=([0-9.]+)", t)
    g["eta"] = float(etas[-1]) if etas else None
    m = re.search(r"^\[E199 경쟁 정규화\] 적용 (\d+)회 .*정규화 뒤 최대 (\S+) · hard=(\d)", t, re.M)
    g["comp"] = (int(m.group(1)), float(m.group(2)), int(m.group(3))) if m else None
    return g


def sums_and_delta(wf, sf, e3):
    W, S, W3 = np.load(wf), np.load(sf), np.load(e3)
    rng_ok = all(W[p].min() >= 0 and W[p].max() <= 300 for p in PO)
    worst = 0.0
    dl = {}
    for kn in "lr":
        tot, cnt = {}, {}
        for m in "lr":
            nm = "kc_%s_to_motor_%s" % (kn, m)
            pre = S[nm + "__pre"].astype(np.int64)
            s_, c_ = np.zeros(3000), np.zeros(3000)
            np.add.at(s_, pre, W[nm].astype(np.float64))
            np.add.at(c_, pre, 1.0)
            tot[m], cnt[m] = s_, c_
        both = (cnt["l"] > 0) & (cnt["r"] > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ml, mr = tot["l"] / cnt["l"], tot["r"] / cnt["r"]
        worst = max(worst, float(np.max(np.abs(ml[both] + mr[both] - 300.0))))
        cross = "r" if kn == "l" else "l"
        d = np.where(both, (tot[cross] / np.maximum(cnt[cross], 1)) - (tot[kn] / np.maximum(cnt[kn], 1)), 0.0)
        dl[kn] = d
    differs = any(not np.array_equal(W[p], W3[p]) for p in PO)
    atb = sum(float(np.count_nonzero((W[p] == 0) | (W[p] == 300))) for p in PO) / sum(W[p].size for p in PO)
    return rng_ok, worst, differs, atb, dl


def main():
    try:
        eta = float(re.search(r"eta=([0-9.]+)", open(os.path.join(R, "logs/E199/calib_eta.txt"), encoding="utf-8").read()).group(1))
    except (OSError, AttributeError) as ex:
        print("독립 판정: 보류(결측 — 보정 %s)" % type(ex).__name__)
        return 0
    fail, eff = [], {}
    for b in BR:
        try:
            g = train(os.path.join(R, "logs/E199/train_b%d.log" % b))
            V = {(w, x): ev(os.path.join(R, "logs/E199/ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
            REF = {x: ev(os.path.join(R, "logs/E193/ev_b%d_none_%s.log" % (b, x)), x)[0] for x in "XY"}
            rng_ok, worst, differs, atb, dl = sums_and_delta(os.path.join(R, "traces/E199/w_b%d.npz" % b), os.path.join(R, "traces/E176/snap/conn_b%d.npz" % b),
                                                             os.path.join(R, "traces/E193/w_b%d.npz" % b))
        except (OSError, ValueError, KeyError, AttributeError) as ex:
            print("독립 판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        c, k = g["ctx"], g["cat"]
        if not (g["catctx"] and g["done"] and g["kt"] > 0 and c is not None and k is not None and c[0] == 3000 and c[1] + c[2] == 3000
                and c[3] + c[4] == 3000 and 1200 <= c[1] <= 1800 and 1200 <= c[3] <= 1800 and k[0] == 3000 and 1200 <= k[1] <= 1800):
            fail.append("b%d 학습" % b)
        if len(g["f"]) != 3 or any(a <= 0 or z != 0 for a, z in g["f"]) or len(g["c"]) != 3 or any(a <= 0 or o != 0 or z != 0 for a, o, z in g["c"]):
            fail.append("b%d 점검 줄" % b)
        if not (g["hard"] and g["comp"] is not None and g["comp"][0] == 3000 and g["comp"][1] <= 1e-3 and g["comp"][2] == 1 and g["eta"] is not None and abs(g["eta"] - eta) < 1e-9):
            fail.append("b%d 규칙" % b)
        if not (rng_ok and worst <= 0.01 and differs):
            fail.append("b%d 가중치" % b)
        val = {}
        for (w, x), (rows, push, kt, fw, sat, bad) in V.items():
            if len(rows) != 4 or bad or sat or kt == 0 or fw > 0 or push != (8 if w == "learn" else 0):
                fail.append("b%d %s %s 평가" % (b, w, x))
                continue
            for (cc, s), v in rows.items():
                val[(w, x, cc, s)] = v
        for x in "XY":
            for (cc, s), v in REF[x].items():
                if ("none", x, cc, s) in val and abs(val[("none", x, cc, s)] - v) > 0.0002 + 1e-9:
                    fail.append("b%d none 회귀" % b)
        if len(val) == 16:
            eff[b] = {(x, cc, s): SIGN[(x, cc)] * (val[("learn", x, cc, s)] - val[("none", x, cc, s)]) for (x, cc) in SIGN for s in ("train", "new")}
    if fail or len(eff) != 5:
        print("독립 판정: 보류(조작검증 실패 — %s)" % ", ".join(sorted(set(fail))))
        return 0
    cnt = {"conc": 0, "memo": 0, "part": 0, "fail": 0}
    for b in BR:
        a = eff[b]
        n = sum(1 for k in SIGN if a[k + ("train",)] >= 0.1 - 1e-9)
        gen = all(a[k + ("new",)] >= 0.05 - 1e-9 and 2 * a[k + ("new",)] >= a[k + ("train",)] - 1e-9 for k in SIGN)
        cnt["conc"] += n == 4 and gen
        cnt["memo"] += n == 4 and not gen
        cnt["part"] += n in (2, 3)
        cnt["fail"] += n <= 1
        print("b%d | %s 칸 %d" % (b, " ".join("%+.4f" % a[k + ("train",)] for k in (("X", "A"), ("X", "B"), ("Y", "A"), ("Y", "B"))), n))
    print("맥락 의존 범주 %d 사례 한정 %d 부분 %d 실패 %d" % (cnt["conc"], cnt["memo"], cnt["part"], cnt["fail"]))
    v = ("경쟁형 규칙으로 L4 학습(H120)" if cnt["conc"] >= 4 else "사례 한정" if cnt["memo"] >= 4 else "부분" if cnt["part"] >= 4
         else "획득 실패(H120-null)" if cnt["fail"] >= 4 else "보류")
    print("독립 판정: " + v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
