#!/usr/bin/env python3
"""E204 독립 대조 — judge_e204 와 따로 구현(로그 정규식 직접 해석·np.add.at KC 합·반복문 범주·판정). 같은 규칙(logs/E204/criteria_fixed.txt).
judge_e193·judge_e199·judge_e204·kc_comp 를 가져오지 않는다. 실행: python3 scripts/verify_e204_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BR = (10, 11, 12, 13, 14)
ARMS = ("G18", "GN", "UN")
GATE = {"G18": 1, "GN": 1, "UN": 0}
WSEED = {"G18": 18300, "GN": 28300, "UN": 28300}
TBASE = {"G18": 18500, "GN": 28500, "UN": 28500}
CBASE = {"G18": 18600, "GN": 28600, "UN": 28600}
SIGN = (("X", "A", -1), ("X", "B", 1), ("Y", "A", 1), ("Y", "B", -1))
P4 = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")

R_RC = re.compile(r"^RCEVAL cat=([AB]) set=(train|new) mod=([+-]?[0-9]+\.[0-9]+) .*?trials=(\d+) feat_on=(\d+) ctx=(\S+) ctx_on=(\d+)")
R_ML = re.compile(r"motor L/R ([0-9.]+)/([0-9.]+)")


def q(s):
    return int(round(float(s) * 10000))


def ev(path, ctx):
    d = {"rc": {}, "pushed": None, "kt": 0, "fw": 0, "sat": False, "bad": False, "world": [], "build": 0}
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = R_RC.match(ln)
        if m:
            d["rc"][(m.group(1), m.group(2))] = q(m.group(3))
            if int(m.group(4)) != 96 or int(m.group(5)) != 96 or m.group(6) != ctx or int(m.group(7)) != 96:
                d["bad"] = True
            for a, b in R_ML.findall(ln):
                if float(a) >= 0.66 and float(b) >= 0.66:
                    d["sat"] = True
            continue
        if ln.startswith("=> DECOMP ") and "rceval" in ln:
            mm = re.search(r"pushed=(\d+)", ln)
            d["pushed"] = int(mm.group(1)) if mm else None
        elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
            d["kt"] += 1
        elif ln.startswith("[E184 특징 가중치 적재]"):
            d["fw"] += 1
        elif ln.startswith("[E204 평가 세계] 세계 시드 "):
            d["world"].append(int(re.search(r"세계 시드 (\d+)", ln).group(1)))
        elif "[E204 맥락 게이팅 KC] 무리 " in ln:
            d["build"] += 1
    return d


def train(path):
    t = {"catctx": 0, "done": 0, "kt": 0, "ctx": None, "cat": None, "f": [], "c": [], "hard": 0, "eta": None, "comp": None,
         "da": None, "dachk": 0, "world": None, "trial": None, "cseed": None, "build": 0, "gchk": [], "gsum": None}
    for ln in open(path, encoding="utf-8", errors="replace"):
        s = ln.strip()
        if s.startswith("[맥락 범주 과제] catctx"):
            t["catctx"] = 1
            mm = re.search(r"지역 난수 (\d+)\)", s)
            if mm:
                t["cseed"] = int(mm.group(1))
        elif s.startswith("[맥락 범주 과제] 시행 "):
            t["ctx"] = tuple(int(x) for x in re.search(r"시행 (\d+) 중 맥락 X (\d+) · Y (\d+) · 정답 같은 쪽 (\d+) · 교차 (\d+)", s).groups())
        elif s.startswith("[범주 과제] 시행 "):
            t["cat"] = tuple(int(x) for x in re.search(r"시행 (\d+) 중 범주 B\(같은 쪽\) (\d+) · A\(교차\) (\d+)", s).groups())
        elif s.startswith("[범주 과제] catrule"):
            mm = re.search(r"세계 시드 (\d+)\).*배정 지역 난수 (\d+)", s)
            if mm:
                t["world"], t["trial"] = int(mm.group(1)), int(mm.group(2))
        elif s.startswith("[범주 과제 점검]"):
            t["f"].append(tuple(int(x) for x in re.search(r"특징 발화 (\d+) · 시행 끝 (\d+)", s).groups()))
        elif s.startswith("[맥락 범주 점검]"):
            t["c"].append(tuple(int(x) for x in re.search(r"켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · 시행 끝 두 맥락 발화 (\d+)", s).groups()))
        elif s.startswith("[학습] "):
            t["done"] = 1
        elif s.startswith("[E153 종류 입력 적재]") and "검증 일치" in s:
            t["kt"] += 1
        elif s.startswith("KC→motor [E199 가산형(경성 상한)"):
            t["hard"] = 1
        elif s.startswith("KC→motor [E109 R-STDP 4방향]"):
            mm = re.search(r"eta=([0-9.eE+-]+)", s)
            t["eta"] = float(mm.group(1)) if mm else None
        elif s.startswith("[E199 경쟁 정규화] 적용 "):
            mm = re.search(r"적용 (\d+)회 .*정규화 뒤 최대 ([0-9.eE+-]+) · hard=(\d)", s)
            t["comp"] = (int(mm.group(1)), float(mm.group(2)), int(mm.group(3))) if mm else None
        elif s.startswith("[E200 도파민 통계] "):
            mm = re.search(r"시행 (\d+) · 평균 DA ([+-]?[0-9.]+) · 평균 \|DA\| [0-9.]+ · 기준선 (\d)", s)
            t["da"] = (int(mm.group(1)), float(mm.group(2)), int(mm.group(3))) if mm else None
        elif s.startswith("[E200 도파민 기준선] 시행 "):
            t["dachk"] += 1
        elif "[E204 맥락 게이팅 KC] 무리 " in s:
            t["build"] += 1
        elif s.startswith("[E204 맥락 게이팅 점검] "):
            mm = re.search(r"활성 무리 (\d+) · 비활성 무리 (\d+)", s)
            t["gchk"].append((int(mm.group(1)), int(mm.group(2))) if mm else (-1, -1))
        elif s.startswith("[E204 맥락 게이팅 KC] 시행 "):
            mm = re.search(r"시행 (\d+) · .*활성 무리 합 (\d+) · 비활성 무리 합 (\d+) · 비\(비활성/활성\) ([0-9.]+)", s)
            t["gsum"] = (int(mm.group(1)), int(mm.group(2)), int(mm.group(3)), float(mm.group(4))) if mm else None
    return t


def weights(wf, sf, e2):
    W, S, W2 = np.load(wf), np.load(sf), np.load(e2)
    rng_ok = all(float(np.min(W[p])) >= 0.0 and float(np.max(W[p])) <= 300.0 for p in P4)
    dev = 0.0
    for kn in "lr":
        mean = []
        cnts = []
        for mn in "lr":
            p = "kc_%s_to_motor_%s" % (kn, mn)
            pre = S[p + "__pre"].astype(np.int64)
            sm = np.zeros(3000); ct = np.zeros(3000)
            np.add.at(sm, pre, np.asarray(W[p], dtype=np.float64)); np.add.at(ct, pre, 1.0)
            mean.append(sm / np.maximum(ct, 1)); cnts.append(ct)
        both = (cnts[0] > 0) & (cnts[1] > 0)
        if both.any():
            dev = max(dev, float(np.max(np.abs(mean[0][both] + mean[1][both] - 300.0))))
    differ = False
    for p in P4:
        if W[p].shape != W2[p].shape or np.any(W[p] != W2[p]):
            differ = True
    return rng_ok, dev, differ


def cellsof(E):
    out = {}
    for x, c, sg in SIGN:
        for st in ("train", "new"):
            out[(x, c, st)] = sg * (E[("learn", x)]["rc"][(c, st)] - E[("none", x)]["rc"][(c, st)])
    return out


def cls(a):
    n = 0
    for x, c, _ in SIGN:
        if a[(x, c, "train")] >= 1000:
            n += 1
    g = True
    for x, c, _ in SIGN:
        if not (a[(x, c, "new")] >= 500 and 2 * a[(x, c, "new")] >= a[(x, c, "train")]):
            g = False
    if n == 4:
        return "conc" if g else "memo"
    return "part" if n >= 2 else "fail"


def mweak(a):
    v = None
    for x, c, _ in SIGN:
        if v is None or a[(x, c, "train")] < v:
            v = a[(x, c, "train")]
    return v


def main():
    why = []
    A = {k: {} for k in ARMS + ("U18",)}
    for arm in ARMS:
        for b in BR:
            L = os.path.join(EXP, "logs", "E204")
            try:
                t = train(os.path.join(L, "train_%s_b%d.log" % (arm, b)))
                E = {(w, x): ev(os.path.join(L, "ev_%s_b%d_%s_%s.log" % (arm, b, w, x)), x) for w in ("learn", "none") for x in "XY"}
                rok, dev, differ = weights(os.path.join(EXP, "traces", "E204", "w_%s_b%d.npz" % (arm, b)),
                                           os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b), os.path.join(EXP, "traces", "E200", "w_b%d.npz" % b))
            except (OSError, ValueError, KeyError, AttributeError) as ex:
                print("독립 판정: 보류(결측 — %s b%d %s)" % (arm, b, type(ex).__name__))
                return 0
            k = "%s b%d" % (arm, b)
            c, g = t["ctx"], t["cat"]
            if not (t["catctx"] and t["done"] and t["kt"] >= 1 and c and g and c[0] == 3000 and c[1] + c[2] == 3000 and c[3] + c[4] == 3000
                    and 1200 <= c[1] <= 1800 and 1200 <= c[3] <= 1800 and g[0] == 3000 and 1200 <= g[1] <= 1800):
                why.append(k + " 학습 줄")
            if len(t["f"]) != 3 or any(x <= 0 or y != 0 for x, y in t["f"]):
                why.append(k + " 특징 점검")
            if len(t["c"]) != 3 or any(x <= 0 or y != 0 or z != 0 for x, y, z in t["c"]):
                why.append(k + " 맥락 점검")
            if not (t["hard"] and t["comp"] and t["comp"][0] == 3000 and t["comp"][1] <= 1e-3 and t["comp"][2] == 1 and t["eta"] is not None and abs(t["eta"] - 2.4) < 1e-9):
                why.append(k + " 규칙 줄")
            if not (t["da"] and t["da"][0] == 3000 and abs(t["da"][1]) <= 0.05 and t["da"][2] == 1 and t["dachk"] == 3):
                why.append(k + " 기준선 줄")
            if (t["world"], t["trial"], t["cseed"]) != (WSEED[arm], TBASE[arm] + b, CBASE[arm] + b):
                why.append(k + " 시드 줄")
            if GATE[arm]:
                gs = t["gsum"]
                if not (t["build"] >= 2 and len(t["gchk"]) == 3 and min(a_ for a_, _ in t["gchk"]) > 0 and gs and gs[0] == 3000 and gs[1] > 0 and gs[3] <= 0.01):
                    why.append(k + " 게이팅")
            elif t["build"] or t["gchk"] or t["gsum"]:
                why.append(k + " 게이팅 없는 팔에 게이팅 줄")
            if not (rok and dev <= 0.01 and differ):
                why.append(k + " 가중치")
            evok = True
            for (w, x), d in E.items():
                if len(d["rc"]) != 4 or d["bad"] or d["sat"] or d["kt"] < 1 or d["fw"] != 0 or d["pushed"] != (8 if w == "learn" else 0):
                    why.append("%s %s %s 평가" % (k, w, x)); evok = False
                if d["world"] != [WSEED[arm]]:
                    why.append("%s %s %s 평가 세계" % (k, w, x))
                if (GATE[arm] and d["build"] < 1) or (not GATE[arm] and d["build"] > 0):
                    why.append("%s %s %s 평가 게이팅 구성" % (k, w, x))
            if evok:
                A[arm][b] = cellsof(E)
    for b in BR:
        try:
            E = {(w, x): ev(os.path.join(EXP, "logs", "E200", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("learn", "none") for x in "XY"}
            A["U18"][b] = cellsof(E)
        except (OSError, ValueError, KeyError) as ex:
            print("독립 판정: 보류(결측 — E200 b%d %s)" % (b, type(ex).__name__))
            return 0
    if why or any(len(A[arm]) != 5 for arm in ARMS):
        print("독립 판정: 보류(조작검증 실패 — %s)" % "; ".join(why))
        return 0
    C = {arm: {b: cls(A[arm][b]) for b in BR} for arm in A}
    M = {arm: {b: mweak(A[arm][b]) for b in BR} for arm in A}
    g_conc = sum(1 for b in BR if C["G18"][b] == "conc")
    g_fail = sum(1 for b in BR if C["G18"][b] == "fail")
    n_conc = sum(1 for b in BR if C["GN"][b] == "conc")
    dev_g = sum(1 for b in BR if M["G18"][b] - M["U18"][b] >= 500)
    new_g = sum(1 for b in BR if M["GN"][b] - M["UN"][b] >= 500)
    for arm in ARMS + ("U18",):
        print("독립 %s: 범주 %s · 가장 약한 칸 %s" % (arm, " ".join("b%d %s" % (b, C[arm][b]) for b in BR), " ".join("%+d" % M[arm][b] for b in BR)))
    if g_fail >= 4:
        v = "H125-null"
    elif g_conc >= 4:
        if dev_g < 4:
            v = "게이팅 무관"
        elif n_conc >= 4:
            v = "H125" if new_g >= 4 else "보류(새 세계는 게이팅 없이도 섬)"
        else:
            v = "H125-세계 한정"
    else:
        v = "보류"
    print("독립 게이팅 효과: 개발 %d/5 · 새 세계 %d/5" % (dev_g, new_g))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
