#!/usr/bin/env python3
"""E196 독립 대조 — judge_e196.py 를 쓰지 않고 다시 구현(정규식 파싱·실수 산술·반올림 비교). 규칙은 logs/E196/criteria_fixed.txt.
실행: python3 scripts/verify_e196_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

B_ = [10, 11, 12, 13, 14]
RX_EV = re.compile(r"^RCEVAL cat=([AB]) set=(train|new) mod=([-+]?\d+\.\d+) .*?trials=(\d+) feat_on=(\d+) ctx=([XY]) ctx_on=(\d+)(.*)$")
RX_MOT = re.compile(r"motor L/R ([0-9.]+)/([0-9.]+)")
XOR_SIGN = {("X", "A"): -1.0, ("X", "B"): 1.0, ("Y", "A"): 1.0, ("Y", "B"): -1.0}
CAT_SIGN = {"A": -1.0, "B": 1.0}


def load(path, ctx):
    d = {"m": {}, "push": None, "kt": False, "fw": False, "sat": False, "bad": False, "orc": None, "ver": False}
    for ln in open(path, encoding="utf-8", errors="replace").read().splitlines():
        mt = RX_EV.match(ln)
        if mt:
            c, s, mod, tr, fo, cx, co, rest = mt.groups()
            d["m"][(c, s)] = float(mod)
            if not (int(tr) == 96 and int(fo) == 96 and cx == ctx and int(co) == 96):
                d["bad"] = True
            for a, b in RX_MOT.findall(rest):
                if float(a) >= 0.66 and float(b) >= 0.66:
                    d["sat"] = True
        elif ln.startswith("RCEVAL "):
            d["bad"] = True
        elif ln.startswith("=> DECOMP") and "rceval" in ln:
            d["push"] = int(re.search(r"pushed=(\d+)", ln).group(1))
        elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
            d["kt"] = True
        elif ln.startswith("[E184 특징 가중치 적재]"):
            d["fw"] = True
        elif ln.startswith("[E196 오라클] "):
            d["orc"] = dict(re.findall(r"(\w+)=(\S+)", ln))
        elif ln.startswith("[E196 오라클 장치 검증] 일치"):
            d["ver"] = True
    return d


def counts(npz, arm):
    z = np.load(npz)
    o = {}
    for sd in ("l", "r"):
        key = ("csi_" if arm == "cat" else "sxi_") + sd
        v = np.round(z[key].astype(np.float64), 9)
        r = z["resp_" + sd].astype(bool)
        o["cross_" + sd], o["same_" + sd], o["resp_" + sd] = int((r & (v >= 0.5)).sum()), int((r & (v <= -0.5)).sum()), int(r.sum())
    return o


def main():
    root = "research/experiments"
    bad = []
    eff = {}
    for b in B_:
        npz = "%s/traces/E196/sxi_b%d.npz" % (root, b)
        logs = {}
        for w in ("none", "cat", "sel", "eq"):
            for x in ("X", "Y"):
                p = "%s/logs/E196/ev_b%d_%s_%s.log" % (root, b, w, x)
                if not os.path.exists(p):
                    print("독립 판정: 보류(결측 b%d %s %s)" % (b, w, x))
                    return 0
                logs[(w, x)] = load(p, x)
        for x in ("X", "Y"):
            p3 = "%s/logs/E193/ev_b%d_none_%s.log" % (root, b, x)
            if not os.path.exists(p3):
                print("독립 판정: 보류(결측 E193 b%d %s)" % (b, x))
                return 0
            r3 = load(p3, x)["m"]
            for k, v in logs[("none", x)]["m"].items():
                if k not in r3 or abs(round((v - r3[k]) * 10000)) > 2:
                    bad.append("b%d none 회귀 %s %s" % (b, x, k))
        for (w, x), d in logs.items():
            if len(d["m"]) != 4 or d["bad"] or d["sat"] or not d["kt"] or d["fw"] or d["push"] != (0 if w == "none" else 4):
                bad.append("b%d %s %s 평가" % (b, w, x))
            if w != "none":
                o = d["orc"]
                if o is None or not d["ver"] or o.get("arm") != w or abs(float(o.get("theta", "9")) - 0.5) > 1e-9 or o.get("file", "").split("/")[-1] != npz.split("/")[-1]:
                    bad.append("b%d %s %s 오라클 줄" % (b, w, x))
                else:
                    try:
                        cn = counts(npz, w)
                    except (OSError, KeyError):
                        bad.append("b%d 지수 파일" % b)
                        continue
                    if any(int(o.get(k, "-1")) != v for k, v in cn.items()):
                        bad.append("b%d %s %s 오라클 수" % (b, w, x))
        if not bad:
            eff[b] = {(w, x, c, s): logs[(w, x)]["m"][(c, s)] - logs[("none", x)]["m"][(c, s)]
                      for w in ("cat", "sel", "eq") for x in ("X", "Y") for c in ("A", "B") for s in ("train", "new")}
    if bad:
        print("독립 조작검증 실패: " + "; ".join(bad))
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    pc = 0
    res = {"sel": [], "eq": []}
    for b in B_:
        e = eff[b]
        catv = [CAT_SIGN[c] * e[("cat", x, c, "train")] for x in ("X", "Y") for c in ("A", "B")]
        pc += all(round(v * 10000) >= 1000 for v in catv)
        out = "b%d 양성 대조 %s" % (b, " ".join("%+.4f" % v for v in catv))
        for w in ("sel", "eq"):
            a = [XOR_SIGN[(x, c)] * e[(w, x, c, "train")] for x, c in (("X", "A"), ("X", "B"), ("Y", "A"), ("Y", "B"))]
            n = sum(round(v * 10000) >= 1000 for v in a)
            res[w].append(n)
            out += " | %s %s 칸 %d" % (w, " ".join("%+.4f" % v for v in a), n)
        print(out)
    sa, sp, sf = sum(n == 4 for n in res["sel"]), sum(2 <= n <= 3 for n in res["sel"]), sum(n <= 1 for n in res["sel"])
    ea, ef = sum(n == 4 for n in res["eq"]), sum(n <= 1 for n in res["eq"])
    if pc < 4:
        v = "보류(양성 대조 실패)"
    elif sa >= 4:
        v = "표현 충분·평형 충분" if ea >= 4 else ("표현 충분·상관 평형 부족" if ef >= 4 else "표현 충분·평형 보류")
    elif sf >= 4:
        v = "표현·출력 병목"
    elif sp >= 4:
        v = "부분"
    else:
        v = "보류"
    print("양성 대조 %d/5 · sel 획득 %d 부분 %d 실패 %d · eq 획득 %d 실패 %d" % (pc, sa, sp, sf, ea, ef))
    print("독립 판정: " + v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
