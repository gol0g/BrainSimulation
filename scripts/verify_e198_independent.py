#!/usr/bin/env python3
"""E198 독립 대조 — judge_e198·judge_e196 을 쓰지 않고 다시 구현(정규식 파싱·np.add.at 누적·실수 산술). 규칙은 logs/E198/criteria_fixed.txt.
실행: python3 scripts/verify_e198_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

B_ = [10, 11, 12, 13, 14]
RX_EV = re.compile(r"^RCEVAL cat=([AB]) set=(train|new) mod=([-+]?\d+\.\d+) .*?trials=(\d+) feat_on=(\d+) ctx=([XY]) ctx_on=(\d+)(.*)$")
RX_MOT = re.compile(r"motor L/R ([0-9.]+)/([0-9.]+)")
XOR_SIGN = {("X", "A"): -1.0, ("X", "B"): 1.0, ("Y", "A"): 1.0, ("Y", "B"): -1.0}
ARMS = ("sgnR", "lsgnR", "lsgnA")


def load(path, ctx):
    d = {"m": {}, "push": None, "kt": False, "fw": False, "sat": False, "bad": False, "orc": None, "ver": False, "lr": None, "cn": None}
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
        elif ln.startswith("[E198 학습 부호]"):
            g = dict(re.findall(r"(가중치|agree_l|agree_r)=(\S+)", ln))
            d["lr"] = (g.get("가중치", "?").split("/")[-1], g.get("agree_l"), g.get("agree_r"))
        elif ln.startswith("[E198 연결 확인]"):
            d["cn"] = {nm: (int(a), int(b)) for nm, a, b in re.findall(r"(kc_[lr]_to_motor_[lr]) n=(\d+) pre_sum=(\d+)", ln)}
    return d


def learned(wf, sf):
    W, S = np.load(wf), np.load(sf)
    dl, conn = {}, {}
    for kn in ("l", "r"):
        oc = {"l": "r", "r": "l"}[kn]
        sums, cnts = {}, {}
        for m in (oc, kn):
            nm = "kc_%s_to_motor_%s" % (kn, m)
            pre = S[nm + "__pre"].astype(np.int64)
            w = W[nm].astype(np.float64)
            if len(pre) != len(w):
                raise ValueError(nm)
            s_, c_ = np.zeros(3000), np.zeros(3000)
            np.add.at(s_, pre, w)
            np.add.at(c_, pre, 1.0)
            sums[m], cnts[m] = s_, c_
            conn[nm] = (len(pre), int(pre.sum()))
        with np.errstate(invalid="ignore", divide="ignore"):
            d = sums[oc] / cnts[oc] - sums[kn] / cnts[kn]
        d[(cnts[oc] == 0) | (cnts[kn] == 0)] = 0.0
        dl[kn] = np.round(d, 9)
    return dl, conn


def want(npz, arm, dl):
    z = np.load(npz)
    o, ag = {}, {}
    for sd in ("l", "r"):
        sx = np.round(z["sxi_" + sd].astype(np.float64), 9)
        r = z["resp_" + sd].astype(bool)
        v = sx if arm == "sgnR" else dl[sd]
        tg = (v != 0) if arm == "lsgnA" else r
        o["cross_" + sd] = int(np.count_nonzero(tg & (v > 0)))
        o["same_" + sd] = int(np.count_nonzero(tg & (v < 0)))
        o["resp_" + sd] = int(np.count_nonzero(tg))
        if arm != "sgnR":
            ag[sd] = int(np.count_nonzero(tg & (v * sx > 0)))
            ag["n" + sd] = int(np.count_nonzero(tg & (v != 0) & (sx != 0)))
    return o, ag


def main():
    root = "research/experiments"
    bad, eff, rate = [], {}, {}
    for b in B_:
        npz = "%s/traces/E196/sxi_b%d.npz" % (root, b)
        wf = "%s/traces/E193/w_b%d.npz" % (root, b)
        sf = "%s/traces/E176/snap/conn_b%d.npz" % (root, b)
        try:
            dl, conn = learned(wf, sf)
        except (OSError, KeyError, ValueError) as ex:
            print("독립 판정: 보류(결측 b%d 가중치·스냅숏 %s)" % (b, type(ex).__name__))
            return 0
        logs = {}
        for w in ("none",) + ARMS:
            for x in ("X", "Y"):
                p = "%s/logs/E198/ev_b%d_%s_%s.log" % (root, b, w, x)
                if not os.path.exists(p):
                    print("독립 판정: 보류(결측 b%d %s %s)" % (b, w, x))
                    return 0
                logs[(w, x)] = load(p, x)
        for w in ("none", "sgnR"):
            for x in ("X", "Y"):
                p7 = "%s/logs/E197/ev_b%d_%s_%s.log" % (root, b, w, x)
                if not os.path.exists(p7):
                    print("독립 판정: 보류(결측 E197 b%d %s %s)" % (b, w, x))
                    return 0
                r7 = load(p7, x)["m"]
                for k, v in logs[(w, x)]["m"].items():
                    if k not in r7 or abs(round((v - r7[k]) * 10000)) > 2:
                        bad.append("b%d %s 회귀 %s %s" % (b, w, x, k))
        for (w, x), d in logs.items():
            if len(d["m"]) != 4 or d["bad"] or d["sat"] or not d["kt"] or d["fw"] or d["push"] != (0 if w == "none" else 4):
                bad.append("b%d %s %s 평가" % (b, w, x))
            if w == "none":
                continue
            o = d["orc"]
            if o is None or not d["ver"] or o.get("arm") != w or abs(float(o.get("theta", "9")) - 0.5) > 1e-9 or o.get("file", "").split("/")[-1] != npz.split("/")[-1]:
                bad.append("b%d %s %s 오라클 줄" % (b, w, x))
                continue
            wo, ag = want(npz, w, dl)
            if any(int(o.get(k, "-1")) != v for k, v in wo.items()):
                bad.append("b%d %s %s 오라클 수" % (b, w, x))
            if w != "sgnR":
                lr = d["lr"]
                if lr is None or lr[0] != wf.split("/")[-1] or lr[1] != str(ag["l"]) or lr[2] != str(ag["r"]):
                    bad.append("b%d %s %s 학습 부호 줄" % (b, w, x))
                if d["cn"] != conn:
                    bad.append("b%d %s %s 연결 확인" % (b, w, x))
                if w == "lsgnR":
                    rate[b] = (ag["l"] / max(ag["nl"], 1), ag["r"] / max(ag["nr"], 1))
        if not bad:
            eff[b] = {(w, x, c, s): logs[(w, x)]["m"][(c, s)] - logs[("none", x)]["m"][(c, s)]
                      for w in ARMS for x in ("X", "Y") for c in ("A", "B") for s in ("train", "new")}
            if max(abs(round(eff[b][("lsgnR", x, c, s)] * 10000)) for x in ("X", "Y") for c in ("A", "B") for s in ("train", "new")) <= 2:
                bad.append("b%d lsgnR 효과 없음(조작 무효 의심)" % b)
    if bad:
        print("독립 조작검증 실패: " + "; ".join(bad))
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    res = {w: [] for w in ARMS}
    for b in B_:
        out = "b%d 부호 일치율 %.3f·%.3f" % (b, rate[b][0], rate[b][1])
        for w in ("lsgnR", "lsgnA", "sgnR"):
            a = [XOR_SIGN[(x, c)] * eff[b][(w, x, c, "train")] for x, c in (("X", "A"), ("X", "B"), ("Y", "A"), ("Y", "B"))]
            n = sum(round(v * 10000) >= 1000 for v in a)
            res[w].append(n)
            out += " | %s %s 칸 %d" % (w, " ".join("%+.4f" % v for v in a), n)
        print(out)
    ra, rp, rf = sum(n == 4 for n in res["lsgnR"]), sum(2 <= n <= 3 for n in res["lsgnR"]), sum(n <= 1 for n in res["lsgnR"])
    if ra >= 4:
        v = "학습 방향 충분"
    elif rf >= 4:
        v = "학습 방향 부족"
    elif rp >= 4:
        v = "부분"
    else:
        v = "보류"
    print("lsgnR 획득 %d 부분 %d 실패 %d · lsgnA 획득 %d · sgnR 획득 %d" % (ra, rp, rf, sum(n == 4 for n in res["lsgnA"]), sum(n == 4 for n in res["sgnR"])))
    print("독립 판정: " + v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
