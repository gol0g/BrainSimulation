#!/usr/bin/env python3
"""E107 판정 — E107.md 4절. 72런 + 회귀 2종 전에는 수치 미출력. --selftest."""
import re
import sys

LINE = re.compile(r"^\s*(same|cross|frz) w(\d) t(\d+): => MINCIRC .*?first=([0-9.]+) .*?\*\*eval=([0-9.]+)\*\* .*?evalCD=([0-9.]+)")
L106 = re.compile(r"^\s*learn w(\d) t(\d+): => MINCIRC .*?first=([0-9.]+) .*?\*\*eval=([0-9.]+)\*\*")
W, TS = [5, 6, 7, 8], list(range(400, 408))


def cond(runs):
    learned = [r for r in runs if r["cd"] >= 90]
    nl = len(learned)
    if nl < 16:
        return nl, float("nan"), "판정 불가(H035-nolearn)"
    frac = sum(r["ab"] >= 90 for r in learned) / nl
    return nl, frac, ("유지" if frac >= 0.875 else ("망각" if frac <= 0.5 else "보류"))


def overall(js, jc):
    if js == "유지" and jc == "유지":
        return "H035 지지"
    if jc == "망각":
        return "H035-forget"
    return "보류"


def selftest():
    mk = lambda ncd, nab: [{"cd": 100 if i < ncd else 50, "ab": 100 if i < nab else 50} for i in range(32)]
    cs = [(cond(mk(32, 28))[2], "유지"), (cond(mk(32, 27))[2], "보류"), (cond(mk(32, 16))[2], "망각"), (cond(mk(15, 15))[2], "판정 불가"),
          (overall("유지", "유지"), "H035 지지"), (overall("유지", "망각"), "H035-forget"), (overall("유지", "보류"), "보류")]
    ok = sum(a.startswith(b) for a, b in cs)
    print("자체 검증 %d/%d 통과" % (ok, len(cs))); return ok == len(cs)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    D = {}
    try:
        txt = open("research/experiments/E107.log", encoding="utf-8").read()
    except FileNotFoundError:
        txt = ""
    for ln in txt.splitlines():
        m = LINE.match(ln)
        if m:
            c, s, t, f, ab, cd = m.groups(); D[(c, int(s), int(t))] = {"first": float(f), "ab": float(ab), "cd": float(cd)}
    need = {(c, s, t) for c in ("same", "cross") for s in W for t in TS} | {("frz", s, t) for s in W for t in (400, 401)}
    regs = ("K38 회귀 통과" in txt, "K50 회귀 통과" in txt)
    if len(need & set(D)) < len(need) or not all(regs):
        print("[E107] %d/%d런, 회귀 K38 %s K50 %s — **판정 보류. 다 모일 때까지 수치 미출력.**" % (len(need & set(D)), len(need), regs[0], regs[1])); sys.exit(0)
    E = {}
    for ln in open("research/experiments/E106.log", encoding="utf-8"):
        m = L106.match(ln)
        if m:
            s, t, f, ev = m.groups(); E[(int(s), int(t))] = (float(f), float(ev))
    print("[E107] %d런 완료, 회귀 K38·K50 통과" % len(need))
    for c in ("same", "cross"):
        tw = sum(D[(c, s, t)]["first"] == E[(s, t)][0] for s in W for t in TS)
        print("조작검증 %s: 1단계 쌍둥이(E106 learn first) 동일 %d/32%s" % (c, tw, "" if tw == 32 else "  ← C/D 추가가 1단계를 바꿈 — 해석 보류"))
    print("C/D 선천(frz): %s" % " ".join("w%d:%.0f" % (s, D[("frz", s, t)]["cd"]) for s in W for t in (400, 401)))
    J = {}
    for c in ("same", "cross"):
        runs = [D[(c, s, t)] for s in W for t in TS]
        for s in W:
            print("  %s 배선 %d: A/B %s | C/D %s | (E106 1단계 직후 A/B %s)" % (c, s, " ".join("%.0f" % D[(c, s, t)]["ab"] for t in TS),
                  " ".join("%.0f" % D[(c, s, t)]["cd"] for t in TS), " ".join("%.0f" % E[(s, t)][1] for t in TS)))
        nl, frac, v = cond(runs); J[c] = v
        drop = sum(D[(c, s, t)]["ab"] < E[(s, t)][1] for s in W for t in TS)
        print("%s: (0) C/D 학습 %d/32 | (a) 유지 비율 %.3f → %s | 1단계 직후 대비 A/B 하락 런 %d/32" % (c, nl, frac, v, drop))
    print("종합: %s" % overall(J["same"], J["cross"]))
