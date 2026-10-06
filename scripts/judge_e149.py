#!/usr/bin/env python3
"""E149 판정 — 기준 logs/E149/criteria_fixed.txt(실행 전 고정). 10줄(뇌 5 × 조건 2)이 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄: "  e149 b10 base: => KCOVERLAP side=l good=.. bad=.. jac=.. cos=.. jac025=.. jac100=.. split_jac=.. split_cos=.. | side=r ... | food_eye_scale=..."
"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
CONDS = ("base", "block")
TL = re.compile(r"^\s*e149 b(\d+) (base|block): => KCOVERLAP (.*)$")
SIDE = re.compile(r"side=([lr]) good=(\d+) bad=(\d+) jac=([-0-9.na]+) cos=([-0-9.na]+) jac025=([-0-9.na]+) jac100=([-0-9.na]+) split_jac=([-0-9.na]+) split_cos=([-0-9.na]+)")


def parse(rest):
    out = {}
    for m in SIDE.finditer(rest):
        out[m.group(1)] = {"good": int(m.group(2)), "bad": int(m.group(3)), "jac": float(m.group(4)), "cos": float(m.group(5)),
                           "j025": float(m.group(6)), "j100": float(m.group(7)), "split": float(m.group(8)), "scos": float(m.group(9))}
    return out if set(out) == {"l", "r"} else None


def avg(d, k):
    v = [d["l"][k], d["r"][k]]
    return sum(v) / 2.0


def judge(D):
    miss = [(b, c) for b in BRAINS for c in CONDS if (b, c) not in D or D[(b, c)] is None]
    if miss:
        return ["[측정 확인] 결측·파싱 실패 %s — **판정 보류, 수치 미출력**" % miss], None
    v1 = sum(avg(D[(b, "base")], "split") >= 0.5 - 1e-9 for b in BRAINS)
    v2 = sum((avg(D[(b, "base")], "good") + avg(D[(b, "base")], "bad")) > 0 for b in BRAINS)
    v3 = sum(avg(D[(b, "base")], "good") >= 5 and avg(D[(b, "base")], "bad") >= 5 for b in BRAINS)
    ok = v1 == v2 == v3 == 5
    checks = ["[측정 검증] V1 기본 반분 신뢰도 J_split ≥ 0.5 %d/5 · V2 스파이크 %d/5 · V3 반응 KC ≥ 5 %d/5 %s" % (v1, v2, v3, "통과" if ok else "실패")]
    Jb = {b: avg(D[(b, "base")], "jac") for b in BRAINS}
    Jk = {b: avg(D[(b, "block")], "jac") for b in BRAINS}
    over = [b for b in BRAINS if Jb[b] >= 0.5 - 1e-9]
    low = [b for b in BRAINS if Jb[b] <= 0.2 + 1e-9]
    sep = [b for b in BRAINS if Jk[b] <= 0.2 + 1e-9
           and avg(D[(b, "block")], "good") >= 0.2 * avg(D[(b, "base")], "good") - 1e-9
           and avg(D[(b, "block")], "bad") >= 0.2 * avg(D[(b, "base")], "bad") - 1e-9]
    if not ok:
        verdict = "보류(측정 검증 실패)"
    elif len(over) >= 4 and len(sep) >= 4:
        verdict = "겹침 확인·차단으로 분리(H072·H072-sep) — E150: 차단 상태로 간섭 하 유지 재시험"
    elif len(over) >= 4:
        verdict = "겹침 확인·차단으로 분리 안 됨(H072) — 좌우 KC 구조 자체"
    elif len(low) >= 4:
        verdict = "겹침 낮음 — 간섭은 표현 밖(출력 쪽)"
    else:
        verdict = "보류"
    return checks, {"Jb": Jb, "Jk": Jk, "over": over, "sep": sep, "low": low, "ok": ok, "verdict": verdict}


def report(checks, res, D=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        x, y = D[(b, "base")], D[(b, "block")]
        print("b%d 기본 J %.3f(l %.3f r %.3f) cos %.3f 반응 good %.0f bad %.0f 반분 %.3f | 차단 J %.3f cos %.3f 반응 good %.0f bad %.0f 반분 %.3f"
              % (b, res["Jb"][b], x["l"]["jac"], x["r"]["jac"], avg(x, "cos"), avg(x, "good"), avg(x, "bad"), avg(x, "split"),
                 res["Jk"][b], avg(y, "cos"), avg(y, "good"), avg(y, "bad"), avg(y, "split")))
    print("겹침(기본 J ≥ 0.5) %d/5 | 분리(차단 J ≤ 0.2 + 반응 유지) %d/5 | 겹침 낮음(기본 J ≤ 0.2) %d/5" % (len(res["over"]), len(res["sep"]), len(res["low"])))
    print("판정: %s" % res["verdict"])


def load():
    D = {}
    try:
        for ln in open(os.path.join(EXP, "E149.log"), encoding="utf-8", errors="replace"):
            m = TL.match(ln.rstrip("\n"))
            if m:
                D[(int(m.group(1)), m.group(2))] = parse(m.group(3))
    except FileNotFoundError:
        pass
    return D


if __name__ == "__main__":
    D = load()
    c, r = judge(D)
    report(c, r, D)
    sys.exit(0)
