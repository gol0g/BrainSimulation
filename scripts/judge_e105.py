#!/usr/bin/env python3
"""E105 판정 — E105.md 4절. frozen 은 E099.log 재사용. 120런 전에는 수치 미출력. --selftest."""
import re
import sys

LINE = re.compile(r"^\s*(frozen|learn|shuf|reg) w(\d) t(\d+): => MINCIRC .*?reward=([0-9.]+) \*\*eval=([0-9.]+)\*\*")


def parse(path):
    r = {}
    try:
        for ln in open(path, encoding="utf-8"):
            m = LINE.match(ln)
            if m:
                c, s, t, rw, ev = m.groups(); r[(c, int(s), int(t))] = (float(rw), float(ev))
    except FileNotFoundError:
        pass
    return r


def need():
    e = set()
    for s in range(5, 10):
        for t in range(200, 208):
            e |= {("learn", s, t), ("shuf", s, t)}
    for s in (1, 2):
        for t in range(200, 208):
            e.add(("learn", s, t))
    for s in (0, 3, 4):
        for t in range(100, 108):
            e.add(("reg", s, t))
    return e


def decide(nl, nsh, nfr, nreg, ndrop):
    a = "지지(일반화 유지)" if nl >= 32 else ("기각(일반화 손실)" if nl <= 16 else "보류")
    b = ("성립" if (nsh <= 8 and nfr <= 4) else "성립 안 함") if a.startswith("지지") else "해당 없음"
    c = "유지" if nreg >= 22 else "손실"
    d = "파괴" if ndrop >= 4 else "파괴 관측 없음"
    adopt = a.startswith("지지") and b == "성립" and c == "유지" and d == "파괴 관측 없음"
    return a, b, c, d, adopt


def selftest():
    cases = [((32, 8, 4, 22, 3), True), ((31, 0, 0, 24, 0), False), ((40, 9, 4, 24, 0), False),
             ((40, 0, 0, 21, 0), False), ((40, 0, 0, 24, 4), False), ((16, 0, 0, 24, 0), False)]
    ok = sum(decide(*a)[4] == e for a, e in cases)
    ok += decide(16, 0, 0, 24, 0)[0].startswith("기각")
    print("자체 검증 %d/%d 통과" % (ok, len(cases) + 1))
    return ok == len(cases) + 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    r = parse("research/experiments/E105.log"); f = parse("research/experiments/E099.log"); w20 = f
    have = len(need() & set(r))
    if have < 120:
        print("[E105] %d/120런 완료 — **판정 보류. 결과가 다 모일 때까지 수치를 출력하지 않는다.**" % have); sys.exit(0)
    ok = lambda e: e >= 90.0
    L = [r[("learn", s, t)][1] for s in range(5, 10) for t in range(200, 208)]
    SH = [r[("shuf", s, t)][1] for s in range(5, 10) for t in range(200, 208)]
    FR = [f[("frozen", s, t)][1] for s in range(5, 10) for t in range(200, 204)]
    REG = [r[("reg", s, t)][1] for s in (0, 3, 4) for t in range(100, 108)]
    bad = [(s, t) for s in range(5, 10) for t in range(200, 208) if r[("shuf", s, t)][0] != r[("learn", s, t)][0]]
    print("조작검증: shuffled 보상 총량 = learn %d/40 일치%s" % (40 - len(bad), "" if not bad else "  ← 불일치 %s" % bad))
    chg = sum(r[("learn", s, t)] != w20[("learn", s, t)] for s in range(5, 10) for t in range(200, 208))
    print("조작검증: learn 요약이 E099(w_max 20)와 다른 런 %d/40%s" % (chg, "" if chg > 0 else "  ← w_max 조작 무효 의심"))
    drops = []
    for s in (1, 2):
        base = sum(f[("frozen", s, t)][1] for t in range(200, 204)) / 4
        drops += [(s, t) for t in range(200, 208) if r[("learn", s, t)][1] < base - 10]
    for s in range(5, 10):
        print("  배선 %d: learn %s | shuf %s | (E099 w20 learn %s)" % (s, " ".join("%.0f" % r[("learn", s, t)][1] for t in range(200, 208)),
              " ".join("%.0f" % r[("shuf", s, t)][1] for t in range(200, 208)), " ".join("%.0f" % w20[("learn", s, t)][1] for t in range(200, 208))))
    print("  배선 1·2 learn: %s" % " ".join("%.0f" % r[("learn", s, t)][1] for s in (1, 2) for t in range(200, 208)))
    print("  탐색 표본 회귀: %s" % " ".join("%.0f" % x for x in REG))
    n = lambda xs: sum(ok(x) for x in xs)
    a, b, c, d, adopt = decide(n(L), n(SH), n(FR), n(REG), len(drops))
    print("(a) 일반화: %s — learn %d/40 | (b) 귀속: %s — shuffled %d/40, frozen(E099) %d/20 | (c) 탐색 표본: %s %d/24 | (d) %s %d/16" % (
        a, n(L), b, n(SH), n(FR), c, n(REG), d, len(drops)))
    print("종합: %s" % ("w_max 2 기본값 채택" if adopt else "채택 보류"))
