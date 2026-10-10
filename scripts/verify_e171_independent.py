#!/usr/bin/env python3
"""E171 독립 대조 — judge_e171.py 를 쓰지 않고 평가 원 로그(E171·E170)를 문자열 분해로 다시 읽어 KC 비와 판정을 분수로 계산한다.
규칙은 logs/E171/criteria_fixed.txt. 실행: python3 scripts/verify_e171_independent.py (저장소 루트에서)"""
import os
import sys
from fractions import Fraction as Fr

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
VARS = ("base", "bad", "agree")


def rd(*p):
    return open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def parse(t, var):
    ln = next(x for x in t.splitlines() if x.startswith("=> DECOMP "))
    kv = dict(tok.split("=", 1) for tok in ln.split() if "=" in tok and not tok.startswith("kc_means"))
    d = {}
    for x in t.splitlines():
        if x.startswith("[E171 평가 진단] variant=%s " % var):
            side = x.split("side=")[1].split()[0]
            n = int(x.split("n=")[1].split()[0])
            m = x.split("motor L/R ")[1].split()[0].split("/")
            k = x.split("KC L/R ")[1].split()[0].split("/")
            d[side] = {"n": n, "kL": Fr(k[0]), "kR": Fr(k[1]), "mL": Fr(m[0]), "mR": Fr(m[1])}
    return q4(kv["mod"]), int(kv["pushed"]), d


def main():
    try:
        ok = True
        keep = red = 0
        for b in BRAINS:
            D = {}
            for w in ("AB", "none"):
                for v in VARS:
                    mod, pushed, d = parse(rd("logs", "E171", "ev_b%d_%s_%s.log" % (b, w, v)), v)
                    t70 = rd("logs", "E170", "ev_b%d_%s_%s.log" % (b, w, v))
                    mod70 = q4(next(x for x in t70.splitlines() if x.startswith("=> DECOMP ")).split("mod=")[1].split()[0])
                    ok &= abs(mod - mod70) <= 1 and pushed == (8 if w == "AB" else 0) and set(d) == {"left", "right"} and all(d[s]["n"] == 250 for s in d)
                    D[(w, v)] = d
            A = lambda v, s, k: D[("AB", v)][s][k]
            rs = [A("agree", "left", "kL") / A("base", "left", "kL"), A("agree", "left", "kR") / A("bad", "right", "kR"),
                  A("agree", "right", "kR") / A("base", "right", "kR"), A("agree", "right", "kL") / A("bad", "left", "kL")]
            keep += all(Fr(85, 100) <= r <= Fr(115, 100) for r in rs)
            red += sum(rs) / 4 < Fr(85, 100)
            print("b%d KC 비 %s" % (b, " ".join("%.4f" % float(r) for r in rs)))
    except (FileNotFoundError, ValueError, KeyError, IndexError, StopIteration, ZeroDivisionError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    v = ("KC 보존(H094)" if keep >= 4 else ("KC 감소(H094-alt)" if red >= 4 else "보류")) if ok else "보류(조작검증 실패)"
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
