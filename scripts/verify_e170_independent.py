#!/usr/bin/env python3
"""E170 독립 대조 — judge_e170.py 를 쓰지 않고 평가 원 로그(E170)·E162.log 를 문자열 분해로 다시 읽어 판정 1·2 를 분수로 계산한다.
규칙은 logs/E170/criteria_fixed.txt. 실행: python3 scripts/verify_e170_independent.py (저장소 루트에서)"""
import os
import sys
from fractions import Fraction as Fr

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
VARS = ("base", "bad", "agree", "conflict")
DES = {"agree": {"left": "0.90/0.00 0.00/0.90 0.90/0.90", "right": "0.00/0.90 0.90/0.00 0.90/0.90"},
       "conflict": {"left": "0.90/0.00 0.90/0.00 0.90/0.00", "right": "0.00/0.90 0.00/0.90 0.00/0.90"}}


def rd(*p):
    return open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def decomp(t, var):
    if not any(ln.startswith("[E146 변형] variant=%s " % var) for ln in t.splitlines()):
        raise ValueError("variant")
    ln = next(x for x in t.splitlines() if x.startswith("=> DECOMP "))
    kv = dict(tok.split("=", 1) for tok in ln.split() if "=" in tok and not tok.startswith("kc_means"))
    return q4(kv["mod"]), int(kv["pushed"])


def stim_ok(t, var):
    got = {}
    for ln in t.splitlines():
        if ln.startswith("[E170 자극] variant=%s " % var):
            side = ln.split("side=")[1].split()[0]
            vals = [ln.split(k)[1].split()[0] for k in ("good L/R ", "bad L/R ", "food L/R ")]
            got[side] = " ".join(vals)
    return got == DES[var]


def main():
    try:
        e162 = {}
        for ln in rd("E162.log").splitlines():
            p = ln.split()
            if len(p) >= 6 and p[0] == "e162" and p[2] in ("AB", "none") and p[3].rstrip(":") in ("base", "bad"):
                e162[(int(p[1][1:]), p[2], p[3].rstrip(":"))] = q4(p[-1])
        ok = True
        succ = none = add2 = 0
        for b in BRAINS:
            m = {}
            for w in ("AB", "none"):
                for v in VARS:
                    t = rd("logs", "E170", "ev_b%d_%s_%s.log" % (b, w, v))
                    mod, pushed = decomp(t, v)
                    m[(w, v)] = mod
                    ok &= pushed == (8 if w == "AB" else 0)
                    if v in DES:
                        ok &= stim_ok(t, v)
                    else:
                        ok &= abs(mod - e162[(b, w, v)]) <= 1
            e = {v: m[("AB", v)] - m[("none", v)] for v in VARS}
            ok &= e["base"] <= -1000 and e["bad"] >= 1000
            rho = Fr(e["agree"], e["base"] - e["bad"])
            succ += rho >= Fr(4, 5)
            none += abs(e["agree"]) * 10 < 11 * max(abs(e["base"]), abs(e["bad"]))
            add2 += Fr(abs(e["conflict"] - (e["base"] + e["bad"])), abs(e["base"]) + abs(e["bad"])) <= Fr(3, 20)
            print("b%d ρ %.4f e agree %+d conflict %+d" % (b, float(rho), e["agree"], e["conflict"]))
    except (FileNotFoundError, ValueError, KeyError, IndexError, StopIteration, ZeroDivisionError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    else:
        v1 = "합성 성공(H093)" if succ >= 4 else ("합성 없음(H093-null)" if none >= 4 else "부분")
        v2 = "가산 상쇄" if add2 >= 4 else "비가산"
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s / %s" % (v1, v2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
