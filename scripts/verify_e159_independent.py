#!/usr/bin/env python3
"""E159 독립 대조 — judge_e159.py 를 쓰지 않고 원 로그에서 다시 계산한다(문자열 분해·분수 비교).
재현 기준값은 E157·E158 원 로그(logs/E157/r25_Fk_b*, learn_Fk_b*, logs/E158/F500_b*)에서 직접 읽는다(판정 코드의 상수를 쓰지 않는다).
실행: python3 scripts/verify_e159_independent.py (저장소 루트에서)"""
import os
import sys
from fractions import Fraction

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
CELLS = ("none_R0", "none_R25", "W0_R0", "W0_R25", "W25_R0", "W25_R25")


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def rd(*p):
    return open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()


def mod_line(t, head):
    for ln in t.splitlines():
        if ln.startswith(head):
            return q4(ln.split("변조폭")[1].split("**")[0])
    raise ValueError("줄 없음 %s" % head)


def cell(c, b):
    t = rd("logs", "E159", "%s_b%d.log" % (c, b))
    for ln in t.splitlines():
        if ln.startswith("=> DECOMP "):
            tok = dict(x.split("=", 1) for x in ln.split()[2:6] if "=" in x)
            pushed = int(ln.split("pushed=")[1].split()[0])
            nld = sum(1 for x in t.splitlines() if x.startswith("[E153 종류 입력 적재]") and "검증 일치" in x)
            nsc = sum(1 for x in t.splitlines() if x.startswith("[E157 종류 입력 배율]") and "검증 일치" in x)
            return q4(tok["mod"]), pushed, nld, nsc
    raise ValueError("DECOMP 없음")


def main():
    try:
        V = {(c, b): cell(c, b) for c in CELLS for b in BRAINS}
        ref = {b: (mod_line(rd("logs", "E157", "r25_Fk_b%d.log" % b), "[사전]"), mod_line(rd("logs", "E157", "learn_Fk_b%d.log" % b), "[사전]"),
                   mod_line(rd("logs", "E157", "learn_Fk_b%d.log" % b), "[사후]"), mod_line(rd("logs", "E158", "F500_b%d.log" % b), "[사후]"))
               for b in BRAINS}
    except (FileNotFoundError, ValueError, KeyError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    ok = True
    cats = {}
    for b in BRAINS:
        v = {c: V[(c, b)][0] for c in CELLS}
        okb = (abs(v["none_R25"] - ref[b][0]) <= 20 and abs(v["none_R0"] - ref[b][1]) <= 20 and abs(v["W0_R0"] - ref[b][2]) <= 20
               and abs(v["W25_R25"] - ref[b][3]) <= 20
               and all(V[(c, b)][1] == (0 if c.startswith("none") else 8) and V[(c, b)][2] >= 2 and V[(c, b)][3] >= 2 for c in CELLS))
        e00 = v["W0_R0"] - v["none_R0"]
        e025 = v["W0_R25"] - v["none_R25"]
        e250 = v["W25_R0"] - v["none_R0"]
        okb = okb and e00 <= -1000
        ok &= okb
        O = Fraction(e025, e00); C = Fraction(e250, e00)
        lim = Fraction(7, 10)
        c_ = ("출력(H082)" if O <= lim < C else "내용(H082-content)" if C <= lim < O else "둘 다(H082-both)" if (O <= lim and C <= lim) else "둘 다 아님(H082-neither)")
        cats[b] = c_
        print("b%d E00 %+d E0_25 %+d E25_0 %+d | O %.3f C %.3f | 재현·이식 %s | %s" % (b, e00, e025, e250, float(O), float(C), "✓" if okb else "✗", c_))
    top = [c for c in set(cats.values()) if sum(x == c for x in cats.values()) >= 4]
    v = "보류(조작검증 실패)" if not ok else (top[0] if top else "보류")
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
