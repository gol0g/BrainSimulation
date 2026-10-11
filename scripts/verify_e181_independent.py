#!/usr/bin/env python3
"""E181 독립 대조 — judge_e181.py 를 쓰지 않고 원 로그(logs/E181/{칸}_b{B}.log)를 문자열 분해로 다시 읽어 분수로 판정한다.
E159 대조를 옮기고 재현 기준값(E157·E158) 검사 자리에 동작점(−4000)·진단·비포화·반사 발현 검사를 따로 구현했다. 규칙은 logs/E181/criteria_fixed.txt.
실행: python3 scripts/verify_e181_independent.py (저장소 루트에서)"""
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


def cell(c, b):
    t = rd("logs", "E181", "%s_b%d.log" % (c, b))
    lines = t.splitlines()
    for ln in lines:
        if ln.startswith("=> DECOMP "):
            mod = q4(ln.split("mod=")[1].split()[0])
            pushed = int(ln.split("pushed=")[1].split()[0])
            nld = sum(1 for x in lines if x.startswith("[E153 종류 입력 적재]") and "검증 일치" in x)
            nsc = sum(1 for x in lines if x.startswith("[E157 종류 입력 배율]") and "검증 일치" in x)
            tv = [x.split("기저값 ")[1].split("(")[0] for x in lines if x.strip().startswith("[motor 동작점]")]
            ton = len(tv) >= 1 and all(q4(x) == -4000 * 10000 for x in tv)
            d = {}
            for x in lines:
                if x.startswith("[E171 평가 진단]"):
                    side = x.split("side=")[1].split()[0]
                    a, bb = x.split("motor L/R ")[1].split()[0].split("/")
                    d[side] = [q4(a), q4(bb)]
            m4 = (d["left"] + d["right"]) if set(d) == {"left", "right"} else None
            unsat = m4 is not None and not all(x >= 6600 for x in m4)
            return mod, pushed, nld, nsc, ton, unsat
    raise ValueError("DECOMP 없음")


def main():
    try:
        V = {(c, b): cell(c, b) for c in CELLS for b in BRAINS}
    except (FileNotFoundError, ValueError, KeyError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    ok = True
    rec = loss = l2 = 0
    for b in BRAINS:
        v = {c: V[(c, b)][0] for c in CELLS}
        okb = all(V[(c, b)][1] == (0 if c.startswith("none") else 8) and V[(c, b)][2] >= 2 and V[(c, b)][3] >= 2 and V[(c, b)][4] and V[(c, b)][5]
                  for c in CELLS)
        e00 = v["W0_R0"] - v["none_R0"]
        e025 = v["W0_R25"] - v["none_R25"]
        okb = okb and e00 <= -1000 and (v["none_R25"] - v["none_R0"]) >= 1000
        ok &= okb
        O = Fraction(e025, e00)
        rec += O >= Fraction(85, 100)
        loss += O <= Fraction(7, 10)
        l2 += v["W25_R25"] <= -200
        print("b%d E00 %+d E0_25 %+d | O %.4f | v W25·R25 %+d | 반사 발현 %+d | 조작 %s" % (b, e00, e025, float(O), v["W25_R25"], v["none_R25"] - v["none_R0"], "✓" if okb else "✗"))
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    else:
        v1 = "회복(H102)" if rec >= 4 else ("손실 유지(H102-null)" if loss >= 4 else "부분")
        v2 = "반사 역전" if l2 >= 4 else "역전 없음"
    print("조작검증 %s | O ≥ 0.85 %d/5 · O ≤ 0.70 %d/5 · 역전 %d/5" % ("통과" if ok else "실패", rec, loss, l2))
    print("독립 판정: %s / %s" % (v1, v2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
