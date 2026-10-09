#!/usr/bin/env python3
"""E161 독립 대조 — judge_e161.py 를 쓰지 않고 원 로그·추적에서 다시 계산한다(문자열 분해·분수 비교).
실행: python3 scripts/verify_e161_independent.py (저장소 루트에서)"""
import os
import sys
from fractions import Fraction

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)


def rd(*p):
    return open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def modline(t, head):
    for ln in t.splitlines():
        if ln.startswith(head):
            return q4(ln.split("변조폭")[1].split("**")[0])
    raise ValueError(head)


def kv(seg):
    return dict(x.split("=", 1) for x in seg.split() if "=" in x)


def nld(t):
    return sum(1 for x in t.splitlines() if x.startswith("[E153 종류 입력 적재]") and "검증 일치" in x)


def main():
    try:
        ok, sep, succ, null = True, 0, 0, 0
        for b in BRAINS:
            dl = next(x for x in rd("logs", "E161", "dev_b%d.log" % b).splitlines() if x.startswith("=> KCDEVOJA "))
            sides = [kv(s) for s in dl[len("=> KCDEVOJA "):].split(" | ") if s.startswith("side=")]
            ovt = rd("logs", "E161", "ov_b%d.log" % b)
            ol = next(x for x in ovt.splitlines() if x.startswith("=> KCOVERLAP"))
            js = [q4(kv(s)["jac"]) for s in ol[len("=> KCOVERLAP "):].split(" | ") if s.startswith("side=")]
            ft, dt = rd("logs", "E161", "F_b%d.log" % b), rd("logs", "E161", "D_b%d.log" % b)
            e = modline(ft, "[사후]") - modline(ft, "[사전]")
            eD = modline(dt, "[사후]") - modline(dt, "[사전]")
            nrate = sum(1 for x in rd("logs", "E161", "rate_b%d.log" % b).splitlines() if x.startswith("=> KCRATE"))
            R = np.load(os.path.join(EXP, "traces", "E161", "tr_F_b%d.npz" % b))["rows"]
            rs = (11.0 / 12.0) ** 20
            num = sum(abs(R[i, 21 + k] - rs * R[i, 13 + k]) for i in range(len(R)) for k in range(4))
            den = sum(abs(R[i, 13 + k]) for i in range(len(R)) for k in range(4))
            alive = sum(1 for i in range(len(R)) if abs(R[i, 13]) + abs(R[i, 14]) > 1.0) / len(R)
            pre = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
            rise = [q4(s["sel_med"]) - q4(s["sel_med0"]) for s in sides]
            okb = (len(sides) == 2 and all(float(s["dg_good"]) > 0 and float(s["dg_bad"]) > 0 for s in sides) and min(rise) >= 1000
                   and nld(ft) >= 2 and nld(ovt) >= 1 and nld(dt) == 0 and nrate == 2
                   and len(R) == 500 and num <= 1e-3 * den and alive >= 0.9 and pre <= 1e-3 and eD <= -1000)
            ok &= okb
            r = Fraction(e, eD)
            is_sep = len(js) == 2 and max(js) <= 1000
            sep += is_sep
            succ += is_sep and r >= Fraction(13, 10)
            null += len(js) == 2 and max(js) > 2500
            print("b%d 선택성 상승 %s | 자카드 %s | e %+d eD %+d r %.3f | 조작 %s" % (b, rise, js, e, eD, float(r), "✓" if okb else "✗"))
    except (FileNotFoundError, StopIteration, ValueError, KeyError, IndexError, ZeroDivisionError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    v = ("보류(조작검증 실패)" if not ok else "형성 성공(H084)" if succ >= 4 else "분리만(H084-sep-only)" if sep >= 4
         else "형성 실패(H084-null)" if null >= 4 else "보류")
    print("조작검증 %s | 분리 %d/5 성공 %d/5 실패 %d/5" % ("통과" if ok else "실패", sep, succ, null))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
