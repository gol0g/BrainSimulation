#!/usr/bin/env python3
"""E169 독립 대조 — judge_e169.py 를 쓰지 않고 원 로그(E169 FW1·FFW1, E161 F, E166 FNF)·추적을 문자열 분해로 다시 읽어 판정 1·2 를 분수로 계산한다.
규칙은 logs/E169/criteria_fixed.txt. 실행: python3 scripts/verify_e169_independent.py (저장소 루트에서)"""
import os
import sys
from fractions import Fraction as Fr

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


def val(t, head):
    for ln in t.splitlines():
        if ln.startswith(head):
            return q4(ln.split("변조폭")[1].split("**")[0])
    raise ValueError(head)


def main():
    try:
        ok = True
        succ = fail = red = none = 0
        pre2 = True
        for b in BRAINS:
            t = {"FW1": rd("logs", "E169", "FW1_b%d.log" % b), "FFW1": rd("logs", "E169", "FFW1_b%d.log" % b),
                 "F": rd("logs", "E161", "F_b%d.log" % b), "FNF": rd("logs", "E166", "FNF_b%d.log" % b)}
            e = {k: val(v, "[사후]") - val(v, "[사전]") for k, v in t.items()}
            okb = e["F"] <= -1000 and all(abs(val(t[a], "[사전]") - val(t["F"], "[사전]")) <= 20 for a in ("FW1", "FFW1"))
            okb &= all(sum(ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln for ln in t[a].splitlines()) >= 2 for a in ("FW1", "FFW1"))
            r1 = (11.0 / 12.0) ** 10
            for a, cond in (("FFW1", lambda x: x <= 1e-3), ("FW1", lambda x: x >= 0.05)):
                R = np.load(os.path.join(EXP, "traces", "E169", "tr_%s_b%d.npz" % (a, b)))["rows"]
                num = sum(abs(R[i, 21 + k] - r1 * R[i, 13 + k]) for i in range(len(R)) for k in range(4))
                den = sum(abs(R[i, 13 + k]) for i in range(len(R)) for k in range(4))
                pre = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
                okb &= len(R) == 500 and pre <= 1e-3 and cond(num / max(den, 1e-12))
            ok &= okb
            q1 = Fr(e["FW1"], e["F"])
            succ += q1 >= Fr(4, 5); fail += q1 <= Fr(1, 2)
            if e["FFW1"] <= -1000:
                d = Fr(e["FW1"], e["FFW1"]) - Fr(e["FNF"], e["F"])
                red += d >= Fr(1, 5); none += abs(d) < Fr(1, 10)
            else:
                pre2 = False
            print("b%d q₁ %.4f FFW1/F %.4f | 조작 %s" % (b, float(q1), e["FFW1"] / e["F"], "✓" if okb else "✗"))
    except (FileNotFoundError, ValueError, IndexError, ZeroDivisionError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    else:
        v1 = "대체 성공(H092)" if succ >= 4 else ("실패(H092-null)" if fail >= 4 else "부분")
        v2 = ("창 단축이 오염을 줄임" if red >= 4 else ("줄이지 않음" if none >= 4 else "중간")) if pre2 else "판정 불가"
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s / %s" % (v1, v2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
