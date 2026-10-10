#!/usr/bin/env python3
"""E168 독립 대조 — judge_e168.py 를 쓰지 않고 원 로그(E168 FI·FR, E161 F)·추적을 문자열 분해로 다시 읽어 판정 1·2 를 분수(Fraction)로 계산한다.
규칙은 logs/E168/criteria_fixed.txt. 실행: python3 scripts/verify_e168_independent.py (저장소 루트에서)"""
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


def kc(t):
    for ln in t.splitlines():
        if ln.startswith("[E168 KC 발화]"):
            parts = ln.split("평균 ")
            return Fr(parts[1].split()[0]), Fr(parts[2].split()[0])   # 결정 단계, 보상 창 보상 시행
    raise ValueError("KC")


def main():
    try:
        ok = True
        succ = fail = eff = none = 0
        for b in BRAINS:
            tI, tR, tF = rd("logs", "E168", "FI_b%d.log" % b), rd("logs", "E168", "FR_b%d.log" % b), rd("logs", "E161", "F_b%d.log" % b)
            eI = val(tI, "[사후]") - val(tI, "[사전]")
            eR = val(tR, "[사후]") - val(tR, "[사전]")
            eF = val(tF, "[사후]") - val(tF, "[사전]")
            okb = eF <= -1000 and all(abs(val(t, "[사전]") - val(tF, "[사전]")) <= 20 for t in (tI, tR))
            nconn = lambda t: sum(ln.startswith("  [E168 도파민→KC억제]") for ln in t.splitlines())
            okb &= nconn(tI) == 1 and nconn(tR) == 0
            okb &= all(any(ln.startswith("[구현 점검] 첫 보상 창 끝 도파민 뉴런 I_input") and ln.rstrip().endswith("→ 0.0") for ln in t.splitlines()) for t in (tI, tR))
            dI, rI = kc(tI)
            dR, rR = kc(tR)
            okb &= rI <= rR / 10 and dI >= dR * Fr(9, 10)
            for a in ("FI", "FR"):
                R = np.load(os.path.join(EXP, "traces", "E168", "tr_%s_b%d.npz" % (a, b)))["rows"]
                okb &= len(R) == 500 and abs(float(R[:, 17:21].sum())) <= 1e-3 * max(abs(float(R[:, 12].sum())), 1e-12)
            ok &= okb
            qI, qR = Fr(eI, eF), Fr(eR, eF)
            succ += qI >= Fr(4, 5); fail += qI <= Fr(1, 2)
            eff += qI - qR >= Fr(1, 5); none += abs(qI - qR) < Fr(1, 10)
            print("b%d q_I %.4f q_R %.4f | KC 보상 창 %.6f/%.6f 결정 %.6f/%.6f | 조작 %s" % (b, float(qI), float(qR), float(rI), float(rR), float(dI), float(dR), "✓" if okb else "✗"))
    except (FileNotFoundError, ValueError, IndexError, ZeroDivisionError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    else:
        v1 = "대체 성공(H091)" if succ >= 4 else ("대체 실패(H091-null)" if fail >= 4 else "부분")
        v2 = "연결 효과" if eff >= 4 else ("효과 없음" if none >= 4 else "중간")
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s / %s" % (v1, v2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
