#!/usr/bin/env python3
"""E166 독립 대조 — judge_e166.py 를 쓰지 않고 원 로그(E166 FNF·DNF, E161 F·D)와 추적을 문자열 분해로 다시 읽어 판정 1·2 를 계산한다.
규칙은 logs/E166/criteria_fixed.txt. 비율 비교는 분수(Fraction)로 — 판정 코드의 정수 곱 변환과 다른 경로.
실행: python3 scripts/verify_e166_independent.py (저장소 루트에서)"""
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
        keep = lose = red = same = 0
        for b in BRAINS:
            t = {a: rd("logs", "E166", "%s_b%d.log" % (a, b)) for a in ("FNF", "DNF")}
            r = {a: rd("logs", "E161", "%s_b%d.log" % (k, b)) for a, k in (("FNF", "F"), ("DNF", "D"))}
            e = {a: val(t[a], "[사후]") - val(t[a], "[사전]") for a in t}
            e0 = {a: val(r[a], "[사후]") - val(r[a], "[사전]") for a in r}
            okb = all(abs(val(t[a], "[사전]") - val(r[a], "[사전]")) <= 20 for a in t) and e0["FNF"] <= -1000 and e0["DNF"] <= -1000
            nld = {a: sum(ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln for ln in t[a].splitlines()) for a in t}
            okb &= nld["FNF"] >= 2 and nld["DNF"] == 0
            for a in t:
                R = np.load(os.path.join(EXP, "traces", "E166", "tr_%s_b%d.npz" % (a, b)))["rows"]
                rs = (11.0 / 12.0) ** 20
                num = sum(abs(R[i, 21 + k] - rs * R[i, 13 + k]) for i in range(len(R)) for k in range(4))
                den = sum(abs(R[i, 13 + k]) for i in range(len(R)) for k in range(4))
                pre = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
                okb &= len(R) == 500 and num >= 0.05 * den and pre <= 1e-3
            ok &= okb
            qF, qD = Fr(e["FNF"], e0["FNF"]), Fr(e["DNF"], e0["DNF"])
            keep += qF >= Fr(4, 5); lose += qF <= Fr(1, 2)
            red += qF - qD >= Fr(1, 5); same += abs(qF - qD) < Fr(1, 10)
            print("b%d q_F %.4f q_D %.4f 차 %+.4f | 조작 %s" % (b, float(qF), float(qD), float(qF - qD), "✓" if okb else "✗"))
    except (FileNotFoundError, ValueError, IndexError, ZeroDivisionError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    else:
        v1 = "남음(H089)" if keep >= 4 else ("손실(H089-null)" if lose >= 4 else "부분")
        v2 = "형성이 동결 의존을 줄임" if red >= 4 else ("줄이지 않음" if same >= 4 else "중간")
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s / %s" % (v1, v2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
