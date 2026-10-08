#!/usr/bin/env python3
"""E157 독립 대조 — judge_e157.py 를 쓰지 않고 원 로그·추적에서 다시 계산한다(문자열 분해·분수 비교로 따로 구현).
학습 e: D = logs/E157/learn_D_b{B}.log 가 있으면 그것, 없으면 logs/E141/b{B}.log; F = learn_F 또는 logs/E153/train_b{B}.log; Fk·Dk = logs/E157/learn_*.
실행: python3 scripts/verify_e157_independent.py (저장소 루트에서)"""
import os
import sys
from fractions import Fraction

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def txt(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def q4(s):
    """'+0.0195' → 195 (1e-4 정수, 문자열에서 직접)."""
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def modline(t, head):
    for ln in t.splitlines():
        if ln.startswith(head):
            return q4(ln.split("변조폭")[1].split("**")[0])
    return None


def learn_src(a, b):
    own = ("logs", "E157", "learn_%s_b%d.log" % (a, b))
    if a in ("Fk", "Dk") or txt(*own) is not None:
        return own
    return ("logs", "E141", "b%d.log" % b) if a == "D" else ("logs", "E153", "train_b%d.log" % b)


def count(t, head):
    return sum(1 for ln in t.splitlines() if ln.startswith(head) and "검증 일치" in ln)


def spk(t):
    v = {}
    for ln in t.splitlines():
        if ln.startswith("=> KCRATE kc_"):
            v[ln[len("=> KCRATE kc_")]] = int(ln.split("제시 스파이크 ")[1].split()[0])
    return v.get("l", 0) + v.get("r", 0) if len(v) == 2 else None


def jac(t):
    for ln in t.splitlines():
        if ln.startswith("=> KCOVERLAP"):
            segs = [s for s in ln.split(" | ") if "side=" in s]
            return tuple(q4(s.split(" jac=")[1].split()[0]) for s in segs[:2])
    return None


def trace_ok(a, b):
    f = os.path.join(EXP, "traces", "E157", "tr_%s_b%d.npz" % (a, b))
    R = np.load(f)["rows"]
    rs = (11.0 / 12.0) ** 20
    num = sum(abs(R[i, 21 + k] - rs * R[i, 13 + k]) for i in range(len(R)) for k in range(4))
    den = sum(abs(R[i, 13 + k]) for i in range(len(R)) for k in range(4))
    alive = sum(1 for i in range(len(R)) if abs(R[i, 13]) + abs(R[i, 14]) > 1.0) / len(R)
    pre = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
    return len(R) == 500 and num <= 1e-3 * den and alive >= 0.9 and pre <= 1e-3


def main():
    try:
        e, sp, jj, ok = {}, {}, {}, True
        for b in BRAINS:
            for a in ("D", "F", "Fk", "Dk"):
                t = txt(*learn_src(a, b))
                e[(a, b)] = modline(t, "[사후]") - modline(t, "[사전]")
                sp[(a, b)] = spk(txt("logs", "E157", "kcrate_%s_b%d.log" % (a, b)))
                if modline(txt("logs", "E157", "r25_%s_b%d.log" % (a, b)), "[사전]") is None:
                    raise ValueError("r25 결측")
            for a in ("Fk", "Dk"):
                jj[(a, b)] = jac(txt("logs", "E157", "ov_%s_b%d.log" % (a, b)))
            tf, td_ = txt(*learn_src("Fk", b)), txt(*learn_src("Dk", b))
            okb = (Fraction(85, 100) <= Fraction(sp[("Fk", b)], sp[("D", b)]) <= Fraction(115, 100)
                   and Fraction(85, 100) <= Fraction(sp[("Dk", b)], sp[("F", b)]) <= Fraction(115, 100)
                   and max(jj[("Fk", b)]) <= 500 and min(jj[("Dk", b)]) >= 2500
                   and count(tf, "[E153 종류 입력 적재]") >= 2 and count(tf, "[E157 종류 입력 배율]") >= 2
                   and count(td_, "[E157 종류 입력 배율]") >= 2 and count(td_, "[E153 종류 입력 적재]") == 0
                   and trace_ok("Fk", b) and trace_ok("Dk", b) and e[("D", b)] <= -1000)
            ok &= okb
    except (TypeError, AttributeError, ValueError, FileNotFoundError, IndexError, ZeroDivisionError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    cats = {}
    for b in BRAINS:
        rF = Fraction(e[("Fk", b)], e[("D", b)]); rD = Fraction(e[("Dk", b)], e[("D", b)])
        s_small = rF <= Fraction(13, 10); g_big = rD > Fraction(13, 10)
        c = ("이득(H080)" if s_small and g_big else "분리(H080-sep)" if not s_small and not g_big
             else "둘 다(H080-both)" if g_big else "둘 다 아님(H080-int)")
        cats[b] = c
        print("b%d e D %+d F %+d Fk %+d Dk %+d | r Fk %.3f Dk %.3f | 발화 Fk/D %.3f Dk/F %.3f | J Fk %s Dk %s | %s" % (
            b, e[("D", b)], e[("F", b)], e[("Fk", b)], e[("Dk", b)], float(rF), float(rD),
            sp[("Fk", b)] / sp[("D", b)], sp[("Dk", b)] / sp[("F", b)], jj[("Fk", b)], jj[("Dk", b)], c))
    top = [c for c in set(cats.values()) if sum(v == c for v in cats.values()) >= 4]
    v = "보류(조작검증 실패)" if not ok else (top[0] if top else "보류")
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
