#!/usr/bin/env python3
"""E182 독립 대조 — judge_e182.py 를 쓰지 않고 평가 원 로그(logs/E182/ev_b{B}_{learn|none}_{A|B}.log)를 문자열 분해·정수 계산으로 다시 판정한다.
규칙은 logs/E182/criteria_fixed.txt. 실행: python3 scripts/verify_e182_independent.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def one(b, w, c):
    t = open(os.path.join(EXP, "logs", "E182", "ev_b%d_%s_%s.log" % (b, w, c)), encoding="utf-8", errors="replace").read()
    lines = t.splitlines()
    dl = [ln for ln in lines if ln.startswith("=> DECOMP ")][0]
    mod = q4(dl.split("mod=")[1].split()[0]); pushed = int(dl.split("pushed=")[1].split()[0])
    tl = [ln for ln in lines if ln.startswith("[맥락 평가 시행]")]
    trial_ok = (len(tl) == 1 and "시행 100 " in tl[0] and "맥락 켬 100회·끔 100회" in tl[0] and "사이 간격 10처리" in tl[0])
    hasB = any(ln.startswith("[맥락 평가] ctx=켬") for ln in lines)
    hasA = any(ln.startswith("[맥락 평가 A]") for ln in lines)
    tv = [ln.split("기저값 ")[1].split("(")[0] for ln in lines if ln.strip().startswith("[motor 동작점]")]
    ton = len(tv) >= 1 and all(q4(x) == -4000 * 10000 for x in tv)
    d = {}
    for ln in lines:
        if ln.startswith("[E171 평가 진단]"):
            side = ln.split("side=")[1].split()[0]
            a, bb = ln.split("motor L/R ")[1].split()[0].split("/")
            d[side] = [q4(a), q4(bb)]
    m4 = (d["left"] + d["right"]) if set(d) == {"left", "right"} else None
    unsat = m4 is not None and not all(x >= 6600 for x in m4)
    ok = trial_ok and hasB == (c == "B") and hasA == (c == "A") and ton and unsat and pushed == (8 if w == "learn" else 0)
    return mod, ok


def main():
    try:
        R = {(b, w, c): one(b, w, c) for b in BRAINS for w in ("learn", "none") for c in ("A", "B")}
    except (FileNotFoundError, ValueError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    ok = all(v[1] for v in R.values())
    acq = elem = pB = pA = 0
    for b in BRAINS:
        eA = R[(b, "learn", "A")][0] - R[(b, "none", "A")][0]
        eB = R[(b, "learn", "B")][0] - R[(b, "none", "B")][0]
        acq += eA <= -1000 and eB >= 1000
        elem += abs(eB - eA) < 1000
        pB += eB >= 1000 and eA > -1000
        pA += eA <= -1000 and eB < 1000
        print("b%d e_A %+d e_B %+d" % (b, eA, eB))
    v = ("보류(조작검증 실패)" if not ok else "획득(H103)" if acq >= 4 else "요소식" if elem >= 4
         else ("부분 B만" if pB >= 4 else "부분 A만" if pA >= 4 else "부분") if pB + pA >= 4 else "보류")
    print("조작검증 %s | 획득 %d/5 요소식 %d/5 B만 %d/5 A만 %d/5" % ("통과" if ok else "실패", acq, elem, pB, pA))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
