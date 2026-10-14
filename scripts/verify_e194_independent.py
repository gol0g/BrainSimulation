#!/usr/bin/env python3
"""E194 독립 대조 — judge_e194.py 를 쓰지 않고 원 로그를 정규식 분해·정수(1e-4)로 다시 판정((가) L4 E187 규칙·(나) 학습량 Δ = 칸 평균 9,000 − 3,000(E193)). 규칙은 logs/E194/criteria_fixed.txt.
실행: python3 scripts/verify_e194_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
SG = {("X", "A"): -1, ("X", "B"): 1, ("Y", "A"): 1, ("Y", "B"): -1}


def i4(s):
    m = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", s.strip())
    v = int(m.group(2)) * 10000 + int((m.group(3) + "0000")[:4])
    return -v if m.group(1) == "-" else v


def train_ok(path, NT):
    t = open(path, encoding="utf-8", errors="replace").read()
    c = re.search(r"^\[맥락 범주 과제\] 시행 (\d+) 중 맥락 X (\d+) · Y (\d+) · 정답 같은 쪽 (\d+) · 교차 (\d+)", t, re.M)
    k = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+)", t, re.M)
    f = re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)
    x = re.findall(r"^\[맥락 범주 점검\].*켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · 시행 끝 두 맥락 발화 (\d+)", t, re.M)
    if not (c and k and re.search(r"^\[맥락 범주 과제\] catctx", t, re.M) and re.search(r"^\[학습\] ", t, re.M)
            and re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M)):
        return False
    c = [int(v) for v in c.groups()]
    lo, hi = NT * 4 // 10, NT * 6 // 10
    return (c[0] == NT and c[1] + c[2] == NT and c[3] + c[4] == NT and lo <= c[1] <= hi and lo <= c[3] <= hi
            and int(k.group(1)) == NT and lo <= int(k.group(2)) <= hi
            and len(f) == 3 and all(int(a) > 0 and int(z) == 0 for a, z in f)
            and len(x) == 3 and all(int(a) > 0 and int(o) == 0 and int(z) == 0 for a, o, z in x))


def ev(path, w, ctx):
    t = open(path, encoding="utf-8", errors="replace").read()
    rc, bad = {}, False
    for m in re.finditer(r"^RCEVAL cat=([AB]) set=(train|new) mod=([-+0-9.]+) off=\S+ acc=\S+ trials=(\d+) feat_on=(\d+) ctx=(\w+) ctx_on=(\d+)(.*)$", t, re.M):
        if m.group(4) != "96" or m.group(5) != "96" or m.group(6) != ctx or m.group(7) != "96":
            bad = True
        if any(i4(a) >= 6600 and i4(b) >= 6600 for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", m.group(8))):
            bad = True
        rc[(m.group(1), m.group(2))] = i4(m.group(3))
    pu = re.findall(r"^=> DECOMP .*rceval.* pushed=(\d+)", t, re.M)
    if len(rc) != 4 or not pu or int(pu[0]) != (8 if w == "learn" else 0) or not re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M):
        bad = True
    return rc, bad


def cells(exp, NT):
    """한 실험 로그 → 뇌마다 칸별 a(노출·새), 조작검증 실패면 None."""
    out = {}
    for b in BRAINS:
        if not train_ok(os.path.join(EXP, "logs", exp, "train_b%d.log" % b), NT):
            return None
        R = {}
        for w in ("learn", "none"):
            for x in "XY":
                rc, bad = ev(os.path.join(EXP, "logs", exp, "ev_b%d_%s_%s.log" % (b, w, x)), w, x)
                if bad:
                    return None
                R[(w, x)] = rc
        out[b] = {(x, c, s): SG[(x, c)] * (R[("learn", x)][(c, s)] - R[("none", x)][(c, s)]) for x in "XY" for c in "AB" for s in ("train", "new")}
    return out


def main():
    try:
        A9 = cells("E194", 9000)
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if A9 is None:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    try:
        A3 = cells("E193", 3000)
    except (OSError, KeyError):
        A3 = None
    if A3 is None:
        print("독립 판정: 보류(결측 — 3,000 기준 E193)")
        return 0
    conc = memo = part = fail = grow = flat = 0
    for b in BRAINS:
        a = A9[b]
        n = sum(a[(x, c, "train")] >= 1000 for x in "XY" for c in "AB")
        g = all(a[(x, c, "new")] >= 500 and a[(x, c, "new")] * 2 >= a[(x, c, "train")] for x in "XY" for c in "AB")
        conc += n == 4 and g
        memo += n == 4 and not g
        part += 2 <= n <= 3
        fail += n <= 1
        d = sum(A9[b][(x, c, "train")] for x in "XY" for c in "AB") / 4.0 - sum(A3[b][(x, c, "train")] for x in "XY" for c in "AB") / 4.0
        grow += d >= 300
        flat += abs(d) < 150
        print("b%d 획득 칸 %d 일반화 %s Δ %+.0f" % (b, n, g, d))
    v = ("맥락 의존 범주" if conc >= 4 else "맥락 의존(사례 한정)" if memo >= 4 else "부분" if part >= 4 else "획득 실패" if fail >= 4 else "보류")
    dv = "자람(H115)" if grow >= 4 else "상한(H115-null)" if flat >= 4 else "보류"
    print("독립 판정: 학습량 %s · L4 %s" % (dv, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
