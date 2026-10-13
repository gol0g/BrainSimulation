#!/usr/bin/env python3
"""E187 독립 대조 — judge_e187.py 를 쓰지 않고 원 로그를 정규식 분해·정수(1e-4)로 다시 판정. 규칙은 logs/E187/criteria_fixed.txt.
실행: python3 scripts/verify_e187_independent.py (저장소 루트에서)"""
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


def train_ok(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    c = re.search(r"^\[맥락 범주 과제\] 시행 (\d+) 중 맥락 X (\d+) · Y (\d+) · 정답 같은 쪽 (\d+) · 교차 (\d+)", t, re.M)
    k = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+)", t, re.M)
    f = re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)
    x = re.findall(r"^\[맥락 범주 점검\].*켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · 시행 끝 두 맥락 발화 (\d+)", t, re.M)
    if not (c and k and re.search(r"^\[맥락 범주 과제\] catctx", t, re.M) and re.search(r"^\[학습\] ", t, re.M)
            and re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M)):
        return False
    c = [int(v) for v in c.groups()]
    return (c[0] == 3000 and c[1] + c[2] == 3000 and c[3] + c[4] == 3000 and 1200 <= c[1] <= 1800 and 1200 <= c[3] <= 1800
            and int(k.group(1)) == 3000 and 1200 <= int(k.group(2)) <= 1800
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


def main():
    ok, E = True, {}
    try:
        for b in BRAINS:
            ok &= train_ok(os.path.join(EXP, "logs", "E187", "train_b%d.log" % b))
            R = {}
            for w in ("learn", "none"):
                for x in "XY":
                    rc, bad = ev(os.path.join(EXP, "logs", "E187", "ev_b%d_%s_%s.log" % (b, w, x)), w, x)
                    ok &= not bad
                    R[(w, x)] = rc
            E[b] = {(x, c, s): R[("learn", x)][(c, s)] - R[("none", x)][(c, s)] for x in "XY" for c in "AB" for s in ("train", "new")}
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    conc = memo = part = fail = 0
    for b in BRAINS:
        a = {(x, c, s): SG[(x, c)] * E[b][(x, c, s)] for x in "XY" for c in "AB" for s in ("train", "new")}
        n = sum(a[(x, c, "train")] >= 1000 for x in "XY" for c in "AB")
        g = all(a[(x, c, "new")] >= 500 and a[(x, c, "new")] * 2 >= a[(x, c, "train")] for x in "XY" for c in "AB")
        conc += n == 4 and g
        memo += n == 4 and not g
        part += 2 <= n <= 3
        fail += n <= 1
        print("b%d 획득 칸 %d 일반화 %s" % (b, n, g))
    v = ("맥락 의존 범주(H108)" if conc >= 4 else "맥락 의존(사례 한정)" if memo >= 4 else "부분" if part >= 4 else "획득 실패(H108-null)" if fail >= 4 else "보류")
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
