#!/usr/bin/env python3
"""E186 독립 대조 — judge_e186.py 를 쓰지 않고 원 로그를 정규식 분해·정수(1e-4) 계산으로 다시 판정. 규칙은 logs/E186/criteria_fixed.txt.
실행: python3 scripts/verify_e186_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
SAMPLES = (("주(16~20)", (16, 17, 18, 19, 20)), ("재현(10~14)", (10, 11, 12, 13, 14)))


def i4(s):
    m = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", s.strip())
    v = int(m.group(2)) * 10000 + int((m.group(3) + "0000")[:4])
    return -v if m.group(1) == "-" else v


def ev(path, w):
    t = open(path, encoding="utf-8", errors="replace").read()
    rc, bad = {}, False
    for m in re.finditer(r"^RCEVAL cat=([AB]) set=(train|new|proto|far) mod=([-+0-9.]+) off=\S+ acc=\S+ trials=(\d+) feat_on=(\d+)(.*)$", t, re.M):
        if m.group(4) != "96" or m.group(5) != "96":
            bad = True
        if any(i4(a) >= 6600 and i4(b) >= 6600 for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", m.group(6))):
            bad = True
        rc[(m.group(1), m.group(2))] = i4(m.group(3))
    pu = re.findall(r"^=> DECOMP .*rceval.* pushed=(\d+)", t, re.M)
    if len(rc) != 8 or not pu or int(pu[0]) != (8 if w == "learn" else 0):
        bad = True
    if not re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M) or re.search(r"^\[E184 특징 가중치 적재\]", t, re.M):
        bad = True
    return rc, bad


def main():
    res, ok = {}, True
    try:
        for name, brains in SAMPLES:
            rows = []
            for b in brains:
                L, b1 = ev(os.path.join(EXP, "logs", "E186", "ev_b%d_learn.log" % b), "learn")
                N, b2 = ev(os.path.join(EXP, "logs", "E186", "ev_b%d_none.log" % b), "none")
                if b1 or b2:
                    ok = False
                    continue
                a = {}
                for (c, s), v in L.items():
                    e = v - N[(c, s)]
                    a[(c, s)] = -e if c == "A" else e
                strong = all(a[(c, "proto")] >= a[(c, "train")] - 50 and a[(c, "proto")] > 0 for c in "AB")
                adv = all(a[(c, "proto")] >= a[(c, "new")] + 50 and a[(c, "proto")] > 0 for c in "AB")
                grad = all(a[(c, "new")] >= a[(c, "far")] + 50 and a[(c, "new")] > 0 for c in "AB")
                inf = all(a[(c, "proto")] <= a[(c, "new")] - 50 for c in "AB")
                rows.append((strong, adv, grad, inf))
            n = [sum(r[i] for r in rows) for i in range(4)]
            res[name] = ("원형 추상화" if n[0] >= 4 and n[2] >= 4 else "원형 우위·기울기" if n[1] >= 4 and n[2] >= 4
                         else "기울기만" if n[2] >= 4 else "원형 열세" if n[3] >= 4 else "보류")
            print("%s 강 %d 우위 %d 기울기 %d 열세 %d → %s" % (name, n[0], n[1], n[2], n[3], res[name]))
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    p, r = res["주(16~20)"], res["재현(10~14)"]
    print("독립 판정: %s%s" % (p, " (재현)" if p == r else " (재현 표본: %s)" % r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
