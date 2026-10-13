#!/usr/bin/env python3
"""E188 독립 대조 — judge_e188.py 를 쓰지 않고 원 로그를 정규식 분해·정수(1e-4)로 다시 판정. 규칙은 logs/E188/criteria_fixed.txt.
실행: python3 scripts/verify_e188_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def i4(s):
    m = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", s.strip())
    v = int(m.group(2)) * 10000 + int((m.group(3) + "0000")[:4])
    return -v if m.group(1) == "-" else v


def ev(path, learn):
    t = open(path, encoding="utf-8", errors="replace").read()
    rc, bad = {}, False
    for m in re.finditer(r"^RCEVAL cat=([AB]) set=(train|new) mod=([-+0-9.]+) off=\S+ acc=\S+ trials=(\d+) feat_on=(\d+)(.*)$", t, re.M):
        if m.group(4) != "96" or m.group(5) != "96":
            bad = True
        if any(i4(a) >= 6600 and i4(b) >= 6600 for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", m.group(6))):
            bad = True
        rc[(m.group(1), m.group(2))] = i4(m.group(3))
    pu = re.findall(r"^=> DECOMP .*rceval.* pushed=(\d+)", t, re.M)
    if len(rc) != 4 or not pu or int(pu[0]) != (8 if learn else 0) or not re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M) \
            or re.search(r"^\[E184 특징 가중치 적재\]", t, re.M):
        bad = True
    return rc, bad


def tr_ok(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    s = re.search(r"^\[범주 반전\] 시행 (\d+) 부터", t, re.M)
    n = re.search(r"^\[범주 반전\] 반전 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)", t, re.M)
    k = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+)", t, re.M)
    c = re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)
    return bool(s and n and k and s.group(1) == "1500" and n.groups() == ("1500", "1500", "3000") and k.group(1) == "3000"
                and 1200 <= int(k.group(2)) <= 1800 and len(c) == 3 and all(int(a) > 0 and int(z) == 0 for a, z in c)
                and re.search(r"^\[학습\] ", t, re.M) and re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M))


def main():
    ok, rows = True, {}
    try:
        for b in BRAINS:
            ok &= tr_ok(os.path.join(EXP, "logs", "E188", "train_b%d.log" % b))
            L, bl = ev(os.path.join(EXP, "logs", "E188", "ev_b%d_learn.log" % b), True)
            N, bn = ev(os.path.join(EXP, "logs", "E188", "ev_b%d_none.log" % b), False)
            RL, _ = ev(os.path.join(EXP, "logs", "E184", "ev_b%d_U_learn.log" % b), True)
            RN, _ = ev(os.path.join(EXP, "logs", "E184", "ev_b%d_U_none.log" % b), False)
            ok &= not (bl or bn) and len(RL) == 4 and len(RN) == 4
            rows[b] = ({k: L[k] - N[k] for k in L}, {k: RL[k] - RN[k] for k in RL})
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    s = m = p = st = 0
    for b in BRAINS:
        e, R = rows[b]
        aT, aN, bT, bN = e[("A", "train")], e[("A", "new")], e[("B", "train")], e[("B", "new")]
        dA, dB = aT - R[("A", "train")], bT - R[("B", "train")]
        acq = aT >= 1000 and bT <= -1000
        gen = aN >= 500 and bN <= -500 and aN * 2 >= aT and bN * 2 <= bT
        s += acq and gen
        m += acq and not gen
        p += (not acq) and dA >= 1000 and dB <= -1000
        st += abs(dA) < 500 and abs(dB) < 500
        print("b%d A %+d/%+d B %+d/%+d Δ A %+d B %+d" % (b, aT, aN, bT, bN, dA, dB))
    v = "반전 성공(H109)" if s >= 4 else "반전(사례 한정)" if m >= 4 else "부분 반전" if p >= 4 else "고착(H109-null)" if st >= 4 else "보류"
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
