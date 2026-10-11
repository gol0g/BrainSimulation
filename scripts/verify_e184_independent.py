#!/usr/bin/env python3
"""E184 독립 대조 — judge_e184.py 를 쓰지 않고 원 로그를 문자열 분해·정수(1e-4) 계산으로 다시 판정한다. 규칙은 logs/E184/criteria_fixed.txt.
실행: python3 scripts/verify_e184_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARM = "U"


def i4(s):
    m = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", s.strip())
    v = int(m.group(2)) * 10000 + int((m.group(3) + "0000")[:4])
    return -v if m.group(1) == "-" else v


def ev(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    rc = {}
    for m in re.finditer(r"^RCEVAL cat=([AB]) set=(train|new) mod=([-+0-9.]+) off=\S+ acc=\S+ trials=(\d+) feat_on=(\d+)(.*)$", t, re.M):
        mots = re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", m.group(6))
        sat = any(i4(a) >= 6600 and i4(b) >= 6600 for a, b in mots)
        rc[(m.group(1), m.group(2))] = (i4(m.group(3)), int(m.group(4)), int(m.group(5)), sat)
    pu = re.findall(r"^=> DECOMP .*rceval.* pushed=(\d+)", t, re.M)
    ld = len(re.findall(r"^\[E184 특징 가중치 적재\].*검증 일치", t, re.M))
    return rc, (int(pu[0]) if pu else None), ld


def tr(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+) · A\(교차\) (\d+)", t, re.M)
    chk = [(int(a), int(b)) for a, b in re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)]
    done = re.search(r"^\[학습\] ", t, re.M) is not None
    ld = len(re.findall(r"^\[E184 특징 가중치 적재\].*검증 일치", t, re.M))
    return (tuple(int(x) for x in m.groups()) if m else None), chk, done, ld


def main():
    ok = True
    E = {}
    try:
        for b in BRAINS:
            cnt, chk, done, ldt = tr(os.path.join(EXP, "logs", "E184", "train_b%d_%s.log" % (b, ARM)))
            if cnt is None or cnt[0] != 1500 or cnt[1] + cnt[2] != 1500 or not 600 <= cnt[1] <= 900 or not done or ldt != 0:
                ok = False
            if len(chk) != 3 or any(a <= 0 or z != 0 for a, z in chk):
                ok = False
            R = {w: ev(os.path.join(EXP, "logs", "E184", "ev_b%d_%s_%s.log" % (b, ARM, w))) for w in ("learn", "none")}
            for w, (rc, pu, ld) in R.items():
                if len(rc) != 4 or pu != (8 if w == "learn" else 0) or ld != 0:
                    ok = False
                if any(v[1] != 96 or v[2] != 96 or v[3] for v in rc.values()):
                    ok = False
            E[b] = {k: R["learn"][0][k][0] - R["none"][0][k][0] for k in R["learn"][0] if k in R["none"][0]}
            if len(E[b]) != 4:
                ok = False
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    for b in BRAINS:
        e = E[b]
        print("b%d A train %+d new %+d · B train %+d new %+d" % (b, e.get(("A", "train"), 0), e.get(("A", "new"), 0), e.get(("B", "train"), 0), e.get(("B", "new"), 0)))
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    n = {"gen": 0, "memo": 0, "A": 0, "B": 0, "fail": 0}
    for b in BRAINS:
        aT, aN, bT, bN = E[b][("A", "train")], E[b][("A", "new")], E[b][("B", "train")], E[b][("B", "new")]
        acq = aT <= -1000 and bT >= 1000
        gen = aN <= -500 and bN >= 500 and aN * 2 <= aT and bN * 2 >= bT
        n["gen"] += acq and gen
        n["memo"] += acq and not gen
        n["A"] += aT <= -1000 and bT < 1000
        n["B"] += bT >= 1000 and aT > -1000
        n["fail"] += aT > -1000 and bT < 1000
    v = ("개념 수준(H105)" if n["gen"] >= 4 else "사례 한정(H105-null)" if n["memo"] >= 4 else "한 범주만 A(교차)" if n["A"] >= 4
         else "한 범주만 B(같은 쪽)" if n["B"] >= 4 else "획득 실패" if n["fail"] >= 4 else "보류")
    print("개념 %d · 사례 한정 %d · A만 %d · B만 %d · 실패 %d" % (n["gen"], n["memo"], n["A"], n["B"], n["fail"]))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
