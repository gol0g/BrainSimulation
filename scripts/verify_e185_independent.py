#!/usr/bin/env python3
"""E185 독립 대조 — judge_e185.py 를 쓰지 않고 원 로그를 정규식 분해·정수(1e-4) 계산으로 다시 판정한다. 규칙은 logs/E185/criteria_fixed.txt.
실행: python3 scripts/verify_e185_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)


def i4(s):
    m = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", s.strip())
    v = int(m.group(2)) * 10000 + int((m.group(3) + "0000")[:4])
    return -v if m.group(1) == "-" else v


def ev(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    rc = {}
    bad = False
    for m in re.finditer(r"^RCEVAL cat=([AB]) set=(train|new) mod=([-+0-9.]+) off=\S+ acc=\S+ trials=(\d+) feat_on=(\d+)(.*)$", t, re.M):
        if int(m.group(4)) != 96 or int(m.group(5)) != 96:
            bad = True
        if any(i4(a) >= 6600 and i4(b) >= 6600 for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", m.group(6))):
            bad = True
        rc[(m.group(1), m.group(2))] = i4(m.group(3))
    pu = re.findall(r"^=> DECOMP .*rceval.* pushed=(\d+)", t, re.M)
    kt = len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M))
    fw = len(re.findall(r"^\[E184 특징 가중치 적재\]", t, re.M))
    return rc, (int(pu[0]) if pu else None), kt, fw, bad


def tr(path, n, lo, hi):
    t = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+) · A\(교차\) (\d+)", t, re.M)
    chk = re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)
    ok = (m is not None and int(m.group(1)) == n and int(m.group(2)) + int(m.group(3)) == n and lo <= int(m.group(2)) <= hi
          and len(chk) == 3 and all(int(a) > 0 and int(z) == 0 for a, z in chk) and re.search(r"^\[학습\] ", t, re.M) is not None
          and len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M)) >= 1 and not re.search(r"^\[E184 특징 가중치 적재\]", t, re.M))
    return ok


def main():
    ok = True
    E = {}
    try:
        for b in BRAINS:
            ok &= tr(os.path.join(EXP, "logs", "E185", "train_b%d_500.log" % b), 500, 200, 300)
            ok &= tr(os.path.join(EXP, "logs", "E185", "train_b%d_1500.log" % b), 1500, 600, 900)
            R = {w: ev(os.path.join(EXP, "logs", "E185", "ev_b%d_%s.log" % (b, w))) for w in ("learn500", "learn1500", "none")}
            for w, (rc, pu, kt, fw, bad) in R.items():
                if len(rc) != 4 or pu != (0 if w == "none" else 8) or kt < 1 or fw != 0 or bad:
                    ok = False
            E[b] = {T: {k: R["learn" + T][0][k] - R["none"][0][k] for k in R["none"][0] if k in R["learn" + T][0]} for T in ("500", "1500")}
            if any(len(E[b][T]) != 4 for T in E[b]):
                ok = False
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    conf = dose = memo = oA = oB = fail = 0
    for b in BRAINS:
        e5, e = E[b]["500"], E[b]["1500"]
        aT, aN, bT, bN = e[("A", "train")], e[("A", "new")], e[("B", "train")], e[("B", "new")]
        acq = aT <= -1000 and bT >= 1000
        gen = aN <= -500 and bN >= 500 and aN * 2 <= aT and bN * 2 >= bT
        conf += acq and gen
        memo += acq and not gen
        oA += aT <= -1000 and bT < 1000
        oB += bT >= 1000 and aT > -1000
        fail += aT > -1000 and bT < 1000
        dose += (aN - e5[("A", "new")]) <= -200 and (bN - e5[("B", "new")]) >= 200
        print("b%d 1500 A %+d/%+d B %+d/%+d · 새 사례 증가 A %+d B %+d" % (b, aT, aN, bT, bN, aN - e5[("A", "new")], bN - e5[("B", "new")]))
    v = ("확증·용량-반응(개념 조건 1~4, 범주 수준)" if conf >= 4 and dose >= 4 else "확증(용량-반응 미확인)" if conf >= 4
         else "확증 실패(사례 한정)" if memo >= 4 else "확증 실패(한 범주만)" if (oA >= 4 or oB >= 4) else "확증 실패(획득 실패)" if fail >= 4 else "보류")
    print("확증 %d · 용량-반응 %d · 사례 한정 %d · A만 %d · B만 %d · 실패 %d" % (conf, dose, memo, oA, oB, fail))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
