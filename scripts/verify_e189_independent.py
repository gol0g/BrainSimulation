#!/usr/bin/env python3
"""E189 독립 대조 — judge_e189.py 를 쓰지 않고 원 로그를 정규식 분해·정수(1e-4)로 다시 판정. 규칙은 logs/E189/criteria_fixed.txt.
실행: python3 scripts/verify_e189_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def i4(s):
    m = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", s.strip())
    v = int(m.group(2)) * 10000 + int((m.group(3) + "0000")[:4])
    return -v if m.group(1) == "-" else v


def ev(path, learn, cats):
    t = open(path, encoding="utf-8", errors="replace").read()
    rc, bad = {}, False
    for m in re.finditer(r"^RCEVAL cat=([A-D]) set=(train|new) mod=([-+0-9.]+) off=\S+ acc=\S+ trials=(\d+) feat_on=(\d+)(.*)$", t, re.M):
        if m.group(4) != "96" or m.group(5) != "96":
            bad = True
        if any(i4(a) >= 6600 and i4(b) >= 6600 for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", m.group(6))):
            bad = True
        rc[(m.group(1), m.group(2))] = i4(m.group(3))
    pu = re.findall(r"^=> DECOMP .*rceval.* pushed=(\d+)", t, re.M)
    if sorted(rc) != sorted((c, s) for c in cats for s in ("train", "new")) or not re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M) \
            or re.search(r"^\[E184 특징 가중치 적재\]", t, re.M):
        bad = True
    if learn is not None and (not pu or int(pu[0]) != (8 if learn else 0)):
        bad = True
    return rc, bad


def tr_ok(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    s = re.search(r"^\[둘째 범주 쌍\] 시행 (\d+) 부터", t, re.M)
    n = re.search(r"^\[둘째 범주 쌍\] 둘째 쌍 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)", t, re.M)
    k = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+)", t, re.M)
    c = re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)
    return bool(s and n and k and s.group(1) == "1500" and n.groups() == ("1500", "1500", "3000") and k.group(1) == "3000"
                and 1200 <= int(k.group(2)) <= 1800 and len(c) == 3 and all(int(a) > 0 and int(z) == 0 for a, z in c)
                and re.search(r"^\[학습\] ", t, re.M) and re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M))


def main():
    ok, rows = True, {}
    try:
        for b in BRAINS:
            ok &= tr_ok(os.path.join(EXP, "logs", "E189", "train_b%d.log" % b))
            L, bl = ev(os.path.join(EXP, "logs", "E189", "ev_b%d_learn.log" % b), True, "ABCD")
            N, bn = ev(os.path.join(EXP, "logs", "E189", "ev_b%d_none.log" % b), False, "ABCD")
            RL, rl = ev(os.path.join(EXP, "logs", "E184", "ev_b%d_U_learn.log" % b), None, "AB")
            RN, rn = ev(os.path.join(EXP, "logs", "E184", "ev_b%d_U_none.log" % b), None, "AB")
            ok &= not (bl or bn or rl or rn) and len(RL) == 4 and len(RN) == 4
            if ok:
                rows[b] = ({k: L[k] - N[k] for k in L}, {k: RL[k] - RN[k] for k in RL})
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    pr = kp = it = 0
    for b in BRAINS:
        e, R = rows[b]
        a1, b1, c2, d2 = e[("A", "train")], e[("B", "train")], e[("C", "train")], e[("D", "train")]
        ra, rb = R[("A", "train")], R[("B", "train")]
        p = c2 >= 1000 and d2 <= -1000
        base = ra <= -1000 and rb >= 1000
        keep = a1 * 2 <= ra and b1 * 2 >= rb
        pr += p
        kp += base and p and keep
        it += base and p and not keep
        print("b%d C %+d D %+d | A %+d/%+d B %+d/%+d" % (b, c2, d2, a1, ra, b1, rb))
    if pr < 4:
        print("독립 판정: 보류(전제 미달 — %d/5)" % pr)
        return 0
    v = "유지(H110)" if kp >= 4 else "간섭(H110-null)" if it >= 4 else "보류"
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
