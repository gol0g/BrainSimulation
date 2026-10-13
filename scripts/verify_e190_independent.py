#!/usr/bin/env python3
"""E190 독립 대조 — judge_e190.py 를 쓰지 않고 원 로그를 정규식 분해·정수(1e-4)로 다시 판정. 규칙은 logs/E190/criteria_fixed.txt.
실행: python3 scripts/verify_e190_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def i4(s):
    m = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", s.strip())
    v = int(m.group(2)) * 10000 + int((m.group(3) + "0000")[:4])
    return -v if m.group(1) == "-" else v


def txt(name, b):
    return open(os.path.join(EXP, "logs", "E190", name % b), encoding="utf-8", errors="replace").read()


def ev(t, pushed):
    rc, bad = {}, False
    for m in re.finditer(r"^RCEVAL cat=([A-D]) set=(train|new) mod=([-+0-9.]+) off=\S+ acc=\S+ trials=(\d+) feat_on=(\d+)(.*)$", t, re.M):
        if m.group(4) != "96" or m.group(5) != "96":
            bad = True
        if any(i4(a) >= 6600 and i4(b) >= 6600 for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", m.group(6))):
            bad = True
        rc[(m.group(1), m.group(2))] = i4(m.group(3))
    pu = re.findall(r"^=> DECOMP .*rceval.* pushed=(\d+)", t, re.M)
    if len(rc) != 8 or not pu or int(pu[0]) != pushed or not re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M) \
            or not re.search(r"^\[E184 특징 가중치 적재\].*검증 일치", t, re.M):
        bad = True
    return rc, bad


def tr_ok(t, phase2):
    k = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+)", t, re.M)
    c = re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)
    common = bool(k and len(c) == 3 and all(int(a) > 0 and int(z) == 0 for a, z in c) and re.search(r"^\[학습\] ", t, re.M)
                  and re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M) and re.search(r"^\[E184 특징 가중치 적재\].*검증 일치", t, re.M))
    if not common:
        return False
    if phase2:
        s = re.search(r"^\[둘째 범주 쌍\] 시행 (\d+) 부터", t, re.M)
        n = re.search(r"^\[둘째 범주 쌍\] 둘째 쌍 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)", t, re.M)
        return bool(s and n and s.group(1) == "1500" and n.groups() == ("1500", "1500", "3000") and k.group(1) == "3000" and 1200 <= int(k.group(2)) <= 1800)
    return not re.search(r"^\[둘째 범주 쌍\]", t, re.M) and k.group(1) == "1500" and 600 <= int(k.group(2)) <= 900


def form_ok(t):
    e = re.search(r"^\[원시 특징 노출\] 제시 (\d+)\(주기 \d+ × 사례 \d+ × 범주 (\d+) × 쪽 2\).* — 노출 범주 (\S+)$", t, re.M)
    w = re.findall(r"^RCW side=[lr] .* dg_abs=([0-9.]+) .* g0_init=(\d)$", t, re.M)
    j = re.findall(r"^RCJ side=([lr]) pair=([A-D]{2}) ", t, re.M)
    ck = re.findall(r"^RCCHK tag=(\S+) npres=(\d+) \S+ feat_inactive_spikes=(\d+) \S+ feat_other_side=(\d+) goodbad=(\d+) g_same=(\d+)", t, re.M)
    return bool(e and e.group(1) == "816" and e.group(2) == "4" and e.group(3) == "A,B,C,D" and len(w) == 2 and all(float(d) > 0 and g == "1" for d, g in w)
                and sorted(j) == sorted((s, p) for s in "lr" for p in ("AB", "AC", "AD", "BC", "BD", "CD")) and len(re.findall(r"^=> RAWCAT ", t, re.M)) == 1
                and sorted(x[0] for x in ck) == ["post", "pre", "pre2"] and all(x[1:] == ("288", "0", "0", "0", "1") for x in ck))


def main():
    ok, rows = True, {}
    try:
        for b in BRAINS:
            ok &= form_ok(txt("form_b%d.log", b)) and tr_ok(txt("train_R_b%d.log", b), False) and tr_ok(txt("train_P_b%d.log", b), True)
            R, br = ev(txt("ev_b%d_R_learn.log", b), 8)
            P, bp = ev(txt("ev_b%d_P_learn.log", b), 8)
            N, bn = ev(txt("ev_b%d_none.log", b), 0)
            ok &= not (br or bp or bn)
            if ok:
                rows[b] = ({k: R[k] - N[k] for k in N}, {k: P[k] - N[k] for k in N})
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    nb = npr = kp = it = 0
    for b in BRAINS:
        R, P = rows[b]
        ra, rb = R[("A", "train")], R[("B", "train")]
        base = ra <= -1000 and rb >= 1000
        p = P[("C", "train")] >= 1000 and P[("D", "train")] <= -1000
        keep = P[("A", "train")] * 2 <= ra and P[("B", "train")] * 2 >= rb
        nb += base
        npr += p
        kp += base and p and keep
        it += base and p and not keep
        print("b%d 기준 A %+d B %+d | C %+d D %+d | 뒤 A %+d B %+d" % (b, ra, rb, P[("C", "train")], P[("D", "train")], P[("A", "train")], P[("B", "train")]))
    if nb < 4:
        print("독립 판정: 보류(기준 미달 — %d/5)" % nb)
        return 0
    if npr < 4:
        print("독립 판정: 보류(전제 미달 — %d/5)" % npr)
        return 0
    v = "유지(H111)" if kp >= 4 else "간섭(H111-null)" if it >= 4 else "보류"
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
