#!/usr/bin/env python3
"""E191 독립 대조 — judge_e191.py 를 쓰지 않고 원 로그를 정규식 분해·정수(1e-4)로 다시 판정. 규칙은 logs/E191/criteria_fixed.txt.
실행: python3 scripts/verify_e191_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)


def i4(s):
    m = re.fullmatch(r"([+-]?)(\d+)\.(\d+)", s.strip())
    v = int(m.group(2)) * 10000 + int((m.group(3) + "0000")[:4])
    return -v if m.group(1) == "-" else v


def txt(name):
    return open(os.path.join(EXP, "logs", "E191", name), encoding="utf-8", errors="replace").read()


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


def tr_ok(t, kind):
    k = re.search(r"^\[범주 과제\] 시행 (\d+) 중 범주 B\(같은 쪽\) (\d+)", t, re.M)
    c = re.findall(r"^\[범주 과제 점검\].*행동 창 끝 특징 발화 (\d+) · 시행 끝 (\d+)", t, re.M)
    if not (k and len(c) == 3 and all(int(a) > 0 and int(z) == 0 for a, z in c) and re.search(r"^\[학습\] ", t, re.M)
            and re.search(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M) and re.search(r"^\[E184 특징 가중치 적재\].*검증 일치", t, re.M)):
        return False
    p2 = re.search(r"^\[둘째 범주 쌍\]", t, re.M)
    rv = re.search(r"^\[범주 반전\]", t, re.M)
    if kind == "R":
        return not p2 and not rv and k.group(1) == "1500" and 600 <= int(k.group(2)) <= 900
    full = k.group(1) == "3000" and 1200 <= int(k.group(2)) <= 1800
    if kind == "P":
        n = re.search(r"^\[둘째 범주 쌍\] 둘째 쌍 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)", t, re.M)
        return bool(full and not rv and re.search(r"^\[둘째 범주 쌍\] 시행 1500 부터", t, re.M) and n and n.groups() == ("1500", "1500", "3000"))
    n = re.search(r"^\[범주 반전\] 반전 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)", t, re.M)
    return bool(full and not p2 and re.search(r"^\[범주 반전\] 시행 1500 부터", t, re.M) and n and n.groups() == ("1500", "1500", "3000"))


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
            ok &= form_ok(txt("form_b%d.log" % b)) and all(tr_ok(txt("train_%s_b%d.log" % (k, b)), k) for k in "RPV")
            N, bn = ev(txt("ev_b%d_none.log" % b), 0)
            got = {}
            for k in "RPV":
                X, bx = ev(txt("ev_b%d_%s_learn.log" % (b, k)), 8)
                ok &= not bx
                got[k] = X
            ok &= not bn
            if ok:
                rows[b] = {k: {kk: got[k][kk] - N[kk] for kk in N} for k in "RPV"}
    except (OSError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    c = dict(base=0, prem=0, keep=0, lose=0, gen=0, memo=0, rs=0, rm=0, rp=0, rt=0)
    for b in BRAINS:
        R, P, V = rows[b]["R"], rows[b]["P"], rows[b]["V"]
        ra, rb = R[("A", "train")], R[("B", "train")]
        bs = ra <= -1000 and rb >= 1000
        pr = P[("C", "train")] >= 1000 and P[("D", "train")] <= -1000
        kp = P[("A", "train")] * 2 <= ra and P[("B", "train")] * 2 >= rb
        g = R[("A", "new")] <= -500 and R[("B", "new")] >= 500 and R[("A", "new")] * 2 <= ra and R[("B", "new")] * 2 >= rb
        va, vb = V[("A", "train")], V[("B", "train")]
        aq = va >= 1000 and vb <= -1000
        vg = V[("A", "new")] >= 500 and V[("B", "new")] <= -500 and V[("A", "new")] * 2 >= va and V[("B", "new")] * 2 <= vb
        c["base"] += bs; c["prem"] += pr; c["keep"] += bs and pr and kp; c["lose"] += bs and pr and not kp
        c["gen"] += bs and g; c["memo"] += bs and not g
        c["rs"] += aq and vg; c["rm"] += aq and not vg
        c["rp"] += (not aq) and va - ra >= 1000 and vb - rb <= -1000; c["rt"] += abs(va - ra) < 500 and abs(vb - rb) < 500
        print("b%d R %+d %+d | P 뒤 %+d %+d | V %+d %+d" % (b, ra, rb, P[("A", "train")], P[("B", "train")], va, vb))
    a = "보류" if c["base"] < 4 or c["prem"] < 4 else "확증" if c["keep"] >= 4 else "불확증" if c["lose"] >= 4 else "보류"
    g_ = "보존" if c["gen"] >= 4 else "약화(사례 한정)" if c["memo"] >= 4 else "보류"
    r_ = "보존" if c["rs"] >= 4 else "사례 한정" if c["rm"] >= 4 else "부분" if c["rp"] >= 4 else "고착" if c["rt"] >= 4 else "보류"
    if a == "확증":
        v = "기준선 확증(H112)" if g_ == "보존" and r_ == "보존" else "부분(유지만 확증 — 일반화 %s·반전 %s)" % (g_, r_)
    else:
        v = "불확증(H112-null)" if a == "불확증" else "보류"
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
