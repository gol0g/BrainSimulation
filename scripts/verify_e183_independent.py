#!/usr/bin/env python3
"""E183 독립 대조 — judge_e183.py·rawcat.py 를 쓰지 않고 원 로그(logs/E183/rc_b{B}_{O|Z}.log)를 문자열 분해·분수(Fraction) 계산으로 다시 판정한다.
규칙은 logs/E183/criteria_fixed.txt(경계 비교 1e-4 단위 — 여기서는 분수를 반올림(절반 올림)한 정수).
실행: python3 scripts/verify_e183_independent.py (저장소 루트에서)"""
import os
import sys
from fractions import Fraction as Fr

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def q(x):
    v = x * 10000
    return (v.numerator * 2 + v.denominator) // (2 * v.denominator)   # floor(v + 1/2)


def load(path):
    t = open(path, encoding="utf-8", errors="replace").read().splitlines()
    if not any(ln.startswith("=> RAWCAT ") for ln in t):
        raise ValueError("완료 줄 없음")
    S, C, W = {}, {}, {}
    for ln in t:
        f = ln.split(" ")
        if f[0] == "RCKC":
            d = {}
            for x in f[1:]:
                k, v = x.split("=", 1)
                d[k] = v
            ids = frozenset(int(v) for v in d["ids"].split(",") if v)
            S[(d["tag"], d["side"], d["set"], d["cat"], int(d["i"]))] = ids
        elif f[0] == "RCCHK":
            d = dict(x.split("=", 1) for x in f[1:])
            C[d["tag"]] = d
        elif f[0] == "RCW":
            d = dict(x.split("=", 1) for x in f[1:])
            W[d["side"]] = d
    return S, C, W


def J(a, b):
    u = len(a | b)
    return Fr(len(a & b), u) if u else Fr(0)


def catset(lst):
    need = (len(lst) + 1) // 2
    cnt = {}
    for s in lst:
        for x in s:
            cnt[x] = cnt.get(x, 0) + 1
    return frozenset(x for x, c in cnt.items() if c >= need)


def sep(S, tag, side, test):
    tr = {c: [S[(tag, side, "train", c, i)] for i in range(6)] for c in "AB"}
    K = {c: catset(tr[c]) for c in "AB"}
    vals = []
    for c, o in (("A", "B"), ("B", "A")):
        for i in range(6):
            x = S[(tag, side, test, c, i)]
            vals.append(J(x, K[c]) - J(x, K[o]))
    return sum(vals, Fr(0)) / len(vals), J(K["A"], K["B"])


def main():
    ok = True
    rows = []
    try:
        for b in BRAINS:
            L = {a: load(os.path.join(EXP, "logs", "E183", "rc_b%d_%s.log" % (b, a))) for a in ("O", "Z")}
            for a in ("O", "Z"):
                for tag in ("pre", "pre2", "post"):
                    c = L[a][1][tag]
                    if not (q(Fr(c["feat_active_frac"])) >= 9000 and 100 * int(c["feat_inactive_spikes"]) <= int(c["feat_active_spikes"])
                            and c["feat_other_side"] == "0" and c["goodbad"] == "0" and c["g_same"] == "1" and c["npres"] == "144"):
                        ok = False
                for sd in "lr":
                    w = L[a][2][sd]
                    if a == "O" and not Fr(w["dg_abs"]) > 0:
                        ok = False
                    if a == "Z" and not (Fr(w["dg_abs"]) == 0 and Fr(w["frac_changed"]) == 0):
                        ok = False
            keys = [k for k in L["O"][0] if k[0] == "pre"]
            if len(keys) != 48:
                raise ValueError("형성 전 사례 수")
            pj = sum((J(L["O"][0][k], L["Z"][0][k]) for k in keys), Fr(0)) / len(keys)
            if q(pj) < 9500:
                ok = False
            for a in ("O", "Z"):
                rep = sum((abs(sep(L[a][0], "pre2", sd, "new")[0] - sep(L[a][0], "pre", sd, "new")[0]) for sd in "lr"), Fr(0)) / 2
                if q(rep) >= 250:
                    ok = False
                for sd in "lr":
                    ks = [len(L[a][0][k]) for k in keys if k[1] == sd]
                    if Fr(sum(ks), len(ks)) < 10:
                        ok = False
            gn = sum((sep(L["O"][0], "post", sd, "new")[0] - sep(L["Z"][0], "post", sd, "new")[0] for sd in "lr"), Fr(0)) / 2
            gt = sum((sep(L["O"][0], "post", sd, "train")[0] - sep(L["Z"][0], "post", sd, "train")[0] for sd in "lr"), Fr(0)) / 2
            jab = max(sep(L["O"][0], "post", sd, "new")[1] for sd in "lr")
            rows.append((b, q(gn), q(gt), q(jab)))
    except (OSError, KeyError, ValueError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    for b, gn, gt, jab in rows:
        print("b%d G_new %+d G_train %+d J_AB %d (1e-4)" % (b, gn, gt, jab))
    if not ok:
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    n3 = sum(j >= 8000 for _, _, _, j in rows)
    sup = sum(gn >= 500 and (gt <= 0 or 2 * gn >= gt) for _, gn, gt, _ in rows)
    mem = sum(gt >= 500 and 2 * gn < gt for _, gn, gt, _ in rows)
    harm = sum(gn <= -500 for _, gn, _, _ in rows)
    n1 = sum(gn < 500 for _, gn, _, _ in rows)
    v = ("붕괴(H104-null-3)" if n3 >= 4 else "지지(H104)" if sup >= 4 else "암기(H104-null-2)" if mem >= 4
         else "역효과" if harm >= 4 else "이득 없음(H104-null-1)" if n1 >= 4 else "보류")
    print("붕괴 %d · 지지 %d · 암기 %d · 역효과 %d · 이득 없음 %d" % (n3, sup, mem, harm, n1))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
