#!/usr/bin/env python3
"""E165 독립 대조 — judge_e165.py 를 쓰지 않고 원 로그·추적(E165)과 기준 원 로그(E161 D)에서 문자열 분해로 다시 계산한다.
판정 규칙은 logs/E165/criteria_fixed.txt 를 따른다(팔별 E161 규칙, 종합 NR·NU).
실행: python3 scripts/verify_e165_independent.py (저장소 루트에서)"""
import os
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
WANT = {"NR": ("1", ["100", "100", "100", "100"]), "NU": ("3", ["100", "300", "100", "300"])}


def rd(*p):
    return open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def first(t, head):
    for ln in t.splitlines():
        if ln.startswith(head):
            return ln
    raise ValueError(head)


def kv(ln):
    """'k=v' 토큰을 사전으로(같은 키가 둘이면 목록)."""
    out = {}
    for tok in ln.replace("|", " ").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out.setdefault(k, []).append(v)
    return out


def eff(t):
    pre = q4(first(t, "[사전]").split("변조폭")[1].split("**")[0])
    post = q4(first(t, "[사후]").split("변조폭")[1].split("**")[0])
    return post - pre


def main():
    try:
        ok = True
        eD = {}
        for b in BRAINS:
            eD[b] = eff(rd("logs", "E161", "D_b%d.log" % b))
            ok &= eD[b] <= -1000
        lv = {}
        for a, (mult, cnt) in WANT.items():
            sep = succ = null = 0
            for b in BRAINS:
                td = rd("logs", "E165", "%s_dev_b%d.log" % (a, b))
                x = kv(first(td, "[E165 노출]"))
                okx = (x["order"] == ["random"] and x["bad_mult"] == [mult] and [x[k][0] for k in ("good_l", "bad_l", "good_r", "bad_r")] == cnt
                       and q4(x["int_min"][0]) >= 5000 and q4(x["int_max"][0]) <= 9000 and abs(q4(x["int_mean"][0]) - 7000) <= 300
                       and q4(x["cyc_match"][0]) <= 4000)
                d = kv(first(td, "=> KCDEVOJA"))      # side=l … | side=r … 순서로 두 값씩
                oko = (sum(ln.startswith("[E160 종류 입력 Oja]") for ln in td.splitlines()) >= 1
                       and all(float(v) > 0 for k in ("dg_good", "dg_bad") for v in d[k]) and len(d["dg_good"]) == 2)
                oks = all(q4(d["sel_med"][i]) - q4(d["sel_med0"][i]) >= 1000 for i in (0, 1))
                to = rd("logs", "E165", "%s_ov_b%d.log" % (a, b))
                o = kv(first(to, "=> KCOVERLAP"))
                jl, jr = q4(o["jac"][0]), q4(o["jac"][1])
                tf = rd("logs", "E165", "%s_F_b%d.log" % (a, b))
                nld = lambda t: sum(ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln for ln in t.splitlines())
                okl = nld(to) >= 1 and nld(tf) >= 2
                eF = eff(tf)
                R = np.load(os.path.join(EXP, "traces", "E165", "tr_%s_F_b%d.npz" % (a, b)))["rows"]
                rs = (11.0 / 12.0) ** 20
                res = sum(abs(R[i, 21 + k] - rs * R[i, 13 + k]) for i in range(len(R)) for k in range(4)) / max(sum(abs(R[i, 13 + k]) for i in range(len(R)) for k in range(4)), 1e-12)
                pre = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
                alive = float(np.mean([abs(R[i, 13]) + abs(R[i, 14]) > 1.0 for i in range(len(R))]))
                okt = len(R) == 500 and res <= 1e-3 and pre <= 1e-3 and alive >= 0.9
                okb = okx and oko and oks and okl and okt
                ok &= okb
                s_ = jl <= 1000 and jr <= 1000
                sep += s_
                succ += s_ and 10 * eF <= 13 * eD[b]
                null += jl > 2500 or jr > 2500
                print("%s b%d 노출 %s 자카드 %d/%d e_F %+d e_D %+d | 조작 %s" % (a, b, "✓" if okx else "✗", jl, jr, eF, eD[b], "✓" if okb else "✗"))
            lv[a] = "성공" if succ >= 4 else ("분리만" if sep >= 4 else ("실패" if null >= 4 else "보류"))
            print("%s: %s (분리 %d · 성공 %d · 실패 %d)" % (a, lv[a], sep, succ, null))
    except (FileNotFoundError, ValueError, IndexError, KeyError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    if not ok:
        v = "보류(조작검증 실패)"
    elif lv["NR"] == "성공" and lv["NU"] == "성공":
        v = "견고(H088)"
    elif lv["NR"] == "성공":
        v = "빈도 의존(H088-freq)"
    elif lv["NR"] in ("실패", "분리만"):
        v = "순서·강도 의존(H088-null)"
    else:
        v = "보류"
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
