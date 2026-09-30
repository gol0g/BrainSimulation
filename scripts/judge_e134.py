#!/usr/bin/env python3
"""E134 판정 — 기준 고정 logs/E134/criteria_fixed.txt. 발달 32 + 과제 128 이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): novel_lbal = 처음 보는 항목(4~7) 라벨 균형 정답률(%). 과제 tid = 같은 위치 같음/다름, tsh = 7칸 이동 같음/다름.
발달 corr = identity(같은 항목 같은 위치), shift = 반쪽2 가 반쪽1 의 7칸 이동. 독립 단위 = 배선(두 난수열 평균).
"""
import os
import re
import sys
from math import comb

EXP = "research/experiments"
WIRES = tuple(range(62, 78))
TSEEDS = (600, 601)
TD = re.compile(r"^\s*ds dev (corr|shift) w(\d+): => DEVHEBB .*? \|\| => DEVSHIFT seed=(\d+) env=(\w+) dev_shift=(\d+) \| 같은 위치: 일치형 ([0-9.]+) 불일치형 ([0-9.]+) \| \d+칸 이동 위치: 일치형 ([0-9.]+) 불일치형 ([0-9.]+)")
TT = re.compile(r"^\s*ds (corr|shift) (tid|tsh) w(\d+) t(\d+): => \[KC불러옴\] (\S+) \| .*? \|\| \[KC불러옴이동\] k=(\d+) \| .*?\(과제 sd_shift=(\d+)\) \|\| => SDLAB .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)")


def sign_p(d):
    n = sum(x > 0 for x in d); nz = sum(x != 0 for x in d)
    if nz == 0:
        return 1.0, n, nz
    k = max(n, nz - n)
    return min(1.0, 2 * sum(comb(nz, i) for i in range(k, nz + 1)) / 2 ** nz), n, nz


def med(v):
    v = sorted(v); n = len(v)
    return (v[n // 2 - 1] + v[n // 2]) / 2 if n % 2 == 0 else v[n // 2]


def judge(D, T):
    needD = [(e, w) for e in ("corr", "shift") for w in WIRES]
    needT = [(d, tk, w, t) for d in ("corr", "shift") for tk in ("tid", "tsh") for w in WIRES for t in TSEEDS]
    miss = [k for k in needD if k not in D] + [k for k in needT if k not in T]
    if miss:
        return ["[측정 확인] 결측 %d/160: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    checks, ok = [], True
    idf = {k: (D[k]["ms"] + D[k]["xs"]) / 2 for k in needD}; shf = {k: (D[k]["mk"] + D[k]["xk"]) / 2 for k in needD}
    a_ok = med([idf[("corr", w)] for w in WIRES]) >= 0.40 and med([shf[("corr", w)] for w in WIRES]) <= 0.05
    b_ok = med([shf[("shift", w)] for w in WIRES]) >= 0.40 and med([idf[("shift", w)] for w in WIRES]) <= 0.05
    checks.append("[측정 확인] identity 발달: 같은 위치 중앙 %.3f·이동 중앙 %.3f %s | shift 발달: 이동 중앙 %.3f·같은 위치 중앙 %.3f %s"
                  % (med([idf[("corr", w)] for w in WIRES]), med([shf[("corr", w)] for w in WIRES]), "✓" if a_ok else "✗",
                     med([shf[("shift", w)] for w in WIRES]), med([idf[("shift", w)] for w in WIRES]), "✓" if b_ok else "✗"))
    ok &= a_ok and b_ok
    lab = [k for k in needD if D[k]["env"] != k[0] or D[k]["seed"] != k[1] or D[k]["k"] != 7]
    fb = [k for k in needT if ("dev_%s_w%d.npz" % (k[0], k[2])) not in T[k]["file"] or T[k]["sh"] != (7 if k[1] == "tsh" else 0) or T[k]["k"] != 7]
    checks.append("[측정 확인] 발달 표기 %d/32, 과제 파일·이동 표기 %d/128" % (32 - len(lab), 128 - len(fb)))
    ok &= not lab and not fb
    wm = lambda d, tk, w: sum(T[(d, tk, w, t)]["nl"] for t in TSEEDS) / 2
    d_sh = [wm("shift", "tsh", w) - wm("corr", "tsh", w) for w in WIRES]
    d_id = [wm("corr", "tid", w) - wm("shift", "tid", w) for w in WIRES]
    p_sh, n_sh, nz_sh = sign_p(d_sh); p_id, n_id, nz_id = sign_p(d_id)
    m_sh = sum(d_sh) / 16; m_id = sum(d_id) / 16
    sh_ok = m_sh >= 10 and p_sh < 0.01; id_ok = m_id >= 10 and p_id < 0.01
    sh_no = m_sh < 5 or p_sh >= 0.05; id_no = m_id < 5 or p_id >= 0.05
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif sh_ok and id_ok:
        verdict = "이중 해리 지지 — 발달 경험의 대응 구조가 회로가 적용하는 관계를 정한다"
    elif sh_no and id_no:
        verdict = "해리 없음 — 경험 대응 구조가 적용 관계를 바꾸지 못함(설계가 관계를 정함)"
    elif sh_ok or id_ok:
        verdict = "보류(단일 해리)"
    else:
        verdict = "보류(혼재)"
    return checks, {"m_sh": m_sh, "p_sh": p_sh, "n_sh": n_sh, "nz_sh": nz_sh, "m_id": m_id, "p_id": p_id, "n_id": n_id, "nz_id": nz_id,
                    "idf": idf, "shf": shf, "ok": ok, "verdict": verdict}


def report(checks, res, D=None, T=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        g = lambda d, tk: "/".join("%.0f" % T[(d, tk, w, t)]["nl"] for t in TSEEDS)
        print("w%d 발달 id(같은 %.2f/이동 %.2f) sh(같은 %.2f/이동 %.2f) | 새 항목 tid: id발달 %s sh발달 %s | tsh: id발달 %s sh발달 %s" % (
            w, res["idf"][("corr", w)], res["shf"][("corr", w)], res["idf"][("shift", w)], res["shf"][("shift", w)],
            g("corr", "tid"), g("shift", "tid"), g("corr", "tsh"), g("shift", "tsh")))
    for d in ("corr", "shift"):
        for tk in ("tid", "tsh"):
            ks = [(d, tk, w, t) for w in WIRES for t in TSEEDS]
            print("발달 %s × 과제 %s: 새 항목 평균 %.1f%% 훈련 %.1f%% | ≥75%%: %d/32" % (d, tk, sum(T[k]["nl"] for k in ks) / 32, sum(T[k]["tl"] for k in ks) / 32, sum(T[k]["nl"] >= 75 for k in ks)))
    print("shift 과제: shift발달 − id발달 배선 평균 %+.1f%%p, 양수 %d/%d, p=%.5f" % (res["m_sh"], res["n_sh"], res["nz_sh"], res["p_sh"]))
    print("identity 과제: id발달 − shift발달 배선 평균 %+.1f%%p, 양수 %d/%d, p=%.5f" % (res["m_id"], res["n_id"], res["nz_id"], res["p_id"]))
    print("판정: %s" % res["verdict"])


def load():
    D, T = {}, {}
    try:
        for ln in open(os.path.join(EXP, "E134.log"), encoding="utf-8"):
            m = TD.match(ln)
            if m:
                g = m.groups()
                D[(g[0], int(g[1]))] = {"seed": int(g[2]), "env": g[3], "k": int(g[4]), "ms": float(g[5]), "xs": float(g[6]), "mk": float(g[7]), "xk": float(g[8])}
                continue
            m = TT.match(ln)
            if m:
                g = m.groups()
                T[(g[0], g[1], int(g[2]), int(g[3]))] = {"file": g[4], "k": int(g[5]), "sh": int(g[6]), "tl": float(g[7]), "nl": float(g[8])}
    except FileNotFoundError:
        pass
    return D, T


if __name__ == "__main__":
    D, T = load()
    c, r = judge(D, T)
    report(c, r, D, T)
    sys.exit(0)
