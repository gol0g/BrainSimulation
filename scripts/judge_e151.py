#!/usr/bin/env python3
"""E151 판정 — 기준 logs/E151/criteria_fixed.txt(보정·실행 전 고정). 학습 10줄 + 평가 25줄이 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄: 학습 "  e151 train A b10: => 사전 ... 사후 ... 보상 N || ..." / 평가 "  e151 b10 AB bad: => mod +0.1000".
η* = logs/E151/eta_star.txt("eta_star 0.9" 또는 "eta_star none"). 지표(1e-4 정수): eA1 = base(A) − base(none), eA = base(AB) − base(none),
rA = eA/eA1, eB = bad(AB) − bad(none), T = (eA − eA1)/eB. 판정 1 기전(T), 판정 2 능력(rA).
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
R20 = (1.0 - 1.0 / 12.0) ** 20
NEEDED = (("A", "base"), ("AB", "base"), ("AB", "bad"), ("none", "base"), ("none", "bad"))
TT = re.compile(r"^\s*e151 train (A|AB) b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")
TE = re.compile(r"^\s*e151 b(\d+) (A|AB|none) (base|bad): => mod ([-+0-9.]+)")
TE150 = re.compile(r"^\s*e150 b(\d+) none (base|bad): => mod ([-+0-9.]+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")
ETA = re.compile(r"KC→motor \[E109 R-STDP 4방향\]: .*eta=([0-9.eE+-]+),")


def stats(rows, switch=None):
    n = len(rows)
    act = rows[:, 6] >= 0
    if switch is None:
        rule = rows[:, 6] != rows[:, 2]
    else:
        rule = np.where(np.arange(n) < switch, rows[:, 6] != rows[:, 2], rows[:, 6] == rows[:, 2])
    eda = rows[:, 13:17]
    return {"n": n, "agree": float(np.mean((rows[act, 7] == 1) == rule[act])) if act.any() else float("nan"),
            "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "sabs": float(np.abs(rows[:, 12]).sum())}


def rawcheck(path, need_b, eta_star):
    """(과제 B 줄 맞음, 반사 0→0, eta 줄 = η*) 또는 None(파일 없음)."""
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None
    got = {}
    for ln in txt.splitlines():
        m = RW.match(ln)
        if m:
            got[m.group(1)] = (m.group(2), m.group(3))
    okb = ("[과제 B] 시행 1500 부터" in txt) if need_b else ("[과제 B]" not in txt)
    etas = [float(x) for x in ETA.findall(txt)]
    oke = eta_star is not None and len(etas) > 0 and all(abs(e - eta_star) < 1e-9 for e in etas)
    return okb, (set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values())), oke


def read_eta(path):
    try:
        tok = open(path, encoding="utf-8").read().split()
    except FileNotFoundError:
        return "missing"
    if len(tok) >= 2 and tok[0] == "eta_star":
        return None if tok[1] == "none" else float(tok[1])
    return "missing"


def judge(TR, EV, S, RC, REF, S150, eta_star):
    if eta_star == "missing":
        return ["[보정] eta_star.txt 없음 — **판정 보류, 수치 미출력**"], None
    if eta_star is None:
        return ["[보정] 해당 η 없음(학습 크기 미회복) — 본실험 없음, 판정: 보류(학습 크기 미회복)"], None
    miss = [(a, b) for a in ("A", "AB") for b in BRAINS if (a, b) not in TR or (a, b) not in S or RC.get((a, b)) is None]
    miss += [(b, w, s) for b in BRAINS for (w, s) in NEEDED if (b, w, s) not in EV]
    miss += [(b, s) for b in BRAINS for s in ("base", "bad") if (b, s) not in REF]
    miss += [b for b in BRAINS if b not in S150]
    if miss:
        return ["[측정 확인] 결측 %d — **판정 보류, 수치 미출력**" % len(miss)], None
    I = {k: int(round(v * 1e4)) for k, v in EV.items()}
    m_tr = sum(S[(a, b)]["res"] <= 1e-3 and S[(a, b)]["pre_ratio"] <= 1e-3 and S[(a, b)]["n"] == (1500 if a == "A" else 3000)
               and S[(a, b)]["agree"] == 1.0 and all(RC[(a, b)]) for a in ("A", "AB") for b in BRAINS)
    m_ref = sum(I[(b, "none", s)] == int(round(REF[(b, s)] * 1e4)) for b in BRAINS for s in ("base", "bad"))
    m_eta = sum(S[("A", b)]["sabs"] > S150[b] for b in BRAINS)
    ok = m_tr == 10 and m_ref == 10 and m_eta == 5
    eA1 = {b: I[(b, "A", "base")] - I[(b, "none", "base")] for b in BRAINS}
    eA = {b: I[(b, "AB", "base")] - I[(b, "none", "base")] for b in BRAINS}
    eB = {b: I[(b, "AB", "bad")] - I[(b, "none", "bad")] for b in BRAINS}
    p1 = sum(eA1[b] <= -2000 for b in BRAINS)
    p2 = sum(eB[b] >= 1500 for b in BRAINS)
    sep = sum(eB[b] >= 1500 and 10 * (eA[b] - eA1[b]) <= 8 * eB[b] for b in BRAINS)
    full = sum(eB[b] >= 1500 and (eA[b] - eA1[b]) >= eB[b] for b in BRAINS)
    keep = sum(eA1[b] < 0 and 2 * eA[b] <= eA1[b] for b in BRAINS)
    checks = ["[조작검증] η* %g · 학습 런 10 개(동결·시행·도파민전·보상-규칙·과제 B 줄·반사 0·eta 줄) %d/10 · 무학습 평가 = E150 %d/10 · "
              "eta 도달(Σ|Δg| > E150) %d/5 → %s · 전제 P1' 과제 A 학습 %d/5 · P2' 과제 B 학습 %d/5"
              % (eta_star, m_tr, m_ref, m_eta, "통과" if ok else "실패", p1, p2)]
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    elif p1 < 4 or p2 < 4:
        v1 = v2 = "보류(학습 크기 미회복 — %s)" % ("과제 A" if p1 < 4 else "과제 B")
    else:
        if sep >= 4:
            v1 = "H074(전이 감소) — 같은 학습 크기에서도 차단이 과제 B 의 과제 A 자극 전이를 줄인다: E150 유지는 약한 학습만의 산물 아님"
        elif full >= 4:
            v1 = "H074-mag(기본 수준 전이) — 학습 크기를 올리면 전이가 기본 수준: E150 의 낮은 전이는 약한 학습 영역 탓"
        else:
            v1 = "보류"
        if keep >= 4:
            v2 = "전체 강도 유지 — 학습 크기를 회복해도 과제 B 뒤 과제 A 효과 절반 이상 유지"
        elif 5 - keep >= 4:
            v2 = "전체 강도 간섭 — 학습 크기를 회복하면 과제 A 효과가 절반 미만"
        else:
            v2 = "보류"
    return checks, {"eA1": {b: eA1[b] / 1e4 for b in BRAINS}, "eA": {b: eA[b] / 1e4 for b in BRAINS}, "eB": {b: eB[b] / 1e4 for b in BRAINS},
                    "rA": {b: (eA[b] / eA1[b]) if eA1[b] else float("nan") for b in BRAINS},
                    "T": {b: ((eA[b] - eA1[b]) / eB[b]) if eB[b] else float("nan") for b in BRAINS},
                    "sabs_ratio": {b: S[("A", b)]["sabs"] / S150[b] if S150[b] else float("nan") for b in BRAINS},
                    "sep": sep, "full": full, "keep": keep, "p1": p1, "p2": p2, "ok": ok, "v1": v1, "v2": v2}


def report(checks, res, TR=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d: eA1 %+.4f | A→B 뒤 eA %+.4f(유지 몫 %.2f) | eB %+.4f | T %+.2f | Σ|Δg| E150 대비 %.2f배 | 학습 보상 A %d·AB %d"
              % (b, res["eA1"][b], res["eA"][b], res["rA"][b], res["eB"][b], res["T"][b], res["sabs_ratio"][b], TR[("A", b)]["rew"], TR[("AB", b)]["rew"]))
    print("전이 감소(T ≤ 0.8) %d/5 · 기본 수준 전이(T ≥ 1.0) %d/5 · 유지(rA ≥ 0.5) %d/5 | P1' %d/5 · P2' %d/5"
          % (res["sep"], res["full"], res["keep"], res["p1"], res["p2"]))
    print("판정 1(기전): %s" % res["v1"])
    print("판정 2(능력): %s" % res["v2"])


def load():
    TR, EV, S, RC, REF, S150 = {}, {}, {}, {}, {}, {}
    eta_star = read_eta(os.path.join(EXP, "logs", "E151", "eta_star.txt"))
    try:
        for ln in open(os.path.join(EXP, "E151.log"), encoding="utf-8", errors="replace"):
            m = TT.match(ln)
            if m:
                TR[(m.group(1), int(m.group(2)))] = {"pre": float(m.group(3)), "post": float(m.group(4)), "rew": int(m.group(5))}
                continue
            m = TE.match(ln)
            if m:
                EV[(int(m.group(1)), m.group(2), m.group(3))] = float(m.group(4))
    except FileNotFoundError:
        pass
    try:
        for ln in open(os.path.join(EXP, "E150.log"), encoding="utf-8", errors="replace"):
            m = TE150.match(ln)
            if m:
                REF[(int(m.group(1)), m.group(2))] = float(m.group(3))
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E150", "tr_A_b%d.npz" % b)
        if os.path.exists(f):
            S150[b] = float(np.abs(np.load(f)["rows"][:, 12]).sum())
    for a in ("A", "AB"):
        for b in BRAINS:
            f = os.path.join(EXP, "traces", "E151", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                S[(a, b)] = stats(np.load(f)["rows"], None if a == "A" else 1500)
            RC[(a, b)] = rawcheck(os.path.join(EXP, "logs", "E151", "train_%s_b%d.log" % (a, b)), a == "AB",
                                  eta_star if isinstance(eta_star, float) else None)
    return TR, EV, S, RC, REF, S150, eta_star


if __name__ == "__main__":
    TR, EV, S, RC, REF, S150, eta_star = load()
    c, r = judge(TR, EV, S, RC, REF, S150, eta_star)
    report(c, r, TR)
    sys.exit(0)
