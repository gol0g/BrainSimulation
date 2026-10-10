#!/usr/bin/env python3
"""E172 판정(유지) — 망 안 형성 표현의 간섭 하 유지 독립 확증(뇌 16~20, E161 형성 가중치). 기준 logs/E172/criteria_fixed.txt. judge_e162.py 를 경로·뇌·가설 번호만 바꿔 옮김. 학습 10줄 + 평가 25줄이 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄: 학습 "  e172 train A b10: => 사전 ... 사후 ... 보상 N || ..." / "  e172 train AB b10: => ...", 평가 "  e172 b16 AB bad: => mod +0.1000"
(가중치 A·AB·none × 자극 base·bad; A 는 base 만). eA1 = base(A) − base(none), eA = base(AB) − base(none), rA = eA/eA1, eB = bad(AB) − bad(none),
T = (eA − eA1)/eB. 1e-4 정수. 적재 검증: 원 학습 로그 ≥ 2줄, 원 평가 로그 ≥ 1줄 '[E153 종류 입력 적재] ... 검증 일치'.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
R20 = (1.0 - 1.0 / 12.0) ** 20
NEEDED = (("A", "base"), ("AB", "base"), ("AB", "bad"), ("none", "base"), ("none", "bad"))
TT = re.compile(r"^\s*e172 train (A|AB) b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")
TE = re.compile(r"^\s*e172 b(\d+) (A|AB|none) (base|bad): => mod ([-+0-9.]+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")
LD = re.compile(r"^\[E153 종류 입력 적재\].*검증 일치")


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
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12))}


def rawcheck(path, need_b):
    """(과제 B 줄 맞음, 반사 0→0, 적재 줄 수) 또는 None."""
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return None
    got = {}
    nld = 0
    for ln in txt.splitlines():
        m = RW.match(ln)
        if m:
            got[m.group(1)] = (m.group(2), m.group(3))
        nld += bool(LD.match(ln))
    okb = ("[과제 B] 시행 1500 부터" in txt) if need_b else ("[과제 B]" not in txt)
    return okb, (set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values())), nld


def judge(TR, EV, S, RC, EL):
    miss = [(a, b) for a in ("A", "AB") for b in BRAINS if (a, b) not in TR or (a, b) not in S or RC.get((a, b)) is None]
    miss += [(b, w, s) for b in BRAINS for (w, s) in NEEDED if (b, w, s) not in EV or EL.get((b, w, s)) is None]
    if miss:
        return ["[측정 확인] 결측 %d — **판정 보류, 수치 미출력**" % len(miss)], None
    I = {k: int(round(v * 1e4)) for k, v in EV.items()}
    m_tr = sum(S[(a, b)]["res"] <= 1e-3 and S[(a, b)]["pre_ratio"] <= 1e-3 and S[(a, b)]["n"] == (1500 if a == "A" else 3000)
               and S[(a, b)]["agree"] == 1.0 and RC[(a, b)][0] and RC[(a, b)][1] and RC[(a, b)][2] >= 2 for a in ("A", "AB") for b in BRAINS)
    m_ev = sum(EL[(b, w, s)] >= 1 for b in BRAINS for (w, s) in NEEDED)
    ok = m_tr == 10 and m_ev == 25
    eA1 = {b: I[(b, "A", "base")] - I[(b, "none", "base")] for b in BRAINS}
    eA = {b: I[(b, "AB", "base")] - I[(b, "none", "base")] for b in BRAINS}
    eB = {b: I[(b, "AB", "bad")] - I[(b, "none", "bad")] for b in BRAINS}
    p1 = sum(eA1[b] <= -2000 for b in BRAINS)
    p2 = sum(eB[b] >= 1500 for b in BRAINS)
    keep = sum(eA1[b] < 0 and 2 * eA[b] <= eA1[b] for b in BRAINS)
    big = sum(eA1[b] < 0 and 5 * eA[b] <= 4 * eA1[b] for b in BRAINS)    # E172 추가 부판정: rA ≥ 0.80(K95 크기 서술 확증)
    checks =["[조작검증] 학습 런 10 개(동결·시행·도파민전·보상-규칙·과제 B 줄·반사 0·적재 ≥2) %d/10 · 평가 적재 %d/25 → %s · 전제 P1 %d/5 · P2 %d/5"
              % (m_tr, m_ev, "통과" if ok else "실패", p1, p2)]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif p1 < 4:
        verdict = "보류(형성 표현 과제 A 전체 강도 미학습)"
    elif p2 < 4:
        verdict = "보류(과제 B 미학습)"
    elif keep >= 4:
        verdict = "유지(H095) — 경험 형성 표현에서 전체 모델이 상충 연합을 함께 담는다(K51 의 전체 모델 보존, 전체 강도)"
    elif 5 - keep >= 4:
        verdict = "간섭(H095-null) — KC 표현을 갈라도 간섭: 원인은 KC 겹침 밖"
    else:
        verdict = "보류"
    return checks, {"eA1": {b: eA1[b] / 1e4 for b in BRAINS}, "eA": {b: eA[b] / 1e4 for b in BRAINS},
                    "rA": {b: (eA[b] / eA1[b]) if eA1[b] else float("nan") for b in BRAINS}, "eB": {b: eB[b] / 1e4 for b in BRAINS},
                    "T": {b: ((eA[b] - eA1[b]) / eB[b]) if eB[b] else float("nan") for b in BRAINS},
                    "keep": keep, "big": big, "p1": p1, "p2": p2, "ok": ok, "verdict": verdict}


def report(checks, res, TR=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d 형성: 과제 A 단독 eA1 %+.4f | A→B 뒤 eA %+.4f(유지 몫 %.2f) | 과제 B eB %+.4f | T %+.2f | 학습 보상 A %d·AB %d"
              % (b, res["eA1"][b], res["eA"][b], res["rA"][b], res["eB"][b], res["T"][b], TR[("A", b)]["rew"], TR[("AB", b)]["rew"]))
    print("유지(rA ≥ 0.5) %d/5 | P1 %d/5 | P2 %d/5" % (res["keep"], res["p1"], res["p2"]))
    print("판정: %s" % res["verdict"])
    print("부판정 크기(rA ≥ 0.80 — K95 크기 서술) %d/5 → %s" % (res["big"], "확증 범위" if res["big"] >= 4 else "개발 뇌 한정"))


def load():
    TR, EV, S, RC, EL = {}, {}, {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E172.log"), encoding="utf-8", errors="replace"):
            m = TT.match(ln)
            if m:
                TR[(m.group(1), int(m.group(2)))] = {"pre": float(m.group(3)), "post": float(m.group(4)), "rew": int(m.group(5))}
                continue
            m = TE.match(ln)
            if m:
                EV[(int(m.group(1)), m.group(2), m.group(3))] = float(m.group(4))
    except FileNotFoundError:
        pass
    for a in ("A", "AB"):
        for b in BRAINS:
            f = os.path.join(EXP, "traces", "E172", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                S[(a, b)] = stats(np.load(f)["rows"], None if a == "A" else 1500)
            RC[(a, b)] = rawcheck(os.path.join(EXP, "logs", "E172", "train_%s_b%d.log" % (a, b)), a == "AB")
    for b in BRAINS:
        for (w, s) in NEEDED:
            try:
                txt = open(os.path.join(EXP, "logs", "E172", "ev_b%d_%s_%s.log" % (b, w, s)), encoding="utf-8", errors="replace").read()
                EL[(b, w, s)] = sum(bool(LD.match(ln)) for ln in txt.splitlines())
            except FileNotFoundError:
                EL[(b, w, s)] = None
    return TR, EV, S, RC, EL


if __name__ == "__main__":
    TR, EV, S, RC, EL = load()
    c, r = judge(TR, EV, S, RC, EL)
    report(c, r, TR)
    sys.exit(0)
