#!/usr/bin/env python3
"""E142 판정 — 기준 logs/E142/criteria_fixed.txt(실행 전 고정). 두 팔(F500·F1500) × 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
e = [사후] − [사전] 이식 변조폭(음수 = 교차 = 반사 반대), m = [사후](음수 = 학습된 교차가 반사를 이김).
조작검증 M1·M1b·M3 은 추적 npz(traces/E142/tr_{팔}_b*.npz), M4 는 런별 원 로그(logs/E142/{팔}_b*.log)의 [반사가중치] 줄.
rows 열(E139 와 같음): 7 correct 8~11 dg(교차·같은쪽·비선택·무활동) 12 dg_전체 13~16 e_da 17~20 dg_도파민전 21~24 e_end.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = {"F500": 500, "F1500": 1500, "NF1500": 1500}   # NF1500 = 무동결(수정 1, 23:04:43)
E119_PRE = {10: 0.4148, 11: 0.4195, 12: 0.3954, 13: 0.3773, 14: 0.4248}
E119_EFF = {10: -0.0022, 11: 0.0018, 12: 0.0079, 13: 0.0211, 14: -0.0114}
R_STAR = (1.0 - 1.0 / 12.0) ** 20
TL = re.compile(r"^\s*e142 (F500|F1500|NF1500) b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def stats(rows):
    rw = rows[:, 7] == 1
    A, B = rows[rw, 8].sum(), rows[rw, 9].sum()
    C, P = rows[~rw, 8].sum(), rows[~rw, 9].sum()
    eda, eend = rows[:, 13:17], rows[:, 21:25]
    n = len(rows)
    blk = [float((rows[i:i + 100, 8] - rows[i:i + 100, 9]).sum()) for i in range(0, n, 100)]
    return {"n": n,
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(eend - R_STAR * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "alive": float(np.mean((np.abs(rows[:, 13]) + np.abs(rows[:, 14])) > 1.0)),
            "BA": (B / A) if A else float("nan"), "CP": (C / P) if P else float("nan"),
            "dD": float((A + C) - (B + P)), "blk": blk}


def reflex_ok(path):
    """[반사가중치] good_food_to_motor_l·r 가 둘 다 25.0000→25.0000 이면 True, 줄이 없으면 None."""
    got = {}
    try:
        for ln in open(path, encoding="utf-8"):
            m = RW.match(ln)
            if m:
                got[m.group(1)] = (m.group(2), m.group(3))
    except FileNotFoundError:
        return None
    if set(got) != {"l", "r"}:
        return None
    return all(v == ("25.0000", "25.0000") for v in got.values())


def judge(T, S, RF):
    """T[arm][b] = {pre, post, rew}, S[arm][b] = stats, RF[arm][b] = reflex_ok."""
    miss = [(a, b) for a in ARMS for b in BRAINS if b not in T.get(a, {}) or b not in S.get(a, {}) or RF.get(a, {}).get(b) is None]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss], None
    checks, ok, ok_nf = [], True, True
    for a, n_exp in ARMS.items():
        if a == "NF1500":
            m1n = sum(S[a][b]["res"] > 0.1 for b in BRAINS)
            m2 = sum(abs(T[a][b]["pre"] - E119_PRE[b]) <= 0.002 + 1e-9 for b in BRAINS)
            m3 = sum(S[a][b]["n"] == n_exp and S[a][b]["pre_ratio"] <= 1e-3 for b in BRAINS)
            m4 = sum(bool(RF[a][b]) for b in BRAINS)
            ok_nf = (m1n == m2 == m3 == m4 == 5)
            checks.append("[NF1500 조작검증] M1 반대(동결 없음, 잔차 > 0.1) %d/5 · M2 출발점 %d/5 · M3 추적 %d/5 · M4 반사 25 불변 %d/5 %s"
                          % (m1n, m2, m3, m4, "통과" if ok_nf else "실패"))
            continue
        m1 = sum(S[a][b]["res"] <= 1e-3 for b in BRAINS)
        m1b = sum(S[a][b]["alive"] >= 0.9 for b in BRAINS)
        m2 = sum(abs(T[a][b]["pre"] - E119_PRE[b]) <= 0.002 + 1e-9 for b in BRAINS)
        m3 = sum(S[a][b]["n"] == n_exp and S[a][b]["pre_ratio"] <= 1e-3 for b in BRAINS)
        m4 = sum(bool(RF[a][b]) for b in BRAINS)
        good = (m1 == m1b == m2 == m3 == m4 == 5)
        checks.append("[%s 조작검증] M1 동결 %d/5 (잔차 최대 %.1e) · M1b 되돌림 %d/5 · M2 출발점 %d/5 · M3 추적 %d/5 · M4 반사 25 불변 %d/5 %s"
                      % (a, m1, max(S[a][b]["res"] for b in BRAINS), m1b, m2, m3, m4, "통과" if good else "실패"))
        ok &= good
    e = {a: {b: round(T[a][b]["post"] - T[a][b]["pre"], 6) for b in BRAINS} for a in ARMS}
    m = {a: {b: T[a][b]["post"] for b in BRAINS} for a in ARMS}
    l2 = [b for b in BRAINS if m["F1500"][b] <= -0.02 + 1e-9]
    opp = [b for b in BRAINS if e["F500"][b] <= -0.10 + 1e-9]
    nul = [b for b in BRAINS if abs(e["F500"][b]) < 0.03]
    acc = [b for b in BRAINS if e["F1500"][b] < e["F500"][b]]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif len(l2) >= 4:
        verdict = "L2 달성(H065) — 1,500시행 사후 변조폭이 교차 쪽(≤ −0.02, ≥4/5): 학습된 매핑이 반사 25 를 이긴다"
    elif len(opp) == 5:
        verdict = "반사를 거스름(H065-partial) — 500시행 효과 ≤ −0.10 5/5, 1,500시행에도 반사 쪽이 남음"
    elif len(nul) >= 4:
        verdict = "효과 없음(H065-null) — 동결해도 반사 25 에서 500시행 효과 |e| < 0.03(≥4/5)"
    else:
        verdict = "보류"
    nf_rev = [b for b in BRAINS if m["NF1500"][b] <= -0.02 + 1e-9]
    if not verdict.startswith("L2"):
        need = "해당 없음(L2 아님)"
    elif not ok_nf:
        need = "필요성 미결(NF1500 조작검증 실패)"
    elif len(BRAINS) - len(nf_rev) >= 4:
        need = "동결이 L2 에 필요(이 학습량에서) — 무동결 1,500시행 m > −0.02 %d/5" % (len(BRAINS) - len(nf_rev))
    elif len(nf_rev) >= 4:
        need = "학습량만으로도 L2 — 동결 불필요(무동결 m ≤ −0.02 %d/5)" % len(nf_rev)
    else:
        need = "필요성 미결"
    return checks, {"e": e, "m": m, "l2": l2, "opp": opp, "nul": nul, "acc": acc, "ok": ok, "ok_nf": ok_nf,
                    "nf_rev": nf_rev, "verdict": verdict, "need": need}


def report(checks, res, T=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        s5, s15 = S["F500"][b], S["F1500"][b]
        print("b%d [사전] %+.4f | F500 사후 %+.4f e %+.4f (E119 %+.4f) 보상 %d | F1500 사후 %+.4f e %+.4f 보상 %d | B/A %+.2f/%+.2f C/P %+.2f/%+.2f | 블록 ΔD 첫·끝 %+.3g→%+.3g"
              % (b, T["F500"][b]["pre"], res["m"]["F500"][b], res["e"]["F500"][b], E119_EFF[b], T["F500"][b]["rew"],
                 res["m"]["F1500"][b], res["e"]["F1500"][b], T["F1500"][b]["rew"], s5["BA"], s15["BA"], s5["CP"], s15["CP"],
                 s15["blk"][0], s15["blk"][-1]))
    print("F1500 m ≤ −0.02 %d/5 | F500 e ≤ −0.10 %d/5 | F500 |e| < 0.03 %d/5 | 누적(e1500 < e500) %d/5 | F500 e 평균 %+.4f · F1500 e 평균 %+.4f (E119 %+.4f)"
          % (len(res["l2"]), len(res["opp"]), len(res["nul"]), len(res["acc"]), sum(res["e"]["F500"].values()) / 5,
             sum(res["e"]["F1500"].values()) / 5, sum(E119_EFF.values()) / 5))
    print("NF1500(무동결): 사후 %s | e %s | 보상 %s"
          % (" ".join("%+.4f" % res["m"]["NF1500"][b] for b in BRAINS), " ".join("%+.4f" % res["e"]["NF1500"][b] for b in BRAINS),
             " ".join("%d" % T["NF1500"][b]["rew"] for b in BRAINS)))
    print("판정: %s" % res["verdict"])
    print("필요성(수정 1 해석 규칙): %s" % res["need"])


def load():
    T = {a: {} for a in ARMS}; S = {a: {} for a in ARMS}; RF = {a: {} for a in ARMS}
    try:
        # 2026-10-06 수리: 러너의 `cut -c1-200` 이 바이트 단위라 요약 줄 끝 한글을 반쯤 자른다(E142.log 3918바이트) —
        # 읽는 칸(사전·사후·보상)은 줄 앞부분이라 영향 없음, 디코딩만 관대하게.
        for ln in open(os.path.join(EXP, "E142.log"), encoding="utf-8", errors="replace"):
            mm = TL.match(ln)
            if mm:
                T[mm.group(1)][int(mm.group(2))] = {"pre": float(mm.group(3)), "post": float(mm.group(4)), "rew": int(mm.group(5))}
    except FileNotFoundError:
        pass
    for a in ARMS:
        for b in BRAINS:
            f = os.path.join(EXP, "traces", "E142", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                S[a][b] = stats(np.load(f)["rows"])
            RF[a][b] = reflex_ok(os.path.join(EXP, "logs", "E142", "%s_b%d.log" % (a, b)))
    return T, S, RF


if __name__ == "__main__":
    T, S, RF = load()
    c, r = judge(T, S, RF)
    report(c, r, T, S)
    sys.exit(0)
