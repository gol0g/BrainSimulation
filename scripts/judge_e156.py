#!/usr/bin/env python3
"""E156 판정 — 기준 logs/E156/criteria_fixed.txt(실행 전 고정). 두 팔(F500·F1500) × 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
E142 규칙 그대로(판정 함수는 E142 와 같은 계산), 바뀐 것: M2 출발점 = 두 팔 [사전] 일치(형성 표현), M5 적재 줄 ≥ 2, NF1500 팔 없음.
e = [사후] − [사전] 이식 변조폭(음수 = 교차 = 반사 반대), m = [사후](음수 = 학습된 교차가 반사를 이김). 1e-4 정수 비교.
수정 1(00:44:40): W1500 팔(반사 W* — logs/E156/wstar.txt) 판정 2(맞춤) — 조작검증 M1·M1b·M3·M4w(W*→W*)·M5·MW([사전] − 같은 뇌 기본 [사전] ∈ [−0.05, +0.10]) 5/5,
W1500 m ≤ −0.02 ≥4/5 → '반사 발현을 맞춰도 학습이 이김'. 종합(H079)은 판정 1·2 조합. W1500 이 다 모이기 전에는 수치를 출력하지 않는다.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = {"F500": 500, "F1500": 1500}
E142_F1500_M = {10: 0.1026, 11: 0.1141, 12: 0.0953, 13: 0.0833, 14: 0.1266}
ARMS_W = {"W1500": 1500}
E142_PRE = {10: 0.4148, 11: 0.4195, 12: 0.3954, 13: 0.3773, 14: 0.4248}          # 같은 뇌 기본 표현 반사 25 [사전](E142 F500 원 로그)
E142_F1500_E = {10: -0.3122, 11: -0.3054, 12: -0.3001, 13: -0.2940, 14: -0.2982}  # 기본 표현 1,500시행 효과(부지표)
R_STAR = (1.0 - 1.0 / 12.0) ** 20
TL = re.compile(r"^\s*e156 (F500|F1500|W1500) b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+) \|\| 적재 (\d+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def stats(rows):
    rw = rows[:, 7] == 1
    A, B = rows[rw, 8].sum(), rows[rw, 9].sum()
    C, P = rows[~rw, 8].sum(), rows[~rw, 9].sum()
    eda, eend = rows[:, 13:17], rows[:, 21:25]
    n = len(rows)
    return {"n": n, "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(eend - R_STAR * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "alive": float(np.mean((np.abs(rows[:, 13]) + np.abs(rows[:, 14])) > 1.0)),
            "BA": (B / A) if A else float("nan"), "CP": (C / P) if P else float("nan"),
            "blk": [float((rows[i:i + 100, 8] - rows[i:i + 100, 9]).sum()) for i in range(0, n, 100)]}


def reflex_ok(path, w="25.0000"):
    got = {}
    try:
        for ln in open(path, encoding="utf-8", errors="replace"):
            m = RW.match(ln)
            if m:
                got[m.group(1)] = (m.group(2), m.group(3))
    except FileNotFoundError:
        return None
    if set(got) != {"l", "r"}:
        return None
    return all(v == (w, w) for v in got.values())


def i4(x):
    return int(round(x * 1e4))


def judge(T, S, RF):
    miss = [(a, b) for a in ARMS for b in BRAINS if b not in T.get(a, {}) or b not in S.get(a, {}) or RF.get(a, {}).get(b) is None]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss], None
    checks, ok = [], True
    m2 = sum(abs(i4(T["F500"][b]["pre"]) - i4(T["F1500"][b]["pre"])) <= 20 for b in BRAINS)
    for a, n_exp in ARMS.items():
        m1 = sum(S[a][b]["res"] <= 1e-3 for b in BRAINS)
        m1b = sum(S[a][b]["alive"] >= 0.9 for b in BRAINS)
        m3 = sum(S[a][b]["n"] == n_exp and S[a][b]["pre_ratio"] <= 1e-3 for b in BRAINS)
        m4 = sum(bool(RF[a][b]) for b in BRAINS)
        m5 = sum(T[a][b]["load"] >= 2 for b in BRAINS)
        good = (m1 == m1b == m3 == m4 == m5 == m2 == 5)
        checks.append("[%s 조작검증] M1 동결 %d/5 · M1b 되돌림 %d/5 · M2 두 팔 출발점 %d/5 · M3 추적 %d/5 · M4 반사 25 불변 %d/5 · M5 적재 %d/5 %s"
                      % (a, m1, m1b, m2, m3, m4, m5, "통과" if good else "실패"))
        ok &= good
    e = {a: {b: i4(T[a][b]["post"]) - i4(T[a][b]["pre"]) for b in BRAINS} for a in ARMS}
    m = {a: {b: i4(T[a][b]["post"]) for b in BRAINS} for a in ARMS}
    l2 = [b for b in BRAINS if m["F1500"][b] <= -200]
    opp = [b for b in BRAINS if e["F500"][b] <= -1000]
    nul = [b for b in BRAINS if abs(e["F500"][b]) < 300]
    acc = [b for b in BRAINS if e["F1500"][b] < e["F500"][b]]
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif len(l2) >= 4:
        verdict = "L2 달성(H079) — 형성 표현에서 1,500시행 사후 변조폭이 교차 쪽(≤ −0.02, ≥4/5): 학습된 매핑이 반사 25 를 이긴다"
    elif len(opp) == 5:
        verdict = "반사를 거스름(H079-partial) — 500시행 효과 ≤ −0.10 5/5, 1,500시행에도 반사 쪽이 남음"
    elif len(nul) >= 4:
        verdict = "효과 없음(H079-null) — 500시행 효과 |e| < 0.03(≥4/5)"
    else:
        verdict = "보류"
    return checks, {"e": e, "m": m, "l2": l2, "opp": opp, "nul": nul, "acc": acc, "ok": ok, "verdict": verdict}


def report(checks, res, T=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d [사전] %+.4f | F500 사후 %+.4f e %+.4f 보상 %d | F1500 사후 %+.4f e %+.4f 보상 %d (기본 E142 F1500 사후 %+.4f) | B/A %+.2f/%+.2f C/P %+.2f/%+.2f"
              % (b, T["F500"][b]["pre"], res["m"]["F500"][b] / 1e4, res["e"]["F500"][b] / 1e4, T["F500"][b]["rew"],
                 res["m"]["F1500"][b] / 1e4, res["e"]["F1500"][b] / 1e4, T["F1500"][b]["rew"], E142_F1500_M[b],
                 S["F500"][b]["BA"], S["F1500"][b]["BA"], S["F500"][b]["CP"], S["F1500"][b]["CP"]))
    print("F1500 m ≤ −0.02 %d/5 | F500 e ≤ −0.10 %d/5 | F500 |e| < 0.03 %d/5 | 누적(e1500 < e500) %d/5"
          % (len(res["l2"]), len(res["opp"]), len(res["nul"]), len(res["acc"])))
    print("판정 1: %s" % res["verdict"])


def judge_w(T, S, RF, wstar):
    """수정 1 판정 2(맞춤). wstar None → 보정 실패(판정 2 없음)."""
    if wstar is None:
        return ["[W1500] 보정 실패 — 판정 2 없음"], {"verdict": "없음(보정 실패)", "ok": False}
    a = "W1500"
    miss = [b for b in BRAINS if b not in T.get(a, {}) or b not in S.get(a, {}) or RF.get(a, {}).get(b) is None]
    if miss:
        return ["[W1500] 결측 %s — 판정 2 보류, 수치 미출력" % miss], {"verdict": "보류(결측)", "ok": False}
    m1 = sum(S[a][b]["res"] <= 1e-3 for b in BRAINS)
    m1b = sum(S[a][b]["alive"] >= 0.9 for b in BRAINS)
    m3 = sum(S[a][b]["n"] == ARMS_W[a] and S[a][b]["pre_ratio"] <= 1e-3 for b in BRAINS)
    m4 = sum(bool(RF[a][b]) for b in BRAINS)
    m5 = sum(T[a][b]["load"] >= 2 for b in BRAINS)
    dpre = {b: i4(T[a][b]["pre"]) - i4(E142_PRE[b]) for b in BRAINS}
    mw = sum(-500 <= dpre[b] <= 1000 for b in BRAINS)
    ok = (m1 == m1b == m3 == m4 == m5 == mw == 5)
    chk = ("[W1500 조작검증, W*=%d] M1 동결 %d/5 · M1b 되돌림 %d/5 · M3 추적 %d/5 · M4w 반사 W* 불변 %d/5 · M5 적재 %d/5 · MW 반사 발현 맞춤 %d/5 %s"
           % (wstar, m1, m1b, m3, m4, m5, mw, "통과" if ok else "실패"))
    m = {b: i4(T[a][b]["post"]) for b in BRAINS}
    e = {b: i4(T[a][b]["post"]) - i4(T[a][b]["pre"]) for b in BRAINS}
    win = [b for b in BRAINS if m[b] <= -200]
    if not ok:
        v = "보류(조작검증 실패)"
    elif len(win) >= 4:
        v = "반사 발현을 맞춰도 학습이 이김"
    else:
        v = "반사 발현을 맞추면 L2 아님"
    return [chk], {"verdict": v, "ok": ok, "m": m, "e": e, "dpre": dpre, "win": win, "wstar": wstar}


def combine(v1, v2):
    """종합(H079) — 수정 1 규칙. v1 판정 1 문자열, v2 판정 2 문자열."""
    head = v1.split(" —")[0]
    if v2.startswith("없음") or v2.startswith("보류"):
        return "판정 1(%s) + 식별 불가(반사 발현 약화 혼입 — 판정 2 %s)" % (head, v2)
    l2 = v1.startswith("L2 달성")
    win = v2 == "반사 발현을 맞춰도 학습이 이김"
    if l2 and win:
        return "H079 지지 — L2 달성(반사 발현을 기본에 맞춰도 형성 표현 학습이 이김)"
    if l2:
        return "H079 부분 — L2 는 반사 발현 약화 동반(학습 강화만으로는 아님)"
    if win:
        return "보류(예측 밖 — 판정 1 L2 아님, 판정 2 이김)"
    return "판정 1 그대로(%s) — L2 미해결" % head


def report_w(checks, r2, res):
    for c in checks:
        print(c)
    if "m" not in r2:
        print("판정 2: %s" % r2["verdict"])
        return
    for b in BRAINS:
        ew, e1 = r2["e"][b], res["e"]["F1500"][b]
        print("b%d W1500 [사전] %+.4f (기본 %+.4f, 차 %+.4f) 사후 %+.4f e %+.4f | 기본 E142 F1500 e %+.4f (비 %.2f) | 형성 F1500 e %+.4f (W/F 비 %.2f)"
              % (b, (r2["dpre"][b] + i4(E142_PRE[b])) / 1e4, E142_PRE[b], r2["dpre"][b] / 1e4, r2["m"][b] / 1e4, ew / 1e4,
                 E142_F1500_E[b], (ew / 1e4) / E142_F1500_E[b], e1 / 1e4, (ew / e1) if e1 else float("nan")))
    print("W1500 m ≤ −0.02 %d/5" % len(r2["win"]))
    print("판정 2: %s" % r2["verdict"])


def read_wstar():
    try:
        mm = re.match(r"^W\*=(\d+) ", open(os.path.join(EXP, "logs", "E156", "wstar.txt"), encoding="utf-8").read())
    except FileNotFoundError:
        return None
    return int(mm.group(1)) if mm else None


def load():
    T = {a: {} for a in list(ARMS) + list(ARMS_W)}; S = {a: {} for a in list(ARMS) + list(ARMS_W)}; RF = {a: {} for a in list(ARMS) + list(ARMS_W)}
    wstar = read_wstar()
    try:
        for ln in open(os.path.join(EXP, "E156.log"), encoding="utf-8", errors="replace"):
            mm = TL.match(ln)
            if mm:
                T[mm.group(1)][int(mm.group(2))] = {"pre": float(mm.group(3)), "post": float(mm.group(4)), "rew": int(mm.group(5)), "load": int(mm.group(6))}
    except FileNotFoundError:
        pass
    for a in list(ARMS) + (list(ARMS_W) if wstar is not None else []):
        w = "%.4f" % (wstar if a in ARMS_W else 25)
        for b in BRAINS:
            f = os.path.join(EXP, "traces", "E156", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                S[a][b] = stats(np.load(f)["rows"])
            RF[a][b] = reflex_ok(os.path.join(EXP, "logs", "E156", "%s_b%d.log" % (a, b)), w)
    return T, S, RF, wstar


if __name__ == "__main__":
    T, S, RF, W = load()
    c, r = judge(T, S, RF)
    c2, r2 = judge_w(T, S, RF, W)
    if r is None or r2["verdict"] == "보류(결측)":
        for x in (c if r is None else []) + c2:
            print(x)
        print("판정 보류 — 결측, 수치 미출력")
        sys.exit(0)
    report(c, r, T, S)
    report_w(c2, r2, r)
    print("종합(H079): %s" % combine(r["verdict"], r2["verdict"]))
    sys.exit(0)
