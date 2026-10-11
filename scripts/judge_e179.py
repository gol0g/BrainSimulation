#!/usr/bin/env python3
"""E179 판정 — 대칭 맥락(맥락 A = zz_ctx2 정답 교차, B = zz_ctx 정답 같은 쪽, 둘 다 200 → KC p 0.10 w 4·독립 연결, motor 기저값 −4000)으로 쌍조건 변별을 배우는가.
E178 판정을 옮기고 맥락 끔·켬을 A·B 로 바꿈. 기준 logs/E179/criteria_fixed.txt.
뇌 10~14 자료(kcctx 5 + 학습 5 + 평가 20)가 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄(E179.log): "  e179 kcctx b10: => KCCTX ..." / "  e179 train b10: => 사전 .. 사후 .. 보상 N || 맥락 켬 n || 적재 K || ..." / "  e179 b10 learn A: => mod -0.1000".
e_A = m(learn A) − m(none A), e_B = m(learn B) − m(none B). 1e-4 정수.
판정(적용 순서): 조작검증 실패 → 보류. 획득(H100) e_A ≤ −0.10 이면서 e_B ≥ +0.10 ≥ 4/5 → 요소식(H100-null) |e_B − e_A| < 0.10 ≥ 4/5
→ 부분 한 맥락만(B만: e_B ≥ +0.10·e_A > −0.10 → H100-bias 쪽, A만: e_A ≤ −0.10·e_B < +0.10) ≥ 4/5 → 그 밖 보류.
조작검증(뇌마다, 모두): kcctx B 집단 발화 끔(= A 자리) 0·켬 > 0 · A 집단 발화 끔 > 0·켬 0 / 학습 로그 '[맥락 입력 2]' 줄 / 맥락 B 수 1,350~1,650 / 추적 3,000행·열 37(1 = B)·
맥락별 보상-규칙 일치 1.000·동결 잔차 ≤ 1e-3·도파민 전 ≤ 1e-3 / 반사 0→0 / 학습 적재 줄 ≥ 2 / 평가 적재 줄 ≥ 1·B 평가에만 '[맥락 평가] ctx=켬'·A 평가에만 '[맥락 평가 A]' /
동작점: 학습·평가 로그 모두 '[motor 동작점]' 줄 ≥ 1·값 = −4000 / 비포화: 평가 4개 모두 진단 motor 4개가 모두 ≥ 0.66 이 아님 / 결정 표현: 맥락 A·B 탐욕 시행 각각 |v| ≤ 0.02 비율 ≤ 0.50.
실행: python3 scripts/judge_e179.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
R20 = (1.0 - 1.0 / 12.0) ** 20
EVS = (("learn", "A"), ("learn", "B"), ("none", "A"), ("none", "B"))
TT = re.compile(r"^\s*e179 train b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+) \|\| 맥락 켬 (\d+) \|\| 적재 (\d+)")
TE = re.compile(r"^\s*e179 b(\d+) (learn|none) (A|B): => mod ([-+0-9.]+)")
TK = re.compile(r"^\s*e179 kcctx b(\d+): => KCCTX (.*)$")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")
LD = re.compile(r"^\[E153 종류 입력 적재\].*검증 일치")
CTXN = re.compile(r"^\[맥락 과제\] 시행 (\d+) 중 맥락 켬 (\d+)", re.M)
SP = re.compile(r"맥락 집단 발화 끔 (\d+) 켬 (\d+)")
SP2 = re.compile(r"맥락2 집단 발화 끔 (\d+) 켬 (\d+)")
C2L = "[맥락 입력 2] 맥락 전용 집단 zz_ctx2"
X179 = -4000.0
TONIC = re.compile(r"^\s*\[motor 동작점\] motor 좌·우 Ioffset 기저값 ([-0-9.]+)", re.M)
DG = re.compile(r"^\[E171 평가 진단\] variant=\w+ side=(left|right) n=\d+ motor L/R ([0-9.]+)/([0-9.]+)", re.M)
SAT = 6600          # 1e-4 정수 — 포화 경계 0.66(E177 정정 1 과 같음)
VSMALL = 0.02       # 결정 조향 '작음'(학습 판단 임계와 같음)
VFRAC_MAX = 0.50    # 맥락 켬 탐욕 시행 중 작은 |v| 비율 상한


def i4(x):
    return int(round(float(x) * 1e4))


def bicond_stats(rows):
    """맥락별 규칙: 열 37 = 0 → 교차(실행 ≠ 쪽), 1 → 같은 쪽(실행 = 쪽). 열 6 실행(−1 = 행동 창 없음), 7 보상, 2 쪽, 3 탐색, 5 결정 조향 v.
    dec_on·dec_off = 맥락 켬·끔 탐욕 시행 중 |v| ≤ 0.02 비율(해당 시행 0 이면 nan)."""
    n = len(rows)
    if rows.shape[1] < 38:
        return {"n": n, "has_ctx": False, "agree": float("nan"), "res": float("nan"), "pre_ratio": float("nan"), "frac_ctx": float("nan"),
                "dec_on": float("nan"), "dec_off": float("nan")}
    act = rows[:, 6] >= 0
    ctx = rows[:, 37] == 1
    greedy = rows[:, 3] == 0
    small = np.abs(rows[:, 5]) <= VSMALL
    rule = np.where(ctx, rows[:, 6] == rows[:, 2], rows[:, 6] != rows[:, 2])
    eda = rows[:, 13:17]
    gon, goff = greedy & ctx, greedy & ~ctx
    return {"n": n, "has_ctx": True, "frac_ctx": float(ctx.mean()),
            "agree": float(np.mean((rows[act, 7] == 1) == rule[act])) if act.any() else float("nan"),
            "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "dec_on": float(small[gon].mean()) if gon.any() else float("nan"),
            "dec_off": float(small[goff].mean()) if goff.any() else float("nan")}


def train_raw(txt):
    """(맥락 B 수 또는 None, 시행 수 또는 None, 반사 0→0, 적재 줄 수, 동작점 기저값 목록, '[맥락 입력 2]' 줄 수)."""
    m = CTXN.search(txt)
    got = {}
    nld = 0
    for ln in txt.splitlines():
        r = RW.match(ln)
        if r:
            got[r.group(1)] = (r.group(2), r.group(3))
        nld += bool(LD.match(ln))
    return (int(m.group(2)) if m else None, int(m.group(1)) if m else None,
            set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values()), nld,
            [float(x) for x in TONIC.findall(txt)], txt.count(C2L))


def eval_raw(t):
    """평가 로그 하나 → (적재 줄 수, B 표지 '[맥락 평가] ctx=켬' 있음, 동작점 기저값 목록, motor 발화율 4개(좌 자극 L·R, 우 자극 L·R) 또는 None, A 표지 '[맥락 평가 A]' 있음)."""
    dg = {d.group(1): (float(d.group(2)), float(d.group(3))) for d in DG.finditer(t)}
    return (sum(bool(LD.match(x)) for x in t.splitlines()), "[맥락 평가] ctx=켬" in t, [float(x) for x in TONIC.findall(t)],
            (dg["left"] + dg["right"]) if set(dg) == {"left", "right"} else None, "[맥락 평가 A]" in t)


def tonic_ok(vals, X):
    return X is not None and len(vals) >= 1 and all(abs(v - X) <= 1e-6 for v in vals)


def judge(K, TRN, EV, S, RAW, EL, X=X179):
    """K[b] = kcctx 줄 꼬리, TRN[b] = 학습 요약, EV[(b, w, c)] = mod(c = A·B), S[b] = bicond_stats(dec_on = B, dec_off = A), RAW[b] = train_raw, EL[(b, w, c)] = eval_raw."""
    miss = [b for b in BRAINS if b not in K or b not in TRN or b not in S or RAW.get(b) is None]
    miss += [(b, w, c) for b in BRAINS for (w, c) in EVS if (b, w, c) not in EV or EL.get((b, w, c)) is None]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % (miss[:6],)], None
    m = {}
    for b in BRAINS:
        sp, sp2 = SP.search(K[b]), SP2.search(K[b])
        st, rw = S[b], RAW[b]
        ev4 = [EL[(b, w, c)] for (w, c) in EVS]
        m[b] = {"kc": bool(sp) and int(sp.group(1)) == 0 and int(sp.group(2)) > 0 and bool(sp2) and int(sp2.group(1)) > 0 and int(sp2.group(2)) == 0,
                "ctx2": rw[5] >= 1,
                "nctx": rw[0] is not None and rw[1] == 3000 and 1350 <= rw[0] <= 1650 and rw[0] == TRN[b]["nctx"],
                "trace": st["has_ctx"] and st["n"] == 3000 and st["agree"] == 1.0 and st["res"] <= 1e-3 and st["pre_ratio"] <= 1e-3,
                "refl": rw[2], "load": rw[3] >= 2 and TRN[b]["load"] >= 2,
                "ev": all(EL[(b, w, c)][0] >= 1 and EL[(b, w, c)][1] == (c == "B") and EL[(b, w, c)][4] == (c == "A") for (w, c) in EVS),
                "tonic": tonic_ok(rw[4], X) and all(tonic_ok(e[2], X) for e in ev4),
                "unsat": all(e[3] is not None for e in ev4) and all(not all(i4(x) >= SAT for x in e[3]) for e in ev4),
                "dec": all(st.get(k) == st.get(k) and st[k] <= VFRAC_MAX for k in ("dec_on", "dec_off"))}
    keys = ("kc", "ctx2", "nctx", "trace", "refl", "load", "ev", "tonic", "unsat", "dec")
    ok = all(all(v.values()) for v in m.values())
    cnt = {k: sum(m[b][k] for b in BRAINS) for k in keys}
    checks = ["[조작검증] 두 맥락 집단 발화(A·B 자리) %d/5 · 맥락 입력 2 줄 %d/5 · 맥락 B 수 %d/5 · 추적(3,000·열37·맥락별 규칙 일치·동결·도파민전) %d/5 · 반사 0 %d/5 · 학습 적재 %d/5"
              " · 평가(적재·A·B 표지) %d/5 · 동작점(X=%g) %d/5 · 평가 비포화 %d/5 · A·B 결정 표현 %d/5 → %s"
              % (cnt["kc"], cnt["ctx2"], cnt["nctx"], cnt["trace"], cnt["refl"], cnt["load"], cnt["ev"], X, cnt["tonic"], cnt["unsat"], cnt["dec"], "통과" if ok else "실패")]
    I = {k: i4(v) for k, v in EV.items()}
    eA = {b: I[(b, "learn", "A")] - I[(b, "none", "A")] for b in BRAINS}
    eB = {b: I[(b, "learn", "B")] - I[(b, "none", "B")] for b in BRAINS}
    acq = [b for b in BRAINS if eA[b] <= -1000 and eB[b] >= 1000]
    elem = [b for b in BRAINS if abs(eB[b] - eA[b]) < 1000]
    p_B = [b for b in BRAINS if eB[b] >= 1000 and eA[b] > -1000]
    p_A = [b for b in BRAINS if eA[b] <= -1000 and eB[b] < 1000]
    if not ok:
        v = "보류(조작검증 실패)"
    elif len(acq) >= 4:
        v = "획득(H100) — 맥락 A·B 가 각자 전용 입력을 받으면 같은 자극의 정답을 맥락에 따라 반대로 배운다(쌍조건)"
    elif len(elem) >= 4:
        v = "요소식(H100-null) — 맥락을 무시하고 두 규칙이 겹친다"
    elif len(p_B) + len(p_A) >= 4:
        v = "부분 — 한 맥락만(B(같은 쪽)만 %d·A(교차)만 %d)%s" % (len(p_B), len(p_A), " — H100-bias 쪽" if len(p_B) >= 4 else "")
    else:
        v = "보류"
    return checks, {"eA": eA, "eB": eB, "acq": acq, "elem": elem, "p_B": p_B, "p_A": p_A, "ok": ok, "verdict": v, "m": m}


def report(checks, res, K=None, TRN=None, S=None, EL=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        mot = " ".join("%s%s %s" % (w[0], c, "/".join("%.4f" % x for x in EL[(b, w, c)][3])) for (w, c) in EVS)
        print("b%d e_A %+.4f e_B %+.4f (차 %+.4f) | 학습 보상 %d 맥락 B %d | 탐욕 |v|≤0.02 A %.2f B %.2f | motor %s | KCCTX %s"
              % (b, res["eA"][b] / 1e4, res["eB"][b] / 1e4, (res["eB"][b] - res["eA"][b]) / 1e4, TRN[b]["rew"], TRN[b]["nctx"],
                 S[b]["dec_off"], S[b]["dec_on"], mot, K[b][:120]))
    print("획득 %d/5 | 요소식 %d/5 | 부분 B만 %d/5·A만 %d/5" % (len(res["acq"]), len(res["elem"]), len(res["p_B"]), len(res["p_A"])))
    print("판정: %s" % res["verdict"])


def load(base=EXP):
    K, TRN, EV, S, RAW, EL = {}, {}, {}, {}, {}, {}
    try:
        for ln in open(os.path.join(base, "E179.log"), encoding="utf-8", errors="replace"):
            m = TT.match(ln)
            if m:
                TRN[int(m.group(1))] = {"pre": float(m.group(2)), "post": float(m.group(3)), "rew": int(m.group(4)), "nctx": int(m.group(5)), "load": int(m.group(6))}
                continue
            m = TE.match(ln)
            if m:
                EV[(int(m.group(1)), m.group(2), m.group(3))] = float(m.group(4))
                continue
            m = TK.match(ln)
            if m:
                K[int(m.group(1))] = m.group(2)
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(base, "traces", "E179", "tr_bc_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = bicond_stats(np.load(f)["rows"])
        g = os.path.join(base, "logs", "E179", "train_b%d.log" % b)
        RAW[b] = train_raw(open(g, encoding="utf-8", errors="replace").read()) if os.path.exists(g) else None
        for (w, c) in EVS:
            h = os.path.join(base, "logs", "E179", "ev_b%d_%s_%s.log" % (b, w, c))
            EL[(b, w, c)] = eval_raw(open(h, encoding="utf-8", errors="replace").read()) if os.path.exists(h) else None
    return K, TRN, EV, S, RAW, EL


if __name__ == "__main__":
    K, TRN, EV, S, RAW, EL = load()
    c, r = judge(K, TRN, EV, S, RAW, EL)
    report(c, r, K, TRN, S, EL)
    sys.exit(0)
