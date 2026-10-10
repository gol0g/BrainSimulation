#!/usr/bin/env python3
"""(E176 판으로부터 실험 번호만 바꿔 옮김 — 맥락 세기 w 4 고정) E177 판정 — 맥락 전용 입력(초기 연결 스냅숏 + zz_ctx → KC)으로 쌍조건 변별(맥락 끔 good → 교차, 켬 → 같은 쪽)을 배우는가. judge_e173.py 를 실험 번호·조작검증(맥락 집단 발화)만 바꿔 옮김. 기준 logs/E177/criteria_fixed.txt.
뇌 10~14 자료(kcctx 5 + 학습 5 + 평가 20)가 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄(E177.log): "  e177 kcctx b10: => KCCTX ..." / "  e177 train b10: => 사전 .. 사후 .. 보상 N || 맥락 켬 n || 적재 K || ..." / "  e177 b10 learn on: => mod +0.1000".
e_off = m(learn off) − m(none off), e_on = m(learn on) − m(none on). 1e-4 정수.
판정(적용 순서): 조작검증 실패 → 보류. 획득(H098) e_off ≤ −0.10 이면서 e_on ≥ +0.10 ≥ 4/5 → 요소식(H098-null) |e_on − e_off| < 0.10 ≥ 4/5
→ 부분(H098-partial) 한 맥락만(e_on ≥ +0.10·e_off > −0.10, 또는 e_off ≤ −0.10·e_on < +0.10) ≥ 4/5 → 그 밖 보류.
조작검증(뇌마다, 모두): kcctx 맥락 전용 집단 발화 끔 0·켬 > 0 / 학습 '[맥락 과제] 시행 3000 중 맥락 켬 n' 1,350~1,650 / 추적 3,000행·열 37 / 맥락별 보상-규칙 일치 1.000 /
동결 잔차 ≤ 1e-3·도파민 전 ≤ 1e-3 / 반사 0→0 / 학습 적재 줄 ≥ 2 / 평가 적재 줄 ≥ 1 / 켬 평가에만 '[맥락 평가]' 줄.
실행: python3 scripts/judge_e177.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
R20 = (1.0 - 1.0 / 12.0) ** 20
EVS = (("learn", "off"), ("learn", "on"), ("none", "off"), ("none", "on"))
TT = re.compile(r"^\s*e177 train b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+) \|\| 맥락 켬 (\d+) \|\| 적재 (\d+)")
TE = re.compile(r"^\s*e177 b(\d+) (learn|none) (off|on): => mod ([-+0-9.]+)")
TK = re.compile(r"^\s*e177 kcctx b(\d+): => KCCTX (.*)$")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")
LD = re.compile(r"^\[E153 종류 입력 적재\].*검증 일치")
CTXN = re.compile(r"^\[맥락 과제\] 시행 (\d+) 중 맥락 켬 (\d+)", re.M)
SP = re.compile(r"맥락 집단 발화 끔 (\d+) 켬 (\d+)")


def i4(x):
    return int(round(float(x) * 1e4))


def bicond_stats(rows):
    """맥락별 규칙: 열 37 = 0 → 교차(실행 ≠ 쪽), 1 → 같은 쪽(실행 = 쪽). 열 6 실행(−1 = 행동 창 없음), 7 보상, 2 쪽."""
    n = len(rows)
    if rows.shape[1] < 38:
        return {"n": n, "has_ctx": False, "agree": float("nan"), "res": float("nan"), "pre_ratio": float("nan"), "frac_ctx": float("nan")}
    act = rows[:, 6] >= 0
    ctx = rows[:, 37] == 1
    rule = np.where(ctx, rows[:, 6] == rows[:, 2], rows[:, 6] != rows[:, 2])
    eda = rows[:, 13:17]
    return {"n": n, "has_ctx": True, "frac_ctx": float(ctx.mean()),
            "agree": float(np.mean((rows[act, 7] == 1) == rule[act])) if act.any() else float("nan"),
            "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12))}


def train_raw(txt):
    """(맥락 켬 수 또는 None, 시행 수 또는 None, 반사 0→0, 적재 줄 수)."""
    m = CTXN.search(txt)
    got = {}
    nld = 0
    for ln in txt.splitlines():
        r = RW.match(ln)
        if r:
            got[r.group(1)] = (r.group(2), r.group(3))
        nld += bool(LD.match(ln))
    return (int(m.group(2)) if m else None, int(m.group(1)) if m else None,
            set(got) == {"l", "r"} and all(v == ("0.0000", "0.0000") for v in got.values()), nld)


def judge(K, TRN, EV, S, RAW, EL):
    """K[b] = kcctx 줄 꼬리, TRN[b] = 학습 요약, EV[(b, w, c)] = mod, S[b] = bicond_stats, RAW[b] = train_raw, EL[(b, w, c)] = (적재 줄 수, 맥락 평가 줄 있음)."""
    miss = [b for b in BRAINS if b not in K or b not in TRN or b not in S or RAW.get(b) is None]
    miss += [(b, w, c) for b in BRAINS for (w, c) in EVS if (b, w, c) not in EV or EL.get((b, w, c)) is None]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % (miss[:6],)], None
    m = {}
    for b in BRAINS:
        sp = SP.search(K[b])
        st, rw = S[b], RAW[b]
        m[b] = {"kc": bool(sp) and int(sp.group(1)) == 0 and int(sp.group(2)) > 0,
                "nctx": rw[0] is not None and rw[1] == 3000 and 1350 <= rw[0] <= 1650 and rw[0] == TRN[b]["nctx"],
                "trace": st["has_ctx"] and st["n"] == 3000 and st["agree"] == 1.0 and st["res"] <= 1e-3 and st["pre_ratio"] <= 1e-3,
                "refl": rw[2], "load": rw[3] >= 2 and TRN[b]["load"] >= 2,
                "ev": all(EL[(b, w, c)][0] >= 1 and EL[(b, w, c)][1] == (c == "on") for (w, c) in EVS)}
    ok = all(all(v.values()) for v in m.values())
    cnt = {k: sum(m[b][k] for b in BRAINS) for k in ("kc", "nctx", "trace", "refl", "load", "ev")}
    checks = ["[조작검증] 맥락 집단 발화 %d/5 · 맥락 켬 수 %d/5 · 추적(3,000·열37·맥락별 규칙 일치·동결·도파민전) %d/5 · 반사 0 %d/5 · 학습 적재 %d/5 · 평가(적재·맥락 평가 줄) %d/5 → %s"
              % (cnt["kc"], cnt["nctx"], cnt["trace"], cnt["refl"], cnt["load"], cnt["ev"], "통과" if ok else "실패")]
    I = {k: i4(v) for k, v in EV.items()}
    eoff = {b: I[(b, "learn", "off")] - I[(b, "none", "off")] for b in BRAINS}
    eon = {b: I[(b, "learn", "on")] - I[(b, "none", "on")] for b in BRAINS}
    acq = [b for b in BRAINS if eoff[b] <= -1000 and eon[b] >= 1000]
    elem = [b for b in BRAINS if abs(eon[b] - eoff[b]) < 1000]
    p_on = [b for b in BRAINS if eon[b] >= 1000 and eoff[b] > -1000]
    p_off = [b for b in BRAINS if eoff[b] <= -1000 and eon[b] < 1000]
    if not ok:
        v = "보류(조작검증 실패)"
    elif len(acq) >= 4:
        v = "획득(H098) — 같은 자극의 정답을 맥락에 따라 반대로 배운다(맥락 의존 규칙)"
    elif len(elem) >= 4:
        v = "요소식(H098-null) — 맥락을 무시하고 두 규칙이 겹친다"
    elif len(p_on) + len(p_off) >= 4:
        v = "부분(H098-partial) — 한 맥락만(켬만 %d·끔만 %d)" % (len(p_on), len(p_off))
    else:
        v = "보류"
    return checks, {"eoff": eoff, "eon": eon, "acq": acq, "elem": elem, "p_on": p_on, "p_off": p_off, "ok": ok, "verdict": v, "m": m}


def report(checks, res, K=None, TRN=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d e_off %+.4f e_on %+.4f (차 %+.4f) | 학습 보상 %d 맥락 켬 %d | KCCTX %s"
              % (b, res["eoff"][b] / 1e4, res["eon"][b] / 1e4, (res["eon"][b] - res["eoff"][b]) / 1e4, TRN[b]["rew"], TRN[b]["nctx"], K[b][:150]))
    print("획득 %d/5 | 요소식 %d/5 | 부분 켬만 %d/5·끔만 %d/5" % (len(res["acq"]), len(res["elem"]), len(res["p_on"]), len(res["p_off"])))
    print("판정: %s" % res["verdict"])


def load():
    K, TRN, EV, S, RAW, EL = {}, {}, {}, {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E177.log"), encoding="utf-8", errors="replace"):
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
        f = os.path.join(EXP, "traces", "E177", "tr_bc_b%d.npz" % b)
        if os.path.exists(f):
            S[b] = bicond_stats(np.load(f)["rows"])
        g = os.path.join(EXP, "logs", "E177", "train_b%d.log" % b)
        RAW[b] = train_raw(open(g, encoding="utf-8", errors="replace").read()) if os.path.exists(g) else None
        for (w, c) in EVS:
            h = os.path.join(EXP, "logs", "E177", "ev_b%d_%s_%s.log" % (b, w, c))
            if os.path.exists(h):
                t = open(h, encoding="utf-8", errors="replace").read()
                EL[(b, w, c)] = (sum(bool(LD.match(x)) for x in t.splitlines()), "[맥락 평가]" in t)
            else:
                EL[(b, w, c)] = None
    return K, TRN, EV, S, RAW, EL


if __name__ == "__main__":
    K, TRN, EV, S, RAW, EL = load()
    c, r = judge(K, TRN, EV, S, RAW, EL)
    report(c, r, K, TRN)
    sys.exit(0)
