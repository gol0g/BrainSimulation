#!/usr/bin/env python3
"""E146 판정 — 기준 logs/E146/criteria_fixed.txt(실행 전 고정). 학습 5줄 + 평가 75줄이 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄: 학습 "  e146 train b10: => 사전 +0.0195 사후 -0.3000 보상 300 || ...", 평가 "  e146 b10 E141 int05: => mod -0.1500".
e_v(W) = mod_v(W) − mod_v(none). V4 는 학습 추적(traces/E146/tr_b*.npz)과 원 로그(logs/E146/train_b*.log)의 [반사가중치]에서.
"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
WSETS = ("none", "E141", "R0F1500")
STIMS = ("base", "int05", "int07", "occ", "noise")
VARS = ("int05", "int07", "occ", "noise")
E141_POST = {10: -0.2189, 11: -0.2433, 12: -0.2471, 13: -0.2194, 14: -0.2550}
E119_PRE0 = {10: 0.0195, 11: 0.0150, 12: 0.0262, 13: 0.0320, 14: 0.0165}
R20 = (1.0 - 1.0 / 12.0) ** 20
TT = re.compile(r"^\s*e146 train b(\d+): => 사전 ([-+0-9.]+) 사후 ([-+0-9.]+) 보상 (\d+)")
TE = re.compile(r"^\s*e146 b(\d+) (none|E141|R0F1500) (base|int05|int07|occ|noise): => mod ([-+0-9.]+)")
RW = re.compile(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)")


def train_stats(rows):
    eda = rows[:, 13:17]
    return {"n": len(rows), "res": float(np.abs(rows[:, 21:25] - R20 * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "blk": [float((rows[i:i + 100, 8] - rows[i:i + 100, 9]).sum()) for i in range(0, len(rows), 100)]}


def reflex0_ok(path):
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
    return all(v == ("0.0000", "0.0000") for v in got.values())


def judge(TR, EV, TS, RF):
    miss = [b for b in BRAINS if b not in TR or b not in TS or RF.get(b) is None]
    miss += [(b, w, s) for b in BRAINS for w in WSETS for s in STIMS if (b, w, s) not in EV]
    if miss:
        return ["[측정 확인] 결측 %d — **판정 보류, 수치 미출력**" % len(miss)], None
    v1 = sum(abs(EV[(b, "E141", "base")] - E141_POST[b]) <= 0.002 + 1e-9 for b in BRAINS)
    v2 = sum(abs(EV[(b, "none", "base")] - E119_PRE0[b]) <= 0.002 + 1e-9 for b in BRAINS)
    v3 = sum(abs(EV[(b, "R0F1500", "base")] - TR[b]["post"]) <= 0.002 + 1e-9 for b in BRAINS)
    v4 = sum(TS[b]["res"] <= 1e-3 and TS[b]["n"] == 1500 and abs(TR[b]["pre"] - E119_PRE0[b]) <= 0.002 + 1e-9 and bool(RF[b]) for b in BRAINS)
    ok = v1 == v2 == v3 == v4 == 5
    checks = ["[측정 검증] V1 base·E141 = E141 사후 %d/5 · V2 base·none = 사전 %d/5 · V3 base·R0F1500 = 학습 사후 %d/5 · V4 학습 조작검증 %d/5 %s"
              % (v1, v2, v3, v4, "통과" if ok else "실패")]
    # 1e-4 단위 정수로 비교(변조폭은 4자리 출력 — 비율 경계에서 부동소수 반올림 오차를 없앤다). 몫 ≥ 0.5 ⇔ 2·e_v ≤ e_base(e_base < 0).
    I = {k: int(round(v * 1e4)) for k, v in EV.items()}
    ei = {(b, w, s): I[(b, w, s)] - I[(b, "none", s)] for b in BRAINS for w in ("E141", "R0F1500") for s in STIMS}
    e = {k: v / 1e4 for k, v in ei.items()}
    c1 = {v: sum(ei[(b, "E141", v)] <= -500 for b in BRAINS) for v in VARS}
    c3 = {v: sum(ei[(b, "E141", "base")] < 0 and 2 * ei[(b, "E141", v)] <= ei[(b, "E141", "base")] for b in BRAINS) for v in VARS}
    c4 = {s: sum(ei[(b, "R0F1500", s)] < ei[(b, "E141", s)] for b in BRAINS) for s in STIMS}
    c1_ok = all(c1[v] == 5 for v in VARS)
    c3_ok = all(c3[v] >= 4 for v in VARS)
    c4_ok = all(c4[s] >= 4 for s in STIMS)
    c3_fail = [v for v in VARS if c3[v] < 4]
    c4_fail = [s for s in STIMS if c4[s] < 4]
    if not ok:
        verdict = "보류(측정 검증 실패)"
    elif c1_ok and c3_ok and c4_ok:
        verdict = "충족(H069) — 반사 차단 전체 모델이 규칙 수준에서 헌장 개념 조건 1~4 충족(무학습 초과·자극 변형 일반화·용량-반응)"
    elif len(c3_fail) >= 2:
        verdict = "일반화 실패(H069-spec) — 훈련 자극 효과의 절반 이상을 못 지키는 변형 %s" % "·".join(c3_fail)
    elif c1_ok and c3_ok and len(c4_fail) >= 3:
        verdict = "용량 포화(H069-sat) — 1,500시행이 500시행보다 크지 않은 자극 %s" % "·".join(c4_fail)
    else:
        verdict = "부분(보류) — C1 %s · C3 실패 %s · C4 실패 %s" % ("통과" if c1_ok else "실패", c3_fail or "없음", c4_fail or "없음")
    return checks, {"e": e, "c1": c1, "c3": c3, "c4": c4, "ok": ok, "verdict": verdict}


def report(checks, res, EV=None, TR=None, TS=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for s in STIMS:
        print("%-5s none %s | e500 %s | e1500 %s"
              % (s, " ".join("%+.4f" % EV[(b, "none", s)] for b in BRAINS), " ".join("%+.4f" % res["e"][(b, "E141", s)] for b in BRAINS),
                 " ".join("%+.4f" % res["e"][(b, "R0F1500", s)] for b in BRAINS)))
    print("C1(e500 ≤ −0.05) %s | C3(e_v/e_base ≥ 0.5) %s | C4(e1500 < e500) %s"
          % (" ".join("%s %d/5" % (v, res["c1"][v]) for v in VARS), " ".join("%s %d/5" % (v, res["c3"][v]) for v in VARS),
             " ".join("%s %d/5" % (s, res["c4"][s]) for s in STIMS)))
    for b in BRAINS:
        print("b%d 학습 사전 %+.4f 사후 %+.4f 보상 %d | 블록 ΔD 첫·끝 %+.3g→%+.3g | 잔차 %.1e"
              % (b, TR[b]["pre"], TR[b]["post"], TR[b]["rew"], TS[b]["blk"][0], TS[b]["blk"][-1], TS[b]["res"]))
    print("판정: %s" % res["verdict"])


def load():
    TR, EV, TS, RF = {}, {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E146.log"), encoding="utf-8", errors="replace"):
            m = TT.match(ln)
            if m:
                TR[int(m.group(1))] = {"pre": float(m.group(2)), "post": float(m.group(3)), "rew": int(m.group(4))}
                continue
            m = TE.match(ln)
            if m:
                EV[(int(m.group(1)), m.group(2), m.group(3))] = float(m.group(4))
    except FileNotFoundError:
        pass
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E146", "tr_b%d.npz" % b)
        if os.path.exists(f):
            TS[b] = train_stats(np.load(f)["rows"])
        RF[b] = reflex0_ok(os.path.join(EXP, "logs", "E146", "train_b%d.log" % b))
    return TR, EV, TS, RF


if __name__ == "__main__":
    TR, EV, TS, RF = load()
    c, r = judge(TR, EV, TS, RF)
    report(c, r, EV, TR, TS)
    sys.exit(0)
