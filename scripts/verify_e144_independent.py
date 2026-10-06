#!/usr/bin/env python3
"""E144 독립 대조 — judge_e144.py 를 쓰지 않고 런별 원 로그(logs/E144/b*.log·logs/E142/F1500_b*.log)와 추적에서 다시 계산한다.
- 변조폭·보상: 원 로그 [사전]·[사후]·[학습] 줄을 1e-4 정수로. 반사 불변: [반사가중치] good_food_to_motor 문자열 직접.
- R0: 보정 원 추적(traces/E144/calib/tr_R0_5000_b15.npz 열 36)을 시행별 값의 중앙값·평균 둘 다 보고, 판정은 평균(기준 문장).
- 반대쪽 침묵·보상 시행 같은 쪽 흔적: 추적에서 직접(보상 시행만 골라 합).
실행: python3 scripts/verify_e144_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
PRE119 = {10: 4148, 11: 4195, 12: 3954, 13: 3773, 14: 4248}
R20 = (11.0 / 12.0) ** 20


def i4(s):
    m = re.fullmatch(r"([-+]?)(\d+)\.(\d{4})", s)
    if not m:
        raise ValueError("4자리 값 아님: %r" % s)
    v = int(m.group(2)) * 10000 + int(m.group(3))
    return -v if m.group(1) == "-" else v


def raw(path):
    out = {"pre": None, "post": None, "rew": None, "refl": {}}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[사전]"):
            out["pre"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[사후]"):
            out["post"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[학습]"):
            out["rew"] = int(re.search(r"보상 (\d+)회", ln).group(1))
        elif ln.startswith("[반사가중치] good_food_to_motor_"):
            side = ln.split("good_food_to_motor_")[1][0]
            w = re.search(r"w_mean (\S+)→(\S+)", ln)
            out["refl"][side] = (w.group(1), w.group(2))
    return out


def main():
    Rc = np.load(os.path.join(EXP, "traces", "E144", "calib", "tr_R0_5000_b15.npz"))["rows"]
    r0 = float(np.mean(Rc[:, 36]))
    print("R0 = 평균 %.4f (중앙값 %.4f, 시행 %d) → 문턱 %.4f" % (r0, float(np.median(Rc[:, 36])), len(Rc), r0 + 0.01))
    c = {"동결": 0, "되돌림": 0, "출발점": 0, "추적": 0, "반사": 0, "반대쪽침묵": 0, "같은쪽LTD": 0, "보상일치": 0}
    d = {}
    print("뇌  [사전] [사후]    e   (F1500 사후·e)     d    보상 | 반대 발화 평균 | 보상 시행 같은쪽 흔적")
    for b in PRE119:
        r = raw(os.path.join(EXP, "logs", "E144", "b%d.log" % b))
        f = raw(os.path.join(EXP, "logs", "E142", "F1500_b%d.log" % b))
        R = np.load(os.path.join(EXP, "traces", "E144", "tr_b%d.npz" % b))["rows"]
        rw = R[:, 7] == 1
        eda = R[:, 13:17]
        res = float(np.abs(R[:, 21:25] - R20 * eda).sum() / np.abs(eda).sum())
        alive = float(np.count_nonzero((np.abs(R[:, 13]) + np.abs(R[:, 14])) > 1.0) / len(R))
        ot = float(np.mean(R[:, 36])) if R.shape[1] >= 37 else float("nan")
        same_rw = float(R[rw, 14].sum())
        c["동결"] += res <= 1e-3
        c["되돌림"] += alive >= 0.9
        c["출발점"] += abs(r["pre"] - PRE119[b]) <= 20
        c["추적"] += (len(R) == 1500 and abs(R[:, 17:21].sum()) <= 1e-3 * abs(R[:, 12].sum()))
        c["반사"] += (set(r["refl"]) == {"l", "r"} and all(v == ("25.0000", "25.0000") for v in r["refl"].values()))
        c["반대쪽침묵"] += ot <= r0 + 0.01 + 1e-9   # 평균의 부동소수 오차 허용
        c["같은쪽LTD"] += same_rw < 0
        c["보상일치"] += r["rew"] == int(rw.sum())
        e, ef = r["post"] - r["pre"], f["post"] - f["pre"]
        d[b] = e - ef
        print("b%d %+5d %+6d %+6d (%+6d %+6d) %+6d %4d | %.4f | %+.3e" % (b, r["pre"], r["post"], e, f["post"], ef, d[b], r["rew"], ot, same_rw))
    ok = all(v == 5 for v in c.values())
    print("조작검증: %s → %s" % (" ".join("%s %d/5" % kv for kv in c.items()), "통과" if ok else "실패"))
    m = {b: raw(os.path.join(EXP, "logs", "E144", "b%d.log" % b))["post"] for b in PRE119}
    l2 = sum(m[b] <= -200 for b in PRE119)
    par = sum(d[b] <= -300 for b in PRE119)
    nul = sum(abs(d[b]) < 300 for b in PRE119)
    rev = sum(d[b] >= 300 for b in PRE119)
    if not ok:
        v = "보류(조작검증 실패)"
    elif l2 >= 4:
        v = "L2 달성(H067)"
    elif par == 5:
        v = "반대쪽 발화가 정체 원인(H067-partial)"
    elif nul >= 4:
        v = "무관(H067-null)"
    elif rev >= 4:
        v = "반대(H067-rev)"
    else:
        v = "보류"
    print("m ≤ −0.02 %d/5 | d ≤ −0.03 %d/5 | |d| < 0.03 %d/5 | d ≥ +0.03 %d/5" % (l2, par, nul, rev))
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
