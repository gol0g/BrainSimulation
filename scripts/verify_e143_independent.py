#!/usr/bin/env python3
"""E143 독립 대조 — judge_e143.py 를 쓰지 않고 런별 원 로그(logs/E143/b*.log)와 추적에서 다시 계산한다.
- 변조폭·보상은 원 로그 [사전]·[사후]·[학습] 줄에서 1e-4 정수로. 짝 F1500 도 E142 런별 원 로그(logs/E142/F1500_b*.log)에서 직접 읽는다.
- 반사 불변은 원 로그 [반사가중치] good_food_to_motor 줄 문자열 직접.
- 동결 잔차는 시행·역할별 비 편차의 최대(|흔적| ≥ 1 인 칸)와 합 기준 둘 다(보상 창 r20, 결정 단계 r30).
- 판정은 criteria_fixed.txt 문장을 정수 비교로 다시 구현.
실행: python3 scripts/verify_e143_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
PRE119 = {10: 4148, 11: 4195, 12: 3954, 13: 3773, 14: 4248}
R20, R30 = (11.0 / 12.0) ** 20, (11.0 / 12.0) ** 30


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


def tr(path):
    R = np.load(path)["rows"]
    eda = R[:, 13:17]
    den = np.abs(eda).sum()
    o = {"n": R.shape[0], "ncol": R.shape[1], "res20": float(np.abs(R[:, 21:25] - R20 * eda).sum() / den),
         "alive": float(np.count_nonzero((np.abs(R[:, 13]) + np.abs(R[:, 14])) > 1.0) / R.shape[0]),
         "pre": float(abs(R[:, 17:21].sum()) / abs(R[:, 12].sum())), "rew_rows": int((R[:, 7] == 1).sum())}
    if R.shape[1] >= 35:
        o["res30"] = float(np.abs(R[:, 31:35] - R30 * R[:, 27:31]).sum() / den)
        big = np.abs(R[:, 27:31]) >= 1.0
        o["dev30"] = float(np.max(np.abs(R[:, 31:35][big] / R[:, 27:31][big] - R30))) if big.any() else float("nan")
    else:
        o["res30"], o["dev30"] = float("nan"), float("nan")
    return o


def main():
    rows, c = {}, {"보상창": 0, "결정": 0, "되돌림": 0, "출발점": 0, "추적": 0, "반사": 0, "보상일치": 0}
    for b in PRE119:
        r = raw(os.path.join(EXP, "logs", "E143", "b%d.log" % b))
        f = raw(os.path.join(EXP, "logs", "E142", "F1500_b%d.log" % b))
        t = tr(os.path.join(EXP, "traces", "E143", "tr_b%d.npz" % b))
        rows[b] = (r, f, t)
        c["보상창"] += t["res20"] <= 1e-3
        c["결정"] += t["res30"] <= 1e-3
        c["되돌림"] += t["alive"] >= 0.9
        c["출발점"] += abs(r["pre"] - PRE119[b]) <= 20
        c["추적"] += (t["n"] == 1500 and t["pre"] <= 1e-3)
        c["반사"] += (set(r["refl"]) == {"l", "r"} and all(v == ("25.0000", "25.0000") for v in r["refl"].values()))
        c["보상일치"] += r["rew"] == t["rew_rows"]
    print("뇌  [사전] [사후]    e   (F1500 사후·e)     d    보상 | 잔차 r20·r30·r30편차최대 | 흔적살아있음")
    d = {}
    for b, (r, f, t) in rows.items():
        e, ef = r["post"] - r["pre"], f["post"] - f["pre"]
        d[b] = e - ef
        print("b%d %+5d %+6d %+6d (%+6d %+6d) %+6d %4d | %.1e %.1e %.1e | %.3f" % (b, r["pre"], r["post"], e, f["post"], ef, d[b], r["rew"], t["res20"], t["res30"], t["dev30"], t["alive"]))
    ok = all(v == 5 for v in c.values())
    print("조작검증: %s → %s" % (" ".join("%s %d/5" % kv for kv in c.items()), "통과" if ok else "실패"))
    m = {b: rows[b][0]["post"] for b in PRE119}
    l2 = sum(m[b] <= -200 for b in PRE119)
    dec = sum(d[b] <= -300 for b in PRE119)
    nul = sum(abs(d[b]) < 300 for b in PRE119)
    rev = sum(d[b] >= 300 for b in PRE119)
    if not ok:
        v = "보류(조작검증 실패)"
    elif l2 >= 4:
        v = "L2 달성(H066)"
    elif dec == 5:
        v = "결정 단계 흔적이 정체 원인(H066-dec)"
    elif nul >= 4:
        v = "결정 단계 흔적 무관(H066-null)"
    elif rev >= 4:
        v = "반대(H066-rev)"
    else:
        v = "보류"
    print("m ≤ −0.02 %d/5 | d ≤ −0.03 %d/5 | |d| < 0.03 %d/5 | d ≥ +0.03 %d/5" % (l2, dec, nul, rev))
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
