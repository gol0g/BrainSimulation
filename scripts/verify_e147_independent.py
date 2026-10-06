#!/usr/bin/env python3
"""E147 독립 대조 — judge_e147.py 를 쓰지 않고 런별 원 로그(logs/E147/b*.log)·추적·E146 학습 원 로그(반전 직전 기준)에서 다시 계산한다.
- 사후·사전: 원 로그 [사전]·[사후] 1e-4 정수. 기준 R: E146 학습 원 로그(logs/E146/train_b*.log)의 [사후] 를 직접 읽는다(상수 표 안 씀).
- 규칙 일치: 추적을 시행 순서대로 다시 훑어 앞/뒤 구간을 따로 센다.
실행: python3 scripts/verify_e147_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
PRE0 = {10: 195, 11: 150, 12: 262, 13: 320, 14: 165}


def i4(s):
    m = re.fullmatch(r"([-+]?)(\d+)\.(\d{4})", s)
    if not m:
        raise ValueError("4자리 값 아님: %r" % s)
    v = int(m.group(2)) * 10000 + int(m.group(3))
    return -v if m.group(1) == "-" else v


def raw(path):
    out = {"refl": {}, "rev": False}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[사전]"):
            out["pre"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[사후]"):
            out["post"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[반전] 시행 1500 부터"):
            out["rev"] = True
        elif ln.startswith("[반사가중치] good_food_to_motor_"):
            w = re.search(r"good_food_to_motor_([lr])\s+n=\d+ w_mean (\S+)→(\S+)", ln)
            out["refl"][w.group(1)] = (w.group(2), w.group(3))
    return out


def main():
    c = {"반전줄": 0, "규칙일치": 0, "동결": 0, "반사0": 0, "출발점": 0}
    d, m = {}, {}
    for b in BRAINS:
        r = raw(os.path.join(EXP, "logs", "E147", "b%d.log" % b))
        ref = raw(os.path.join(EXP, "logs", "E146", "train_b%d.log" % b))["post"]
        R = np.load(os.path.join(EXP, "traces", "E147", "tr_b%d.npz" % b))["rows"]
        good = tot = 0
        for i in range(len(R)):
            if R[i, 6] < 0:
                continue
            want = (R[i, 6] == R[i, 2]) if i >= 1500 else (R[i, 6] != R[i, 2])
            good += int((R[i, 7] == 1) == want); tot += 1
        res = float(np.abs(R[:, 21:25] - ((11.0 / 12.0) ** 20) * R[:, 13:17]).sum() / np.abs(R[:, 13:17]).sum())
        c["반전줄"] += r["rev"]
        c["규칙일치"] += (good == tot and tot > 0)
        c["동결"] += (res <= 1e-3 and len(R) == 3000)
        c["반사0"] += r["refl"] == {"l": ("0.0000", "0.0000"), "r": ("0.0000", "0.0000")}
        c["출발점"] += abs(r["pre"] - PRE0[b]) <= 20
        m[b], d[b] = r["post"], r["post"] - ref
        rw_rev = [int((R[i:i + 100, 7] == 1).sum()) for i in range(1500, len(R), 100)]
        print("b%d 사후 %+5d 기준(E146 원 로그) %+5d Δ %+5d | 규칙 일치 %d/%d | 잔차 %.1e | 반전 구간 블록 보상 %s" % (b, r["post"], ref, d[b], good, tot, res, rw_rev))
    ok = all(v == 5 for v in c.values())
    print("조작검증: %s → %s" % (" ".join("%s %d/5" % kv for kv in c.items()), "통과" if ok else "실패"))
    succ = sum(m[b] >= 200 for b in BRAINS); part = sum(d[b] >= 1000 for b in BRAINS); stuck = sum(d[b] < 500 for b in BRAINS)
    if not ok:
        v = "보류(조작검증 실패)"
    elif succ >= 4:
        v = "반전 성공(H070)"
    elif part == 5:
        v = "부분(H070-partial)"
    elif stuck >= 4:
        v = "고착(H070-stuck)"
    else:
        v = "보류"
    print("m ≥ +0.02 %d/5 | Δ ≥ +0.10 %d/5 | Δ < +0.05 %d/5" % (succ, part, stuck))
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
