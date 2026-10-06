#!/usr/bin/env python3
"""E148 독립 대조 — judge_e148.py 를 쓰지 않고 런별 원 로그(logs/E148/train_b*.log·ev_b*_*_*.log)·추적·E146 학습 원 로그에서 다시 계산한다.
- 과제 A 단독 기준 eA1: E146 학습 원 로그의 [사후] − E148 학습 원 로그의 [사전](같은 시드 출발점)을 직접 읽는다.
- 평가: 원 로그 DECOMP mod 를 1e-4 정수로, 변형 표시 줄이 파일 이름과 맞는지 확인.
실행: python3 scripts/verify_e148_independent.py (저장소 루트에서)"""
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


def train(path):
    out = {"refl": {}, "b": False}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[사전]"):
            out["pre"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[사후]"):
            out["post"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[과제 B] 시행 1500 부터"):
            out["b"] = True
        elif ln.startswith("[반사가중치] good_food_to_motor_"):
            w = re.search(r"good_food_to_motor_([lr])\s+n=\d+ w_mean (\S+)→(\S+)", ln)
            out["refl"][w.group(1)] = (w.group(2), w.group(3))
    return out


def ev(b, w, s):
    txt = open(os.path.join(EXP, "logs", "E148", "ev_b%d_%s_%s.log" % (b, w, s)), encoding="utf-8", errors="replace").read()
    if ("[E146 변형] variant=%s " % s) not in txt:
        raise RuntimeError("변형 표시 불일치 b%d %s %s" % (b, w, s))
    return i4(re.search(r"^=> DECOMP mode=\S+ mod=([-+]?\d+\.\d{4})", txt, re.M).group(1))


def main():
    c = {"과제B줄": 0, "규칙일치": 0, "동결": 0, "반사0": 0, "출발점": 0, "무학습재현": 0}
    keep = mb = 0
    for b in BRAINS:
        t = train(os.path.join(EXP, "logs", "E148", "train_b%d.log" % b))
        ref = train(os.path.join(EXP, "logs", "E146", "train_b%d.log" % b))
        R = np.load(os.path.join(EXP, "traces", "E148", "tr_b%d.npz" % b))["rows"]
        good = tot = 0
        for i in range(len(R)):
            if R[i, 6] < 0:
                continue
            want = (R[i, 6] == R[i, 2]) if i >= 1500 else (R[i, 6] != R[i, 2])
            good += int((R[i, 7] == 1) == want); tot += 1
        res = float(np.abs(R[:, 21:25] - ((11.0 / 12.0) ** 20) * R[:, 13:17]).sum() / np.abs(R[:, 13:17]).sum())
        M = {(w, s): ev(b, w, s) for w in ("learn", "none") for s in ("base", "bad")}
        c["과제B줄"] += t["b"]; c["규칙일치"] += (good == tot and tot > 0); c["동결"] += (res <= 1e-3 and len(R) == 3000)
        c["반사0"] += t["refl"] == {"l": ("0.0000", "0.0000"), "r": ("0.0000", "0.0000")}
        c["출발점"] += abs(t["pre"] - PRE0[b]) <= 20; c["무학습재현"] += abs(M[("none", "base")] - PRE0[b]) <= 20
        eA = M[("learn", "base")] - M[("none", "base")]
        eA1 = ref["post"] - t["pre"]
        eB = M[("learn", "bad")] - M[("none", "bad")]
        k = eA1 < 0 and 2 * eA <= eA1
        keep += k; mb += eB >= 500
        print("b%d eA %+5d eA1(E146 원 로그) %+5d 몫 %.2f %s | eB %+5d | 규칙 일치 %d/%d | 잔차 %.1e" % (b, eA, eA1, eA / eA1 if eA1 else float("nan"), "유지" if k else "-", eB, good, tot, res))
    ok = all(v == 5 for v in c.values())
    print("조작검증: %s → %s | 과제 B 학습 %d/5 | 유지 %d/5" % (" ".join("%s %d/5" % kv for kv in c.items()), "통과" if ok else "실패", mb, keep))
    if not ok:
        v = "보류(조작검증 실패)"
    elif mb < 4:
        v = "보류(과제 B 미학습)"
    elif keep >= 4:
        v = "유지(H071)"
    elif 5 - keep >= 4:
        v = "간섭(H071-int)"
    else:
        v = "보류"
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
