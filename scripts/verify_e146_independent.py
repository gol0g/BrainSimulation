#!/usr/bin/env python3
"""E146 독립 대조 — judge_e146.py 를 쓰지 않고 런별 원 로그(logs/E146/train_b*.log·ev_b*_*_*.log)와 학습 추적에서 다시 계산한다.
- 변조폭: 평가 원 로그의 "=> DECOMP ... mod=" 를 1e-4 정수로(요약 E146.log 를 쓰지 않음), 변형 표시 줄([E146 변형] variant=)이 파일 이름과 맞는지 확인.
- 학습: 원 로그 [사전]·[사후]·[학습]·[반사가중치], 추적 잔차·시행 수.
- 판정: criteria_fixed.txt 문장을 정수 비교로 다시 구현.
실행: python3 scripts/verify_e146_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
STIMS = ("base", "int05", "int07", "occ", "noise")
VARS = STIMS[1:]
E141_POST = {10: -2189, 11: -2433, 12: -2471, 13: -2194, 14: -2550}
PRE0 = {10: 195, 11: 150, 12: 262, 13: 320, 14: 165}
R20 = (11.0 / 12.0) ** 20


def i4(s):
    m = re.fullmatch(r"([-+]?)(\d+)\.(\d{4})", s)
    if not m:
        raise ValueError("4자리 값 아님: %r" % s)
    v = int(m.group(2)) * 10000 + int(m.group(3))
    return -v if m.group(1) == "-" else v


def ev(b, w, s):
    txt = open(os.path.join(EXP, "logs", "E146", "ev_b%d_%s_%s.log" % (b, w, s)), encoding="utf-8", errors="replace").read()
    if ("[E146 변형] variant=%s " % s) not in txt:
        raise RuntimeError("변형 표시 불일치 b%d %s %s" % (b, w, s))
    return i4(re.search(r"^=> DECOMP mode=\S+ mod=([-+]?\d+\.\d{4})", txt, re.M).group(1))


def train(b):
    out = {"refl": {}}
    for ln in open(os.path.join(EXP, "logs", "E146", "train_b%d.log" % b), encoding="utf-8", errors="replace"):
        if ln.startswith("[사전]"):
            out["pre"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[사후]"):
            out["post"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[반사가중치] good_food_to_motor_"):
            w = re.search(r"good_food_to_motor_([lr])\s+n=\d+ w_mean (\S+)→(\S+)", ln)
            out["refl"][w.group(1)] = (w.group(2), w.group(3))
    R = np.load(os.path.join(EXP, "traces", "E146", "tr_b%d.npz" % b))["rows"]
    out["n"] = len(R)
    out["res"] = float(np.abs(R[:, 21:25] - R20 * R[:, 13:17]).sum() / np.abs(R[:, 13:17]).sum())
    return out


def main():
    M = {(b, w, s): ev(b, w, s) for b in BRAINS for w in ("none", "E141", "R0F1500") for s in STIMS}
    T = {b: train(b) for b in BRAINS}
    v1 = sum(abs(M[(b, "E141", "base")] - E141_POST[b]) <= 20 for b in BRAINS)
    v2 = sum(abs(M[(b, "none", "base")] - PRE0[b]) <= 20 for b in BRAINS)
    v3 = sum(abs(M[(b, "R0F1500", "base")] - T[b]["post"]) <= 20 for b in BRAINS)
    v4 = sum(T[b]["res"] <= 1e-3 and T[b]["n"] == 1500 and abs(T[b]["pre"] - PRE0[b]) <= 20
             and T[b]["refl"] == {"l": ("0.0000", "0.0000"), "r": ("0.0000", "0.0000")} for b in BRAINS)
    ok = v1 == v2 == v3 == v4 == 5
    e = {(b, w, s): M[(b, w, s)] - M[(b, "none", s)] for b in BRAINS for w in ("E141", "R0F1500") for s in STIMS}
    print("측정 검증 V1 %d/5 V2 %d/5 V3 %d/5 V4 %d/5 → %s" % (v1, v2, v3, v4, "통과" if ok else "실패"))
    for s in STIMS:
        print("%-5s e500 %s | e1500 %s" % (s, " ".join("%+5d" % e[(b, "E141", s)] for b in BRAINS), " ".join("%+5d" % e[(b, "R0F1500", s)] for b in BRAINS)))
    c1 = {v: sum(e[(b, "E141", v)] <= -500 for b in BRAINS) for v in VARS}
    c3 = {v: sum(e[(b, "E141", "base")] < 0 and 2 * e[(b, "E141", v)] <= e[(b, "E141", "base")] for b in BRAINS) for v in VARS}
    c4 = {s: sum(e[(b, "R0F1500", s)] < e[(b, "E141", s)] for b in BRAINS) for s in STIMS}
    c1_ok, c3_ok, c4_ok = all(c1[v] == 5 for v in VARS), all(c3[v] >= 4 for v in VARS), all(c4[s] >= 4 for s in STIMS)
    c3_fail, c4_fail = [v for v in VARS if c3[v] < 4], [s for s in STIMS if c4[s] < 4]
    if not ok:
        v = "보류(측정 검증 실패)"
    elif c1_ok and c3_ok and c4_ok:
        v = "충족(H069)"
    elif len(c3_fail) >= 2:
        v = "일반화 실패(H069-spec)"
    elif c1_ok and c3_ok and len(c4_fail) >= 3:
        v = "용량 포화(H069-sat)"
    else:
        v = "부분(보류)"
    print("C1 %s | C3 %s | C4 %s" % (c1, c3, c4))
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
