#!/usr/bin/env python3
"""E172 독립 대조(유지) — verify_e162_independent.py 를 경로·뇌·가설 번호만 바꿔 옮김. 원 설명: E162 독립 대조 — judge_e162.py 를 쓰지 않고 런별 원 로그(logs/E162/train_*·ev_*)·추적에서 다시 계산한다.
평가 mod 는 원 로그 DECOMP 줄을 1e-4 정수로, 변형 표시·적재 검증 줄 확인. 학습 런은 원 로그(과제 B 줄 유무·반사 0·적재 ≥2)·추적(시행별 규칙 일치·동결 잔차).
실행: python3 scripts/verify_e172_ret.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)


def i4(s):
    m = re.fullmatch(r"([-+]?)(\d+)\.(\d{4})", s)
    if not m:
        raise ValueError("4자리 값 아님: %r" % s)
    v = int(m.group(2)) * 10000 + int(m.group(3))
    return -v if m.group(1) == "-" else v


def ev(b, w, s):
    txt = open(os.path.join(EXP, "logs", "E172", "ev_b%d_%s_%s.log" % (b, w, s)), encoding="utf-8", errors="replace").read()
    if ("[E146 변형] variant=%s " % s) not in txt:
        raise RuntimeError("변형 표시 불일치 b%d %s %s" % (b, w, s))
    if not any(ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln for ln in txt.splitlines()):
        raise RuntimeError("평가 적재 줄 없음 b%d %s %s" % (b, w, s))
    return i4(re.search(r"^=> DECOMP mode=\S+ mod=([-+]?\d+\.\d{4})", txt, re.M).group(1))


def train_ok(arm, b):
    txt = open(os.path.join(EXP, "logs", "E172", "train_%s_b%d.log" % (arm, b)), encoding="utf-8", errors="replace").read()
    n_ld = sum(1 for ln in txt.splitlines() if ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln)
    has_b = "[과제 B] 시행 1500 부터" in txt
    refl = re.findall(r"^\[반사가중치\] good_food_to_motor_[lr]\s+n=\d+ w_mean (\S+)→(\S+)", txt, re.M)
    R = np.load(os.path.join(EXP, "traces", "E172", "tr_%s_b%d.npz" % (arm, b)))["rows"]
    good = tot = 0
    for i in range(len(R)):
        if R[i, 6] < 0:
            continue
        want = (R[i, 6] == R[i, 2]) if (arm == "AB" and i >= 1500) else (R[i, 6] != R[i, 2])
        good += int((R[i, 7] == 1) == want); tot += 1
    res = float(np.abs(R[:, 21:25] - ((11.0 / 12.0) ** 20) * R[:, 13:17]).sum() / np.abs(R[:, 13:17]).sum())
    n_ok = len(R) == (1500 if arm == "A" else 3000)
    return (has_b == (arm == "AB")) and len(refl) == 2 and all(r == ("0.0000", "0.0000") for r in refl) and good == tot and tot > 0 and res <= 1e-3 and n_ok and n_ld >= 2


def main():
    ok = sum(train_ok(a, b) for a in ("A", "AB") for b in BRAINS)
    keep = p1 = p2 = big = 0
    for b in BRAINS:
        M = {(w, s): ev(b, w, s) for (w, s) in (("A", "base"), ("AB", "base"), ("AB", "bad"), ("none", "base"), ("none", "bad"))}
        eA1 = M[("A", "base")] - M[("none", "base")]; eA = M[("AB", "base")] - M[("none", "base")]; eB = M[("AB", "bad")] - M[("none", "bad")]
        p1 += eA1 <= -2000; p2 += eB >= 1500
        k = eA1 < 0 and 2 * eA <= eA1
        keep += k
        big += eA1 < 0 and 5 * eA <= 4 * eA1
        print("b%d eA1 %+5d eA %+5d 몫 %.2f %s | eB %+5d | T %s" % (b, eA1, eA, eA / eA1 if eA1 else float("nan"), "유지" if k else "-", eB, ("%.2f" % ((eA - eA1) / eB)) if eB else "nan"))
    print("학습 런 조작검증 %d/10 | P1 %d/5 P2 %d/5 | 유지 %d/5 | 크기(rA ≥ 0.80) %d/5" % (ok, p1, p2, keep, big))
    if ok < 10:
        v = "보류(조작검증 실패)"
    elif p1 < 4:
        v = "보류(형성 표현 과제 A 전체 강도 미학습)"
    elif p2 < 4:
        v = "보류(과제 B 미학습)"
    elif keep >= 4:
        v = "유지(H095)"
    elif 5 - keep >= 4:
        v = "간섭(H095-null)"
    else:
        v = "보류"
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
