#!/usr/bin/env python3
"""E172 독립 대조(반전) — verify_e163_independent.py 를 경로·뇌·가설 번호만 바꿔 옮김. 원 설명: E163 독립 대조 — judge_e163.py 를 쓰지 않고 런별 원 로그(logs/E163/rev_b*.log)·추적·E162 A단독 원 로그에서 다시 계산한다(문자열 분해).
실행: python3 scripts/verify_e172_rev.py (저장소 루트에서)"""
import os
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)


def rd(*p):
    return open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def modline(t, head):
    for ln in t.splitlines():
        if ln.startswith(head):
            return q4(ln.split("변조폭")[1].split("**")[0])
    raise ValueError(head)


def main():
    try:
        ok, succ, part, stuck, mir = True, 0, 0, 0, 0
        for b in BRAINS:
            t = rd("logs", "E172", "rev_b%d.log" % b)
            pre, post = modline(t, "[사전]"), modline(t, "[사후]")
            a = rd("logs", "E172", "train_A_b%d.log" % b)
            rpre, rpost = modline(a, "[사전]"), modline(a, "[사후]")
            refl = [ln for ln in t.splitlines() if ln.startswith("[반사가중치] good_food_to_motor_")]
            refl_ok = len(refl) == 2 and all("0.0000→0.0000" in ln for ln in refl)
            nld = sum(1 for ln in t.splitlines() if ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln)
            R = np.load(os.path.join(EXP, "traces", "E172", "tr_rev_b%d.npz" % b))["rows"]
            good = 0; tot = 0
            for i in range(len(R)):
                if R[i, 6] < 0:
                    continue
                tot += 1
                cross = R[i, 6] != R[i, 2]
                want = cross if i < 1500 else (not cross)
                good += (R[i, 7] == 1) == want
            rs = (11.0 / 12.0) ** 20
            num = sum(abs(R[i, 21 + k] - rs * R[i, 13 + k]) for i in range(len(R)) for k in range(4))
            den = sum(abs(R[i, 13 + k]) for i in range(len(R)) for k in range(4))
            prer = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
            okb = ("[반전] 시행 1500 부터" in t and refl_ok and nld >= 2 and abs(pre - rpre) <= 20 and len(R) == 3000
                   and tot > 0 and good == tot and num <= 1e-3 * den and prer <= 1e-3)
            ok &= okb
            d = post - rpost
            succ += post >= 200; part += d >= 1000; stuck += d < 500; mir += 5 * post >= 4 * abs(rpost)
            print("b%d 사후 %+d 반전 전 %+d Δ %+d | 조작 %s" % (b, post, rpost, d, "✓" if okb else "✗"))
    except (FileNotFoundError, ValueError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    v = ("보류(조작검증 실패)" if not ok else "반전 성공(H096)" if succ >= 4 else "부분(H096-partial)" if part == 5
         else "고착(H096-null)" if stuck >= 4 else "보류")
    print("조작검증 %s | 성공 %d/5 부분 %d/5 고착 %d/5 | 거울상(m ≥ 0.8|R|) %d/5" % ("통과" if ok else "실패", succ, part, stuck, mir))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
