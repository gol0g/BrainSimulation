#!/usr/bin/env python3
"""E175 독립 대조(verify_e173·e175 를 옮김 — 조작검증 지표만 연합 결합 발화) — judge_e175.py 를 쓰지 않고 뇌별 원 로그(logs/E175/kcctx_b*·train_b*·ev_b*)와 추적(traces/E175/tr_bc_b*)에서
문자열 분해·정수 계산으로 다시 판정한다(E175.log 요약 줄은 읽지 않는다). 규칙은 logs/E175/criteria_fixed.txt.
실행: python3 scripts/verify_e175_independent.py (저장소 루트에서)"""
import os
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def rd(*p):
    return open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def mod_of(t):
    for ln in t.splitlines():
        if ln.startswith("=> DECOMP "):
            return q4(ln.split("mod=")[1].split()[0])
    raise ValueError("DECOMP 없음")


def nload(t):
    return sum(1 for ln in t.splitlines() if ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln)


def main():
    try:
        ok = True
        res = {}
        for b in BRAINS:
            k = rd("logs", "E175", "kcctx_b%d.log" % b)
            kl = [ln for ln in k.splitlines() if ln.startswith("=> KCCTX")][0]
            sp = kl.split("연합 결합 발화 끔 ")[1]
            k_off = int(sp.split(" 켬 ")[0]); k_on = int(sp.split(" 켬 ")[1].split()[0])
            t = rd("logs", "E175", "train_b%d.log" % b)
            cl = [ln for ln in t.splitlines() if ln.startswith("[맥락 과제] 시행 ")][0]
            ntr = int(cl.split("시행 ")[1].split()[0]); non = int(cl.split("맥락 켬 ")[1].split()[0])
            refl = [ln for ln in t.splitlines() if ln.startswith("[반사가중치] good_food_to_motor_")]
            refl_ok = len(refl) == 2 and all("0.0000→0.0000" in ln for ln in refl)
            R = np.load(os.path.join(EXP, "traces", "E175", "tr_bc_b%d.npz" % b))["rows"]
            good = tot = 0
            for i in range(len(R)):
                if R[i, 6] < 0:
                    continue
                tot += 1
                same = R[i, 6] == R[i, 2]
                want = same if R[i, 37] == 1 else (not same)
                good += (R[i, 7] == 1) == want
            rs = (11.0 / 12.0) ** 20
            num = float(np.abs(R[:, 21:25] - rs * R[:, 13:17]).sum()); den = float(np.abs(R[:, 13:17]).sum())
            prer = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
            M = {}
            ev_ok = True
            for w in ("learn", "none"):
                for c in ("off", "on"):
                    e = rd("logs", "E175", "ev_b%d_%s_%s.log" % (b, w, c))
                    M[(w, c)] = mod_of(e)
                    ev_ok &= nload(e) >= 1 and (("[맥락 평가]" in e) == (c == "on"))
            okb = (k_on > k_off and ntr == 3000 and 1350 <= non <= 1650 and refl_ok and nload(t) >= 2 and len(R) == 3000
                   and tot > 0 and good == tot and num <= 1e-3 * den and prer <= 1e-3 and ev_ok)
            ok &= okb
            eo = M[("learn", "off")] - M[("none", "off")]; en = M[("learn", "on")] - M[("none", "on")]
            res[b] = (eo, en)
            print("b%d e_off %+d e_on %+d | 맥락 켬 %d | 조작 %s" % (b, eo, en, non, "✓" if okb else "✗"))
    except (FileNotFoundError, ValueError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    acq = sum(eo <= -1000 and en >= 1000 for eo, en in res.values())
    elem = sum(abs(en - eo) < 1000 for eo, en in res.values())
    part = sum((en >= 1000 and eo > -1000) or (eo <= -1000 and en < 1000) for eo, en in res.values())
    v = ("보류(조작검증 실패)" if not ok else "획득(H097)" if acq >= 4 else "요소식(H097-null)" if elem >= 4 else "부분(H097-partial)" if part >= 4 else "보류")
    print("조작검증 %s | 획득 %d/5 요소식 %d/5 부분 %d/5" % ("통과" if ok else "실패", acq, elem, part))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
