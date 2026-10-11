#!/usr/bin/env python3
"""E178 독립 대조 — judge_e178.py 를 쓰지 않고 뇌별 원 로그(logs/E178/kcctx_b*·train_b*·ev_b*)·추적(traces/E178/tr_bc_b*)·pick.txt 에서
문자열 분해·정수 계산으로 다시 판정한다(E178.log 요약 줄은 읽지 않는다). 규칙은 logs/E178/criteria_fixed.txt.
verify_e177_independent.py 를 옮기고 조작검증 셋(동작점 줄 값 = X, 맥락 켬 평가 비포화, 학습 중 맥락 켬 탐욕 결정 표현)을 따로 구현했다.
실행: python3 scripts/verify_e178_independent.py (저장소 루트에서)"""
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


def tonics(t):
    """'[motor 동작점] ... 기저값 <값>(...)' 줄의 값(정수 — 소수점 아래는 버리지 않고 0 이어야 함)."""
    out = []
    for ln in t.splitlines():
        if ln.strip().startswith("[motor 동작점]"):
            v = ln.split("기저값 ")[1].split("(")[0]
            out.append(q4(v))
    return out


def motor4(t):
    """진단 줄 → [좌 자극 L, 좌 자극 R, 우 자극 L, 우 자극 R](1e-4 정수) 또는 None."""
    d = {}
    for ln in t.splitlines():
        if ln.startswith("[E171 평가 진단]"):
            side = ln.split("side=")[1].split()[0]
            a, b = ln.split("motor L/R ")[1].split()[0].split("/")
            d[side] = [q4(a), q4(b)]
    return (d["left"] + d["right"]) if set(d) == {"left", "right"} else None


def main():
    try:
        pk = rd("logs", "E178", "pick.txt").strip()
        X = int(pk.split("X=")[1]) * 10000          # 'X=none' 이면 ValueError → 결측
        ok = True
        res = {}
        for b in BRAINS:
            k = rd("logs", "E178", "kcctx_b%d.log" % b)
            kl = [ln for ln in k.splitlines() if ln.startswith("=> KCCTX")][0]
            sp = kl.split("맥락 집단 발화 끔 ")[1]
            k_off = int(sp.split(" 켬 ")[0]); k_on = int(sp.split(" 켬 ")[1].split()[0])
            t = rd("logs", "E178", "train_b%d.log" % b)
            cl = [ln for ln in t.splitlines() if ln.startswith("[맥락 과제] 시행 ")][0]
            ntr = int(cl.split("시행 ")[1].split()[0]); non = int(cl.split("맥락 켬 ")[1].split()[0])
            refl = [ln for ln in t.splitlines() if ln.startswith("[반사가중치] good_food_to_motor_")]
            refl_ok = len(refl) == 2 and all("0.0000→0.0000" in ln for ln in refl)
            tn = tonics(t)
            ton_ok = len(tn) >= 1 and all(v == X for v in tn)
            R = np.load(os.path.join(EXP, "traces", "E178", "tr_bc_b%d.npz" % b))["rows"]
            good = tot = g_on = s_on = 0
            for i in range(len(R)):
                if R[i, 3] == 0 and R[i, 37] == 1:
                    g_on += 1
                    s_on += abs(R[i, 5]) <= 0.02
                if R[i, 6] < 0:
                    continue
                tot += 1
                same = R[i, 6] == R[i, 2]
                want = same if R[i, 37] == 1 else (not same)
                good += (R[i, 7] == 1) == want
            dec_ok = g_on > 0 and 2 * s_on <= g_on          # 비율 ≤ 0.50
            rs = (11.0 / 12.0) ** 20
            num = float(np.abs(R[:, 21:25] - rs * R[:, 13:17]).sum()); den = float(np.abs(R[:, 13:17]).sum())
            prer = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
            M = {}
            ev_ok = True
            unsat_ok = True
            for w in ("learn", "none"):
                for c in ("off", "on"):
                    e = rd("logs", "E178", "ev_b%d_%s_%s.log" % (b, w, c))
                    M[(w, c)] = mod_of(e)
                    te = tonics(e)
                    m4 = motor4(e)
                    ev_ok &= nload(e) >= 1 and (("[맥락 평가]" in e) == (c == "on")) and len(te) >= 1 and all(v == X for v in te) and m4 is not None
                    if c == "on":
                        unsat_ok &= m4 is not None and not all(x >= 6600 for x in m4)
            okb = (k_off == 0 and k_on > 0 and ntr == 3000 and 1350 <= non <= 1650 and refl_ok and nload(t) >= 2 and len(R) == 3000
                   and tot > 0 and good == tot and num <= 1e-3 * den and prer <= 1e-3 and ev_ok and ton_ok and unsat_ok and dec_ok)
            ok &= okb
            eo = M[("learn", "off")] - M[("none", "off")]; en = M[("learn", "on")] - M[("none", "on")]
            res[b] = (eo, en)
            print("b%d e_off %+d e_on %+d | 맥락 켬 %d | 동작점 %s 켬 비포화 %s 켬 결정 작음 %d/%d | 조작 %s"
                  % (b, eo, en, non, "✓" if ton_ok else "✗", "✓" if unsat_ok else "✗", s_on, g_on, "✓" if okb else "✗"))
    except (FileNotFoundError, ValueError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    acq = sum(eo <= -1000 and en >= 1000 for eo, en in res.values())
    elem = sum(abs(en - eo) < 1000 for eo, en in res.values())
    part = sum((en >= 1000 and eo > -1000) or (eo <= -1000 and en < 1000) for eo, en in res.values())
    v = ("보류(조작검증 실패)" if not ok else "획득(H099)" if acq >= 4 else "요소식(H099-null)" if elem >= 4 else "부분(H099-partial)" if part >= 4 else "보류")
    print("조작검증 %s | 획득 %d/5 요소식 %d/5 부분 %d/5" % ("통과" if ok else "실패", acq, elem, part))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
