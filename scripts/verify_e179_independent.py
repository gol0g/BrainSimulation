#!/usr/bin/env python3
"""E179 독립 대조 — judge_e179.py 를 쓰지 않고 뇌별 원 로그(logs/E179/kcctx_b*·train_b*·ev_b*)·추적(traces/E179/tr_bc_b*)에서
문자열 분해·정수 계산으로 다시 판정한다(E179.log 요약 줄은 읽지 않는다). 규칙은 logs/E179/criteria_fixed.txt.
E178 대조를 옮기고 맥락 끔·켬을 대칭 맥락 A·B 로 바꿨다(두 집단 발화 자리·A·B 평가 표지·기저값 −4000 고정·A·B 결정 표현·평가 4개 비포화를 따로 구현).
실행: python3 scripts/verify_e179_independent.py (저장소 루트에서)"""
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
        X = -4000 * 10000
        ok = True
        res = {}
        for b in BRAINS:
            k = rd("logs", "E179", "kcctx_b%d.log" % b)
            kl = [ln for ln in k.splitlines() if ln.startswith("=> KCCTX")][0]
            sp = kl.split("맥락 집단 발화 끔 ")[1]
            kB_off = int(sp.split(" 켬 ")[0]); kB_on = int(sp.split(" 켬 ")[1].split()[0])
            sp2 = kl.split("맥락2 집단 발화 끔 ")[1]
            kA_off = int(sp2.split(" 켬 ")[0]); kA_on = int(sp2.split(" 켬 ")[1].split()[0])
            t = rd("logs", "E179", "train_b%d.log" % b)
            c2_ok = "[맥락 입력 2] 맥락 전용 집단 zz_ctx2" in t
            cl = [ln for ln in t.splitlines() if ln.startswith("[맥락 과제] 시행 ")][0]
            ntr = int(cl.split("시행 ")[1].split()[0]); nB = int(cl.split("맥락 켬 ")[1].split()[0])
            refl = [ln for ln in t.splitlines() if ln.startswith("[반사가중치] good_food_to_motor_")]
            refl_ok = len(refl) == 2 and all("0.0000→0.0000" in ln for ln in refl)
            tn = tonics(t)
            ton_ok = len(tn) >= 1 and all(v == X for v in tn)
            R = np.load(os.path.join(EXP, "traces", "E179", "tr_bc_b%d.npz" % b))["rows"]
            good = tot = 0
            g = {0: 0, 1: 0}; sm = {0: 0, 1: 0}
            for i in range(len(R)):
                if R[i, 3] == 0:
                    c = int(R[i, 37] == 1)
                    g[c] += 1
                    sm[c] += abs(R[i, 5]) <= 0.02
                if R[i, 6] < 0:
                    continue
                tot += 1
                same = R[i, 6] == R[i, 2]
                want = same if R[i, 37] == 1 else (not same)
                good += (R[i, 7] == 1) == want
            dec_ok = all(g[c] > 0 and 2 * sm[c] <= g[c] for c in (0, 1))
            rs = (11.0 / 12.0) ** 20
            num = float(np.abs(R[:, 21:25] - rs * R[:, 13:17]).sum()); den = float(np.abs(R[:, 13:17]).sum())
            prer = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
            M = {}
            ev_ok = unsat_ok = True
            for w in ("learn", "none"):
                for c in ("A", "B"):
                    e = rd("logs", "E179", "ev_b%d_%s_%s.log" % (b, w, c))
                    M[(w, c)] = mod_of(e)
                    te = tonics(e)
                    m4 = motor4(e)
                    hasB = "[맥락 평가] ctx=켬" in e
                    hasA = "[맥락 평가 A]" in e
                    ev_ok &= nload(e) >= 1 and hasB == (c == "B") and hasA == (c == "A") and len(te) >= 1 and all(v == X for v in te) and m4 is not None
                    unsat_ok &= m4 is not None and not all(x >= 6600 for x in m4)
            okb = (kB_off == 0 and kB_on > 0 and kA_off > 0 and kA_on == 0 and c2_ok and ntr == 3000 and 1350 <= nB <= 1650 and refl_ok
                   and nload(t) >= 2 and len(R) == 3000 and tot > 0 and good == tot and num <= 1e-3 * den and prer <= 1e-3
                   and ev_ok and ton_ok and unsat_ok and dec_ok)
            ok &= okb
            eA = M[("learn", "A")] - M[("none", "A")]; eB = M[("learn", "B")] - M[("none", "B")]
            res[b] = (eA, eB)
            print("b%d e_A %+d e_B %+d | 맥락 B %d | 집단 A %d/%d B %d/%d | 결정 작음 A %d/%d B %d/%d | 조작 %s"
                  % (b, eA, eB, nB, kA_off, kA_on, kB_off, kB_on, sm[0], g[0], sm[1], g[1], "✓" if okb else "✗"))
    except (FileNotFoundError, ValueError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    acq = sum(eA <= -1000 and eB >= 1000 for eA, eB in res.values())
    elem = sum(abs(eB - eA) < 1000 for eA, eB in res.values())
    pB = sum(eB >= 1000 and eA > -1000 for eA, eB in res.values())
    pA = sum(eA <= -1000 and eB < 1000 for eA, eB in res.values())
    v = ("보류(조작검증 실패)" if not ok else "획득(H100)" if acq >= 4 else "요소식(H100-null)" if elem >= 4
         else ("부분 B만" if pB >= 4 else "부분") if pB + pA >= 4 else "보류")
    print("조작검증 %s | 획득 %d/5 요소식 %d/5 부분 B만 %d/5 A만 %d/5" % ("통과" if ok else "실패", acq, elem, pB, pA))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
