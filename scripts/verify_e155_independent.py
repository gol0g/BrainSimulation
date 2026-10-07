#!/usr/bin/env python3
"""E155 독립 대조 — judge_e155.py 를 쓰지 않고 원 평가 로그(logs/E155/ev_b*_*_*.log)·원 반전 로그(logs/E155/rev_b*.log)·추적(traces/E155/tr_rev_b*.npz)에서 다시 계산한다.
평가: DECOMP 줄 mod(4자리 문자열 → 정수), 변형 표시, 적재 검증 줄. 반전: [사전]/[사후] 변조폭 줄, 반전 줄, 반사 줄, 적재 줄, 추적 행마다 규칙 일치·동결.
실행: python3 scripts/verify_e155_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
STIMS = ("base", "int05", "int07", "occ", "noise")
REF = {"E153_POST": ("-0.4893", "-0.4801", "-0.5024", "-0.4582", "-0.5143"), "E153_PRE": ("+0.0117", "+0.0015", "+0.0064", "+0.0231", "+0.0116"),
       "E154A_POST": ("-0.6015", "-0.5886", "-0.5881", "-0.5568", "-0.6153")}


def q(x):
    sgn = -1 if x.startswith("-") else 1
    a, b = x.lstrip("+-").split(".")
    return sgn * (int(a) * 10000 + int((b + "0000")[:4]))


def ref(name, b):
    return q(REF[name][BRAINS.index(b)])


def evmod(b, w, s):
    txt = open(os.path.join(EXP, "logs", "E155", "ev_b%d_%s_%s.log" % (b, w, s)), encoding="utf-8", errors="replace").read()
    if ("[E146 변형] variant=%s " % s) not in txt:
        raise RuntimeError("변형 표시 불일치 b%d %s %s" % (b, w, s))
    loaded = any(ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln for ln in txt.splitlines())
    return q(re.search(r"^=> DECOMP mode=\S+ mod=([-+]?\d+\.\d{4})", txt, re.M).group(1)), loaded


def main():
    M, L = {}, {}
    for b in BRAINS:
        for w in ("none", "E153", "E154A"):
            for s in STIMS:
                M[(b, w, s)], L[(b, w, s)] = evmod(b, w, s)
    v_ok = all(abs(M[(b, "E153", "base")] - ref("E153_POST", b)) <= 20 and abs(M[(b, "none", "base")] - ref("E153_PRE", b)) <= 20
               and abs(M[(b, "E154A", "base")] - ref("E154A_POST", b)) <= 20 for b in BRAINS) and all(L.values())
    e = {(b, w, s): M[(b, w, s)] - M[(b, "none", s)] for b in BRAINS for w in ("E153", "E154A") for s in STIMS}
    c1 = all(sum(e[(b, "E153", v)] <= -500 for b in BRAINS) == 5 for v in STIMS[1:])
    c3_fail = [v for v in STIMS[1:] if sum(2 * e[(b, "E153", v)] <= e[(b, "E153", "base")] < 0 for b in BRAINS) < 4]
    c4_fail = [s for s in STIMS if sum(e[(b, "E154A", s)] < e[(b, "E153", s)] for b in BRAINS) < 4]
    for s in STIMS:
        print("%-5s e500 %s | e1500 %s" % (s, [e[(b, "E153", s)] for b in BRAINS], [e[(b, "E154A", s)] for b in BRAINS]))
    if not v_ok:
        v1 = "보류(측정 검증 실패)"
    elif c1 and not c3_fail and not c4_fail:
        v1 = "통과(K80 보존)"
    elif len(c3_fail) >= 2:
        v1 = "일반화 실패"
    elif c1 and not c3_fail and len(c4_fail) >= 3:
        v1 = "용량 포화"
    else:
        v1 = "부분(보류)"
    print("판정 1 측정 검증 %s | C1 %s | C3 실패 %s | C4 실패 %s → %s" % ("통과" if v_ok else "실패", "통과" if c1 else "실패", c3_fail, c4_fail, v1))
    ok2 = 0
    succ = part = stuck = 0
    for b in BRAINS:
        txt = open(os.path.join(EXP, "logs", "E155", "rev_b%d.log" % b), encoding="utf-8", errors="replace").read()
        pre = q(re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d{4})", txt, re.M).group(1))
        post = q(re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d{4})", txt, re.M).group(1))
        refl = re.findall(r"^\[반사가중치\] good_food_to_motor_[lr]\s+n=\d+ w_mean (\S+)→(\S+)", txt, re.M)
        nld = sum(1 for ln in txt.splitlines() if ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln)
        R = np.load(os.path.join(EXP, "traces", "E155", "tr_rev_b%d.npz" % b))["rows"]
        agree = tot = 0
        for i in range(len(R)):
            if R[i, 6] < 0:
                continue
            tot += 1
            want = (R[i, 6] == R[i, 2]) if i >= 1500 else (R[i, 6] != R[i, 2])
            agree += int((R[i, 7] == 1) == want)
        res = float(np.abs(R[:, 21:25] - ((11.0 / 12.0) ** 20) * R[:, 13:17]).sum() / np.abs(R[:, 13:17]).sum())
        prer = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
        ok = ("[반전] 시행 1500 부터" in txt and len(refl) == 2 and all(r == ("0.0000", "0.0000") for r in refl) and nld >= 2
              and abs(pre - ref("E153_PRE", b)) <= 20 and len(R) == 3000 and tot > 0 and agree == tot and res <= 1e-3 and prer <= 1e-3)
        ok2 += ok
        d = post - ref("E154A_POST", b)
        succ += post >= 200; part += d >= 1000; stuck += d < 500
        print("b%d 반전 사후 %+d Δ %+d | 적재 %d 일치 %d/%d 동결 %.1e | %s" % (b, post, d, nld, agree, tot, res, "통과" if ok else "조작검증 실패"))
    if ok2 < 5:
        v2 = "보류(조작검증 실패)"
    elif succ >= 4:
        v2 = "통과(K81 보존 — 반전 성공)"
    elif part == 5:
        v2 = "부분"
    elif stuck >= 4:
        v2 = "고착"
    else:
        v2 = "보류"
    print("판정 2 조작검증 %d/5 → %s" % (ok2, v2))
    tot_v = "형성 표현 기준선 확립(H078)" if (v1.startswith("통과") and v2.startswith("통과")) else "기준선 미확립"
    print("독립 판정 1: %s" % v1)
    print("독립 판정 2: %s" % v2)
    print("독립 종합: %s" % tot_v)


if __name__ == "__main__":
    sys.exit(main())
