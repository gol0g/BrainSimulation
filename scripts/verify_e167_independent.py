#!/usr/bin/env python3
"""E167 독립 대조 — judge_e167.py 를 쓰지 않고 원 로그(E167 N·E136 H)를 문자열 분해로, 발달 저장본을 직접 열어 다시 계산한다.
규칙은 logs/E167/criteria_fixed.txt(남음 = ΣΔ_N ≥ 0.8·ΣΔ_H·Δ_N > 0 ≥ 14/16, 손실 = 평균 Δ_N < 5%p, 그 밖 부분). 백분율은 실수 평균으로(판정 코드의 0.1%p 정수 합과 다른 경로).
실행: python3 scripts/verify_e167_independent.py (저장소 루트에서)"""
import os
import sys

import numpy as np

EXP = "research/experiments"
WIRES = tuple(range(78, 94))


def rd(*p):
    return open(os.path.join(EXP, *p), encoding="utf-8", errors="replace").read()


def field(t, head, key, mode=None):
    for ln in t.splitlines():
        if ln.startswith(head):
            kv = dict(tok.split("=", 1) for tok in ln.split() if "=" in tok and not tok.startswith("|"))
            if mode is not None and kv.get("mode") != mode:
                raise ValueError("mode")
            return kv[key]
    raise ValueError(head)


def main():
    try:
        ok = True
        dN, dH = {}, {}
        for w in WIRES:
            z7 = np.load(os.path.join(EXP, "traces", "E167", "dev_corr_w%d.npz" % w))
            z6 = np.load(os.path.join(EXP, "traces", "E136", "dev_corr_w%d.npz" % w))
            okw = all(z7[k].tolist() == z6[k].tolist() for k in ("a", "b", "e", "i"))
            nv = {}
            for g, e, pat in (("N", "E167", "N_%s_w%d_t%d.log"), ("H", "E136", "corr_%s_w%d_t%d.log")):
                for mode in ("learn", "frozen"):
                    vals = []
                    for ts in (600, 601):
                        t = rd("logs", e, pat % (mode, w, ts))
                        vals.append(float(field(t, "=> SDLAB", "novel_lbal", mode)))
                        if g == "N":
                            ln = next(x for x in t.splitlines() if x.startswith("[KC망안]"))
                            nc = int(ln.split("흥분 후보 ")[1].split("개")[0]); ni = int(ln.split("억제 후보 ")[1].split("개")[0])
                            dm = float(ln.rsplit("|차|", 1)[1].strip())
                            okw &= nc == z7["cpre"].size and ni == z7["ipre"].size and dm <= 1e-6
                    nv[(g, mode)] = sum(vals) / 2.0
            ok &= okw
            dN[w] = nv[("N", "learn")] - nv[("N", "frozen")]
            dH[w] = nv[("H", "learn")] - nv[("H", "frozen")]
            print("w%d Δ_N %+.2f Δ_H %+.2f | 조작 %s" % (w, dN[w], dH[w], "✓" if okw else "✗"))
    except (FileNotFoundError, ValueError, KeyError, IndexError, StopIteration) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    sN, sH = sum(dN.values()), sum(dH.values())
    pos = sum(v > 0 for v in dN.values())
    if not ok:
        v = "보류(조작검증 실패)"
    elif sN >= 0.8 * sH - 1e-9 and pos >= 14:
        v = "남음(H090)"
    elif sN / len(WIRES) < 5.0 - 1e-9:
        v = "손실(H090-null)"
    else:
        v = "부분"
    print("평균 Δ_N %+.2f%%p · Δ_H %+.2f%%p · 비 %.3f · Δ_N > 0 %d/16" % (sN / 16, sH / 16, sN / sH if sH else float("nan"), pos))
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
