#!/usr/bin/env python3
"""E164 독립 대조 — judge_e164.py 를 쓰지 않고 런별 원 로그·추적·기준 원 로그(E141·E142)에서 다시 계산한다(문자열 분해).
실행: python3 scripts/verify_e164_independent.py (저장소 루트에서)"""
import os
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
BASE = {"R0X": ("E141", "b%d.log"), "R25X": ("E142", "F500_b%d.log")}


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
        ok = True
        verdicts = {}
        for a, (be, pat) in BASE.items():
            ds = []
            for b in BRAINS:
                t = rd("logs", "E164", "%s_b%d.log" % (a, b))
                tb = rd("logs", be, pat % b)
                px, mx = modline(t, "[사전]"), modline(t, "[사후]")
                pb, mb = modline(tb, "[사전]"), modline(tb, "[사후]")
                chk = [ln for ln in t.splitlines() if ln.startswith("[구현 점검]")]
                da = [ln for ln in chk if "첫 보상 창 끝 도파민 뉴런 I_input" in ln]
                da_ok = len(da) == 1 and da[0].rstrip().endswith("→ 0.0")
                R = np.load(os.path.join(EXP, "traces", "E164", "tr_%s_b%d.npz" % (a, b)))["rows"]
                rs = (11.0 / 12.0) ** 20
                num = sum(abs(R[i, 21 + k] - rs * R[i, 13 + k]) for i in range(len(R)) for k in range(4))
                den = sum(abs(R[i, 13 + k]) for i in range(len(R)) for k in range(4))
                pre = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
                okb = (len(chk) >= 3 and any("rw_da_reset=True offset_steps=3" in ln for ln in chk)
                       and any(ln.startswith("[구현 점검] 오프셋(조향 3처리 합)") for ln in chk) and da_ok
                       and abs(px - pb) <= 20 and len(R) == 500 and num <= 1e-3 * den and pre <= 1e-3)
                ok &= okb
                d = (mx - px) - (mb - pb)
                ds.append(d)
                print("%s b%d e 수정 %+d 기준 %+d Δ %+d | 조작 %s" % (a, b, mx - px, mb - pb, d, "✓" if okb else "✗"))
            up = sum(d >= 500 for d in ds); dn = sum(d <= -500 for d in ds); sm = sum(abs(d) < 300 for d in ds)
            verdicts[a] = "영향 큼" if (up >= 4 or dn >= 4) else ("무시 가능" if sm >= 4 else "중간")
            print("%s: %s" % (a, verdicts[a]))
    except (FileNotFoundError, ValueError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    v = ("보류(조작검증 실패)" if not ok else "영향 큼(H087)" if any(x == "영향 큼" for x in verdicts.values())
         else "무시 가능(H087-null)" if all(x == "무시 가능" for x in verdicts.values()) else "보류(중간)")
    print("조작검증 %s" % ("통과" if ok else "실패"))
    print("독립 판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
