#!/usr/bin/env python3
"""E138 독립 대조 — judge_e138.py·kc_selectivity.py 와 다른 입력·다른 코드로 다시 계산한다(기록 전 독립 대조).
입력: 요약 줄(E138.log)이 아니라 런별 원 로그 logs/E138/b*_<모드>.log 의 DECOMP 줄, 발화 수 npz traces/E138/rate_b*.npz.
분류는 정수 계수로 직접(부동소수 나눗셈 없이): 좌선택 ⇔ (e_L − e_R) ≥ θ·(e_L + e_R) 를 공통 분모로 바꾼 정수 부등식.
  e_L·D = max(cL·B − c0·3·P, 0) (D = P·B, P = 좌우 제시 수, B = 기준선 스텝) — 정수 산술.
  유발 합 > 0.02 ⇔ (eL_D + eR_D) > 0.02·D ⇔ 100·(eL_D + eR_D) > 2·D (정수)."""
import os
import re
import sys

import numpy as np

E = "research/experiments"
L = os.path.join(E, "logs", "E138")
T = os.path.join(E, "traces", "E138")
BR = [10, 11, 12, 13, 14]
MODES = ["none", "all", "kc_only", "kcpop", "kcsel", "kcselonly"]
PRE = {10: 0.0195, 11: 0.0150, 12: 0.0262, 13: 0.0320, 14: 0.0165}
POST = {10: -0.0838, 11: -0.0574, 12: -0.0428, 13: -0.0489, 14: -0.0442}

mod, bad = {}, []
FIXD = os.path.join(L, "fix")   # 2026-10-03 수리 재실행(kcrate·kcsel·kcselonly) — 있으면 그 원 로그를 쓴다
for b in BR:
    for m in MODES:
        p = os.path.join(FIXD if (m in ("kcsel", "kcselonly") and os.path.isdir(FIXD)) else L, "b%d_%s.log" % (b, m))
        s = open(p, encoding="utf-8").read() if os.path.exists(p) else ""
        g = re.search(r"^=> DECOMP mode=(\w+) mod=([-+0-9.]+)", s, flags=re.M)
        if not g or g.group(1) != m:
            bad.append(p); continue
        mod[(b, m)] = float(g.group(2))
print("DECOMP 파싱 %d/30, 결측·불일치 %d" % (len(mod), len(bad)))
if bad:
    print(" 예:", bad[:3]); sys.exit(1)


def cls_int(cL, cR, c0, P, Bs, num, den):
    """θ = num/den. 반환: 0 무활동, 1 좌선택, 2 우선택, 3 비선택 — 정수 산술."""
    cL = cL.astype(np.int64); cR = cR.astype(np.int64); c0 = c0.astype(np.int64)
    D = P * Bs
    eL = np.maximum(cL * Bs - c0 * 3 * P, 0); eR = np.maximum(cR * Bs - c0 * 3 * P, 0)
    tot = eL + eR
    resp = 100 * tot > 2 * D
    selL = resp & (den * (eL - eR) >= num * tot)
    selR = resp & (den * (eR - eL) >= num * tot)
    out = np.zeros(cL.size, dtype=np.int64)
    out[(cL + cR) > 0] = 3; out[selL] = 1; out[selR] = 2
    return out


print("뇌  집단  좌선택 우선택 비선택 무활동  희석    (원 로그 KCRATE 와 비교)")
agree = 0
for b in BR:
    Z = np.load(os.path.join(T, "fix", "rate_b%d.npz" % b)); Z1 = np.load(os.path.join(T, "rate_b%d.npz" % b))
    same_counts = all(np.array_equal(Z[k], Z1[k]) for k in Z.files)
    P, Bs = int(Z["n_pres"]), int(Z["base_steps"])
    s = open(os.path.join(FIXD, "b%d_kcrate.log" % b), encoding="utf-8").read()
    print("b%d 발화 수 npz 1차 = 수리 재실행: %s" % (b, same_counts))
    for p in "lr":
        c = cls_int(Z["cL_" + p], Z["cR_" + p], Z["c0_" + p], P, Bs, 1, 2)
        sp = (Z["cL_" + p] + Z["cR_" + p]).astype(np.float64)
        D = sp[c == 3].sum() / sp[c > 0].sum()
        n = [int((c == k).sum()) for k in (1, 2, 3, 0)]
        g = re.search(r"=> KCRATE kc_%s \| 좌선택 (\d+) 우선택 (\d+) 비선택 (\d+) 무활동 (\d+) \| 희석 ([0-9.]+)" % p, s)
        same = g is not None and [int(x) for x in g.groups()[:4]] == n and abs(float(g.group(5)) - D) < 0.0006
        agree += same
        print("b%d  kc_%s  %5d %5d %5d %5d   %.3f   %s" % (b, p, n[0], n[1], n[2], n[3], D, "일치" if same else "불일치 %s" % (str(g.groups() if g else None),)))
print("분류·희석 일치 %d/10" % agree)
print("뇌  none(E119)  all(E119)  e=kc_only−none  R=none−kcpop  R_sel=none−kcsel  e_so  ρ  κ")
rho, kap = {}, {}
for b in BR:
    n = mod[(b, "none")]; e = round(mod[(b, "kc_only")] - n, 6); R = round(n - mod[(b, "kcpop")], 6); Rs = round(n - mod[(b, "kcsel")], 6)
    eso = round(mod[(b, "kcselonly")] - n, 6)
    rho[b] = Rs / abs(e) if e < -0.005 else float("nan"); kap[b] = -eso / abs(e) if e < -0.005 else float("nan")
    print("b%d  %+.4f(%+.4f)  %+.4f(%+.4f)  %+.4f  %.4f  %.4f  %+.4f  %.2f  %.2f"
          % (b, n, PRE[b], mod[(b, "all")], POST[b], e, R, Rs, eso, rho[b], kap[b]))
print("ρ ≤ 2.5: %d/5, ρ ≥ 4: %d/5 | κ ≥ 1.5: %d/5, κ ≤ 1.1: %d/5"
      % (sum(rho[b] <= 2.5 for b in BR), sum(rho[b] >= 4 for b in BR), sum(kap[b] >= 1.5 for b in BR), sum(kap[b] <= 1.1 for b in BR)))
