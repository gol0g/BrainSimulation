#!/usr/bin/env python3
"""E151 독립 대조 — judge_e151.py 를 쓰지 않고 런별 원 로그(logs/E151/train_*·ev_*)·추적·E150 원 로그에서 다시 계산한다.
평가 mod 는 원 로그 DECOMP 줄을 1e-4 정수로(변형 표시 확인). 무학습 기준 재현은 E150 원 평가 로그와 비교.
학습 런: 과제 B 줄 유무·반사 0·KC→motor eta 줄 = η*·시행별 규칙 일치·동결 잔차·도파민 전 변화. eta 도달: A단독 Σ|시행 Δg| > E150.
실행: python3 scripts/verify_e151_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def i4(s):
    m = re.fullmatch(r"([-+]?)(\d+)\.(\d{4})", s)
    if not m:
        raise ValueError("4자리 값 아님: %r" % s)
    v = int(m.group(2)) * 10000 + int(m.group(3))
    return -v if m.group(1) == "-" else v


def mod_of(path, s):
    txt = open(path, encoding="utf-8", errors="replace").read()
    if ("[E146 변형] variant=%s " % s) not in txt:
        raise RuntimeError("변형 표시 불일치 %s" % path)
    return i4(re.search(r"^=> DECOMP mode=\S+ mod=([-+]?\d+\.\d{4})", txt, re.M).group(1))


def eta_star():
    tok = open(os.path.join(EXP, "logs", "E151", "eta_star.txt"), encoding="utf-8").read().split()
    return None if tok[1] == "none" else float(tok[1])


def train_ok(arm, b, eta):
    txt = open(os.path.join(EXP, "logs", "E151", "train_%s_b%d.log" % (arm, b)), encoding="utf-8", errors="replace").read()
    has_b = "[과제 B] 시행 1500 부터" in txt
    refl = re.findall(r"^\[반사가중치\] good_food_to_motor_[lr]\s+n=\d+ w_mean (\S+)→(\S+)", txt, re.M)
    etas = re.findall(r"KC→motor \[E109 R-STDP 4방향\]: init_w=\S+ w_max=\S+ eta=([^,]+),", txt)
    R = np.load(os.path.join(EXP, "traces", "E151", "tr_%s_b%d.npz" % (arm, b)))["rows"]
    good = tot = 0
    for i in range(len(R)):
        if R[i, 6] < 0:
            continue
        want = (R[i, 6] == R[i, 2]) if (arm == "AB" and i >= 1500) else (R[i, 6] != R[i, 2])
        good += int((R[i, 7] == 1) == want); tot += 1
    res = float(np.abs(R[:, 21:25] - ((11.0 / 12.0) ** 20) * R[:, 13:17]).sum() / np.abs(R[:, 13:17]).sum())
    pre = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
    n_ok = len(R) == (1500 if arm == "A" else 3000)
    eta_ok = len(etas) > 0 and all(abs(float(e) - eta) < 1e-9 for e in etas)
    return ((has_b == (arm == "AB")) and len(refl) == 2 and all(r == ("0.0000", "0.0000") for r in refl) and good == tot and tot > 0
            and res <= 1e-3 and pre <= 1e-3 and n_ok and eta_ok)


def main():
    eta = eta_star()
    if eta is None:
        print("독립 판정: 보류(학습 크기 미회복 — 보정에서 해당 η 없음, 본실험 없음)")
        return
    ok = sum(train_ok(a, b, eta) for a in ("A", "AB") for b in BRAINS)
    ref = sum(mod_of(os.path.join(EXP, "logs", "E151", "ev_b%d_none_%s.log" % (b, s)), s)
              == mod_of(os.path.join(EXP, "logs", "E150", "ev_b%d_none_%s.log" % (b, s)), s) for b in BRAINS for s in ("base", "bad"))
    grow = sum(np.abs(np.load(os.path.join(EXP, "traces", "E151", "tr_A_b%d.npz" % b))["rows"][:, 12]).sum()
               > np.abs(np.load(os.path.join(EXP, "traces", "E150", "tr_A_b%d.npz" % b))["rows"][:, 12]).sum() for b in BRAINS)
    p1 = p2 = sep = full = keep = 0
    for b in BRAINS:
        M = {(w, s): mod_of(os.path.join(EXP, "logs", "E151", "ev_b%d_%s_%s.log" % (b, w, s)), s)
             for (w, s) in (("A", "base"), ("AB", "base"), ("AB", "bad"), ("none", "base"), ("none", "bad"))}
        eA1 = M[("A", "base")] - M[("none", "base")]; eA = M[("AB", "base")] - M[("none", "base")]; eB = M[("AB", "bad")] - M[("none", "bad")]
        d = eA - eA1
        p1 += eA1 <= -2000; p2 += eB >= 1500
        s_ = eB >= 1500 and 10 * d <= 8 * eB; f_ = eB >= 1500 and d >= eB; k_ = eA1 < 0 and 2 * eA <= eA1
        sep += s_; full += f_; keep += k_
        print("b%d eA1 %+5d eA %+5d eB %+5d | T %s 몫 %s %s%s%s" % (b, eA1, eA, eB, ("%.2f" % (d / eB)) if eB else "nan",
                                                          ("%.2f" % (eA / eA1)) if eA1 else "nan", "전이감소 " if s_ else "", "기본전이 " if f_ else "", "유지" if k_ else ""))
    print("η* %g | 학습 런 조작검증 %d/10 · 무학습 = E150 %d/10 · Σ|Δg| > E150 %d/5 | P1' %d/5 P2' %d/5 | 전이 감소 %d/5 기본 전이 %d/5 유지 %d/5"
          % (eta, ok, ref, grow, p1, p2, sep, full, keep))
    if ok < 10 or ref < 10 or grow < 5:
        v1 = v2 = "보류(조작검증 실패)"
    elif p1 < 4 or p2 < 4:
        v1 = v2 = "보류(학습 크기 미회복)"
    else:
        v1 = "H074(전이 감소)" if sep >= 4 else ("H074-mag(기본 수준 전이)" if full >= 4 else "보류")
        v2 = "전체 강도 유지" if keep >= 4 else ("전체 강도 간섭" if 5 - keep >= 4 else "보류")
    print("독립 판정 1(기전): %s" % v1)
    print("독립 판정 2(능력): %s" % v2)


if __name__ == "__main__":
    sys.exit(main())
