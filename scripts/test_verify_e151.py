#!/usr/bin/env python3
"""verify_e151_independent.py 합성 시험: 전이 감소+유지, 기본 전이+간섭, eta 줄 불일치·무학습 ≠ E150·Σ|Δg| 안 커짐(조작검증 실패), P2' 실패, η* 없음.
실행: python3 scripts/test_verify_e151.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e151_independent as V


def rows(n, ab, dg):
    rng = np.random.default_rng(n + ab)
    R = np.zeros((n, 27)); R[:, 2] = rng.integers(0, 2, n); R[:, 6] = rng.integers(0, 2, n)
    idx = np.arange(n)
    R[:, 7] = (np.where(idx < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]) if ab else (R[:, 6] != R[:, 2])).astype(float)
    R[:, 13:17] = 1.0; R[:, 21:25] = (11.0 / 12.0) ** 20; R[:, 12] = dg
    return R


def tlog(b_line, eta):
    return (("[과제 B] 시행 1500 부터 자극 = bad food, 정답 = 같은 쪽\n" if b_line else "") +
            "    KC→motor [E109 R-STDP 4방향]: init_w=150.0, w_max=300.0, eta=%s, tau_e=12.0, sparsity=0.25\n" % eta +
            "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")


def run(T, eA1=-3000, eB=2500, eta_line="0.9", ref_shift=0, dg150=1.0, star="0.9"):
    f = lambda x: "%+.4f" % (x / 1e4)
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E151", "logs/E150", "traces/E151", "traces/E150"):
            os.makedirs(os.path.join(td, d))
        open(os.path.join(td, "logs", "E151", "eta_star.txt"), "w", encoding="utf-8").write("eta_star %s\n" % star)
        for b in V.BRAINS:
            for a in ("A", "AB"):
                open(os.path.join(td, "logs", "E151", "train_%s_b%d.log" % (a, b)), "w", encoding="utf-8").write(tlog(a == "AB", eta_line))
                np.savez_compressed(os.path.join(td, "traces", "E151", "tr_%s_b%d.npz" % (a, b)), rows=rows(1500 if a == "A" else 3000, a == "AB", 2.0))
            np.savez_compressed(os.path.join(td, "traces", "E150", "tr_A_b%d.npz" % b), rows=rows(1500, False, dg150))
            d = int(round(T * eB))
            vals = {("none", "base"): 300, ("none", "bad"): 250, ("A", "base"): 300 + eA1, ("AB", "base"): 300 + eA1 + d, ("AB", "bad"): 250 + eB}
            for (w, s), m in vals.items():
                open(os.path.join(td, "logs", "E151", "ev_b%d_%s_%s.log" % (b, w, s)), "w", encoding="utf-8").write(
                    "=> DECOMP mode=all mod=%s acc=0.0\n[E146 변형] variant=%s vseed=0\n" % (f(m), s))
            for s, m in (("base", 300 + ref_shift), ("bad", 250)):
                open(os.path.join(td, "logs", "E150", "ev_b%d_none_%s.log" % (b, s)), "w", encoding="utf-8").write(
                    "=> DECOMP mode=none mod=%s acc=0.0\n[E146 변형] variant=%s vseed=0\n" % (f(m), s))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, args, kw, want in (("전이 감소 + 유지", (0.5,), {}, ("H074(전이 감소)", "전체 강도 유지")),
                             ("기본 전이 + 간섭", (1.2,), {}, ("H074-mag(기본 수준 전이)", "전체 강도 간섭")),
                             ("eta 줄 불일치", (0.5,), {"eta_line": "0.15"}, ("보류(조작검증 실패)", "보류(조작검증 실패)")),
                             ("무학습 ≠ E150", (0.5,), {"ref_shift": 1}, ("보류(조작검증 실패)", "보류(조작검증 실패)")),
                             ("Σ|Δg| 안 커짐", (0.5,), {"dg150": 2.0}, ("보류(조작검증 실패)", "보류(조작검증 실패)")),
                             ("P2' 실패", (0.5,), {"eB": 1400}, ("보류(학습 크기 미회복)", "보류(학습 크기 미회복)"))):
    out = run(*args, **kw)
    good = ("독립 판정 1(기전): %s" % want[0]) in out and ("독립 판정 2(능력): %s" % want[1]) in out
    ok_all &= good
    print("%-16s → %s %s" % (name, " / ".join(out.strip().splitlines()[-2:])[:70], "✓" if good else "✗\n" + out))
out = run(0.5, star="none")
good = "보류(학습 크기 미회복 — 보정에서 해당 η 없음" in out; ok_all &= good
print("%-16s → %s" % ("η* 없음", "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
