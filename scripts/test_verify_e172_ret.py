#!/usr/bin/env python3
"""verify_e172_ret.py 합성 시험: 유지·간섭·P1 실패·A 팔에 과제 B 줄·학습 적재 1줄·평가 적재 없음(조작검증 실패).
실행: python3 scripts/test_verify_e162.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e172_ret as V


def rows(n, ab):
    rng = np.random.default_rng(n + ab)
    R = np.zeros((n, 27)); R[:, 2] = rng.integers(0, 2, n); R[:, 6] = rng.integers(0, 2, n)
    idx = np.arange(n)
    R[:, 7] = (np.where(idx < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]) if ab else (R[:, 6] != R[:, 2])).astype(float)
    R[:, 13:17] = 1.0; R[:, 21:25] = (11.0 / 12.0) ** 20
    return R


LD = "[E153 종류 입력 적재] k.npz 검증 일치 — x\n"


def tlog(b_line, nld=2):
    return (("[과제 B] 시행 1500 부터 자극 = bad food, 정답 = 같은 쪽\n" if b_line else "") + LD * nld +
            "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")


def run(rA, eA1=-3000, eB=2000, a_has_b=False, nld=2, ev_ld=True):
    f = lambda x: "%+.4f" % (x / 1e4)
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E172", "traces/E172"):
            os.makedirs(os.path.join(td, d))
        for b in V.BRAINS:
            for a in ("A", "AB"):
                open(os.path.join(td, "logs", "E172", "train_%s_b%d.log" % (a, b)), "w", encoding="utf-8").write(tlog(a == "AB" or a_has_b, nld))
                np.savez_compressed(os.path.join(td, "traces", "E172", "tr_%s_b%d.npz" % (a, b)), rows=rows(1500 if a == "A" else 3000, a == "AB"))
            vals = {("none", "base"): 150, ("none", "bad"): 200, ("A", "base"): 150 + eA1, ("AB", "base"): 150 + int(rA * eA1), ("AB", "bad"): 200 + eB}
            for (w, s), m in vals.items():
                open(os.path.join(td, "logs", "E172", "ev_b%d_%s_%s.log" % (b, w, s)), "w", encoding="utf-8").write(
                    (LD if ev_ld else "") + "=> DECOMP mode=all mod=%s acc=0.0\n[E146 변형] variant=%s vseed=0\n" % (f(m), s))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, args, kw, want in (("유지", (0.8,), {}, "독립 판정: 유지(H095)"), ("간섭", (0.2,), {}, "독립 판정: 간섭(H095-null)"),
                             ("P1 실패(−0.19)", (0.8,), {"eA1": -1900}, "독립 판정: 보류(형성 표현 과제 A 전체 강도 미학습)"),
                             ("A 팔에 과제 B 줄", (0.8,), {"a_has_b": True}, "독립 판정: 보류(조작검증 실패)"),
                             ("학습 적재 1줄", (0.8,), {"nld": 1}, "독립 판정: 보류(조작검증 실패)")):
    out = run(*args, **kw)
    good = want in out
    ok_all &= good
    print("%-16s → %s %s" % (name, out.strip().splitlines()[-1][:40], "✓" if good else "✗\n" + out))
for name, rA, want in (("크기 0.80 정확", 0.8, "크기(rA ≥ 0.80) 5/5"), ("크기 0.79", 0.79, "크기(rA ≥ 0.80) 0/5")):
    out = run(rA)
    good = want in out
    ok_all &= good
    print("%-16s → %s %s" % (name, want, "✓" if good else "✗\n" + out))
try:
    run(0.8, ev_ld=False); ev_ok = False
except RuntimeError as e_:
    ev_ok = "평가 적재 줄 없음" in str(e_)
ok_all &= ev_ok
print("%-16s → %s" % ("평가 적재 없음", "✓ (중단)" if ev_ok else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
