#!/usr/bin/env python3
"""verify_e147_independent.py 합성 시험: 답을 아는 원 로그(E147·E146 학습)·추적으로 성공·부분·고착·규칙 불일치.
실행: python3 scripts/test_verify_e147.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e147_independent as V

REF = {10: -3194, 11: -3309, 12: -3215, 13: -3217, 14: -3486}


def log(pre, post, rev=True):
    f = lambda x: "%+.4f" % (x / 1e4)
    return (("[반전] 시행 1500 부터 정답 = 같은 쪽(good 쪽)\n" if rev else "") +
            "[사전] 오프셋 +0.0 | **변조폭 %s**\n[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n"
            "[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n[사후] 오프셋 +0.0 | **변조폭 %s**\n" % (f(pre), f(post)))


def rows(bad=False):
    rng = np.random.default_rng(2)
    R = np.zeros((3000, 27)); R[:, 2] = rng.integers(0, 2, 3000); R[:, 6] = rng.integers(0, 2, 3000)
    idx = np.arange(3000)
    R[:, 7] = np.where(idx < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]).astype(float)
    if bad:
        R[2000, 7] = 1 - R[2000, 7]
    R[:, 13:17] = 1.0; R[:, 21:25] = (11.0 / 12.0) ** 20
    return R


def run(post, bad=False, rev=True):
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E147", "logs/E146", "traces/E147"):
            os.makedirs(os.path.join(td, d))
        for b in V.BRAINS:
            open(os.path.join(td, "logs", "E147", "b%d.log" % b), "w", encoding="utf-8").write(log(V.PRE0[b], post[b], rev))
            open(os.path.join(td, "logs", "E146", "train_b%d.log" % b), "w", encoding="utf-8").write(log(V.PRE0[b], REF[b], False))
            np.savez_compressed(os.path.join(td, "traces", "E147", "tr_b%d.npz" % b), rows=rows(bad))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


B = list(V.BRAINS)
ok_all = True
for name, post, kw, want in (
        ("성공", {b: 500 for b in B}, {}, "독립 판정: 반전 성공(H070)"),
        ("부분", {b: REF[b] + 1500 for b in B}, {}, "독립 판정: 부분(H070-partial)"),
        ("고착", {b: REF[b] + 200 for b in B}, {}, "독립 판정: 고착(H070-stuck)"),
        ("규칙 불일치", {b: 500 for b in B}, {"bad": True}, "독립 판정: 보류(조작검증 실패)"),
        ("반전 줄 없음", {b: 500 for b in B}, {"rev": False}, "독립 판정: 보류(조작검증 실패)")):
    out = run(post, **kw)
    good = want in out
    ok_all &= good
    print("%-14s → %s %s" % (name, [l for l in out.splitlines() if l.startswith("독립 판정")][0][7:], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
