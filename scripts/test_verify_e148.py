#!/usr/bin/env python3
"""verify_e148_independent.py 합성 시험: 답을 아는 원 로그(E148·E146)·추적으로 유지·간섭·과제 B 미학습·변형 표시 불일치.
실행: python3 scripts/test_verify_e148.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e148_independent as V

F1500 = {10: -3194, 11: -3309, 12: -3215, 13: -3217, 14: -3486}


def tlog(pre, post, b=True):
    f = lambda x: "%+.4f" % (x / 1e4)
    return (("[과제 B] 시행 1500 부터 자극 = bad food, 정답 = 같은 쪽\n" if b else "") +
            "[사전] 오프셋 +0.0 | **변조폭 %s**\n[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n"
            "[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n[사후] 오프셋 +0.0 | **변조폭 %s**\n" % (f(pre), f(post)))


def rows():
    rng = np.random.default_rng(4)
    R = np.zeros((3000, 27)); R[:, 2] = rng.integers(0, 2, 3000); R[:, 6] = rng.integers(0, 2, 3000)
    R[:, 7] = np.where(np.arange(3000) < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]).astype(float)
    R[:, 13:17] = 1.0; R[:, 21:25] = (11.0 / 12.0) ** 20
    return R


def run(rA, eB, bad_tag=False):
    f = lambda x: "%+.4f" % (x / 1e4)
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E148", "logs/E146", "traces/E148"):
            os.makedirs(os.path.join(td, d))
        for b in V.BRAINS:
            pre = V.PRE0[b]; eA1 = F1500[b] - pre
            open(os.path.join(td, "logs", "E148", "train_b%d.log" % b), "w", encoding="utf-8").write(tlog(pre, -1000))
            open(os.path.join(td, "logs", "E146", "train_b%d.log" % b), "w", encoding="utf-8").write(tlog(pre, F1500[b], False))
            np.savez_compressed(os.path.join(td, "traces", "E148", "tr_b%d.npz" % b), rows=rows())
            vals = {("none", "base"): pre, ("learn", "base"): pre + int(rA * eA1), ("none", "bad"): 100, ("learn", "bad"): 100 + eB}
            for (w, s), m in vals.items():
                tag = "base" if (bad_tag and (w, s) == ("learn", "bad") and b == 11) else s
                open(os.path.join(td, "logs", "E148", "ev_b%d_%s_%s.log" % (b, w, s)), "w", encoding="utf-8").write(
                    "=> DECOMP mode=all mod=%s acc=0.0\n[E146 변형] variant=%s vseed=0\n" % (f(m), tag))
        V.EXP = td
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                V.main()
        except RuntimeError as ex:
            return "예외: %s" % ex
    return buf.getvalue()


ok_all = True
for name, args, kw, want in (("유지", (0.8, 1000), {}, "독립 판정: 유지(H071)"), ("간섭", (0.2, 1000), {}, "독립 판정: 간섭(H071-int)"),
                             ("과제 B 미학습", (0.8, 300), {}, "독립 판정: 보류(과제 B 미학습)"), ("변형 표시 불일치", (0.8, 1000), {"bad_tag": True}, "예외: 변형 표시 불일치")):
    out = run(*args, **kw)
    good = want in out
    ok_all &= good
    print("%-14s → %s %s" % (name, out.strip().splitlines()[-1][:40], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
