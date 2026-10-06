#!/usr/bin/env python3
"""verify_e144_independent.py 합성 시험: 답을 아는 원 로그(E144·E142 F1500)·추적·보정 추적으로 L2·부분·무관·조작검증 실패.
실행: python3 scripts/test_verify_e144.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e144_independent as V

F1500 = {10: 1026, 11: 1141, 12: 953, 13: 833, 14: 1266}


def rows(ot=0.0, same_sign=-1.0):
    rng = np.random.default_rng(3)
    R = np.zeros((1500, 37)); R[:300, 7] = 1
    R[:, 8] = 2.0; R[:, 9] = -1.0; R[:, 12] = R[:, 8:12].sum(1)
    R[:, 13:17] = rng.normal(0, 1000, (1500, 4)); R[:300, 14] = same_sign * np.abs(R[:300, 14])
    R[:, 21:25] = V.R20 * R[:, 13:17]
    R[:, 35] = 2.0; R[:, 36] = ot
    return R


def log(pre, post, rew, refl="25.0000"):
    f = lambda x: "%+.4f" % (x / 1e4)
    return ("[사전] 오프셋 +0.001 | 정답률 0.0%% | **변조폭 %s**\n[학습] 15ep 완료, 보상 %d회 (탐색)\n"
            "[반사가중치] good_food_to_motor_l   n=1 w_mean 25.0000→%s (학습 뇌)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (학습 뇌)\n"
            "[사후] 오프셋 +0.002 | **변조폭 %s**\n" % (f(pre), rew, refl, f(post)))


def run(post, ot=0.0, same_sign=-1.0, r0=0.02):
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E144", "logs/E142", "traces/E144/calib"):
            os.makedirs(os.path.join(td, d))
        Rc = np.zeros((100, 37)); Rc[:, 36] = r0
        np.savez_compressed(os.path.join(td, "traces", "E144", "calib", "tr_R0_5000_b15.npz"), rows=Rc)
        for b in V.PRE119:
            open(os.path.join(td, "logs", "E144", "b%d.log" % b), "w", encoding="utf-8").write(log(V.PRE119[b], post[b], 300))
            open(os.path.join(td, "logs", "E142", "F1500_b%d.log" % b), "w", encoding="utf-8").write(log(V.PRE119[b], F1500[b], 436))
            np.savez_compressed(os.path.join(td, "traces", "E144", "tr_b%d.npz" % b), rows=rows(ot, same_sign))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


B = list(V.PRE119)
ok_all = True
for name, post, kw, want in (
        ("L2", {b: -500 for b in B}, {}, "독립 판정: L2 달성(H067)"),
        ("부분", {b: F1500[b] - 500 for b in B}, {}, "독립 판정: 반대쪽 발화가 정체 원인(H067-partial)"),
        ("무관", {b: F1500[b] + 100 for b in B}, {}, "독립 판정: 무관(H067-null)"),
        ("반대쪽 침묵 경계 = R0+0.01", {b: -500 for b in B}, {"ot": 0.03, "r0": 0.02}, "독립 판정: L2 달성(H067)"),
        ("반대쪽 침묵 실패", {b: -500 for b in B}, {"ot": 0.05, "r0": 0.02}, "독립 판정: 보류(조작검증 실패)"),
        ("같은쪽 LTD 실패", {b: -500 for b in B}, {"same_sign": 1.0}, "독립 판정: 보류(조작검증 실패)")):
    out = run(post, **kw)
    good = want in out
    ok_all &= good
    print("%-24s → %s %s" % (name, [l for l in out.splitlines() if l.startswith("독립 판정")][0][7:], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
