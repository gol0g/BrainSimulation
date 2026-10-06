#!/usr/bin/env python3
"""verify_e143_independent.py 합성 시험: 답을 아는 원 로그(E143·E142 F1500)·추적으로 L2·원인·무관·조작검증 실패가 나오는지.
실행: python3 scripts/test_verify_e143.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e143_independent as V

F1500 = {10: 1026, 11: 1141, 12: 953, 13: 833, 14: 1266}


def rows(dec_ok=True, ncol=35):
    rng = np.random.default_rng(7)
    R = np.zeros((1500, ncol)); R[:300, 7] = 1
    R[:, 8] = 2.0; R[:, 9] = -1.0; R[:, 12] = R[:, 8:12].sum(1)
    R[:, 13:17] = rng.normal(0, 1000, (1500, 4)); R[:, 21:25] = V.R20 * R[:, 13:17]
    if ncol >= 35:
        R[:, 27:31] = rng.normal(0, 40, (1500, 4)); R[:, 31:35] = (V.R30 if dec_ok else 1.0) * R[:, 27:31] + (0.0 if dec_ok else 500.0)
    return R


def log(pre, post, rew, refl="25.0000"):
    f = lambda x: "%+.4f" % (x / 1e4)
    return ("[사전] 오프셋 +0.001 | 정답률 0.0%% | **변조폭 %s**\n[학습] 15ep 완료, 보상 %d회 (탐색)\n"
            "[반사가중치] good_food_to_motor_l   n=1 w_mean 25.0000→%s (학습 뇌)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (학습 뇌)\n"
            "[사후] 오프셋 +0.002 | **변조폭 %s**\n" % (f(pre), rew, refl, f(post)))


def run(post, dec_ok=True, refl_bad=None, ncol=35):
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E143", "logs/E142", "traces/E143"):
            os.makedirs(os.path.join(td, d))
        for b in V.PRE119:
            open(os.path.join(td, "logs", "E143", "b%d.log" % b), "w", encoding="utf-8").write(
                log(V.PRE119[b], post[b], 300, "24.0000" if refl_bad == b else "25.0000"))
            open(os.path.join(td, "logs", "E142", "F1500_b%d.log" % b), "w", encoding="utf-8").write(log(V.PRE119[b], F1500[b], 436))
            np.savez_compressed(os.path.join(td, "traces", "E143", "tr_b%d.npz" % b), rows=rows(dec_ok, ncol))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


B = list(V.PRE119)
ok_all = True
for name, post, kw, want in (
        ("L2", {b: -500 for b in B}, {}, "독립 판정: L2 달성(H066)"),
        ("원인(d −0.05)", {b: F1500[b] - 500 for b in B}, {}, "독립 판정: 결정 단계 흔적이 정체 원인(H066-dec)"),
        ("원인 경계 d=−0.03", {b: F1500[b] - 300 for b in B}, {}, "독립 판정: 결정 단계 흔적이 정체 원인(H066-dec)"),
        ("무관", {b: F1500[b] + 100 for b in B}, {}, "독립 판정: 결정 단계 흔적 무관(H066-null)"),
        ("반대", {b: F1500[b] + 400 for b in B}, {}, "독립 판정: 반대(H066-rev)"),
        ("결정 동결 실패", {b: -500 for b in B}, {"dec_ok": False}, "독립 판정: 보류(조작검증 실패)"),
        ("옛 27열 추적", {b: -500 for b in B}, {"ncol": 27}, "독립 판정: 보류(조작검증 실패)"),
        ("반사 변함", {b: -500 for b in B}, {"refl_bad": 12}, "독립 판정: 보류(조작검증 실패)")):
    out = run(post, **kw)
    good = want in out
    ok_all &= good
    print("%-18s → %s %s" % (name, [l for l in out.splitlines() if l.startswith("독립 판정")][0][7:], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
