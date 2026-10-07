#!/usr/bin/env python3
"""verify_e153_independent.py 합성 시험: 형성 성공·분리 실패·권한 상실·적재 줄 부족·동결 실패·형성 안 됨.
실행: python3 scripts/test_verify_e153.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e153_independent as V

BASE = {10: -2384, 11: -2583, 12: -2733, 13: -2514, 14: -2715}
LOADLN = "[E153 종류 입력 적재] x.npz 검증 일치 — good_food_eye_l_to_kc_l=1.00\n"


def rows(res_bad=False):
    rng = np.random.default_rng(7)
    R = np.zeros((500, 37)); R[:, 2] = rng.integers(0, 2, 500); R[:, 6] = rng.integers(0, 2, 500)
    R[:, 7] = (R[:, 6] != R[:, 2]).astype(float); R[:, 13:17] = 1.0; R[:, 21:25] = (11.0 / 12.0) ** 20
    if res_bad:
        R[:, 21:25] *= 1.5
    return R


def run(J="0.1200", ef=1.0, sel="0.9500", n_tr_load=2, res_bad=False):
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E153", "traces/E153"):
            os.makedirs(os.path.join(td, d))
        for b in V.BRAINS:
            dl = "side=%s fired=100 sel_med0=0.5500 sel_med=%s frac09=0.9000 goodfrac=0.5000 relerr=2.22e-16"
            open(os.path.join(td, "logs", "E153", "dev_b%d.log" % b), "w", encoding="utf-8").write(
                "x\n=> KCDEV %s | %s | n=100 eta=0.1 updates=400 save=y\n" % (dl % ("l", sel), dl % ("r", sel)))
            ol = "side=%s good=90 bad=91 jac=%s cos=0.3 jac025=0.1 jac100=0.1 split_jac=0.95 split_cos=0.99"
            open(os.path.join(td, "logs", "E153", "ov_b%d.log" % b), "w", encoding="utf-8").write(
                LOADLN + "=> KCOVERLAP %s | %s | food_eye_scale=1.00\n" % (ol % ("l", J), ol % ("r", J)))
            e = int(round(BASE[b] * ef))
            open(os.path.join(td, "logs", "E153", "train_b%d.log" % b), "w", encoding="utf-8").write(
                LOADLN * n_tr_load + "[사전] 오프셋 +0.0 | 정답률 0.0%% | **변조폭 +0.0200**\n"
                "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n"
                "[사후] 오프셋 +0.0 | 정답률 60.0%% | **변조폭 %+.4f**\n" % ((200 + e) / 1e4))
            np.savez_compressed(os.path.join(td, "traces", "E153", "tr_b%d.npz" % b), rows=rows(res_bad))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("형성 성공", {}, "형성 성공(H076)"), ("분리 실패", {"J": "0.4000"}, "분리 실패(H076-null)"),
                       ("권한 상실", {"ef": 0.5}, "권한 상실(H076-auth)"), ("적재 1줄", {"n_tr_load": 1}, "보류(조작검증 실패)"),
                       ("동결 실패", {"res_bad": True}, "보류(조작검증 실패)"), ("형성 안 됨", {"sel": "0.7000"}, "보류(조작검증 실패)")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out
    ok_all &= good
    print("%-12s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
