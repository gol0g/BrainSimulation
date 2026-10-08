#!/usr/bin/env python3
"""verify_e156_independent.py 합성 시험: L2 달성·거스름·효과 없음·반사 변함·적재 1줄·두 팔 출발점 다름·동결 실패.
실행: python3 scripts/test_verify_e156.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e156_independent as V

LD = "[E153 종류 입력 적재] k 검증 일치 — x\n"


def rows(n, bad=False):
    R = np.zeros((n, 37)); R[:, 13] = 1.0; R[:, 14] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20; R[:, 22] = (11.0 / 12.0) ** 20
    if bad:
        R[:, 21] *= 2.0
    return R


BASEPRE = {10: 0.4148, 11: 0.4195, 12: 0.3954, 13: 0.3773, 14: 0.4248}


def run(m1500=-0.10, e500=-0.40, refl_bad=False, nld=2, pre_shift=0.0, freeze_bad=False, wst=None, mW=-0.10, dW=0.0, wmiss=False, wrefl=None):
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E156", "traces/E156", "logs/E142"):
            os.makedirs(os.path.join(td, d))
        for b in V.BRAINS:
            open(os.path.join(td, "logs", "E142", "F500_b%d.log" % b), "w", encoding="utf-8").write("[사전] x | **변조폭 %+.4f** (y)\n" % BASEPRE[b])
        if wst is not None:
            open(os.path.join(td, "logs", "E156", "wstar.txt"), "w", encoding="utf-8").write("W*=%d 사전 +0.4100 (목표)\n" % wst)
            for b in V.BRAINS:
                if wmiss and b == 13:
                    continue
                wv = "%.4f" % (wrefl if (wrefl is not None and b == 11) else wst)
                open(os.path.join(td, "logs", "E156", "W1500_b%d.log" % b), "w", encoding="utf-8").write(
                    LD * 2 + "[사전] x | **변조폭 %+.4f**\n[반사가중치] good_food_to_motor_l   n=1 w_mean %.4f→%s (x)\n"
                    "[반사가중치] good_food_to_motor_r   n=1 w_mean %.4f→%.4f (x)\n[사후] x | **변조폭 %+.4f**\n" % (BASEPRE[b] + dW, wst, wv, wst, wst, mW))
                np.savez_compressed(os.path.join(td, "traces", "E156", "tr_W1500_b%d.npz" % b), rows=rows(1500))
        for b in V.BRAINS:
            for arm, n in V.ARMS:
                pre = 0.4000 + (pre_shift if arm == "F1500" else 0.0)
                post = (0.4000 + e500) if arm == "F500" else m1500
                rw = "25.0000→24.0000" if (refl_bad and arm == "F500" and b == 12) else "25.0000→25.0000"
                open(os.path.join(td, "logs", "E156", "%s_b%d.log" % (arm, b)), "w", encoding="utf-8").write(
                    LD * nld + "[사전] x | **변조폭 %+.4f**\n[반사가중치] good_food_to_motor_l   n=1 w_mean %s (x)\n"
                    "[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (x)\n[사후] x | **변조폭 %+.4f**\n" % (pre, rw, post))
                np.savez_compressed(os.path.join(td, "traces", "E156", "tr_%s_b%d.npz" % (arm, b)), rows=rows(n, freeze_bad and arm == "F1500" and b == 10))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("L2 달성", {}, "L2 달성(H079)"), ("거스름", {"m1500": 0.10, "e500": -0.30}, "반사를 거스름(H079-partial)"),
                       ("효과 없음", {"m1500": 0.35, "e500": -0.02}, "효과 없음(H079-null)"), ("반사 변함", {"refl_bad": True}, "보류(조작검증 실패)"),
                       ("적재 1줄", {"nld": 1}, "보류(조작검증 실패)"), ("두 팔 출발점 다름", {"pre_shift": 0.0021}, "보류(조작검증 실패)"),
                       ("동결 실패", {"freeze_bad": True}, "보류(조작검증 실패)")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out and "독립 판정 2: 없음(보정 실패)" in out
    ok_all &= good
    print("%-16s → %s %s" % (name, [x for x in out.splitlines() if x.startswith("독립 판정:")][0], "✓" if good else "✗\n" + out))
for name, kw, want2, wantc in (("판정2 이김", {"wst": 90}, "반사 발현을 맞춰도 학습이 이김", "H079 지지"),
                               ("판정2 L2 아님", {"wst": 90, "mW": 0.05}, "반사 발현을 맞추면 L2 아님", "H079 부분"),
                               ("판정2 경계 m −0.02", {"wst": 90, "mW": -0.02}, "반사 발현을 맞춰도 학습이 이김", "H079 지지"),
                               ("판정2 m −0.0199", {"wst": 90, "mW": -0.0199}, "반사 발현을 맞추면 L2 아님", "H079 부분"),
                               ("MW −0.05 경계", {"wst": 90, "dW": -0.05}, "반사 발현을 맞춰도 학습이 이김", "H079 지지"),
                               ("MW −0.0501", {"wst": 90, "dW": -0.0501}, "보류(조작검증 실패)", "판정 1(L2 달성(H079)) + 식별 불가"),
                               ("MW +0.1001", {"wst": 90, "dW": 0.1001}, "보류(조작검증 실패)", "판정 1(L2 달성(H079)) + 식별 불가"),
                               ("W 반사 변함", {"wst": 90, "wrefl": 89}, "보류(조작검증 실패)", "판정 1(L2 달성(H079)) + 식별 불가"),
                               ("W 결측", {"wst": 90, "wmiss": True}, "보류(결측)", "판정 1(L2 달성(H079)) + 식별 불가"),
                               ("미해결 조합", {"wst": 90, "mW": 0.05, "m1500": 0.10, "e500": -0.30}, "반사 발현을 맞추면 L2 아님", "판정 1 그대로(반사를 거스름(H079-partial)) — L2 미해결"),
                               ("예측 밖 조합", {"wst": 90, "m1500": 0.10, "e500": -0.30}, "반사 발현을 맞춰도 학습이 이김", "보류(예측 밖)")):
    out = run(**kw)
    good = ("독립 판정 2: %s" % want2) in out and ("독립 종합: %s" % wantc) in out
    ok_all &= good
    print("%-16s → %s | %s %s" % (name, [x for x in out.splitlines() if x.startswith("독립 판정 2:")][0], out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
