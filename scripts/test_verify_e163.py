#!/usr/bin/env python3
"""verify_e163_independent.py 합성 시험: 성공·부분·고착·보류, 반전 줄·반사·적재·출발점·규칙 일치 실패, 결측.
실행: python3 scripts/test_verify_e163.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e163_independent as V

LDL = "[E153 종류 입력 적재] k 검증 일치 — x\n"


def rows(bad_rule=False):
    R = np.zeros((3000, 37)); R[:, 2] = np.arange(3000) % 2
    R[:, 6] = np.where(np.arange(3000) < 1500, 1 - R[:, 2], R[:, 2]); R[:, 7] = 1
    if bad_rule:
        R[10, 7] = 0
    R[:, 13] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20
    return R


def run(m=0.30, rev_line=True, refl="0.0000→0.0000", nld=2, pre=0.0100, bad_rule=False, miss=False):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E163"), ("logs", "E162"), ("traces", "E163")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            w(("logs", "E162", "train_A_b%d.log" % b), "[사전] x | **변조폭 +0.0100** (y)\n[사후] x | **변조폭 -0.6000**\n")
            if miss and b == 14:
                continue
            w(("logs", "E163", "rev_b%d.log" % b), (("[반전] 시행 1500 부터 규칙 = 같은 쪽\n" if rev_line else "") + LDL * nld
               + "[사전] x | **변조폭 %+.4f** (y)\n[반사가중치] good_food_to_motor_l   n=1 w_mean %s (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n"
               "[사후] x | **변조폭 %+.4f**\n" % (pre if b == 12 else 0.0100, refl if b == 13 else "0.0000→0.0000", m)))
            np.savez_compressed(os.path.join(td, "traces", "E163", "tr_rev_b%d.npz" % b), rows=rows(bad_rule and b == 10))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("성공", {}, "반전 성공(H086)"), ("부분", {"m": -0.40}, "부분(H086-partial)"), ("고착", {"m": -0.58}, "고착(H086-null)"),
                       ("Δ 0.05 정확 → 보류", {"m": -0.55}, "보류"), ("반전 줄 없음", {"rev_line": False}, "보류(조작검증 실패)"),
                       ("반사 변함", {"refl": "0.0000→1.0000"}, "보류(조작검증 실패)"), ("적재 1줄", {"nld": 1}, "보류(조작검증 실패)"),
                       ("출발점 어긋남", {"pre": 0.0121}, "보류(조작검증 실패)"), ("규칙 불일치 1시행", {"bad_rule": True}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out and (want != "보류" or out.strip().endswith("독립 판정: 보류"))
    ok_all &= good
    print("%-20s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
