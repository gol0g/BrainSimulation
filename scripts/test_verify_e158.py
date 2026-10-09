#!/usr/bin/env python3
"""verify_e158_independent.py 합성 시험: L2·거스름·효과 없음, 엄격 달성/아님/경계, 반사 변함·배율 1줄·E157 재현 어긋남·동결 실패·결측.
실행: python3 scripts/test_verify_e158.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e158_independent as V

LD = "[E153 종류 입력 적재] k 검증 일치 — x\n"; SC = "[E157 종류 입력 배율] k=0.7000 검증 일치 — x\n"
FK = {10: 0.3519, 11: 0.3617, 12: 0.3513, 13: 0.3338, 14: 0.3714}
BASE = {10: 0.4148, 11: 0.4195, 12: 0.3954, 13: 0.3773, 14: 0.4248}


def rows(n, bad=False):
    R = np.zeros((n, 37)); R[:, 13] = 1.0; R[:, 14] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20; R[:, 22] = (11.0 / 12.0) ** 20
    if bad:
        R[:, 21] *= 2.0
    return R


def run(m1500=-0.20, e500=-0.40, refl_bad=False, nsc=2, fk_shift=0.0, freeze_bad=False, miss=False, m_per=None):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E158"), ("traces", "E158"), ("logs", "E142"), ("logs", "E157")):
            os.makedirs(os.path.join(td, *d))
        for b in V.BRAINS:
            open(os.path.join(td, "logs", "E142", "F500_b%d.log" % b), "w", encoding="utf-8").write("[사전] x | **변조폭 %+.4f** (y)\n" % BASE[b])
            open(os.path.join(td, "logs", "E157", "r25_Fk_b%d.log" % b), "w", encoding="utf-8").write("[사전] x | **변조폭 %+.4f** (y)\n" % FK[b])
            for arm, n in V.ARMS:
                if miss and arm == "F1500" and b == 13:
                    continue
                pre = FK[b] + fk_shift
                post = (pre + e500) if arm == "F500" else (m_per[b] if m_per else m1500)
                rw = "25.0000→24.0000" if (refl_bad and arm == "F500" and b == 12) else "25.0000→25.0000"
                open(os.path.join(td, "logs", "E158", "%s_b%d.log" % (arm, b)), "w", encoding="utf-8").write(
                    LD * 2 + SC * (nsc if (arm == "F1500" and b == 11) else 2) + "[사전] x | **변조폭 %+.4f**\n[반사가중치] good_food_to_motor_l   n=1 w_mean %s (x)\n"
                    "[반사가중치] good_food_to_motor_r   n=1 w_mean 25.0000→25.0000 (x)\n[사후] x | **변조폭 %+.4f**\n" % (pre, rw, post))
                np.savez_compressed(os.path.join(td, "traces", "E158", "tr_%s_b%d.npz" % (arm, b)), rows=rows(n, freeze_bad and arm == "F1500" and b == 10))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
cases = (("L2·엄격", {}, "L2 달성(H081)", "엄격 L2 달성"),
         ("L2·엄격 아님(m −0.05)", {"m1500": -0.05}, "L2 달성(H081)", "엄격 L2 아님"),
         ("엄격 경계 정확 5/5", {"m_per": {b: round(FK[b] - BASE[b], 4) for b in BASE}}, "L2 달성(H081)", "엄격 L2 달성"),
         ("엄격 0.0001 모자람", {"m_per": {b: round(FK[b] - BASE[b] + 0.0001, 4) for b in BASE}}, "L2 달성(H081)", "엄격 L2 아님"),
         ("거스름", {"m1500": 0.10, "e500": -0.30}, "반사를 거스름(H081-partial)", "엄격 L2 아님"),
         ("효과 없음", {"m1500": 0.35, "e500": -0.02}, "효과 없음(H081-null)", "엄격 L2 아님"),
         ("반사 변함", {"refl_bad": True}, "보류(조작검증 실패)", "보류(조작검증 실패)"),
         ("배율 1줄", {"nsc": 1}, "보류(조작검증 실패)", "보류(조작검증 실패)"),
         ("E157 재현 어긋남 0.0021", {"fk_shift": 0.0021}, "보류(조작검증 실패)", "보류(조작검증 실패)"),
         ("동결 실패", {"freeze_bad": True}, "보류(조작검증 실패)", "보류(조작검증 실패)"))
for name, kw, w1, w2 in cases:
    out = run(**kw)
    good = ("독립 판정 1: %s" % w1) in out and ("독립 판정 2: %s" % w2) in out
    ok_all &= good
    print("%-24s → %s | %s %s" % (name, [x for x in out.splitlines() if x.startswith("독립 판정 1")][0], out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
out = run(miss=True)
good = "독립 판정: 보류(결측" in out
ok_all &= good
print("%-24s → %s %s" % ("결측", out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
