#!/usr/bin/env python3
"""verify_e146_independent.py 합성 시험: 답을 아는 원 로그·추적으로 충족·일반화 실패·용량 포화·측정 검증 실패(변형 표시 불일치 포함).
실행: python3 scripts/test_verify_e146.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e146_independent as V


def make(td, e500, e1500, post1500=-3500, bad_variant=False):
    os.makedirs(os.path.join(td, "logs", "E146")); os.makedirs(os.path.join(td, "traces", "E146"))
    f = lambda x: "%+.4f" % (x / 1e4)
    for b in V.BRAINS:
        R = np.zeros((1500, 27)); R[:, 13:17] = 1.0; R[:, 21:25] = V.R20
        np.savez_compressed(os.path.join(td, "traces", "E146", "tr_b%d.npz" % b), rows=R)
        open(os.path.join(td, "logs", "E146", "train_b%d.log" % b), "w", encoding="utf-8").write(
            "[사전] 오프셋 +0.0 | 정답률 0.0%% | **변조폭 %s**\n[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n"
            "[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n[사후] 오프셋 +0.0 | **변조폭 %s**\n" % (f(V.PRE0[b]), f(post1500)))
        for s in V.STIMS:
            none = V.PRE0[b] if s == "base" else V.PRE0[b] + 100
            vals = {"none": none, "E141": V.E141_POST[b] if s == "base" else none + e500[s],
                    "R0F1500": post1500 if s == "base" else none + e1500[s]}
            for w, m in vals.items():
                tag = "occ" if (bad_variant and b == 12 and w == "E141" and s == "noise") else s
                open(os.path.join(td, "logs", "E146", "ev_b%d_%s_%s.log" % (b, w, s)), "w", encoding="utf-8").write(
                    "=> DECOMP mode=all mod=%s acc=0.0\n[E146 변형] variant=%s vseed=0\n" % (f(m), tag))


def run(*a, **kw):
    with tempfile.TemporaryDirectory() as td:
        make(td, *a, **kw)
        V.EXP = td
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                V.main()
        except RuntimeError as ex:
            return "예외: %s" % ex
    return buf.getvalue()


E500 = {"int05": -1500, "int07": -2000, "occ": -1400, "noise": -1800}
ok_all = True
for name, args, kw, want in (
        ("충족", (E500, {k: v - 500 for k, v in E500.items()}), {}, "독립 판정: 충족(H069)"),
        ("일반화 실패", ({"int05": -600, "int07": -2000, "occ": -500, "noise": -1800}, {"int05": -900, "int07": -2500, "occ": -800, "noise": -2300}), {}, "독립 판정: 일반화 실패(H069-spec)"),
        ("용량 포화", (E500, {k: v + 200 for k, v in E500.items()}), {"post1500": -2000}, "독립 판정: 용량 포화(H069-sat)"),
        ("변형 표시 불일치", (E500, {k: v - 500 for k, v in E500.items()}), {"bad_variant": True}, "예외: 변형 표시 불일치")):
    out = run(*args, **kw)
    good = want in out
    ok_all &= good
    print("%-16s → %s %s" % (name, (out.strip().splitlines() or [""])[-1][:40], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
