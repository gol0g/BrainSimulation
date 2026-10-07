#!/usr/bin/env python3
"""verify_e155_independent.py 합성 시험: 둘 다 통과·일반화 실패·반전 부분·평가 적재 없음·반전 적재 1줄·변형 표시 불일치(중단).
실행: python3 scripts/test_verify_e155.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e155_independent as V

LD = "[E153 종류 입력 적재] k 검증 일치 — x\n"
PRE = {10: 117, 11: 15, 12: 64, 13: 231, 14: 116}
P153 = {10: -4893, 11: -4801, 12: -5024, 13: -4582, 14: -5143}
P154 = {10: -6015, 11: -5886, 12: -5881, 13: -5568, 14: -6153}


def f4(x):
    return "%+.4f" % (x / 1e4)


def rows():
    rng = np.random.default_rng(2)
    R = np.zeros((3000, 37)); R[:, 2] = rng.integers(0, 2, 3000); R[:, 6] = rng.integers(0, 2, 3000)
    R[:, 7] = np.where(np.arange(3000) < 1500, R[:, 6] != R[:, 2], R[:, 6] == R[:, 2]).astype(float); R[:, 13:17] = 1.0; R[:, 21:25] = (11.0 / 12.0) ** 20
    return R


def run(var_ratio=0.9, post=3000, ev_ld=True, rev_ld=2, bad_variant=False):
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E155", "traces/E155"):
            os.makedirs(os.path.join(td, d))
        for b in V.BRAINS:
            for s in V.STIMS:
                n = PRE[b] if s == "base" else 100
                vals = {"none": n, "E153": P153[b] if s == "base" else n + int(round((P153[b] - PRE[b]) * var_ratio)),
                        "E154A": P154[b] if s == "base" else n + int(round((P154[b] - PRE[b]) * var_ratio))}
                for w, m in vals.items():
                    vs = "base" if (bad_variant and s == "noise" and w == "none" and b == 10) else s
                    open(os.path.join(td, "logs", "E155", "ev_b%d_%s_%s.log" % (b, w, s)), "w", encoding="utf-8").write(
                        (LD if ev_ld else "") + "[E146 변형] variant=%s vseed=0\n=> DECOMP mode=all mod=%s acc=0.0\n" % (vs, f4(m)))
            open(os.path.join(td, "logs", "E155", "rev_b%d.log" % b), "w", encoding="utf-8").write(
                LD * rev_ld + "[반전] 시행 1500 부터 정답 = 같은 쪽\n[사전] x | **변조폭 %s**\n" % f4(PRE[b]) +
                "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n"
                "[사후] x | **변조폭 %s**\n" % f4(post))
            np.savez_compressed(os.path.join(td, "traces", "E155", "tr_rev_b%d.npz" % b), rows=rows())
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("둘 다 통과", {}, ("통과(K80 보존)", "통과(K81 보존 — 반전 성공)", "형성 표현 기준선 확립(H078)")),
                       ("일반화 실패(변형 몫 0.4)", {"var_ratio": 0.4}, ("일반화 실패", "통과(K81", "기준선 미확립")),
                       ("반전 부분(사후 −0.30)", {"post": -3000}, ("통과(K80 보존)", "부분", "기준선 미확립")),
                       ("평가 적재 없음", {"ev_ld": False}, ("보류(측정 검증 실패)", "통과(K81", "기준선 미확립")),
                       ("반전 적재 1줄", {"rev_ld": 1}, ("통과(K80 보존)", "보류(조작검증 실패)", "기준선 미확립"))):
    out = run(**kw)
    good = ("독립 판정 1: %s" % want[0]) in out and ("독립 판정 2: %s" % want[1]) in out and ("독립 종합: %s" % want[2]) in out
    ok_all &= good
    print("%-24s → %s %s" % (name, " / ".join(ln.split(": ", 1)[1] for ln in out.strip().splitlines()[-3:]), "✓" if good else "✗\n" + out))
try:
    run(bad_variant=True); g = False
except RuntimeError as e_:
    g = "변형 표시 불일치" in str(e_)
ok_all &= g
print("%-24s → %s" % ("변형 표시 불일치", "✓ (중단)" if g else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
