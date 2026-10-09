#!/usr/bin/env python3
"""verify_e160_independent.py 합성 시험: 성공·분리만·실패·보류, r 경계, 선택성·Oja 변화·적재 실패, 결측.
실행: python3 scripts/test_verify_e160.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e160_independent as V

LDL = "[E153 종류 입력 적재] k 검증 일치 — x\n"


def run(e=-0.40, jl=0.0, jr=0.0, sel=0.88, dg=10.0, nld=2, miss=False):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E160"), ("logs", "E141"), ("traces", "E160")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 14] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20; R[:, 22] = (11.0 / 12.0) ** 20
        for b in V.BRAINS:
            w(("logs", "E160", "dev_b%d.log" % b),
              "=> KCDEVOJA side=l fired=210 sel_med0=0.5500 sel_med=%.4f frac09=0.45 goodfrac=0.5 sum_med=1.0 sum_q10=0.7 sum_q90=1.3 dg_good=%.1f dg_bad=%.1f"
              " | side=r fired=205 sel_med0=0.5400 sel_med=0.8600 frac09=0.42 goodfrac=0.49 sum_med=1.0 sum_q10=0.7 sum_q90=1.3 dg_good=10.0 dg_bad=9.5 | n=100 eta=0.02 save=x\n"
              % (sel if b == 12 else 0.88, dg if b == 13 else 10.0, 10.0))
            if not (miss and b == 14):
                w(("logs", "E160", "ov_b%d.log" % b), LDL * 2 + "=> KCOVERLAP side=l good=95 bad=90 jac=%.4f cos=0.1 jac025=0.9 jac100=0.9 split_jac=0.9 split_cos=0.9 | "
                  "side=r good=93 bad=88 jac=%.4f cos=0.1 jac025=0.9 jac100=0.9 split_jac=0.9 split_cos=0.9 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n" % (jl, jr))
            w(("logs", "E160", "learn_b%d.log" % b), LDL * (nld if b == 11 else 2) + "[사전] x | **변조폭 +0.0000** (y)\n[사후] x | **변조폭 %+.4f**\n" % e)
            w(("logs", "E141", "b%d.log" % b), "[사전] x | **변조폭 +0.0000** (y)\n[사후] x | **변조폭 -0.2500**\n")
            np.savez_compressed(os.path.join(td, "traces", "E160", "tr_b%d.npz" % b), rows=R)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("성공", {}, "형성 성공(H083)"), ("r 경계 1.30 정확", {"e": -0.325}, "형성 성공(H083)"), ("r 1.2996", {"e": -0.3249}, "분리만(H083-sep-only)"),
                       ("분리만", {"e": -0.275}, "분리만(H083-sep-only)"), ("실패", {"jl": 0.3, "jr": 0.3}, "형성 실패(H083-null)"),
                       ("자카드 0.20 → 보류", {"jl": 0.2, "jr": 0.2}, "보류"), ("선택성 0.79", {"sel": 0.79}, "보류(조작검증 실패)"),
                       ("Oja 변화 없음", {"dg": 0.0}, "보류(조작검증 실패)"), ("적재 1줄", {"nld": 1}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out and (want != "보류" or out.strip().endswith("독립 판정: 보류"))
    ok_all &= good
    print("%-20s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
