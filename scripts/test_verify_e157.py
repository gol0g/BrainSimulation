#!/usr/bin/env python3
"""verify_e157_independent.py 합성 시험: 네 범주·S 경계·발화 맞춤 실패·겹침 실패·배율 줄 실패·결측·D 재사용 우선순위.
실행: python3 scripts/test_verify_e157.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e157_independent as V

LDL = "[E153 종류 입력 적재] k 검증 일치 — x\n"; SCL = "[E157 종류 입력 배율] k=0.8000 검증 일치 — x\n"
KR = ("=> KCRATE kc_l | 좌선택 1 우선택 2 | 제시 스파이크 %d 기준선(제시창) 평균 0.0500 | KC별 ΔS y\n"
      "=> KCRATE kc_r | 좌선택 1 우선택 2 | 제시 스파이크 %d 기준선(제시창) 평균 0.0500 | KC별 ΔS y\n")
OV = ("=> KCOVERLAP side=l good=10 bad=12 jac=%.4f cos=0.1 jac025=0.9 jac100=0.9 split_jac=0.9 split_cos=0.9 | "
      "side=r good=11 bad=13 jac=%.4f cos=0.1 jac025=0.9 jac100=0.9 split_jac=0.9 split_cos=0.9 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n")


def ml(pre, post, nld=0, nsc=0):
    return LDL * nld + SCL * nsc + "[사전] x | **변조폭 %+.4f** (y)\n[학습] 5ep 완료, 보상 300회 (z)\n[사후] x | **변조폭 %+.4f**\n" % (pre, post)


def run(eFk=-0.27, eDk=-0.45, spFk=10000, jFk=0.0, scDk=2, miss=None, own_D=None):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E157"), ("logs", "E141"), ("logs", "E153"), ("traces", "E157")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 14] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20; R[:, 22] = (11.0 / 12.0) ** 20
        for b in V.BRAINS:
            w(("logs", "E141", "b%d.log" % b), ml(0.0, -0.25))
            w(("logs", "E153", "train_b%d.log" % b), ml(0.0, -0.50, nld=2))
            w(("logs", "E157", "learn_Fk_b%d.log" % b), ml(0.0, eFk, nld=2, nsc=2))
            w(("logs", "E157", "learn_Dk_b%d.log" % b), ml(0.0, eDk, nsc=scDk if b == 12 else 2))
            for a, s_ in (("D", 10000), ("F", 12500), ("Fk", spFk if b == 11 else 10000), ("Dk", 12500)):
                w(("logs", "E157", "kcrate_%s_b%d.log" % (a, b)), KR % (s_ // 2, s_ - s_ // 2))
                w(("logs", "E157", "r25_%s_b%d.log" % (a, b)), "[사전] x | **변조폭 +0.3000** (y)\n")
            w(("logs", "E157", "ov_Fk_b%d.log" % b), OV % (jFk if b == 13 else 0.0, 0.0))
            w(("logs", "E157", "ov_Dk_b%d.log" % b), OV % (0.40, 0.42))
            for a in ("Fk", "Dk"):
                np.savez_compressed(os.path.join(td, "traces", "E157", "tr_%s_b%d.npz" % (a, b)), rows=R)
        if own_D is not None:
            for b in V.BRAINS:
                w(("logs", "E157", "learn_D_b%d.log" % b), ml(0.0, own_D))
        if miss:
            os.remove(os.path.join(td, *miss))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("이득", {}, "이득(H080)"), ("분리", {"eFk": -0.45, "eDk": -0.27}, "분리(H080-sep)"),
                       ("둘 다", {"eFk": -0.40, "eDk": -0.40}, "둘 다(H080-both)"), ("둘 다 아님", {"eFk": -0.27, "eDk": -0.27}, "둘 다 아님(H080-int)"),
                       ("S 경계 0.30 정확", {"eFk": -0.325}, "이득(H080)"), ("S 0.3004", {"eFk": -0.3251}, "둘 다(H080-both)"),
                       ("발화 Fk/D 0.8499", {"spFk": 8499}, "보류(조작검증 실패)"), ("겹침 Fk 0.0501", {"jFk": 0.0501}, "보류(조작검증 실패)"),
                       ("Dk 배율 1줄", {"scDk": 1}, "보류(조작검증 실패)"), ("결측 ov Dk b14", {"miss": ("logs", "E157", "ov_Dk_b14.log")}, "보류(결측"),
                       ("D 재사용 우선순위(E157 learn_D −0.20)", {"own_D": -0.20, "eFk": -0.27, "eDk": -0.45}, "둘 다(H080-both)"),
                       ("D 재사용 우선순위(E157 learn_D −0.50)", {"own_D": -0.50, "eFk": -0.27, "eDk": -0.45}, "둘 다 아님(H080-int)")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out
    ok_all &= good
    print("%-40s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
