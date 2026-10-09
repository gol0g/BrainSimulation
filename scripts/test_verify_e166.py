#!/usr/bin/env python3
"""verify_e166_independent.py 합성 시험: 남음·손실·부분 × 줄임·줄이지 않음·중간, 경계(0.80·0.50·0.20 정확), 동결 꺼짐 실패, [사전] 어긋남, 적재, 결측.
실행: python3 scripts/test_verify_e166.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e166_independent as V

LD = "[E153 종류 입력 적재] /x/kctype.npz 검증 일치 — good_food_eye_l_to_kc_l=1.0\n"


def log(pre, post, nld):
    return LD * (nld // 2) + "[사전] x | **변조폭 %+.4f** (y)\n" % pre + LD * (nld - nld // 2) + "[사후] x | **변조폭 %+.4f**\n" % post


def run(qF=0.9, qD=0.3, per=None, frozen=False, preshift=0.0, dnf_ld=0, miss=False):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E166"), ("logs", "E161"), ("traces", "E166")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            qf, qd = (per or {}).get(b, (qF, qD))
            w(("logs", "E161", "F_b%d.log" % b), log(0.0150, 0.0150 - 0.6000, 2))
            w(("logs", "E161", "D_b%d.log" % b), log(0.0200, 0.0200 - 0.2500, 0))
            w(("logs", "E166", "FNF_b%d.log" % b), log(0.0150 + (preshift if b == 18 else 0.0), 0.0150 - 0.6000 * qf, 2))
            if not (miss and b == 20):
                w(("logs", "E166", "DNF_b%d.log" % b), log(0.0200, 0.0200 - 0.2500 * qd, dnf_ld))
            for a in ("FNF", "DNF"):
                R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20 if (frozen and a == "DNF" and b == 17) else 2.0
                np.savez_compressed(os.path.join(td, "traces", "E166", "tr_%s_b%d.npz" % (a, b)), rows=R)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("남음 / 줄임", {}, "남음(H089) / 형성이 동결 의존을 줄임"),
                       ("손실 / 줄이지 않음", {"qF": 0.3, "qD": 0.3}, "손실(H089-null) / 줄이지 않음"),
                       ("부분 / 줄임", {"qF": 0.65, "qD": 0.3}, "부분 / 형성이 동결 의존을 줄임"),
                       ("q_F 0.80 정확 → 남음", {"qF": 0.80, "qD": 0.3}, "남음(H089)"),
                       ("q_F 0.50 정확 → 손실", {"qF": 0.50, "qD": 0.45}, "손실(H089-null) / 줄이지 않음"),
                       ("차 0.20 정확 → 줄임", {"qF": 0.60, "qD": 0.40}, "부분 / 형성이 동결 의존을 줄임"),
                       ("3/5 남음 → 부분", {"per": {16: (0.3, 0.3), 17: (0.3, 0.3)}}, "부분 / 중간"),
                       ("동결 꺼짐 실패(DNF 뇌 17 동결)", {"frozen": True}, "보류(조작검증 실패)"),
                       ("[사전] 어긋남 0.0021", {"preshift": 0.0021}, "보류(조작검증 실패)"),
                       ("DNF 적재 2", {"dnf_ld": 2}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want)
    ok_all &= good
    print("%-30s → %-46s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
