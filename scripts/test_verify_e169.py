#!/usr/bin/env python3
"""verify_e169_independent.py 합성 시험: 성공·실패·부분 × 줄임·없음·중간·판정 불가, 경계(0.80·0.50·0.20 정확), 조작검증 실패(창 1 잔차·동결 꺼짐·[사전]·적재), 결측.
실행: python3 scripts/test_verify_e169.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e169_independent as V

LD = "[E153 종류 입력 적재] /x/kctype.npz 검증 일치 — good_food_eye_l_to_kc_l=1.0\n"


def log(pre, post, nld=2):
    return LD * (nld // 2) + "[사전] x | **변조폭 %+.4f** (y)\n" % pre + LD * (nld - nld // 2) + "[사후] x | **변조폭 %+.4f**\n" % post


def run(q1=0.9, qff=0.9, qnf=0.07, per=None, win2=False, unfrozen=False, preshift=0.0, ld1=False, miss=False):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E169"), ("logs", "E161"), ("logs", "E166"), ("traces", "E169")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        r1 = (11.0 / 12.0) ** 10
        for b in V.BRAINS:
            a1, aff = (per or {}).get(b, (q1, qff))
            w(("logs", "E161", "F_b%d.log" % b), log(0.0150, 0.0150 - 0.6))
            w(("logs", "E166", "FNF_b%d.log" % b), log(0.0150, 0.0150 - 0.6 * qnf))
            w(("logs", "E169", "FW1_b%d.log" % b), log(0.0150 + (preshift if b == 17 else 0.0), 0.0150 - 0.6 * a1))
            if not (miss and b == 20):
                w(("logs", "E169", "FFW1_b%d.log" % b), log(0.0150, 0.0150 - 0.6 * aff, 1 if (ld1 and b == 19) else 2))
            Rf = np.zeros((500, 37)); Rf[:, 13] = 1.0; Rf[:, 21] = ((11.0 / 12.0) ** 20) if (win2 and b == 18) else r1
            Rw = np.zeros((500, 37)); Rw[:, 13] = 1.0; Rw[:, 21] = r1 if (unfrozen and b == 16) else 3.0
            np.savez_compressed(os.path.join(td, "traces", "E169", "tr_FFW1_b%d.npz" % b), rows=Rf)
            np.savez_compressed(os.path.join(td, "traces", "E169", "tr_FW1_b%d.npz" % b), rows=Rw)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("성공 / 줄임", {}, "대체 성공(H092) / 창 단축이 오염을 줄임"),
                       ("실패 / 없음", {"q1": 0.07, "qff": 0.9}, "실패(H092-null) / 줄이지 않음"),
                       ("부분 / 줄임", {"q1": 0.65}, "부분 / 창 단축이 오염을 줄임"),
                       ("q₁ 0.80 정확 → 성공", {"q1": 0.80}, "대체 성공(H092)"),
                       ("q₁ 0.50 정확 → 실패", {"q1": 0.50}, "실패(H092-null)"),
                       ("c₁ − c₂ 0.20 정확 → 줄임", {"q1": 0.27, "qff": 1.0}, "실패(H092-null) / 창 단축이 오염을 줄임"),
                       ("판정 2 전제 실패", {"q1": 0.05, "per": {18: (0.05, 0.1665)}}, "실패(H092-null) / 판정 불가"),
                       ("창 2 로 돈 FFW1(잔차)", {"win2": True}, "보류(조작검증 실패)"),
                       ("FW1 동결된 채(잔차 0)", {"unfrozen": True}, "보류(조작검증 실패)"),
                       ("[사전] 어긋남 0.0021", {"preshift": 0.0021}, "보류(조작검증 실패)"),
                       ("적재 1", {"ld1": True}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want)
    ok_all &= good
    print("%-26s → %-46s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
