#!/usr/bin/env python3
"""verify_e168_independent.py 합성 시험: 성공·실패·부분 × 효과·없음·중간, 경계(0.80·0.50·0.20 정확, KC 10%·90% 정확), 조작검증 실패(연결·KC·I_input·[사전]·추적), 결측.
실행: python3 scripts/test_verify_e168.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e168_independent as V


def log(pre, post, conn, kcd, kcr, da="0.0"):
    return (("  [E168 도파민→KC억제] 도파민 뉴런 100 → KC 억제 뉴런 좌·우 400, w=5.00, p=0.20\n" if conn else "")
            + "[사전] x | **변조폭 %+.4f** (y)\n[구현 점검] 첫 보상 창 끝 도파민 뉴런 I_input 53.0 → %s\n" % (pre, da)
            + "[E168 KC 발화] 결정 단계(3처리 끝) 평균 %.6f n=500 | 보상 창 보상 시행 평균 %.6f n=330 | 보상 창 처벌 시행 평균 0.009000 n=170 | da_kc_inh=5.00 p=0.20\n" % (kcd, kcr)
            + "[사후] x | **변조폭 %+.4f**\n" % post)


def run(qI=0.9, qR=0.08, per=None, kcI=(0.045, 0.001), conn_fr=0, da="0.0", preshift=0.0, n_tr=500, miss=False):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E168"), ("logs", "E161"), ("traces", "E168")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            qi, qr = (per or {}).get(b, (qI, qR))
            w(("logs", "E161", "F_b%d.log" % b), "[사전] x | **변조폭 +0.0150** (y)\n[사후] x | **변조폭 %+.4f**\n" % (0.0150 - 0.6000))
            w(("logs", "E168", "FI_b%d.log" % b), log(0.0150 + (preshift if b == 17 else 0.0), 0.0150 - 0.6 * qi, 1, kcI[0], kcI[1], da if b == 18 else "0.0"))
            if not (miss and b == 20):
                w(("logs", "E168", "FR_b%d.log" % b), log(0.0150, 0.0150 - 0.6 * qr, conn_fr if b == 19 else 0, 0.050, 0.010))
            for a in ("FI", "FR"):
                np.savez_compressed(os.path.join(td, "traces", "E168", "tr_%s_b%d.npz" % (a, b)), rows=np.zeros((n_tr if (a == "FR" and b == 16) else 500, 37)))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("성공 / 효과", {}, "대체 성공(H091) / 연결 효과"),
                       ("실패 / 없음", {"qI": 0.1, "qR": 0.08}, "대체 실패(H091-null) / 효과 없음"),
                       ("부분 / 효과", {"qI": 0.65, "qR": 0.08}, "부분 / 연결 효과"),
                       ("q_I 0.80 정확 → 성공", {"qI": 0.80}, "대체 성공(H091)"),
                       ("q_I 0.50 정확 → 실패", {"qI": 0.50, "qR": 0.45}, "대체 실패(H091-null) / 효과 없음"),
                       ("차 0.20 정확 → 효과", {"qI": 0.45, "qR": 0.25}, "대체 실패(H091-null) / 연결 효과"),
                       ("3/5 성공 → 부분", {"per": {16: (0.2, 0.08), 17: (0.2, 0.08)}}, "부분"),
                       ("KC 보상 창 10% 정확 → 통과", {"kcI": (0.045, 0.001)}, "대체 성공(H091)"),
                       ("KC 보상 창 10.1% → 실패", {"kcI": (0.045, 0.00101)}, "보류(조작검증 실패)"),
                       ("KC 결정 89.998%", {"kcI": (0.044999, 0.0005)}, "보류(조작검증 실패)"),
                       ("FR 연결 줄 1", {"conn_fr": 1}, "보류(조작검증 실패)"),
                       ("I_input 0 아님", {"da": "53.0"}, "보류(조작검증 실패)"),
                       ("[사전] 어긋남 0.0021", {"preshift": 0.0021}, "보류(조작검증 실패)"),
                       ("추적 499", {"n_tr": 499}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want)
    ok_all &= good
    print("%-28s → %-46s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
