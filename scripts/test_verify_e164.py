#!/usr/bin/env python3
"""verify_e164_independent.py 합성 시험: 무시 가능·영향 큼·중간, I_input 미초기화·점검 줄 부족·[사전] 어긋남, 결측.
실행: python3 scripts/test_verify_e164.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e164_independent as V


def xlog(post, pre=0.0200, da_to="0.0", nchk=3):
    lines = ["[구현 점검] rw_da_reset=True offset_steps=3 (외부 검토 2026-10-09 ①·③)", "[구현 점검] 오프셋(조향 3처리 합) 첫 에피소드 +0.0120",
             "[구현 점검] 첫 보상 창 끝 도파민 뉴런 I_input 80.0 → %s" % da_to][:nchk]
    return "\n".join(lines) + "\n[사전] x | **변조폭 %+.4f** (y)\n[사후] x | **변조폭 %+.4f**\n" % (pre, post)


def run(d0=0.0, d25=0.0, da_to="0.0", nchk=3, pre_shift=0.0, miss=False, per=None):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E164"), ("logs", "E141"), ("logs", "E142"), ("traces", "E164")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20
        for b in V.BRAINS:
            w(("logs", "E141", "b%d.log" % b), "[사전] x | **변조폭 +0.0200** (y)\n[사후] x | **변조폭 -0.2300**\n")
            w(("logs", "E142", "F500_b%d.log" % b), "[사전] x | **변조폭 +0.4000** (y)\n[사후] x | **변조폭 +0.1300**\n")
            for a, d, pre, eb in (("R0X", d0, 0.02, -0.25), ("R25X", d25, 0.40, -0.27)):
                if miss and a == "R25X" and b == 14:
                    continue
                dd = (per or {}).get((a, b), d)
                w(("logs", "E164", "%s_b%d.log" % (a, b)), xlog(pre + eb + dd, pre + (pre_shift if b == 12 else 0.0),
                                                              da_to if b == 11 else "0.0", nchk if b == 13 else 3))
                np.savez_compressed(os.path.join(td, "traces", "E164", "tr_%s_b%d.npz" % (a, b)), rows=R)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("무시 가능", {}, "무시 가능(H087-null)"), ("영향 큼(반사 0 −0.08)", {"d0": -0.08}, "영향 큼(H087)"),
                       ("영향 큼(반사 25 +0.06)", {"d25": 0.06}, "영향 큼(H087)"), ("중간(−0.04)", {"d0": -0.04}, "보류(중간)"),
                       ("섞인 부호", {"d0": 0.08, "per": {("R0X", 13): -0.08, ("R0X", 14): -0.08}}, "보류(중간)"),
                       ("I_input 미초기화", {"da_to": "80.0"}, "보류(조작검증 실패)"), ("점검 줄 2개", {"nchk": 2}, "보류(조작검증 실패)"),
                       ("[사전] 어긋남", {"pre_shift": 0.0021}, "보류(조작검증 실패)"), ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out
    ok_all &= good
    print("%-24s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
