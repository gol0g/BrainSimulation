#!/usr/bin/env python3
"""verify_e159_independent.py 합성 시험: 네 범주, O 음수, 재현 어긋남, pushed 틀림, 배율 줄, 결측.
실행: python3 scripts/test_verify_e159.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e159_independent as V

R1 = {10: 0.3519, 11: 0.3617, 12: 0.3513, 13: 0.3338, 14: 0.3714}
R2 = {10: 0.0169, 11: 0.0088, 12: 0.0197, 13: 0.0326, 14: 0.0223}
R3 = {10: -0.4316, 11: -0.4306, 12: -0.4337, 13: -0.4114, 14: -0.4473}
R4 = {10: 0.0528, 11: 0.0673, 12: 0.0596, 13: 0.0501, 14: 0.0748}
L = "[E153 종류 입력 적재] k 검증 일치 — x\n[E157 종류 입력 배율] k=0.7000 검증 일치 — x\n"


def run(O=0.5, C=0.9, shift=0, pushed_bad=False, nsc=1, miss=False):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E159"), ("logs", "E157"), ("logs", "E158")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            w(("logs", "E157", "r25_Fk_b%d.log" % b), "[사전] x | **변조폭 %+.4f** (y)\n" % R1[b])
            w(("logs", "E157", "learn_Fk_b%d.log" % b), "[사전] x | **변조폭 %+.4f** (y)\n[사후] x | **변조폭 %+.4f**\n" % (R2[b], R3[b]))
            w(("logs", "E158", "F500_b%d.log" % b), "[사전] x | **변조폭 %+.4f** (y)\n[사후] x | **변조폭 %+.4f**\n" % (R1[b], R4[b]))
            base = R3[b] - R2[b]
            v = {"none_R0": R2[b], "none_R25": R1[b] + (shift if b == 12 else 0), "W0_R0": R3[b], "W25_R25": R4[b],
                 "W0_R25": R1[b] + O * base, "W25_R0": R2[b] + C * base}
            for c in V.CELLS:
                if miss and c == "W25_R0" and b == 14:
                    continue
                pu = 0 if c.startswith("none") else (7 if (pushed_bad and c == "W0_R25" and b == 11) else 8)
                w(("logs", "E159", "%s_b%d.log" % (c, b)), L * (2 if not (nsc == 1 and c == "W25_R0" and b == 10) else 1)
                  + "=> DECOMP mode=%s mod=%+.4f acc=0.0 off=+0.0000 pushed=%d kc_means[x]\n" % ("none" if c.startswith("none") else "all", v[c], pu))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("출력", {"nsc": 2}, "출력(H082)"), ("내용", {"O": 0.9, "C": 0.5, "nsc": 2}, "내용(H082-content)"),
                       ("둘 다", {"O": 0.5, "C": 0.5, "nsc": 2}, "둘 다(H082-both)"), ("둘 다 아님", {"O": 0.9, "C": 0.9, "nsc": 2}, "둘 다 아님(H082-neither)"),
                       ("O 음수", {"O": -0.2, "nsc": 2}, "출력(H082)"), ("재현 어긋남 0.0021", {"shift": 0.0021, "nsc": 2}, "보류(조작검증 실패)"),
                       ("pushed 7", {"pushed_bad": True, "nsc": 2}, "보류(조작검증 실패)"), ("배율 1줄", {"nsc": 1}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True, "nsc": 2}, "보류(결측")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out
    ok_all &= good
    print("%-20s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
