#!/usr/bin/env python3
"""verify_e181_independent.py 합성 시험: 회복·손실 유지·부분, 경계(O 0.85·0.70), 판정 2 역전(−0.02)·없음, 조작검증 실패(pushed·적재·배율·동작점 없음·값·진단 한쪽·포화·기준 효과·반사 발현), 결측.
실행: python3 scripts/test_verify_e181.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e181_independent as V  # noqa: E402

TL = "  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)\n"


def run(O=0.95, w25=0.05, base=-1.0, refl=0.40, pushed7=False, ld1=False, notonic=False, tonic_val=-4000.0, diag_one=False, sat=False, miss=False):
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E181"))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            vals = {"none_R0": 0.01, "none_R25": 0.01 + refl, "W0_R0": 0.01 + base, "W0_R25": 0.01 + refl + O * base, "W25_R0": 0.01 + 0.6 * base, "W25_R25": w25}
            for c in V.CELLS:
                if miss and b == 14 and c == "W25_R0":
                    continue
                pu = 7 if (pushed7 and b == 12 and c == "W0_R25") else (0 if c.startswith("none") else 8)
                ldl = "[E153 종류 입력 적재] k 검증 일치\n[E157 종류 입력 배율] x 검증 일치\n" * (1 if (ld1 and b == 11 and c == "none_R0") else 2)
                tl = "" if (notonic and b == 10 and c == "none_R25") else TL % (tonic_val if (b == 13 and c == "W25_R25") else -4000.0)
                r4 = (0.6664, 0.6662, 0.6663, 0.6600) if (sat and b == 14 and c == "W0_R0") else (0.40, 0.30, 0.31, 0.41)
                dg = "[E171 평가 진단] variant=base side=left n=250 motor L/R %.6f/%.6f KC L/R 0.004/0.001\n" % r4[:2]
                if not (diag_one and b == 12 and c == "W25_R0"):
                    dg += "[E171 평가 진단] variant=base side=right n=250 motor L/R %.6f/%.6f KC L/R 0.001/0.004\n" % r4[2:]
                w(("logs", "E181", "%s_b%d.log" % (c, b)), tl + ldl + dg + "=> DECOMP mode=x mod=%+.4f acc=0.0 off=0.0000 pushed=%d kc_means[x]\n" % (vals[c], pu))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("회복 / 역전 없음", {}, "회복(H102) / 역전 없음"),
                       ("O 0.85 정확 → 회복", {"O": 0.85}, "회복(H102)"),
                       ("O 0.70 정확 → 손실 유지", {"O": 0.70}, "손실 유지(H102-null)"),
                       ("O 0.80 → 부분", {"O": 0.80}, "부분"),
                       ("역전 −0.02 정확", {"w25": -0.02}, "회복(H102) / 반사 역전"),
                       ("역전 없음 −0.0199", {"w25": -0.0199}, "회복(H102) / 역전 없음"),
                       ("pushed 7", {"pushed7": True}, "보류(조작검증 실패)"),
                       ("적재 1", {"ld1": True}, "보류(조작검증 실패)"),
                       ("동작점 줄 없음", {"notonic": True}, "보류(조작검증 실패)"),
                       ("동작점 −1000", {"tonic_val": -1000.0}, "보류(조작검증 실패)"),
                       ("진단 한쪽", {"diag_one": True}, "보류(조작검증 실패)"),
                       ("포화", {"sat": True}, "보류(조작검증 실패)"),
                       ("기준 효과 −0.0999", {"base": -0.0999}, "보류(조작검증 실패)"),
                       ("반사 발현 +0.0999", {"refl": 0.0999}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want)
    ok_all &= good
    print("%-22s → %-44s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
