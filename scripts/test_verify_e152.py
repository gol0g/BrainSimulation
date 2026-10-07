#!/usr/bin/env python3
"""verify_e152_independent.py 합성 시험: 먹이 주도·결합 주도·혼합·M1 실패(ΔJ 0.0501)·M2 실패(먹이 반분 nan)·경계(φ 0.7, ΔJ 0.05 정확).
실행: python3 scripts/test_verify_e152.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e152_independent as V


def side152(k, O, OF, J, FS):
    return "side=%s good=100 bad=105 food=80 both=%d both_food=%d food_in=70 jac_gb=%s food_split_jac=%s" % (k, O, OF, J, FS)


def run(OF_list, J152="0.4300", FS="0.9000", O=30, bad_brain=None, bad_kw=None):
    with tempfile.TemporaryDirectory() as td:
        for d in ("logs/E152", "logs/E149"):
            os.makedirs(os.path.join(td, d))
        for i, b in enumerate(V.BRAINS):
            kw = {"J": J152, "FS": FS}
            if bad_brain is not None and b in bad_brain:
                kw.update(bad_kw)
            open(os.path.join(td, "logs", "E152", "b%d.log" % b), "w", encoding="utf-8").write(
                "x\n=> KCOVERLAP3 %s | %s | food_eye_scale=1.00 n_pres=50\n" % (side152("l", O, OF_list[i], kw["J"], kw["FS"]), side152("r", O, OF_list[i], "0.4300", "0.9000")))
            open(os.path.join(td, "logs", "E149", "b%d_base.log" % b), "w", encoding="utf-8").write(
                "=> KCOVERLAP side=l good=100 bad=105 jac=0.3800 cos=0.6 jac025=0.4 jac100=0.3 split_jac=0.95 split_cos=0.99 | side=r good=99 bad=100 jac=0.4300 cos=0.6 jac025=0.5 jac100=0.4 split_jac=0.93 split_cos=0.99 | food_eye_scale=1.00\n")
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
# E149 J: l 0.3800, r 0.4300. E152 l 기본 0.4300 → ΔJ_l 500(경계 안), r 0.
for name, args, kw, want in (("먹이 주도(φ 0.8)", ([24] * 5,), {}, "먹이 주도(H075)"),
                             ("경계 φ 0.7·ΔJ 0.05", ([21] * 5,), {}, "먹이 주도(H075)"),
                             ("결합 주도(φ 0.2)", ([6] * 5,), {}, "결합 주도(H075-conj)"),
                             ("혼합", ([24, 24, 24, 6, 6],), {}, "혼합(보류)"),
                             ("M1 실패(ΔJ 0.0501, 2뇌)", ([24] * 5,), {"bad_brain": (10, 11), "bad_kw": {"J": "0.4301"}}, "보류(측정 검증 실패)"),
                             ("M2 실패(반분 nan, 2뇌)", ([24] * 5,), {"bad_brain": (12, 13), "bad_kw": {"FS": "nan"}}, "보류(측정 검증 실패)")):
    out = run(*args, **kw)
    good = ("독립 판정: %s" % want) in out
    ok_all &= good
    print("%-26s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
