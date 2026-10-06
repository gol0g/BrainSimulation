#!/usr/bin/env python3
"""judge_e149.py 합성 시험(조건 1): 겹침+분리·겹침+분리 안 됨·겹침 낮음·보류, 경계(J 0.5·0.2, 반응 0.2배), 측정 검증 실패, 결측, 줄 파싱.
실행: python3 scripts/test_judge_e149.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e149 as J


def side(jac, good=100, bad=100, split=0.8, cos=0.7):
    return {"good": good, "bad": bad, "jac": jac, "cos": cos, "j025": jac, "j100": jac, "split": split, "scos": 0.9}


def build(jb, jk, block_good=60, split=0.8, base_good=100):
    D = {}
    for i, b in enumerate(J.BRAINS):
        D[(b, "base")] = {"l": side(jb[i], good=base_good, split=split), "r": side(jb[i], good=base_good, split=split)}
        D[(b, "block")] = {"l": side(jk[i], good=block_good, bad=block_good), "r": side(jk[i], good=block_good, bad=block_good)}
    return D


cases = [
    ("겹침+분리", build([0.7] * 5, [0.05] * 5), "겹침 확인·차단으로 분리(H072·H072-sep)"),
    ("경계 J 0.5·0.2", build([0.5] * 5, [0.2] * 5), "겹침 확인·차단으로 분리(H072·H072-sep)"),
    ("겹침+분리 안 됨", build([0.7] * 5, [0.5] * 5), "겹침 확인·차단으로 분리 안 됨"),
    ("분리했으나 반응 사라짐", build([0.7] * 5, [0.05] * 5, block_good=10), "겹침 확인·차단으로 분리 안 됨"),
    ("반응 0.2배 경계", build([0.7] * 5, [0.05] * 5, block_good=20), "겹침 확인·차단으로 분리(H072·H072-sep)"),
    ("겹침 낮음", build([0.1] * 5, [0.05] * 5), "겹침 낮음"),
    ("보류", build([0.3] * 5, [0.1] * 5), "보류"),
    ("V1 반분 신뢰도 낮음", build([0.7] * 5, [0.05] * 5, split=0.3), "보류(측정 검증 실패)"),
    ("V3 반응 적음", build([0.7] * 5, [0.05] * 5, base_good=3, block_good=3), "보류(측정 검증 실패)"),
]
ok_all = True
for name, D, want in cases:
    c, r = J.judge(D)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-22s 기대 %-26s → %s %s" % (name, want, r["verdict"][:26] if r else c[0][:30], "✓" if good else "✗"))
D = build([0.7] * 5, [0.05] * 5); del D[(12, "block")]
c, r = J.judge(D)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-22s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E149.log"), "w", encoding="utf-8").write(
        "  e149 b10 base: => KCOVERLAP side=l good=120 bad=110 jac=0.6500 cos=0.8000 jac025=0.7000 jac100=0.5000 split_jac=0.8500 split_cos=0.9500 | "
        "side=r good=118 bad=105 jac=0.6000 cos=0.7900 jac025=0.6900 jac100=0.4800 split_jac=0.8400 split_cos=0.9400 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n"
        "  e149 b10 block: => KCOVERLAP side=l good=nan\n")
    J.EXP = td
    Dp = J.load()
good = Dp[(10, "base")]["l"]["jac"] == 0.65 and Dp[(10, "base")]["r"]["split"] == 0.84 and Dp[(10, "block")] is None; ok_all &= good
print("%-22s 기대 base 파싱·block 파싱 실패 None → %s" % ("줄 파싱", "✓" if good else "✗ %s" % Dp))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
