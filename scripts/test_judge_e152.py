#!/usr/bin/env python3
"""judge_e152.py 합성 시험(조건 1): 먹이 주도·결합 주도·혼합, 경계(φ 0.7·0.3 정확), 측정 검증 M1·M2·M3 실패, 결측, 줄 파싱(E152·E149).
실행: python3 scripts/test_judge_e152.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e152 as J


def side(O=60, OF=45, J_=4300, FS=9000, G=100, B=105, F=80, Fin=70):
    return {"G": G, "B": B, "F": F, "O": O, "OF": OF, "Fin": Fin, "J": J_, "FS": FS}


def build(OF_list=None, O=60, OF=45, bad=None):
    M, J149 = {}, {}
    for i, b in enumerate(J.BRAINS):
        of = OF_list[i] if OF_list else OF
        l = side(O=O, OF=of); r = side(O=O, OF=of)
        if bad and bad[0] == b:
            (l if bad[1] == "l" else r).update(bad[2])
        M[b] = {"b": b, "l": l, "r": r}
        J149[b] = {"l": 4300, "r": 4300}
    return M, J149


cases = [
    ("먹이 주도(φ 0.75)", build(OF=45), "먹이 주도(H075)"),
    ("결합 주도(φ 0.2)", build(OF=12), "결합 주도(H075-conj)"),
    ("경계 φ=0.7 정확(OF 42/60)", build(OF=42), "먹이 주도(H075)"),
    ("φ 0.7 바로 아래(41/60)", build(OF=41), "혼합(보류)"),
    ("경계 φ=0.3 정확(18/60)", build(OF=18), "결합 주도(H075-conj)"),
    ("φ 0.3 바로 위(19/60)", build(OF=19), "혼합(보류)"),
    ("섞임 3·2", build(OF_list=[45, 45, 45, 12, 12]), "혼합(보류)"),
    ("M1 실패 2뇌", None, "보류(측정 검증 실패)"),
    ("M2 실패 2뇌", None, "보류(측정 검증 실패)"),
    ("M3 실패 5뇌(O 합 18)", build(O=9, OF=8), "보류(측정 검증 실패)"),
    ("M1 실패 1뇌 → 4/5 통과", None, "먹이 주도(H075)"),
]
ok_all = True
for name, data, want in cases:
    if name.startswith("M1 실패 2뇌"):
        M, Jb = build(OF=45); M[10]["l"]["J"] = 4801; M[11]["r"]["J"] = 3799; data = (M, Jb)
    elif name.startswith("M2 실패 2뇌"):
        M, Jb = build(OF=45); M[12]["l"]["FS"] = 7999; M[13]["r"]["FS"] = None; data = (M, Jb)
    elif name.startswith("M1 실패 1뇌"):
        M, Jb = build(OF=45); M[14]["l"]["J"] = 4801; M[13]["r"]["J"] = 4800; data = (M, Jb)   # 4800 = 경계 ±0.05 안
    c, r = J.judge(*data)
    good = r is not None and r["verdict"].startswith(want) and (want != "혼합(보류)" or r["verdict"] == "혼합(보류)")
    ok_all &= good
    print("%-30s 기대 %-22s → %s %s" % (name, want, r["verdict"][:22] if r else c[0][:30], "✓" if good else "✗"))
M, Jb = build(); del Jb[12]
c, r = J.judge(M, Jb); g = r is None and "결측" in c[0]; ok_all &= g; print("%-30s → %s" % ("결측(E149 기준)", "✓" if g else "✗"))
# 줄 파싱
L152 = ("  e152 b10: => KCOVERLAP3 side=l good=104 bad=113 food=88 both=65 both_food=50 food_in=80 jac_gb=0.4371 food_split_jac=0.9500"
        " | side=r good=99 bad=100 food=70 both=60 both_food=41 food_in=66 jac_gb=0.4632 food_split_jac=0.9100 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n")
L149 = ("  e149 b10 base: => KCOVERLAP side=l good=104 bad=113 jac=0.4371 cos=0.5901 jac025=0.4528 jac100=0.3040 split_jac=0.9717 split_cos=0.9978"
        " | side=r good=99 bad=100 jac=0.4632 cos=0.6577 jac025=0.5329 jac100=0.4234 split_jac=0.9320 split_cos=0.9972 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n"
        "  e149 b10 block: => KCOVERLAP side=l good=812 bad=449 jac=0.1081 cos=0.2057 jac025=0.1081 jac100=0.0209 split_jac=1.0000 split_cos=0.9999"
        " | side=r good=806 bad=445 jac=0.1140 cos=0.2137 jac025=0.1140 jac100=0.1140 split_jac=1.0000 split_cos=1.0000 | food_eye_scale=0.00\n")
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E152.log"), "w", encoding="utf-8").write(L152)
    open(os.path.join(td, "E149.log"), "w", encoding="utf-8").write(L149)
    J.EXP = td
    Mp, Jp = J.load()
g = (Mp[10]["l"] == {"G": 104, "B": 113, "F": 88, "O": 65, "OF": 50, "Fin": 80, "J": 4371, "FS": 9500}
     and Mp[10]["r"]["OF"] == 41 and Mp[10]["r"]["FS"] == 9100 and Jp == {10: {"l": 4371, "r": 4632}})
ok_all &= g
print("%-30s → %s" % ("줄 파싱(E149 base 만, jac= 필드)", "✓" if g else "✗ %s %s" % (Mp, Jp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
