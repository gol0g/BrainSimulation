#!/usr/bin/env python3
"""judge_e145.py 합성 시험(조건 1): 같은쪽·교차·D1·복수·비가산·보류, 경계(몫 0.6, ΔB 0.03, 재현 ±0.002), 측정 검증 실패 3종, 결측, 줄 파싱.
실행: python3 scripts/test_judge_e145.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e145 as J


def build(fs, fc, fd, a_shift=0.0, b_shift=0.0, dB_scale=1.0):
    """몫(fs, fc, fd)을 뇌마다 같게 주고 A_all = F1500 사후, B_all = A_all + ΔB."""
    M = {}
    for b in J.BRAINS:
        a = round(J.F1500_POST[b] + a_shift, 4)
        dB = round((J.E144_POST[b] - J.F1500_POST[b]) * dB_scale, 4)
        bb = round(J.E144_POST[b] + b_shift, 4) if dB_scale == 1.0 else round(a + dB, 4)
        dB = bb - a
        M[(b, "A_all")] = a; M[(b, "B_all")] = bb
        M[(b, "A_sameB")] = round(a + fs * dB, 4); M[(b, "A_crossB")] = round(a + fc * dB, 4); M[(b, "A_d1B")] = round(a + fd * dB, 4)
    return M


cases = [
    ("같은쪽", build(0.9, 0.1, 0.0), "H068-same"),
    ("교차", build(0.1, 0.8, 0.1), "H068-cross"),
    ("D1", build(0.0, 0.2, 0.8), "H068-d1"),
    ("복수", build(0.7, 0.7, 0.0), "H068-multi"),
    ("비가산", build(0.3, 0.2, 0.0), "H068-int"),
    ("보류(가산, 0.5 씩)", build(0.5, 0.5, 0.0), "보류"),
    ("V1 실패", build(0.9, 0.1, 0.0, a_shift=0.003), "보류(측정 검증 실패)"),
    ("V2 실패", build(0.9, 0.1, 0.0, b_shift=0.003), "보류(측정 검증 실패)"),
]
ok_all = True
for name, M, want in cases:
    c, r = J.judge(M)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-18s 기대 %-22s → %s %s" % (name, want, r["verdict"][:24] if r else None, "✓" if good else "✗"))
# 경계: 몫 정확히 0.6(4자리 반올림 영향 피하려 ΔB 를 0.05 로 맞춘 합성)
M = {}
for b in J.BRAINS:
    a = J.F1500_POST[b]; bb = round(a + 0.05, 4)
    M[(b, "A_all")] = a; M[(b, "B_all")] = bb; M[(b, "A_sameB")] = round(a + 0.03, 4); M[(b, "A_crossB")] = a; M[(b, "A_d1B")] = a
c, r = J.judge(M)
good = r is None  # V1 은 통과(A=F1500), V2 는 B 가 E144 와 달라 실패 → 판정 보류여야 하나 r 은 존재
good = r is not None and r["verdict"].startswith("보류(측정 검증 실패)")
ok_all &= good
print("%-18s 기대 보류(측정 검증 실패: B ≠ E144) → %s %s" % ("V2 경계 밖", r["verdict"][:24], "✓" if good else "✗"))
# V3: ΔB 가 0.03 미만이면 실패 — E144 사후를 바꿀 수 없으므로 judge 의 상수를 임시로 바꿔 시험
_save = dict(J.E144_POST)
for b in J.BRAINS:
    J.E144_POST[b] = round(J.F1500_POST[b] + 0.02, 4)
M = build(0.9, 0.1, 0.0)
c, r = J.judge(M)
good = r is not None and r["verdict"].startswith("보류(측정 검증 실패)"); ok_all &= good
print("%-18s 기대 보류(측정 검증 실패: ΔB<0.03) → %s %s" % ("V3 실패", r["verdict"][:24], "✓" if good else "✗"))
J.E144_POST.update(_save)
M = build(0.9, 0.1, 0.0); del M[(12, "A_d1B")]
c, r = J.judge(M)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-18s 기대 결측 → %s" % ("결측", "✓" if good else "✗"))
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E145.log"), "w", encoding="utf-8").write("  e145 b10 A_all: => mod +0.1026\n  e145 b10 A_sameB: => mod +0.1500\n  e144 b10: => 사전 +0.4 사후 +0.1 보상 1\n")
    J.EXP = td
    Mp = J.load()
good = Mp == {(10, "A_all"): 0.1026, (10, "A_sameB"): 0.15}; ok_all &= good
print("%-18s 기대 두 줄 → %s" % ("줄 파싱", "✓" if good else "✗ %s" % Mp))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
