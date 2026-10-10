#!/usr/bin/env python3
"""judge_e172_rev.py 합성 시험: 반전 성공·부분·고착·보류, 경계(m 0.02·Δ 0.10·0.05 정확), 조작검증 M1~M6, 결측, 원 로그·추적 파싱(E162 기준 읽기).
실행: python3 scripts/test_judge_e172_rev.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e172_rev as J

R_POST = -0.60
R_PRE = 0.01


def build(m=0.30, per=None, s_bad=None, rc_bad=None, pre_shift=0.0, load=2, drop_ref=None):
    T, S, RC, REF = {}, {}, {}, {}
    for b in J.BRAINS:
        mm = (per or {}).get(b, m)
        T[b] = {"pre": R_PRE + (pre_shift if b == 18 else 0.0), "post": mm, "rew": 2000, "load": load if b == 17 else 2}
        S[b] = {"n": 3000, "agree": 1.0, "res": 1e-8, "pre_ratio": 0.0, "rew_blk": [50] * 30}
        if s_bad and s_bad[0] == b:
            S[b] = dict(S[b], **s_bad[1])
        RC[b] = (True, True) if rc_bad != b else ((False, True) if b == 19 else (True, False))
        REF[b] = (J.i4(R_PRE), J.i4(R_POST))
    if drop_ref:
        del REF[drop_ref]
    return T, S, RC, REF


ok_all = True


def chk(name, data, want):
    global ok_all
    l, r = J.judge(*data)
    got = r["verdict"] if r else l
    good = (got.startswith(want) and (want != "보류" or got == "보류")) if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-32s 기대 %-20s → %-26s %s" % (name, want[:20], got[:26], "✓" if good else "✗ %s" % l))


chk("반전 성공(m +0.30)", build(), "반전 성공(H096)")
chk("m 경계 +0.02 정확 → 성공", build(m=0.0200), "반전 성공(H096)")
chk("m +0.0199 → 부분(Δ 0.62)", build(m=0.0199), "부분(H096-partial)")
chk("부분(m −0.40, Δ 0.20)", build(m=-0.40), "부분(H096-partial)")
chk("Δ 경계 0.10 정확 → 부분", build(m=-0.50), "부분(H096-partial)")
chk("Δ 0.0999 → 보류", build(m=-0.5001), "보류")
chk("고착(Δ 0.02)", build(m=-0.58), "고착(H096-null)")
chk("Δ 0.05 정확 → 고착 아님 → 보류", build(m=-0.55), "보류")
chk("성공 4/5", build(per={20: -0.58}), "반전 성공(H096)")
chk("M1 반전 줄 없음", build(rc_bad=19), "보류(조작검증 실패)")
chk("M4 반사 변함", build(rc_bad=16), "보류(조작검증 실패)")
chk("M2 규칙 일치 0.99", build(s_bad=(18, {"agree": 0.99})), "보류(조작검증 실패)")
chk("M3 시행 2900", build(s_bad=(18, {"n": 2900})), "보류(조작검증 실패)")
chk("M3 동결 잔차", build(s_bad=(20, {"res": 0.01})), "보류(조작검증 실패)")
chk("M5 출발점 어긋남 0.0021", build(pre_shift=0.0021), "보류(조작검증 실패)")
chk("M6 적재 1줄", build(load=1), "보류(조작검증 실패)")
chk("결측(E172 A 기준 없음)", build(drop_ref=18), "결측")
for name, data, want in (("거울상 m 0.48 정확(|R| 0.60) 5/5", build(m=0.48), 5), ("거울상 m 0.4799 → 0/5", build(m=0.4799), 0),
                         ("거울상 한 뇌 0.47 → 4/5", build(m=0.60, per={20: 0.47}), 4)):
    l, r = J.judge(*data)
    g = r is not None and len(r["mirror"]) == want; ok_all &= g
    print("%-32s 기대 %d → %s %s" % (name, want, len(r["mirror"]) if r else l, "✓" if g else "✗"))
R = np.zeros((3000, 37)); R[:, 2] = np.arange(3000) % 2; R[:, 6] = np.where(np.arange(3000) < 1500, 1 - R[:, 2], R[:, 2]); R[:, 7] = 1
R[:, 13] = 1.0; R[:, 21] = J.R20
st = J.rev_stats(R)
g = st["n"] == 3000 and st["agree"] == 1.0 and st["res"] < 1e-12 and st["rew_blk"][0] == 100; ok_all &= g
print("%-32s → %s" % ("rev_stats(규칙 전환 1500)", "✓" if g else "✗ %s" % st))
with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E172"), ("traces", "E172")):
        os.makedirs(os.path.join(td, *d))
    open(os.path.join(td, "E172.log"), "w", encoding="utf-8").write("  e172 rev b16: => 사전 +0.0080 사후 +0.3000 보상 2100 || 적재 2 || x\n")
    open(os.path.join(td, "logs", "E172", "rev_b16.log"), "w", encoding="utf-8").write(
        "[반전] 시행 1500 부터 규칙 = 같은 쪽\n[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")
    open(os.path.join(td, "logs", "E172", "train_A_b16.log"), "w", encoding="utf-8").write("[사전] x | **변조폭 +0.0080** (y)\n[사후] x | **변조폭 -0.6354**\n")
    np.savez_compressed(os.path.join(td, "traces", "E172", "tr_rev_b16.npz"), rows=R)
    J.EXP = td
    T, S, RC, REF = J.load()
g = (T[16] == {"pre": 0.008, "post": 0.3, "rew": 2100, "load": 2} and RC[16] == (True, True) and REF[16] == (80, -6354)
     and S[16]["n"] == 3000 and RC[17] is None and 17 not in REF)
ok_all &= g
print("%-32s → %s" % ("원 로그·E172 A 기준 파싱", "✓" if g else "✗ %s %s %s" % (T, RC, REF)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
