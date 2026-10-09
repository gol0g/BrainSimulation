#!/usr/bin/env python3
"""judge_e164.py 합성 시험: 영향 큼(같은 부호 4/5)·무시 가능·중간, 경계(Δ ±0.05·0.03 정확), 섞인 부호, 조작검증(점검 줄·I_input·[사전] 재현·추적), 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e164.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e164 as J


def build(d0=0.0, d25=0.0, per=None, over=None, drop=None):
    X = {}
    for a, d in (("R0X", d0), ("R25X", d25)):
        for b in J.BRAINS:
            dd = (per or {}).get((a, b), d)
            X[("b", a, b)] = (200, 200 - 2500)
            X[("x", a, b)] = (200, 200 - 2500 + J.i4(dd))
            X[("c", a, b)] = {"n": 3, "set": True, "off": True, "da": (80.0, 0.0)}
            X[("s", a, b)] = {"n": 500, "res": 1e-8, "pre_ratio": 0.0}
    for k, f in (over or {}).items():
        f(X[k]) if callable(f) else X.__setitem__(k, f)
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, want):
    global ok_all
    c, r = J.judge(X)
    got = r["verdict"] if r else c[0]
    good = (got.startswith(want) and (want != "보류" or got == "보류")) if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-34s 기대 %-16s → %-24s %s" % (name, want[:16], got[:24], "✓" if good else "✗ %s" % c))


chk("무시 가능(Δ 0)", build(), "무시 가능(H087-null)")
chk("영향 큼(반사 0, Δ −0.08)", build(d0=-0.08), "영향 큼(H087)")
chk("영향 큼(반사 25, Δ +0.06)", build(d25=0.06), "영향 큼(H087)")
chk("Δ 경계 −0.05 정확 → 영향 큼", build(d0=-0.05), "영향 큼(H087)")
chk("Δ −0.0499 → 중간 → 보류", build(d0=-0.0499), "보류(중간)")
chk("|Δ| 경계 0.03 정확 → 무시 아님 → 보류", build(d0=0.03, d25=0.0), "보류(중간)")
chk("|Δ| 0.0299 → 무시 가능", build(d0=0.0299, d25=-0.0299), "무시 가능(H087-null)")
chk("섞인 부호(+0.08 3·−0.08 2) → 보류", build(d0=0.08, per={("R0X", 13): -0.08, ("R0X", 14): -0.08}), "보류(중간)")
chk("영향 큼 4/5(한 뇌 0)", build(d25=-0.07, per={("R25X", 14): 0.0}), "영향 큼(H087)")
chk("무시 가능 4/5(한 뇌 0.04)", build(per={("R0X", 12): 0.04}), "무시 가능(H087-null)")
chk("점검 줄 2개", build(over={("c", "R0X", 11): lambda d: d.update(n=2)}), "보류(조작검증 실패)")
chk("I_input 0 이 아님", build(over={("c", "R25X", 12): lambda d: d.update(da=(80.0, 5.0))}), "보류(조작검증 실패)")
chk("I_input 줄 없음", build(over={("c", "R25X", 12): lambda d: d.update(da=None)}), "보류(조작검증 실패)")
chk("설정 줄 없음", build(over={("c", "R0X", 10): lambda d: d.update(set=False)}), "보류(조작검증 실패)")
chk("[사전] 재현 어긋남 0.0021", build(over={("x", "R0X", 13): (221, 221 - 2500)}), "보류(조작검증 실패)")
chk("추적 시행 400", build(over={("s", "R25X", 10): lambda d: d.update(n=400)}), "보류(조작검증 실패)")
chk("동결 잔차", build(over={("s", "R25X", 10): lambda d: d.update(res=0.01)}), "보류(조작검증 실패)")
chk("결측", build(drop=("b", "R25X", 11)), "결측")
with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E164"), ("logs", "E141"), ("logs", "E142"), ("traces", "E164")):
        os.makedirs(os.path.join(td, *d))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    w(("logs", "E164", "R0X_b10.log"), "[구현 점검] rw_da_reset=True offset_steps=3 (외부 검토 2026-10-09 ①·③)\n[사전] x | **변조폭 +0.0195** (y)\n"
      "[구현 점검] 오프셋(조향 3처리 합) 첫 에피소드 +0.0120\n[구현 점검] 첫 보상 창 끝 도파민 뉴런 I_input 80.0 → 0.0\n[사후] x | **변조폭 -0.2500**\n")
    w(("logs", "E141", "b10.log"), "[사전] x | **변조폭 +0.0195** (y)\n[사후] x | **변조폭 -0.2189**\n")
    R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 21] = J.R20
    np.savez_compressed(os.path.join(td, "traces", "E164", "tr_R0X_b10.npz"), rows=R)
    J.EXP = td
    X = J.load()
g = (X[("x", "R0X", 10)] == (195, -2500) and X[("b", "R0X", 10)] == (195, -2189) and X[("c", "R0X", 10)] == {"n": 3, "set": True, "off": True, "da": (80.0, 0.0)}
     and X[("s", "R0X", 10)]["n"] == 500 and ("x", "R25X", 10) not in X)
ok_all &= g
print("%-34s → %s" % ("원 로그 파싱", "✓" if g else "✗ %s" % {k: v for k, v in X.items() if k[2] == 10}))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
