#!/usr/bin/env python3
"""judge_e166.py 합성 시험: 남음·손실·부분, 경계(q_F 0.80·0.50 정확), 4/5, 판정 2(줄임·줄이지 않음·중간, 경계 0.20·0.10),
조작검증(동결 꺼짐·추적·적재·[사전] 재현·전제), 결측, 원 로그 파싱(실제 줄 형식).
실행: python3 scripts/test_judge_e166.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e166 as J


def build(qF=None, qD=None, eF=-6000, eD=-2500, over=None, drop=None):
    """qF[b]·qD[b] = 동결 제거 효과 / 같은 뇌 동결 효과(기본 qF 0.9, qD 0.3)."""
    X = {}
    for b in J.BRAINS:
        for a, e_ref, q in (("FNF", eF, (qF or {}).get(b, 0.9)), ("DNF", eD, (qD or {}).get(b, 0.3))):
            X[("b", a, b)] = {"pre": 200, "post": 200 + e_ref, "rew": 340, "ld": 2 if a == "FNF" else 0}
            X[("x", a, b)] = {"pre": 200, "post": 200 + int(round(e_ref * q)), "rew": 300, "ld": 2 if a == "FNF" else 0}
            X[("s", a, b)] = {"n": 500, "pre_ratio": 0.0, "res": 1.5}
    for k, f in (over or {}).items():
        f(X[k]) if callable(f) else X.__setitem__(k, f)
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, w1, w2=None):
    global ok_all
    c, r = J.judge(X)
    if w1 == "결측":
        good = r is None and "결측" in c[0]
        got = c[0]
    else:
        good = r is not None and r["v1"].startswith(w1) and (w2 is None or r["v2"].startswith(w2))
        got = "%s / %s" % (r["v1"], r["v2"]) if r else c[0]
    ok_all &= good
    print("%-40s 기대 %-30s → %-40s %s" % (name, w1 + (" / " + w2 if w2 else ""), got[:40], "✓" if good else "✗ %s" % c))


allq = lambda v: {b: v for b in J.BRAINS}
chk("남음 + 줄임(qF 0.9, qD 0.3)", build(), "남음(H089)", "형성이 동결 의존을 줄임")
chk("손실 + 줄이지 않음(qF 0.3, qD 0.3)", build(qF=allq(0.3)), "손실(H089-null)", "줄이지 않음")
chk("부분(qF 0.65)", build(qF=allq(0.65)), "부분", "형성이 동결 의존을 줄임")
chk("qF 0.80 정확 → 남음", build(qF=allq(0.80)), "남음(H089)")
chk("qF 0.7998 → 부분", build(qF=allq(0.7998)), "부분")
chk("qF 0.50 정확 → 손실", build(qF=allq(0.50), qD=allq(0.45)), "손실(H089-null)", "줄이지 않음")
chk("qF 0.5002 → 부분", build(qF=allq(0.5002), qD=allq(0.45)), "부분")
chk("4/5 남음(한 뇌 qF 0.4)", build(qF={18: 0.4}), "남음(H089)")
chk("3/5 남음·2 손실 → 부분", build(qF={16: 0.3, 17: 0.3}), "부분")
chk("qF − qD 0.20 정확 → 줄임", build(qF=allq(0.6), qD=allq(0.4)), "부분", "형성이 동결 의존을 줄임")
chk("qF − qD 0.1996 → 중간", build(qF=allq(0.6), qD=allq(0.4004)), "부분", "중간")
chk("|qF − qD| 0.10 정확 → 중간", build(qF=allq(0.4), qD=allq(0.3)), "손실(H089-null)", "중간")
chk("|qF − qD| 0.0996 → 줄이지 않음", build(qF=allq(0.3996), qD=allq(0.3)), "손실(H089-null)", "줄이지 않음")
chk("qF < qD(형성이 더 의존) → 중간", build(qF=allq(0.3), qD=allq(0.6)), "손실(H089-null)", "중간")
chk("e_FNF 양수(반대 방향) → 손실", build(qF=allq(-0.2)), "손실(H089-null)")
F = "보류(조작검증 실패)"
chk("동결 꺼짐 실패(잔차 0.01)", build(over={("s", "FNF", 17): lambda d: d.update(res=0.01)}), F, F)
chk("추적 499", build(over={("s", "DNF", 18): lambda d: d.update(n=499)}), F)
chk("도파민 전 0.002", build(over={("s", "DNF", 18): lambda d: d.update(pre_ratio=0.002)}), F)
chk("FNF 적재 1", build(over={("x", "FNF", 19): lambda d: d.update(ld=1)}), F)
chk("DNF 적재 1", build(over={("x", "DNF", 19): lambda d: d.update(ld=1)}), F)
chk("[사전] 어긋남 0.0021", build(over={("x", "FNF", 20): lambda d: d.update(pre=221)}), F)
chk("전제 e_D −0.0999", build(eD=-999), F)
chk("결측(DNF 뇌 20 추적)", build(drop=("s", "DNF", 20)), "결측")
chk("결측(E161 F 뇌 16)", build(drop=("b", "FNF", 16)), "결측")

with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E166"), ("logs", "E161"), ("traces", "E166")):
        os.makedirs(os.path.join(td, *d))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    LD = "[E153 종류 입력 적재] /x/kctype.npz 검증 일치 — good_food_eye_l_to_kc_l=1.0\n"
    w(("logs", "E161", "F_b16.log"), LD + "[사전] 오프셋 -0.004 | 정답률 22.0% | **변조폭 +0.0149** (양수=반사방향, 음수=역전)\n[학습] 5ep 완료, 보상 341회 (탐색 주입 296회, ε=0.60)\n"
      + LD + "[사후] 오프셋 -0.013 | 정답률 100.0% | **변조폭 -0.5977**\n")
    w(("logs", "E166", "FNF_b16.log"), LD + "[사전] 오프셋 -0.004 | 정답률 22.0% | **변조폭 +0.0149** (양수=반사방향, 음수=역전)\n[학습] 5ep 완료, 보상 300회 (탐색 주입 296회, ε=0.60)\n"
      + LD + "[사후] 오프셋 -0.010 | 정답률 90.0% | **변조폭 -0.3001**\n")
    R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 21] = 3.0
    np.savez_compressed(os.path.join(td, "traces", "E166", "tr_FNF_b16.npz"), rows=R)
    J.EXP = td
    X = J.load()
g = (X[("b", "FNF", 16)] == {"pre": 149, "post": -5977, "rew": 341, "ld": 2} and X[("x", "FNF", 16)] == {"pre": 149, "post": -3001, "rew": 300, "ld": 2}
     and X[("s", "FNF", 16)]["n"] == 500 and abs(X[("s", "FNF", 16)]["res"] - (3.0 - J.R_STAR)) < 1e-9 and ("x", "DNF", 16) not in X)
ok_all &= g
print("%-40s → %s" % ("원 로그 파싱(실제 줄 형식)", "✓" if g else "✗ %s" % X))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
