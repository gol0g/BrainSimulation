#!/usr/bin/env python3
"""judge_e169.py 합성 시험: 대체 성공·실패·부분, 경계(q₁ 0.80·0.50, c₁ − c₂ 0.20·0.10 정확), 판정 2 전제, 조작검증(창 1 잔차·동결 꺼짐·[사전]·추적·적재·전제), 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e169.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e169 as J


def build(q1=None, qff=None, qnf=None, eF=-6000, over=None, drop=None):
    """q1[b] = e_FW1/e_F(기본 0.9), qff[b] = e_FFW1/e_F(기본 0.9), qnf[b] = e_FNF/e_F(기본 0.07)."""
    X = {}
    for b in J.BRAINS:
        X[("F", b)] = {"pre": 150, "post": 150 + eF, "rew": 340, "ld": 2}
        X[("FNF", b)] = {"pre": 150, "post": 150 + int(round(eF * (qnf or {}).get(b, 0.07))), "rew": 335, "ld": 2}
        X[("FW1", b)] = {"pre": 150, "post": 150 + int(round(eF * (q1 or {}).get(b, 0.9))), "rew": 330, "ld": 2}
        X[("FFW1", b)] = {"pre": 150, "post": 150 + int(round(eF * (qff or {}).get(b, 0.9))), "rew": 330, "ld": 2}
        X[("s", "FW1", b)] = {"n": 500, "pre_ratio": 0.0, "res1": 3.0}
        X[("s", "FFW1", b)] = {"n": 500, "pre_ratio": 0.0, "res1": 1e-8}
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
        good, got = (r is None and "결측" in c[0]), c[0]
    else:
        good = r is not None and r["v1"].startswith(w1) and (w2 is None or r["v2"].startswith(w2))
        got = "%s / %s" % (r["v1"], r["v2"]) if r else c[0]
    ok_all &= good
    print("%-40s 기대 %-30s → %-36s %s" % (name, w1 + (" / " + w2 if w2 else ""), got[:36], "✓" if good else "✗ %s" % c))


allb = lambda v: {b: v for b in J.BRAINS}
chk("성공 + 오염 줄임(q₁ 0.9, c₁ 1.0, c₂ 0.07)", build(), "대체 성공(H092)", "창 단축이 오염을 줄임")
chk("실패 + 줄이지 않음(q₁ 0.07·qff 0.9 → c₁ 0.078)", build(q1=allb(0.07)), "실패(H092-null)", "줄이지 않음")
chk("부분(q₁ 0.65)", build(q1=allb(0.65)), "부분", "창 단축이 오염을 줄임")
chk("q₁ 0.80 정확 → 성공", build(q1=allb(0.80)), "대체 성공(H092)")
chk("q₁ 0.7998 → 부분", build(q1=allb(0.7998)), "부분")
chk("q₁ 0.50 정확 → 실패", build(q1=allb(0.50)), "실패(H092-null)")
chk("q₁ 0.5002 → 부분", build(q1=allb(0.5002)), "부분")
chk("c₁ − c₂ 0.20 정확(c₁ 0.27, c₂ 0.07) → 줄임", build(q1=allb(0.27), qff=allb(1.0)), "실패(H092-null)", "창 단축이 오염을 줄임")
chk("c₁ − c₂ 0.1998 → 중간", build(q1=allb(0.2698), qff=allb(1.0)), "실패(H092-null)", "중간")
chk("|c₁ − c₂| 0.10 정확 → 중간", build(q1=allb(0.17), qff=allb(1.0)), "실패(H092-null)", "중간")
chk("|c₁ − c₂| 0.0998 → 줄이지 않음", build(q1=allb(0.1698), qff=allb(1.0)), "실패(H092-null)", "줄이지 않음")
chk("판정 2 전제(FFW1 −0.0999)", build(q1=allb(0.05), qff={**allb(0.9), 18: 0.1665}), "실패(H092-null)", "판정 불가")
chk("4/5 성공(한 뇌 q₁ 0.3)", build(q1={**allb(0.9), 19: 0.3}), "대체 성공(H092)")
chk("3/5 성공·2 실패 → 부분", build(q1={**allb(0.9), 16: 0.2, 17: 0.2}), "부분")
F = "보류(조작검증 실패)"
chk("창 1 확인 실패(FFW1 잔차 0.002)", build(over={("s", "FFW1", 17): lambda d: d.update(res1=0.002)}), F, F)
chk("동결 꺼짐 실패(FW1 잔차 0.04)", build(over={("s", "FW1", 18): lambda d: d.update(res1=0.04)}), F)
chk("[사전] 차 0.0021", build(over={("FW1", 19): lambda d: d.update(pre=171)}), F)
chk("추적 499", build(over={("s", "FFW1", 20): lambda d: d.update(n=499)}), F)
chk("적재 1", build(over={("FFW1", 16): lambda d: d.update(ld=1)}), F)
chk("전제 e_F −0.0999", build(eF=-999), F)
chk("결측(E166 FNF 뇌 20)", build(drop=("FNF", 20)), "결측")
chk("결측(FW1 추적 뇌 16)", build(drop=("s", "FW1", 16)), "결측")

with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E169"), ("logs", "E161"), ("logs", "E166"), ("traces", "E169")):
        os.makedirs(os.path.join(td, *d))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    LD = "[E153 종류 입력 적재] /x/kctype.npz 검증 일치 — good_food_eye_l_to_kc_l=1.0\n"
    w(("logs", "E169", "FFW1_b16.log"), LD + "[사전] 오프셋 -0.004 | 정답률 22.0% | **변조폭 +0.0149** (양수=반사방향, 음수=역전)\n[학습] 5ep 완료, 보상 330회 (탐색 주입 296회, ε=0.60)\n"
      + LD + "[사후] 오프셋 -0.013 | 정답률 100.0% | **변조폭 -0.4500**\n")
    R = np.zeros((500, 37)); R[:, 13] = 2.0; R[:, 21] = 2.0 * J.R1
    np.savez_compressed(os.path.join(td, "traces", "E169", "tr_FFW1_b16.npz"), rows=R)
    J.EXP = td
    X = J.load()
g = (X[("FFW1", 16)] == {"pre": 149, "post": -4500, "rew": 330, "ld": 2} and X[("s", "FFW1", 16)]["n"] == 500 and X[("s", "FFW1", 16)]["res1"] < 1e-9
     and ("FW1", 16) not in X and ("F", 16) not in X)
ok_all &= g
print("%-40s → %s" % ("원 로그 파싱·창 1 잔차(r₁ = (11/12)^10)", "✓" if g else "✗ %s" % X))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
