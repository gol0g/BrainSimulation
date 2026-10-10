#!/usr/bin/env python3
"""judge_e168.py 합성 시험: 대체 성공·실패·부분, 경계(q_I 0.80·0.50, q_I − q_R 0.20·0.10 정확), 4/5, 조작검증(연결 줄·KC 억제·결정 단계·I_input·[사전]·추적·전제), 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e168.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e168 as J


def build(qI=None, qR=None, eF=-6000, over=None, drop=None):
    X = {}
    for b in J.BRAINS:
        X[("F", b)] = {"pre": 150, "post": 150 + eF, "rew": 340, "conn": 0, "kc": None, "da0": False}
        for a, q, conn, kc in (("FI", (qI or {}).get(b, 0.9), 1, {"dec": 45000, "rw": 1000, "pun": 9000, "w": 5.0}),
                               ("FR", (qR or {}).get(b, 0.08), 0, {"dec": 50000, "rw": 20000, "pun": 9000, "w": 0.0})):
            X[(a, b)] = {"pre": 150, "post": 150 + int(round(eF * q)), "rew": 330, "conn": conn, "kc": dict(kc), "da0": True}
            X[("s", a, b)] = {"n": 500, "pre_ratio": 0.0}
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
    print("%-42s 기대 %-26s → %-34s %s" % (name, w1 + (" / " + w2 if w2 else ""), got[:34], "✓" if good else "✗ %s" % c))


allb = lambda v: {b: v for b in J.BRAINS}
chk("대체 성공 + 연결 효과(q_I 0.9, q_R 0.08)", build(), "대체 성공(H091)", "연결 효과")
chk("대체 실패 + 효과 없음(q_I 0.1, q_R 0.08)", build(qI=allb(0.1)), "대체 실패(H091-null)", "효과 없음")
chk("부분 + 연결 효과(q_I 0.65)", build(qI=allb(0.65)), "부분", "연결 효과")
chk("q_I 0.80 정확 → 성공", build(qI=allb(0.80)), "대체 성공(H091)")
chk("q_I 0.7998 → 부분", build(qI=allb(0.7998)), "부분")
chk("q_I 0.50 정확 → 실패", build(qI=allb(0.50), qR=allb(0.45)), "대체 실패(H091-null)", "효과 없음")
chk("q_I 0.5002 → 부분", build(qI=allb(0.5002), qR=allb(0.45)), "부분")
chk("q_I − q_R 0.20 정확 → 연결 효과", build(qI=allb(0.45), qR=allb(0.25)), "대체 실패(H091-null)", "연결 효과")
chk("q_I − q_R 0.1998 → 중간", build(qI=allb(0.45), qR=allb(0.2502)), "대체 실패(H091-null)", "중간")
chk("|q_I − q_R| 0.10 정확 → 중간", build(qI=allb(0.30), qR=allb(0.20)), "대체 실패(H091-null)", "중간")
chk("|q_I − q_R| 0.0998 → 효과 없음", build(qI=allb(0.2998), qR=allb(0.20)), "대체 실패(H091-null)", "효과 없음")
chk("4/5 성공(한 뇌 q_I 0.3)", build(qI={**allb(0.9), 18: 0.3}), "대체 성공(H091)")
chk("3/5 성공·2 실패 → 부분", build(qI={**allb(0.9), 16: 0.2, 17: 0.2}), "부분")
F = "보류(조작검증 실패)"
chk("MC FI 연결 줄 0", build(over={("FI", 17): lambda d: d.update(conn=0)}), F, F)
chk("MC FR 연결 줄 1", build(over={("FR", 17): lambda d: d.update(conn=1)}), F)
chk("MK 보상 창 KC 10.05%", build(over={("FI", 18): lambda d: d["kc"].update(rw=2001)}), F)
chk("MK 보상 창 KC 10% 정확 → 통과", build(over={("FI", 18): lambda d: d["kc"].update(rw=2000)}), "대체 성공(H091)")
chk("MK 결정 단계 89.998%", build(over={("FI", 19): lambda d: d["kc"].update(dec=44999)}), F)
chk("MK KC 줄 없음", build(over={("FR", 19): lambda d: d.update(kc=None)}), F)
chk("MD I_input 0 아님", build(over={("FR", 20): lambda d: d.update(da0=False)}), F)
chk("MP [사전] 차 0.0021", build(over={("FI", 16): lambda d: d.update(pre=171)}), F)
chk("MT 추적 499", build(over={("s", "FR", 16): lambda d: d.update(n=499)}), F)
chk("전제 e_F −0.0999", build(eF=-999), F)
chk("결측(FR 뇌 20)", build(drop=("FR", 20)), "결측")

with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E168"), ("logs", "E161"), ("traces", "E168")):
        os.makedirs(os.path.join(td, *d))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    w(("logs", "E168", "FI_b16.log"), "  [E168 도파민→KC억제] 도파민 뉴런 100 → KC 억제 뉴런 좌·우 400, w=5.00, p=0.20\n[사전] x | **변조폭 +0.0149** (y)\n"
      "[구현 점검] 첫 보상 창 끝 도파민 뉴런 I_input 53.0 → 0.0\n[학습] 5ep 완료, 보상 335회 (탐색 주입 296회, ε=0.60)\n"
      "[E168 KC 발화] 결정 단계(3처리 끝) 평균 0.045000 n=500 | 보상 창 보상 시행 평균 0.001000 n=335 | 보상 창 처벌 시행 평균 0.009000 n=165 | da_kc_inh=5.00 p=0.20\n"
      "[사후] x | **변조폭 -0.5000**\n")
    w(("logs", "E161", "F_b16.log"), "[사전] x | **변조폭 +0.0149** (y)\n[학습] 5ep 완료, 보상 341회\n[사후] x | **변조폭 -0.5977**\n")
    np.savez_compressed(os.path.join(td, "traces", "E168", "tr_FI_b16.npz"), rows=np.zeros((500, 37)))
    J.EXP = td
    X = J.load()
g = (X[("FI", 16)] == {"pre": 149, "post": -5000, "rew": 335, "conn": 1, "kc": {"dec": 45000, "rw": 1000, "pun": 9000, "w": 5.0}, "da0": True}
     and X[("F", 16)]["pre"] == 149 and X[("F", 16)]["post"] == -5977 and X[("s", "FI", 16)]["n"] == 500 and ("FR", 16) not in X)
ok_all &= g
print("%-42s → %s" % ("원 로그 파싱(실제 줄 형식)", "✓" if g else "✗ %s" % X))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
