#!/usr/bin/env python3
"""judge_e157.py 합성 시험(조건 1): 네 범주·경계(S·G 정확히 0.30)·4/5 규칙·섞임 보류, 조작검증 MS·MJ·ML·M1·M1b·M3·MD 경계와 실패, 결측, 원 로그 파싱(D·F 재사용 경로 우선순위).
실행: python3 scripts/test_judge_e157.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e157 as J


def build(eD=-0.25, eF=-0.50, eFk=-0.27, eDk=-0.45, per=None, over=None, drop=None):
    X = {}
    for b in J.BRAINS:
        p = {"eD": eD, "eF": eF, "eFk": eFk, "eDk": eDk}
        p.update((per or {}).get(b, {}))
        for a, k in (("D", "eD"), ("F", "eF"), ("Fk", "eFk"), ("Dk", "eDk")):
            X[("e", a, b)] = J.i4(p[k])
            X[("rew", a, b)] = 300
            X[("r25", a, b)] = 0.3
        X[("ld", "D", b)] = (0, 0); X[("ld", "F", b)] = (2, 0); X[("ld", "Fk", b)] = (2, 2); X[("ld", "Dk", b)] = (0, 2)
        X[("sp", "D", b)] = 10000; X[("sp", "F", b)] = 12500; X[("sp", "Fk", b)] = 10000; X[("sp", "Dk", b)] = 12500
        X[("J", "Fk", b)] = (0.0, 0.0); X[("J", "Dk", b)] = (0.40, 0.42)
        for a in ("Fk", "Dk"):
            X[("st", a, b)] = {"n": 500, "pre_ratio": 0.0, "res": 1e-8, "alive": 1.0}
    for (k, a, b), v in (over or {}).items():
        X[(k, a, b)] = v
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, want):
    global ok_all
    c, r = J.judge(X)
    got = r["verdict"] if r else c[0]
    good = got == want if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-34s 기대 %-20s → %-24s %s" % (name, want, got[:24], "✓" if good else "✗ %s" % c))


chk("이득", build(), "이득(H080)")
chk("분리", build(eFk=-0.45, eDk=-0.27), "분리(H080-sep)")
chk("둘 다", build(eFk=-0.40, eDk=-0.40), "둘 다(H080-both)")
chk("둘 다 아님", build(eFk=-0.27, eDk=-0.27), "둘 다 아님(H080-int)")
chk("S 경계 정확 0.30 → S 작음(이득)", build(eFk=-0.3250), "이득(H080)")
chk("S 0.3004 → 둘 다", build(eFk=-0.3251), "둘 다(H080-both)")
chk("G 경계 정확 0.30 → G 작음(둘 다 아님)", build(eDk=-0.3250), "둘 다 아님(H080-int)")
chk("G 0.3004 → 이득", build(eDk=-0.3251), "이득(H080)")
chk("4/5 이득", build(per={14: {"eFk": -0.45, "eDk": -0.27}}), "이득(H080)")
chk("3/5 → 보류", build(per={13: {"eFk": -0.45, "eDk": -0.27}, 14: {"eFk": -0.45, "eDk": -0.27}}), "보류")
chk("MS Fk/D 0.85 정확 → 통과", build(over={("sp", "Fk", 12): 8500}), "이득(H080)")
chk("MS Fk/D 0.8499 → 실패", build(over={("sp", "Fk", 12): 8499}), "보류(조작검증 실패)")
chk("MS Dk/F 1.15 정확 → 통과", build(over={("sp", "Dk", 11): 14375}), "이득(H080)")
chk("MS Dk/F 1.1501 → 실패", build(over={("sp", "Dk", 11): 14376}), "보류(조작검증 실패)")
chk("MJ Fk 0.0500 정확 → 통과", build(over={("J", "Fk", 10): (0.05, 0.0)}), "이득(H080)")
chk("MJ Fk 0.0501 → 실패", build(over={("J", "Fk", 10): (0.0, 0.0501)}), "보류(조작검증 실패)")
chk("MJ Dk 0.2500 정확 → 통과", build(over={("J", "Dk", 13): (0.25, 0.30)}), "이득(H080)")
chk("MJ Dk 0.2499 → 실패", build(over={("J", "Dk", 13): (0.30, 0.2499)}), "보류(조작검증 실패)")
chk("ML Fk 적재 1 → 실패", build(over={("ld", "Fk", 14): (1, 2)}), "보류(조작검증 실패)")
chk("ML Fk 배율 1 → 실패", build(over={("ld", "Fk", 14): (2, 1)}), "보류(조작검증 실패)")
chk("ML Dk 적재 있음 → 실패", build(over={("ld", "Dk", 10): (2, 2)}), "보류(조작검증 실패)")
chk("M1 동결 실패", build(over={("st", "Dk", 11): {"n": 500, "pre_ratio": 0.0, "res": 0.002, "alive": 1.0}}), "보류(조작검증 실패)")
chk("M1b 되돌림 실패", build(over={("st", "Fk", 12): {"n": 500, "pre_ratio": 0.0, "res": 1e-8, "alive": 0.8}}), "보류(조작검증 실패)")
chk("M3 시행 400", build(over={("st", "Fk", 13): {"n": 400, "pre_ratio": 0.0, "res": 1e-8, "alive": 1.0}}), "보류(조작검증 실패)")
chk("M3 도파민 전 변화", build(over={("st", "Dk", 13): {"n": 500, "pre_ratio": 0.002, "res": 1e-8, "alive": 1.0}}), "보류(조작검증 실패)")
chk("MD e_D −0.1000 정확 → 통과", build(eD=-0.1000, eF=-0.2, eFk=-0.11, eDk=-0.18), "이득(H080)")
chk("MD e_D −0.0999 → 실패", build(eD=-0.0999, eF=-0.2, eFk=-0.11, eDk=-0.18), "보류(조작검증 실패)")
chk("결측(J Dk b12)", build(drop=("J", "Dk", 12)), "결측")
chk("결측(r25 D b10)", build(drop=("r25", "D", 10)), "결측")

# 원 로그 파싱 — D 는 E141, F 는 E153(E157 learn_D 가 있으면 그것)
KR = ("=> KCRATE kc_l | 좌선택 1 우선택 2 | 제시 스파이크 %d 기준선(제시창) 평균 0.0500 | KC별 ΔS y\n"
      "=> KCRATE kc_r | 좌선택 1 우선택 2 | 제시 스파이크 %d 기준선(제시창) 평균 0.0500 | KC별 ΔS y\n")
OV = ("=> KCOVERLAP side=l good=10 bad=12 jac=%.4f cos=0.1 jac025=0.1 jac100=0.1 split_jac=0.9 split_cos=0.9 | "
      "side=r good=11 bad=13 jac=%.4f cos=0.1 jac025=0.1 jac100=0.1 split_jac=0.9 split_cos=0.9 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n")
LDL = "[E153 종류 입력 적재] k 검증 일치 — x\n"; SCL = "[E157 종류 입력 배율] k=0.8000 검증 일치 — x\n"
with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E157"), ("logs", "E141"), ("logs", "E153"), ("traces", "E157")):
        os.makedirs(os.path.join(td, *d))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    w(("logs", "E141", "b10.log"), "[사전] x | **변조폭 +0.0195** (y)\n[학습] 5ep 완료, 보상 300회 (z)\n[사후] x | **변조폭 -0.2189**\n")
    w(("logs", "E153", "train_b10.log"), LDL * 2 + "[사전] x | **변조폭 +0.0117** (y)\n[사후] x | **변조폭 -0.4893**\n")
    w(("logs", "E157", "learn_Fk_b10.log"), LDL * 2 + SCL * 2 + "[사전] x | **변조폭 +0.0100** (y)\n[학습] 5ep 완료, 보상 310회 (z)\n[사후] x | **변조폭 -0.3000**\n")
    w(("logs", "E157", "kcrate_Fk_b10.log"), KR % (24000, 25000))
    w(("logs", "E157", "ov_Fk_b10.log"), OV % (0.0, 0.0123))
    w(("logs", "E157", "r25_Dk_b10.log"), "[사전] x | **변조폭 +0.2000** (y)\n")
    R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 14] = 1.0; R[:, 21] = J.R_STAR; R[:, 22] = J.R_STAR
    np.savez_compressed(os.path.join(td, "traces", "E157", "tr_Fk_b10.npz"), rows=R)
    J.EXP = td
    X1 = J.load()
    w(("logs", "E157", "learn_D_b10.log"), "[사전] x | **변조폭 +0.0000** (y)\n[사후] x | **변조폭 -0.3000**\n")
    X2 = J.load()
g = (X1[("e", "D", 10)] == -2384 and X1[("e", "F", 10)] == -5010 and X1[("e", "Fk", 10)] == -3100 and X1[("rew", "Fk", 10)] == 310
     and X1[("ld", "Fk", 10)] == (2, 2) and X1[("ld", "F", 10)] == (2, 0) and X1[("sp", "Fk", 10)] == 49000 and X1[("J", "Fk", 10)] == (0.0, 0.0123)
     and X1[("r25", "Dk", 10)] == 0.2 and X1[("st", "Fk", 10)]["n"] == 500 and X1[("st", "Fk", 10)]["res"] < 1e-12
     and ("e", "Dk", 10) not in X1 and X2[("e", "D", 10)] == -3000)
ok_all &= g
print("%-34s → %s" % ("원 로그 파싱·재사용 우선순위", "✓" if g else "✗ %s" % {k: v for k, v in X1.items() if k[2] == 10}))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
