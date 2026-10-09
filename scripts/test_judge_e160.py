#!/usr/bin/env python3
"""judge_e160.py 합성 시험(조건 1): 성공·분리만·실패·보류, 경계(r 1.30·자카드 0.10·0.25 정확), 4/5, 조작검증 MO·MS·ML·M1·M1b·M3·MD, 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e160.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e160 as J


def build(e=-0.40, D=-0.25, jl=0.0, jr=0.0, per=None, over=None, drop=None):
    X = {}
    for b in J.BRAINS:
        p = {"e": e, "D": D, "jl": jl, "jr": jr}
        p.update((per or {}).get(b, {}))
        side = {"fired": 200, "sel_med": 0.9, "frac09": 0.5, "sum_med": 1.0, "dg_good": 10.0, "dg_bad": 10.0}
        X[("dev", b)] = {"l": dict(side), "r": dict(side)}
        X[("ov", b)] = {"jl": p["jl"], "jr": p["jr"], "n": (90, 90, 90, 90), "ld": 2}
        X[("learn", b)] = {"pre": 100, "post": 100 + J.i4(p["e"]), "ld": 2, "rew": 330}
        X[("D", b)] = J.i4(p["D"])
        X[("st", b)] = {"n": 500, "pre_ratio": 0.0, "res": 1e-8, "alive": 1.0}
        X[("sp", b)] = (55000, 49000)
    for k, f in (over or {}).items():
        f(X[k])
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, want):
    global ok_all
    c, r = J.judge(X)
    got = r["verdict"] if r else c[0]
    good = got.startswith(want) and (want != "보류" or got == "보류") if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-34s 기대 %-22s → %-28s %s" % (name, want[:22], got[:28], "✓" if good else "✗ %s" % c))


chk("성공(r 1.6, 자카드 0)", build(), "형성 성공(H083)")
chk("r 경계 1.30 정확 → 성공", build(e=-0.3250), "형성 성공(H083)")
chk("r 1.2996 → 분리만", build(e=-0.3249), "분리만(H083-sep-only)")
chk("자카드 경계 0.10 정확 → 분리", build(jl=0.1000, jr=0.1000), "형성 성공(H083)")
chk("자카드 0.1001 → 분리 아님 → 보류", build(jr=0.1001), "보류")
chk("분리만(r 1.1)", build(e=-0.275), "분리만(H083-sep-only)")
chk("실패(자카드 0.30)", build(jl=0.30, jr=0.30), "형성 실패(H083-null)")
chk("자카드 경계 0.25 정확 → 실패 아님 → 보류", build(jl=0.25, jr=0.25), "보류")
chk("자카드 0.2501 한쪽 → 실패", build(jl=0.0, jr=0.2501), "형성 실패(H083-null)")
chk("성공 4/5", build(per={14: {"e": -0.20}}), "형성 성공(H083)")
chk("성공 3/5·분리 5/5 → 분리만", build(per={13: {"e": -0.20}, 14: {"e": -0.20}}), "분리만(H083-sep-only)")
chk("MO Oja 변화 없음", build(over={("dev", 12): lambda d: d["l"].update(dg_bad=0.0)}), "보류(조작검증 실패)")
chk("MS 선택성 0.7999", build(over={("dev", 11): lambda d: d["r"].update(sel_med=0.7999)}), "보류(조작검증 실패)")
chk("MS 경계 0.80 정확 → 통과", build(over={("dev", 11): lambda d: d["r"].update(sel_med=0.80)}), "형성 성공(H083)")
chk("ML 학습 적재 1줄", build(over={("learn", 10): lambda d: d.update(ld=1)}), "보류(조작검증 실패)")
chk("ML 겹침 적재 0줄", build(over={("ov", 10): lambda d: d.update(ld=0)}), "보류(조작검증 실패)")
chk("M1 동결 실패", build(over={("st", 13): lambda d: d.update(res=0.01)}), "보류(조작검증 실패)")
chk("M1b 되돌림 실패", build(over={("st", 13): lambda d: d.update(alive=0.5)}), "보류(조작검증 실패)")
chk("M3 시행 400", build(over={("st", 14): lambda d: d.update(n=400)}), "보류(조작검증 실패)")
chk("MD e_D −0.0999", build(D=-0.0999, e=-0.20), "보류(조작검증 실패)")
chk("결측(ov b12)", build(drop=("ov", 12)), "결측")
# 원 로그 파싱
with tempfile.TemporaryDirectory() as td:
    for d_ in (("logs", "E160"), ("logs", "E141"), ("logs", "E157"), ("traces", "E160")):
        os.makedirs(os.path.join(td, *d_))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    w(("logs", "E160", "dev_b10.log"), "=> KCDEVOJA side=l fired=210 sel_med0=0.5500 sel_med=0.8800 frac09=0.4500 goodfrac=0.5000 sum_med=0.9700 sum_q10=0.7 sum_q90=1.3 dg_good=12.5 dg_bad=11.0"
      " | side=r fired=205 sel_med0=0.5400 sel_med=0.8600 frac09=0.4200 goodfrac=0.4900 sum_med=1.0300 sum_q10=0.7 sum_q90=1.3 dg_good=10.0 dg_bad=9.5 | n=100 eta=0.02 beta=1 mmax=32 tau=20 save=x\n")
    w(("logs", "E160", "ov_b10.log"), "[E153 종류 입력 적재] k 검증 일치 — x\n" * 2 + "=> KCOVERLAP side=l good=95 bad=90 jac=0.0123 cos=0.1 jac025=0.1 jac100=0.1 split_jac=0.9 split_cos=0.9 | "
      "side=r good=93 bad=88 jac=0.0000 cos=0.1 jac025=0.1 jac100=0.1 split_jac=0.9 split_cos=0.9 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n")
    w(("logs", "E160", "learn_b10.log"), "[E153 종류 입력 적재] k 검증 일치 — x\n" * 2 + "[사전] x | **변조폭 +0.0150** (y)\n[학습] 5ep 완료, 보상 331회 (z)\n[사후] x | **변조폭 -0.4000**\n")
    w(("logs", "E141", "b10.log"), "[사전] x | **변조폭 +0.0195** (y)\n[사후] x | **변조폭 -0.2189**\n")
    KR = "=> KCRATE kc_l | a | 제시 스파이크 %d 기준선(제시창) 평균 0.05 | b\n=> KCRATE kc_r | a | 제시 스파이크 %d 기준선(제시창) 평균 0.05 | b\n"
    w(("logs", "E160", "kcrate_b10.log"), KR % (30000, 31000))
    w(("logs", "E157", "kcrate_D_b10.log"), KR % (24000, 25125))
    R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 14] = 1.0; R[:, 21] = J.R_STAR; R[:, 22] = J.R_STAR
    np.savez_compressed(os.path.join(td, "traces", "E160", "tr_b10.npz"), rows=R)
    J.EXP = td
    X = J.load()
g = (X[("dev", 10)]["l"]["sel_med"] == 0.88 and X[("dev", 10)]["r"]["sum_med"] == 1.03 and X[("dev", 10)]["l"]["dg_bad"] == 11.0
     and X[("ov", 10)]["jl"] == 0.0123 and X[("ov", 10)]["ld"] == 2 and X[("learn", 10)] == {"pre": 150, "post": -4000, "ld": 2, "rew": 331}
     and X[("D", 10)] == -2384 and X[("st", 10)]["n"] == 500 and X[("sp", 10)] == (61000, 49125) and ("dev", 11) not in X)
ok_all &= g
print("%-34s → %s" % ("원 로그 파싱", "✓" if g else "✗ %s" % {k: v for k, v in X.items() if k[1] == 10}))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
