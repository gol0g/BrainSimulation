#!/usr/bin/env python3
"""judge_e161.py 합성 시험(조건 1): 성공·분리만·실패·보류, 경계(r 1.30·자카드 0.10·0.25·MS' 상승 0.10 정확), 조작검증 각 항목, 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e161.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e161 as J


def build(e=-0.55, D=-0.25, jl=0.0, jr=0.0, per=None, over=None, drop=None):
    X = {}
    for b in J.BRAINS:
        p = {"e": e, "D": D, "jl": jl, "jr": jr}
        p.update((per or {}).get(b, {}))
        side = {"fired": 220, "sel_med0": 0.55, "sel_med": 0.85, "frac09": 0.47, "sum_med": 0.98, "dg_good": 10.0, "dg_bad": 10.0}
        X[("dev", b)] = {"l": dict(side), "r": dict(side)}
        X[("ov", b)] = {"jl": p["jl"], "jr": p["jr"], "n": (55, 60, 58, 57), "ld": 2}
        X[("F", b)] = {"pre": 100, "post": 100 + J.i4(p["e"]), "ld": 2, "rew": 337}
        X[("D", b)] = {"pre": 200, "post": 200 + J.i4(p["D"]), "ld": 0, "rew": 300}
        X[("st", b)] = {"n": 500, "pre_ratio": 0.0, "res": 1e-8, "alive": 1.0}
        X[("rate", b)] = 2
        X[("sp", b)] = (58000, 49000)
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
    print("%-36s 기대 %-22s → %-28s %s" % (name, want[:22], got[:28], "✓" if good else "✗ %s" % c))


chk("성공(r 2.2)", build(), "형성 성공(H084)")
chk("r 경계 1.30 정확 → 성공", build(e=-0.3250), "형성 성공(H084)")
chk("r 1.2996 → 분리만", build(e=-0.3249), "분리만(H084-sep-only)")
chk("자카드 경계 0.10 정확", build(jl=0.1, jr=0.1), "형성 성공(H084)")
chk("자카드 0.1001 → 보류", build(jr=0.1001), "보류")
chk("실패(0.30)", build(jl=0.3, jr=0.3), "형성 실패(H084-null)")
chk("자카드 0.25 정확 → 보류", build(jl=0.25, jr=0.25), "보류")
chk("성공 4/5", build(per={20: {"e": -0.20}}), "형성 성공(H084)")
chk("성공 3/5·분리 5/5 → 분리만", build(per={19: {"e": -0.2}, 20: {"e": -0.2}}), "분리만(H084-sep-only)")
chk("MS' 상승 정확 0.10 → 통과", build(over={("dev", 17): lambda d: d["l"].update(sel_med0=0.55, sel_med=0.65)}), "형성 성공(H084)")
chk("MS' 상승 0.0999 → 실패", build(over={("dev", 17): lambda d: d["r"].update(sel_med0=0.55, sel_med=0.6499)}), "보류(조작검증 실패)")
chk("선택성 0.745(상승 0.19) → 통과(부지표만)", build(over={("dev", 18): lambda d: d["l"].update(sel_med=0.745)}), "형성 성공(H084)")
chk("MO Oja 변화 없음", build(over={("dev", 16): lambda d: d["r"].update(dg_good=0.0)}), "보류(조작검증 실패)")
chk("ML 형성 학습 적재 1줄", build(over={("F", 16): lambda d: d.update(ld=1)}), "보류(조작검증 실패)")
chk("ML 기본 학습에 적재 있음", build(over={("D", 16): lambda d: d.update(ld=2)}), "보류(조작검증 실패)")
chk("ML 겹침 적재 0", build(over={("ov", 16): lambda d: d.update(ld=0)}), "보류(조작검증 실패)")
chk("M1 동결 실패", build(over={("st", 18): lambda d: d.update(res=0.01)}), "보류(조작검증 실패)")
chk("M1b 되돌림 실패", build(over={("st", 18): lambda d: d.update(alive=0.5)}), "보류(조작검증 실패)")
chk("M3 시행 400", build(over={("st", 19): lambda d: d.update(n=400)}), "보류(조작검증 실패)")
chk("MD e_D −0.0999", build(D=-0.0999, e=-0.2), "보류(조작검증 실패)")
chk("MR rate 1줄", build(over={("rate", 20): 1}), "보류(조작검증 실패)")
chk("결측(F b18)", build(drop=("F", 18)), "결측")
c, r = J.judge(build(over={("dev", 18): lambda d: d["l"].update(sel_med=0.745)}))
g = len(r["sel80"]) == 4; ok_all &= g
print("%-36s → %d/5 %s" % ("부지표 선택성 ≥ 0.80 집계", len(r["sel80"]), "✓" if g else "✗"))
with tempfile.TemporaryDirectory() as td:
    for d_ in (("logs", "E161"), ("traces", "E161")):
        os.makedirs(os.path.join(td, *d_))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    w(("logs", "E161", "dev_b16.log"), "=> KCDEVOJA side=l fired=220 sel_med0=0.5600 sel_med=0.8700 frac09=0.47 goodfrac=0.5 sum_med=0.9800 sum_q10=0.7 sum_q90=1.3 dg_good=12.0 dg_bad=11.0"
      " | side=r fired=215 sel_med0=0.5500 sel_med=0.8500 frac09=0.45 goodfrac=0.49 sum_med=0.9900 sum_q10=0.7 sum_q90=1.3 dg_good=10.0 dg_bad=9.0 | n=100 eta=0.02 save=x\n")
    w(("logs", "E161", "ov_b16.log"), "[E153 종류 입력 적재] k 검증 일치 — x\n" * 2 + "=> KCOVERLAP side=l good=55 bad=60 jac=0.0100 cos=0.1 jac025=0.1 jac100=0.1 split_jac=0.9 split_cos=0.9 | "
      "side=r good=58 bad=57 jac=0.0050 cos=0.1 jac025=0.1 jac100=0.1 split_jac=0.9 split_cos=0.9 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n")
    w(("logs", "E161", "F_b16.log"), "[E153 종류 입력 적재] k 검증 일치 — x\n" * 2 + "[사전] x | **변조폭 +0.0100** (y)\n[학습] 5ep 완료, 보상 337회 (z)\n[사후] x | **변조폭 -0.5500**\n")
    w(("logs", "E161", "D_b16.log"), "[사전] x | **변조폭 +0.0200** (y)\n[학습] 5ep 완료, 보상 300회 (z)\n[사후] x | **변조폭 -0.2300**\n")
    w(("logs", "E161", "rate_b16.log"), "=> KCRATE kc_l | x\n=> KCRATE kc_r | x\n")
    KR = "=> KCRATE kc_l | a | 제시 스파이크 %d 기준선(제시창) 평균 0.05 | b\n=> KCRATE kc_r | a | 제시 스파이크 %d 기준선(제시창) 평균 0.05 | b\n"
    w(("logs", "E161", "kcF_b16.log"), KR % (29000, 29500)); w(("logs", "E161", "kcD_b16.log"), KR % (24500, 25000))
    R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 14] = 1.0; R[:, 21] = J.R_STAR; R[:, 22] = J.R_STAR
    np.savez_compressed(os.path.join(td, "traces", "E161", "tr_F_b16.npz"), rows=R)
    J.EXP = td
    X = J.load()
g = (X[("dev", 16)]["l"]["sel_med0"] == 0.56 and X[("dev", 16)]["r"]["sel_med"] == 0.85 and X[("ov", 16)]["jr"] == 0.005 and X[("ov", 16)]["ld"] == 2
     and X[("F", 16)] == {"pre": 100, "post": -5500, "ld": 2, "rew": 337} and X[("D", 16)] == {"pre": 200, "post": -2300, "ld": 0, "rew": 300}
     and X[("rate", 16)] == 2 and X[("sp", 16)] == (58500, 49500) and X[("st", 16)]["n"] == 500 and ("dev", 17) not in X)
ok_all &= g
print("%-36s → %s" % ("원 로그 파싱", "✓" if g else "✗ %s" % {k: v for k, v in X.items() if k[1] == 16}))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
