#!/usr/bin/env python3
"""judge_e165.py 합성 시험: 견고·빈도 의존(NU 실패·분리만)·순서·강도 의존(NR 실패·분리만)·보류, 경계(자카드 0.10·0.25, r 1.30 정확), 4/5,
조작검증(MX 각 항목·MO·MS'·ML·M1·M1b·M3·MD), 결측, 원 로그 파싱(실제 줄 형식).
실행: python3 scripts/test_judge_e165.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e165 as J


def build(jac=None, r=None, over=None, drop=None):
    """jac[(팔, 뇌)] = (좌, 우) 자카드, r[(팔, 뇌)] = e_F / e_D. 기본: 자카드 0, r 2.0, e_D −0.25."""
    X = {}
    for b in J.BRAINS:
        X[("D", b)] = {"pre": 200, "post": 200 - 2500, "ld": 0, "rew": 320}
        for a, (mult, cnt) in J.ARMS.items():
            jl, jr = (jac or {}).get((a, b), (0.0, 0.0))
            rr = (r or {}).get((a, b), 2.0)
            X[("dev", a, b)] = {"l": {"fired": 230, "sel_med0": 0.56, "sel_med": 0.85, "goodfrac": 0.48, "sum_med": 1.0, "dg_good": 2e4, "dg_bad": 2e4},
                                "r": {"fired": 230, "sel_med0": 0.55, "sel_med": 0.86, "goodfrac": 0.48, "sum_med": 1.0, "dg_good": 2e4, "dg_bad": 2e4}, "oja": 2}
            X[("x", a, b)] = {"order": "random", "mult": mult, "cnt": cnt, "n": sum(cnt), "imin": 5010, "imax": 8990, "imean": 6980, "cyc": 2900}
            X[("ov", a, b)] = {"jl": J.i4(jl), "jr": J.i4(jr), "n": (55, 60, 56, 61), "ld": 2}
            X[("F", a, b)] = {"pre": 150, "post": 150 + int(round(-2500 * rr)), "ld": 2, "rew": 340}
            X[("st", a, b)] = {"n": 500, "pre_ratio": 0.0, "res": 1e-8, "alive": 1.0}
    for k, f in (over or {}).items():
        f(X[k]) if callable(f) else X.__setitem__(k, f)
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, want):
    global ok_all
    c, res = J.judge(X)
    got = res["verdict"] if res else c[0]
    good = (got.startswith(want) and (want != "보류" or got == "보류")) if want != "결측" else (res is None and "결측" in got)
    ok_all &= good
    print("%-40s 기대 %-22s → %-30s %s" % (name, want[:22], got[:30], "✓" if good else "✗ %s" % c))


allb = lambda a, v: {(a, b): v for b in J.BRAINS}
chk("견고(두 팔 성공)", build(), "견고(H088)")
chk("빈도 의존(NU 자카드 0.30 → 실패)", build(jac=allb("NU", (0.30, 0.30))), "빈도 의존(H088-freq)")
chk("빈도 의존(NU 분리만 r 1.2)", build(r=allb("NU", 1.2)), "빈도 의존(H088-freq)")
chk("빈도 의존(NU 보류 — 자카드 0.15)", build(jac=allb("NU", (0.15, 0.15))), "빈도 의존(H088-freq)")
chk("순서·강도 의존(NR 실패)", build(jac=allb("NR", (0.30, 0.0))), "순서·강도 의존(H088-null)")
chk("순서·강도 의존(NR 분리만 r 1.0)", build(r=allb("NR", 1.0)), "순서·강도 의존(H088-null)")
chk("NR 보류(자카드 0.15) → 보류", build(jac=allb("NR", (0.15, 0.15))), "보류")
chk("자카드 경계 0.10 정확 → 분리", build(jac=allb("NR", (0.10, 0.10))), "견고(H088)")
chk("자카드 0.1001 → 분리 아님 → NR 보류", build(jac=allb("NR", (0.1001, 0.0))), "보류")
chk("r 경계 1.30 정확 → 성공", build(r=allb("NR", 1.30)), "견고(H088)")
chk("r 1.2996 → NR 분리만", build(r=allb("NR", 1.2996)), "순서·강도 의존(H088-null)")
chk("자카드 0.25 정확 → 실패 아님(보류)", build(jac=allb("NR", (0.25, 0.25))), "보류")
chk("자카드 0.2501 → NR 실패", build(jac=allb("NR", (0.2501, 0.0))), "순서·강도 의존(H088-null)")
chk("4/5 성공(한 뇌 자카드 0.5)", build(jac={("NR", 18): (0.5, 0.5), ("NU", 20): (0.5, 0.0)}), "견고(H088)")
chk("3/5 성공 + 2 실패(NU) → 빈도 의존", build(jac={("NU", 16): (0.5, 0.5), ("NU", 17): (0.5, 0.5)}), "빈도 의존(H088-freq)")
F = "보류(조작검증 실패)"
chk("MX 순서 cyclic", build(over={("x", "NR", 16): lambda d: d.update(order="cyclic")}), F)
chk("MX NU bad_l 299", build(over={("x", "NU", 17): lambda d: d.update(cnt=(100, 299, 100, 300))}), F)
chk("MX 배수 1(NU)", build(over={("x", "NU", 18): lambda d: d.update(mult=1)}), F)
chk("MX 강도 최소 0.4999", build(over={("x", "NR", 19): lambda d: d.update(imin=4999)}), F)
chk("MX 강도 최대 0.9001", build(over={("x", "NR", 19): lambda d: d.update(imax=9001)}), F)
chk("MX 강도 평균 0.7301", build(over={("x", "NU", 20): lambda d: d.update(imean=7301)}), F)
chk("MX 순환 일치 0.4001", build(over={("x", "NR", 16): lambda d: d.update(cyc=4001)}), F)
chk("MO dg_bad 0", build(over={("dev", "NU", 16): lambda d: d["l"].update(dg_bad=0.0)}), F)
chk("MO Oja 줄 0", build(over={("dev", "NR", 17): lambda d: d.update(oja=0)}), F)
chk("MS' 상승 0.0999", build(over={("dev", "NR", 18): lambda d: d["r"].update(sel_med=0.6499, sel_med0=0.55)}), F)
chk("ML F 적재 1", build(over={("F", "NU", 19): lambda d: d.update(ld=1)}), F)
chk("ML ov 적재 0", build(over={("ov", "NR", 20): lambda d: d.update(ld=0)}), F)
chk("M1 동결 잔차 0.002", build(over={("st", "NR", 16): lambda d: d.update(res=0.002)}), F)
chk("M1b 흔적 생존 0.8", build(over={("st", "NU", 16): lambda d: d.update(alive=0.8)}), F)
chk("M3 추적 499", build(over={("st", "NU", 17): lambda d: d.update(n=499)}), F)
chk("M3 도파민 전 0.002", build(over={("st", "NU", 17): lambda d: d.update(pre_ratio=0.002)}), F)
chk("MD e_D −0.0999", build(over={("D", 18): lambda d: d.update(post=200 - 999)}), F)
chk("결측(NU F 뇌 19)", build(drop=("F", "NU", 19)), "결측")
chk("결측(E161 D 뇌 16)", build(drop=("D", 16)), "결측")

# 원 로그 파싱(실제 줄 형식)
with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E165"), ("logs", "E161"), ("traces", "E165")):
        os.makedirs(os.path.join(td, *d))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    LD = "[E153 종류 입력 적재] /x/kctype.npz 검증 일치 — good_food_eye_l_to_kc_l=1.0\n"
    w(("logs", "E161", "D_b16.log"), "[사전] 오프셋 -0.004 | 정답률 22.0% | **변조폭 +0.0283** (양수=반사방향, 음수=역전)\n[학습] 5ep 완료, 보상 321회 (탐색 주입 296회, ε=0.60)\n"
      "[사후] 오프셋 -0.013 | 정답률 100.0% | **변조폭 -0.2425**\n")
    w(("logs", "E165", "NU_dev_b16.log"), "[E160 종류 입력 Oja] 4집단 망 안 가소성 — tau_pre 20.0\n"
      "[E165 노출] order=random bad_mult=3 good_l=100 bad_l=300 good_r=100 bad_r=300 n=800 int_min=0.5000 int_max=0.8999 int_mean=0.6979 cyc_match=0.1765 seed=16\n"
      "=> KCDEVOJA side=l fired=251 sel_med0=0.5671 sel_med=0.8271 frac09=0.4741 goodfrac=0.4781 sum_med=0.9589 sum_q10=0.0390 sum_q90=1.7942 dg_good=21597.6 dg_bad=22500.0 | "
      "side=r fired=237 sel_med0=0.5525 sel_med=0.7492 frac09=0.4346 goodfrac=0.4852 sum_med=0.9305 sum_q10=0.0407 sum_q90=1.7050 dg_good=21325.3 dg_bad=22573.5 | "
      "n=100 eta=0.02 beta=0.3 mmax=32 tau=20 save=/x.npz\n")
    w(("logs", "E165", "NU_ov_b16.log"), LD + "=> KCOVERLAP side=l good=54 bad=62 jac=0.0000 cos=0.0002 jac025=0.0000 jac100=0.0000 split_jac=0.9818 split_cos=0.9998 | "
      "side=r good=55 bad=63 jac=0.0172 cos=0.0160 jac025=0.0169 jac100=0.0187 split_jac=1.0000 split_cos=0.9999 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50\n")
    w(("logs", "E165", "NU_F_b16.log"), LD + "[사전] 오프셋 -0.004 | 정답률 22.0% | **변조폭 +0.0149** (양수=반사방향, 음수=역전)\n"
      "[학습] 5ep 완료, 보상 341회 (탐색 주입 296회, ε=0.60)\n" + LD + "[사후] 오프셋 -0.013 | 정답률 100.0% | **변조폭 -0.5977**\n")
    R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 21] = J.R_STAR; R[:, 14] = 0.5; R[:, 22] = J.R_STAR * 0.5   # 시험 구성 수정(첫 판: 22열 누락)
    np.savez_compressed(os.path.join(td, "traces", "E165", "tr_NU_F_b16.npz"), rows=R)
    J.EXP = td
    X = J.load()
g = (X[("D", 16)] == {"pre": 283, "post": -2425, "ld": 0, "rew": 321}
     and X[("x", "NU", 16)] == {"order": "random", "mult": 3, "cnt": (100, 300, 100, 300), "n": 800, "imin": 5000, "imax": 8999, "imean": 6979, "cyc": 1765}
     and J.mx_ok(X[("x", "NU", 16)], "NU") and not J.mx_ok(X[("x", "NU", 16)], "NR")
     and X[("dev", "NU", 16)]["oja"] == 1 and X[("dev", "NU", 16)]["l"]["sel_med"] == 0.8271 and X[("dev", "NU", 16)]["r"]["dg_bad"] == 22573.5
     and X[("ov", "NU", 16)] == {"jl": 0, "jr": 172, "n": (54, 62, 55, 63), "ld": 1}
     and X[("F", "NU", 16)] == {"pre": 149, "post": -5977, "ld": 2, "rew": 341}
     and X[("st", "NU", 16)]["n"] == 500 and X[("st", "NU", 16)]["res"] < 1e-9 and X[("st", "NU", 16)]["alive"] == 1.0
     and ("dev", "NR", 16) not in X)
ok_all &= g
print("%-40s → %s" % ("원 로그 파싱(실제 줄 형식)", "✓" if g else "✗ %s" % {k: v for k, v in X.items()}))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
