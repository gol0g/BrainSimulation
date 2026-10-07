#!/usr/bin/env python3
"""judge_e153.py 합성 시험(조건 1): 형성 성공·분리 실패·권한 상실·보류, 경계(J 0.25·효과 2/3 정확), 조작검증 K1·K2·K3 실패, 결측, 줄 파싱·추적 통계.
실행: python3 scripts/test_judge_e153.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e153 as J


def build(J_=1500, ef=1.0, sel=9500, relerr=1e-15, load_ov=1, load_tr=3, res=1e-8, n=500, agree=1.0, refl=True, per=None):
    """ef = 효과/기본 비(1.0 = 기본과 같음). per = {b: {...}} 뇌별 덮어쓰기."""
    DEV, OV, TR, S, RF = {}, {}, {}, {}, {}
    for b in J.BRAINS:
        p = dict(J_=J_, ef=ef, sel=sel, relerr=relerr, load_ov=load_ov, load_tr=load_tr, res=res, n=n, agree=agree, refl=refl)
        p.update((per or {}).get(b, {}))
        DEV[b] = {k: {"fired": 100, "sel_med0": 5500, "sel_med": p["sel"], "frac09": "0.9", "goodfrac": "0.5", "relerr": p["relerr"]} for k in "lr"}
        OV[b] = {"Gl": 90, "Bl": 90, "Jl": p["J_"], "Gr": 90, "Br": 90, "Jr": p["J_"], "load": p["load_ov"]}
        e = int(round(J.E141[b] * p["ef"]))
        TR[b] = {"pre": "+0.0200", "post": "%+.4f" % ((200 + e) / 1e4), "rew": 300, "load": p["load_tr"]}
        S[b] = {"n": p["n"], "agree": p["agree"], "res": p["res"], "pre_ratio": 0.0}
        RF[b] = p["refl"]
    return DEV, OV, TR, S, RF


ok_all = True
cases = [
    ("형성 성공", build(), "형성 성공(H076)"),
    ("분리 실패(J 0.40)", build(J_=4000), "분리 실패(H076-null)"),
    ("권한 상실(효과 0.5×)", build(ef=0.5), "권한 상실(H076-auth)"),
    ("경계 J 0.25 정확", build(J_=2500), "형성 성공(H076)"),
    ("J 0.2501 → 분리 아님", build(J_=2501), "분리 실패(H076-null)"),
    ("경계 효과 2/3 정확(b11 3·e = 2·e141)", build(per={b: {"ef": 2 / 3} for b in J.BRAINS}), None),
    ("섞임 3·2 → 보류", build(per={13: {"J_": 4000}, 14: {"J_": 4000}}), "보류"),
    ("K1 형성 안 됨(sel 0.79)", build(per={12: {"sel": 7999}}), "보류(조작검증 실패)"),
    ("K1 합 보존 실패", build(per={10: {"relerr": 2e-6}}), "보류(조작검증 실패)"),
    ("K2 학습 적재 1줄", build(per={11: {"load_tr": 1}}), "보류(조작검증 실패)"),
    ("K2 겹침 적재 0", build(per={11: {"load_ov": 0}}), "보류(조작검증 실패)"),
    ("K3 동결 실패", build(per={14: {"res": 0.01}}), "보류(조작검증 실패)"),
    ("K3 반사 변함", build(per={13: {"refl": False}}), "보류(조작검증 실패)"),
]
for name, data, want in cases:
    c, r = J.judge(*data)
    if want is None:   # 2/3 경계: 정수 반올림 때문에 뇌마다 3·e ≤ 2·e141 가 정확히 성립하는지 직접 확인
        exp_auth = sum(3 * int(round(J.E141[b] * 2 / 3)) <= 2 * J.E141[b] for b in J.BRAINS)
        good = r is not None and r["auth"] == exp_auth
        print("%-36s 권한 %d/5 (직접 계산 %d/5) %s" % (name, r["auth"], exp_auth, "✓" if good else "✗"))
    else:
        good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
        print("%-36s 기대 %-20s → %s %s" % (name, want, r["verdict"][:20] if r else c[0][:30], "✓" if good else "✗"))
    ok_all &= good
D = build(); del D[2][12]
c, r = J.judge(*D); g = r is None and "결측" in c[0]; ok_all &= g; print("%-36s → %s" % ("결측(학습 줄)", "✓" if g else "✗"))
# 추적 통계: 교차 규칙 일치·동결 잔차
R = np.zeros((500, 37)); rng = np.random.default_rng(3); R[:, 2] = rng.integers(0, 2, 500); R[:, 6] = rng.integers(0, 2, 500)
R[:, 7] = (R[:, 6] != R[:, 2]).astype(float); R[:, 13:17] = 1.0; R[:, 21:25] = J.R20; R[:5, 6] = -1
s = J.stats(R)
g = s["n"] == 500 and s["agree"] == 1.0 and s["res"] < 1e-12 and s["pre_ratio"] == 0.0; ok_all &= g
print("%-36s → %s %s" % ("stats(일치·동결)", s, "✓" if g else "✗"))
# 줄 파싱
LOG = ("  e153 b10 dev: => KCDEV side=l fired=120 sel_med0=0.5500 sel_med=0.9800 frac09=0.9100 goodfrac=0.5100 relerr=2.22e-16"
       " | side=r fired=118 sel_med0=0.5600 sel_med=0.9700 frac09=0.9000 goodfrac=0.4900 relerr=1.11e-16 | n=100 eta=0.1 updates=400 save=x\n"
       "  e153 b10 ov: => KCOVERLAP side=l good=95 bad=97 jac=0.1200 cos=0.3 jac025=0.1 jac100=0.1 split_jac=0.95 split_cos=0.99"
       " | side=r good=93 bad=99 jac=0.1300 cos=0.3 jac025=0.1 jac100=0.1 split_jac=0.95 split_cos=0.99 | food_eye_scale=1.00 bilateral_scale=1.00 n_pres=50 || 적재 1\n"
       "  e153 b10 train: => 사전 +0.0195 사후 -0.2189 보상 273 || 적재 3 || => KCTRACE 시행 500\n")
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E153")); os.makedirs(os.path.join(td, "traces", "E153"))
    open(os.path.join(td, "E153.log"), "w", encoding="utf-8").write(LOG)
    open(os.path.join(td, "logs", "E153", "train_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=1 w_mean 0.0000→0.0000 (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n")
    np.savez_compressed(os.path.join(td, "traces", "E153", "tr_b10.npz"), rows=R)
    J.EXP = td
    DEV, OV, TR, S, RF = J.load()
g = (DEV[10]["l"]["sel_med"] == 9800 and DEV[10]["r"]["relerr"] == 1.11e-16 and DEV[10]["r"]["sel_med0"] == 5600
     and OV[10] == {"Gl": 95, "Bl": 97, "Jl": 1200, "Gr": 93, "Br": 99, "Jr": 1300, "load": 1}
     and TR[10] == {"pre": "+0.0195", "post": "-0.2189", "rew": 273, "load": 3} and RF[10] is True and RF[11] is None and S[10]["n"] == 500)
ok_all &= g
print("%-36s → %s" % ("줄 파싱(dev·ov·train·반사·추적)", "✓" if g else "✗ %s %s %s" % (DEV, OV, TR)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
