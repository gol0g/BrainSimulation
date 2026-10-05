#!/usr/bin/env python3
"""judge_e142.py 합성 시험(조건 1): L2·부분·효과 없음·보류, 경계(m 정확히 −0.02, e 정확히 −0.10), 조작검증 실패 5종, 결측,
[반사가중치] 줄 파싱, 요약 줄 파싱, stats() 를 답을 아는 합성 행으로.
실행: python3 scripts/test_judge_e142.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e142 as J


def S_(n=500, res=1e-8, alive=1.0, pre=0.0):
    return {"n": n, "res": res, "alive": alive, "pre_ratio": pre, "BA": -1.1, "CP": -0.7, "dD": 2e6, "blk": [5e5, 3e5]}


def build(e500, post1500, Skw=None, pre_shift=None, rf_bad=None, nf_post=None):
    T = {"F500": {}, "F1500": {}, "NF1500": {}}; S = {"F500": {}, "F1500": {}, "NF1500": {}}; RF = {"F500": {}, "F1500": {}, "NF1500": {}}
    for i, b in enumerate(J.BRAINS):
        pre = round(J.E119_PRE[b] + (pre_shift.get(b, 0.0) if pre_shift else 0.0), 4)
        T["F500"][b] = {"pre": pre, "post": round(pre + e500[i], 4), "rew": 140}
        T["F1500"][b] = {"pre": pre, "post": round(post1500[i], 4), "rew": 200}
        T["NF1500"][b] = {"pre": pre, "post": round((nf_post or [0.40] * 5)[i], 4), "rew": 140}
        for a, n in (("F500", 500), ("F1500", 1500), ("NF1500", 1500)):
            kw = {"n": n, "res": (3.0 if a == "NF1500" else 1e-8)}
            if Skw:
                kw.update(Skw(a, b))
            S[a][b] = S_(**kw)
            RF[a][b] = not (rf_bad and (a, b) in rf_bad)
    return T, S, RF


cases = [
    ("L2 달성", build([-0.2] * 5, [-0.05, -0.03, -0.10, -0.02, +0.05]), "L2 달성(H065)"),
    ("L2 경계 m=−0.02 4/5", build([-0.2] * 5, [-0.02, -0.02, -0.02, -0.02, +0.10]), "L2 달성(H065)"),
    ("L2 경계 밖 m=−0.0199", build([-0.2] * 5, [-0.0199, -0.0199, -0.0199, -0.0199, +0.10]), "반사를 거스름(H065-partial)"),
    ("부분", build([-0.15] * 5, [+0.10] * 5), "반사를 거스름(H065-partial)"),
    ("부분 경계 e=−0.10", build([-0.10] * 5, [+0.10] * 5), "반사를 거스름(H065-partial)"),
    ("부분 4/5 → 보류", build([-0.15, -0.15, -0.15, -0.15, -0.09], [+0.10] * 5), "보류"),
    ("효과 없음", build([-0.02, 0.01, 0.0, 0.029, -0.01], [+0.40] * 5), "효과 없음(H065-null)"),
    ("보류(혼재)", build([-0.12, -0.12, -0.05, -0.05, -0.05], [+0.20] * 5), "보류"),
    ("M1 동결 실패", build([-0.2] * 5, [-0.05] * 5, Skw=lambda a, b: {"res": 0.1} if (a, b) == ("F1500", 12) else {}), "보류(조작검증 실패)"),
    ("M1b 되돌림 실패", build([-0.2] * 5, [-0.05] * 5, Skw=lambda a, b: {"alive": 0.5} if (a, b) == ("F500", 10) else {}), "보류(조작검증 실패)"),
    ("M2 출발점 다름", build([-0.2] * 5, [-0.05] * 5, pre_shift={13: 0.003}), "보류(조작검증 실패)"),
    ("M3 시행 수 틀림", build([-0.2] * 5, [-0.05] * 5, Skw=lambda a, b: {"n": 500} if (a, b) == ("F1500", 14) else {}), "보류(조작검증 실패)"),
    ("M4 반사 변함", build([-0.2] * 5, [-0.05] * 5, rf_bad={("F1500", 11)}), "보류(조작검증 실패)"),
]
ok_all = True
for name, (T, S, RF), want in cases:
    c, r = J.judge(T, S, RF)
    good = r is not None and r["verdict"].startswith(want) and (want != "보류" or r["verdict"] == "보류")
    ok_all &= good
    print("%-22s 기대 %-24s → %s %s" % (name, want, r["verdict"][:24] if r else None, "✓" if good else "✗"))
for name, kw, want in (
        ("필요성: 동결 필요", {"nf_post": [0.30, 0.25, 0.20, 0.35, -0.05]}, "동결이 L2 에 필요"),
        ("필요성: 불필요", {"nf_post": [-0.05, -0.03, -0.02, -0.04, 0.10]}, "학습량만으로도 L2"),
        ("필요성: 미결", {"nf_post": [-0.05, -0.03, 0.20, 0.25, 0.10]}, "필요성 미결"),
        ("필요성: NF 검사 실패", {"nf_post": [0.30] * 5, "Skw": lambda a, b: {"res": 1e-8} if (a, b) == ("NF1500", 10) else {}}, "필요성 미결(NF1500 조작검증 실패)"),
        ("필요성: L2 아님", {"nf_post": [0.30] * 5}, "해당 없음")):
    post = [+0.10] * 5 if want == "해당 없음" else [-0.05] * 5
    T, S, RF = build([-0.2] * 5, post, **kw)
    c, r = J.judge(T, S, RF)
    good = r is not None and r["need"].startswith(want) and (r["verdict"].startswith("L2") or want == "해당 없음")
    ok_all &= good
    print("%-22s 기대 %-24s → %s %s" % (name, want, r["need"][:28] if r else None, "✓" if good else "✗"))
T, S, RF = build([-0.2] * 5, [-0.05] * 5); del S["F1500"][13]
c, r = J.judge(T, S, RF)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-22s 기대 결측 → %s" % ("결측(추적)", "✓" if good else "✗"))
T, S, RF = build([-0.2] * 5, [-0.05] * 5); RF["F500"][10] = None
c, r = J.judge(T, S, RF)
good = r is None and "결측" in c[0]; ok_all &= good
print("%-22s 기대 결측 → %s" % ("결측(반사 줄 없음)", "✓" if good else "✗"))

# stats: 1,500행, 동결 흔적(잔차 0), 결정 흔적 살아 있음 100%, 블록 15개 각 100×(3−(−2)) 보상/100×(1−(−1)) 처벌
rng = np.random.default_rng(0)
R = np.zeros((1500, 27)); R[:, 7] = (np.arange(1500) % 2 == 0)
R[:, 8] = np.where(R[:, 7] == 1, 3.0, 1.0); R[:, 9] = np.where(R[:, 7] == 1, -2.0, -1.0); R[:, 12] = R[:, 8:12].sum(1)
R[:, 13:17] = rng.normal(0, 1000, (1500, 4)); R[:, 21:25] = J.R_STAR * R[:, 13:17]
s = J.stats(R)
good = (s["n"] == 1500 and s["res"] < 1e-12 and s["alive"] == 1.0 and len(s["blk"]) == 15 and abs(s["blk"][0] - 350.0) < 1e-9
        and abs(s["BA"] + 2 / 3) < 1e-12 and abs(s["CP"] + 1.0) < 1e-12); ok_all &= good
print("%-22s 기대 n 1500·잔차 0·살아있음 1·블록 15(첫 350) → n %d res %.1e alive %.2f 블록 %d(첫 %.0f) %s"
      % ("stats", s["n"], s["res"], s["alive"], len(s["blk"]), s["blk"][0], "✓" if good else "✗"))
R2 = R.copy(); R2[750:, 13:17] = 0.0; R2[750:, 21:25] = 0.0
s2 = J.stats(R2)
good = abs(s2["alive"] - 0.5) < 1e-12; ok_all &= good
print("%-22s 기대 0.5 → %.2f %s" % ("stats 되돌림 실패", s2["alive"], "✓" if good else "✗"))

with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E142")); os.makedirs(os.path.join(td, "traces", "E142"))
    open(os.path.join(td, "E142.log"), "w", encoding="utf-8").write(
        "  e142 F500 b10: => 사전 +0.4148 사후 +0.2000 보상 140 || ...\n  e142 F1500 b10: => 사전 +0.4148 사후 -0.0500 보상 300 || ...\n  e141 b11: => 사전 +0.0150 사후 -0.1 보상 1\n")
    open(os.path.join(td, "logs", "E142", "F500_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] food_explore_motor_l   n=30103 w_mean 10.0000→10.0000 (학습 뇌; 이식 대상 아님)\n"
        "[반사가중치] good_food_to_motor_l   n=15139 w_mean 25.0000→25.0000 (학습 뇌; 이식 대상 아님)\n"
        "[반사가중치] good_food_to_motor_r   n=14818 w_mean 25.0000→25.0000 (학습 뇌; 이식 대상 아님)\n")
    open(os.path.join(td, "logs", "E142", "F1500_b10.log"), "w", encoding="utf-8").write(
        "[반사가중치] good_food_to_motor_l   n=15139 w_mean 25.0000→24.9000 (학습 뇌; 이식 대상 아님)\n"
        "[반사가중치] good_food_to_motor_r   n=14818 w_mean 25.0000→25.0000 (학습 뇌; 이식 대상 아님)\n")
    J.EXP = td
    Tp, Sp, RFp = J.load()
good = (Tp["F500"] == {10: {"pre": 0.4148, "post": 0.2, "rew": 140}} and Tp["F1500"] == {10: {"pre": 0.4148, "post": -0.05, "rew": 300}}
        and RFp["F500"][10] is True and RFp["F1500"][10] is False and RFp["F500"][11] is None); ok_all &= good
print("%-22s 기대 두 팔 파싱·반사 줄(참/거짓/없음) → %s" % ("줄 파싱", "✓" if good else "✗ %s %s" % (Tp, RFp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
