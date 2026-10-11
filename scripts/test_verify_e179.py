#!/usr/bin/env python3
"""verify_e179_independent.py 합성 시험(test_verify_e178 을 옮겨 대칭 맥락 A·B 로): 획득·요소식·부분 B만·차 0.10 → 보류,
조작 실패(B 집단 A 자리 발화·A 집단 A 자리 침묵·A 집단 B 자리 발화·맥락 입력 2 줄 없음·맥락 B 1,700·반사·적재·규칙 불일치·B 표지 없음·A 표지 없음·B 평가에 A 표지·
동작점 학습 값·평가 값·평가 줄 없음·A 평가 포화·진단 한쪽·A 결정 0.51·B 결정 0.51), 경계 통과(포화 하나 0.6599·결정 0.50), 결측.
실행: python3 scripts/test_verify_e179.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e179_independent as V  # noqa: E402

LDL = "[E153 종류 입력 적재] k 검증 일치 — x\n"
TL = "  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)\n"


def rows(bad_rule=False, small_A=0.10, small_B=0.10):
    n = 3000
    R = np.zeros((n, 38)); R[:, 2] = np.arange(n) % 2
    ctx = (np.arange(n) // 2) % 2
    R[:, 37] = ctx
    R[:, 6] = np.where(ctx == 1, R[:, 2], 1 - R[:, 2]); R[:, 7] = 1
    if bad_rule:
        R[7, 7] = 0
    R[:, 13] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20
    R[:, 3] = 0.0
    R[:, 5] = 0.2
    for c, fr in ((0, small_A), (1, small_B)):
        idx = np.where(ctx == c)[0]
        R[idx[:int(round(fr * len(idx)))], 5] = 0.01
    return R


def diag(r4, sides=("left", "right")):
    d = {"left": "[E171 평가 진단] variant=base side=left n=250 motor L/R %.6f/%.6f KC L/R 0.006926/0.003237\n" % (r4[0], r4[1]),
         "right": "[E171 평가 진단] variant=base side=right n=250 motor L/R %.6f/%.6f KC L/R 0.003227/0.007001\n" % (r4[2], r4[3])}
    return "".join(d[s] for s in sides)


def run(eA=-3000, eB=3000, kB_off=0, kA_off=2400, kA_on=0, c2line=True, nB=1500, refl="0.0000→0.0000", nld=2, bad_rule=False,
        noB=False, noA=False, B_hasA=False, tonic_train=-4000.0, tonic_eval=-4000.0, notonic_eval=False,
        A_rates=(0.52, 0.48, 0.47, 0.51), diag_one=False, small_A=0.10, small_B=0.10, miss=False):
    f = lambda x: "%+.4f" % (x / 1e4)
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E179"), ("traces", "E179")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            w(("logs", "E179", "kcctx_b%d.log" % b),
              "x\n=> KCCTX side=l off=60 on=80 ctx=3 | n_pres=40 | ctx_n=200 w=4.00 p=0.100 level=0.90 맥락 집단 발화 끔 %d 켬 2400"
              " | 대칭(E179 — 끔 자리 = 맥락 A) ctx2_n=200 w2=4.00 p2=0.100 맥락2 집단 발화 끔 %d 켬 %d\n"
              % (kB_off if b == 12 else 0, kA_off if b == 12 else 2400, kA_on if b == 12 else 0))
            if miss and b == 14:
                continue
            w(("logs", "E179", "train_b%d.log" % b), TL % (tonic_train if b == 13 else -4000.0)
              + ("  [맥락 입력 2] 맥락 전용 집단 zz_ctx2 200 → KC 좌·우\n" if (c2line or b != 10) else "") + LDL * nld
              + "[맥락 과제] bicond2 frac=0.50\n[맥락 과제] 시행 3000 중 맥락 켬 %d\n" % (nB if b == 11 else 1500)
              + "[반사가중치] good_food_to_motor_l   n=1 w_mean %s (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n" % (refl if b == 13 else "0.0000→0.0000"))
            np.savez_compressed(os.path.join(td, "traces", "E179", "tr_bc_b%d.npz" % b),
                                rows=rows(bad_rule and b == 10, small_A if b == 12 else 0.10, small_B if b == 12 else 0.10))
            vals = {("none", "A"): 150, ("none", "B"): 50, ("learn", "A"): 150 + eA, ("learn", "B"): 50 + eB}
            for (ww, c), m in vals.items():
                if c == "B":
                    tag = ("" if (noB and b == 10) else "[맥락 평가] ctx=켬 x\n") + ("[맥락 평가 A] ctx2=켬\n" if (B_hasA and b == 10) else "")
                else:
                    tag = "" if (noA and b == 11) else "[맥락 평가 A] ctx2=켬\n"
                tl = "" if (notonic_eval and b == 14 and ww == "none" and c == "A") else TL % (tonic_eval if (b == 11 and ww == "learn" and c == "B") else -4000.0)
                r4 = A_rates if (c == "A" and b == 10) else ((0.50, 0.53, 0.54, 0.49) if c == "B" else (0.52, 0.48, 0.47, 0.51))
                dg = diag(r4, ("left",) if (diag_one and b == 12 and ww == "learn" and c == "A") else ("left", "right"))
                w(("logs", "E179", "ev_b%d_%s_%s.log" % (b, ww, c)), tl + LDL + tag + dg + "=> DECOMP mode=all mod=%s acc=0.0\n" % f(m))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("획득", {}, "획득(H100)"), ("요소식", {"eA": 2000, "eB": 2500}, "요소식(H100-null)"),
                       ("부분 B만", {"eA": -500, "eB": 3000}, "부분 B만"), ("차 0.10 정확 → 보류", {"eA": -500, "eB": 500}, "보류"),
                       ("B 집단 A 자리 발화", {"kB_off": 3}, "보류(조작검증 실패)"), ("A 집단 A 자리 침묵", {"kA_off": 0}, "보류(조작검증 실패)"),
                       ("A 집단 B 자리 발화", {"kA_on": 4}, "보류(조작검증 실패)"), ("맥락 입력 2 줄 없음", {"c2line": False}, "보류(조작검증 실패)"),
                       ("맥락 B 1,700", {"nB": 1700}, "보류(조작검증 실패)"), ("반사 변함", {"refl": "0.0000→1.0000"}, "보류(조작검증 실패)"),
                       ("적재 1줄", {"nld": 1}, "보류(조작검증 실패)"), ("규칙 불일치 1시행", {"bad_rule": True}, "보류(조작검증 실패)"),
                       ("B 표지 없음", {"noB": True}, "보류(조작검증 실패)"), ("A 표지 없음", {"noA": True}, "보류(조작검증 실패)"),
                       ("B 평가에 A 표지", {"B_hasA": True}, "보류(조작검증 실패)"),
                       ("동작점 학습 값 −1000", {"tonic_train": -1000.0}, "보류(조작검증 실패)"),
                       ("동작점 평가 값 −2000", {"tonic_eval": -2000.0}, "보류(조작검증 실패)"),
                       ("동작점 평가 줄 없음", {"notonic_eval": True}, "보류(조작검증 실패)"),
                       ("A 평가 포화", {"A_rates": (0.6664, 0.6662, 0.6663, 0.6600)}, "보류(조작검증 실패)"),
                       ("A 평가 하나 0.6599 → 통과", {"A_rates": (0.6664, 0.6662, 0.6663, 0.6599)}, "획득(H100)"),
                       ("진단 한쪽만", {"diag_one": True}, "보류(조작검증 실패)"),
                       ("A 결정 작음 0.51", {"small_A": 0.51}, "보류(조작검증 실패)"),
                       ("B 결정 작음 0.51", {"small_B": 0.51}, "보류(조작검증 실패)"),
                       ("결정 작음 0.50 → 통과", {"small_A": 0.50, "small_B": 0.50}, "획득(H100)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out and (want != "보류" or out.strip().endswith("독립 판정: 보류"))
    ok_all &= good
    print("%-26s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
