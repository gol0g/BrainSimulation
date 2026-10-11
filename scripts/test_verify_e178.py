#!/usr/bin/env python3
"""verify_e178_independent.py 합성 시험(test_verify_e177 을 옮김): 획득·요소식·부분·보류, E177 조작 실패 7종 + 동작점 값 다름(학습·평가)·동작점 줄 없음·
켬 평가 포화(4개 모두 ≥ 0.66) / 하나 0.6599 통과·진단 줄 한쪽만·켬 결정 작음 비율 0.51 실패 / 0.50 통과, pick 'X=none'(결측), 로그 결측.
실행: python3 scripts/test_verify_e178.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e178_independent as V  # noqa: E402

LDL = "[E153 종류 입력 적재] k 검증 일치 — x\n"
TL = "  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)\n"


def rows(bad_rule=False, small_frac=0.10):
    n = 3000
    R = np.zeros((n, 38)); R[:, 2] = np.arange(n) % 2
    ctx = (np.arange(n) // 2) % 2
    R[:, 37] = ctx
    R[:, 6] = np.where(ctx == 1, R[:, 2], 1 - R[:, 2]); R[:, 7] = 1
    if bad_rule:
        R[7, 7] = 0
    R[:, 13] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20
    R[:, 3] = 0.0                                   # 모두 탐욕
    R[:, 5] = 0.2
    on_idx = np.where(ctx == 1)[0]
    k = int(round(small_frac * len(on_idx)))
    R[on_idx[:k], 5] = 0.01                         # 켬 탐욕 시행 중 small_frac 비율이 |v| ≤ 0.02
    return R


def diag(r4, sides=("left", "right")):
    d = {"left": "[E171 평가 진단] variant=base side=left n=250 motor L/R %.6f/%.6f KC L/R 0.006926/0.003237\n" % (r4[0], r4[1]),
         "right": "[E171 평가 진단] variant=base side=right n=250 motor L/R %.6f/%.6f KC L/R 0.003227/0.007001\n" % (r4[2], r4[3])}
    return "".join(d[s] for s in sides)


def run(eoff=-3000, eon=3000, sp_off=0, non=1500, refl="0.0000→0.0000", nld=2, bad_rule=False, on_line=True, off_line=False, miss=False,
        pick="X=-1000", tonic_train=-1000.0, tonic_eval=-1000.0, notonic_eval=False, on_rates=(0.55, 0.58, 0.58, 0.54), diag_one=False, small_frac=0.10):
    f = lambda x: "%+.4f" % (x / 1e4)
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E178"), ("traces", "E178")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        w(("logs", "E178", "pick.txt"), pick + "\n")
        for b in V.BRAINS:
            w(("logs", "E178", "kcctx_b%d.log" % b), "x\n=> KCCTX side=l off=60 on=80 ctx=3 | ctx_i=0.00 | n_pres=40 | ctx_n=200 w=4.00 p=0.100 level=0.90 맥락 집단 발화 끔 %d 켬 2400\n" % (sp_off if b == 12 else 0))
            if miss and b == 14:
                continue
            w(("logs", "E178", "train_b%d.log" % b), TL % (tonic_train if b == 13 else -1000.0) + LDL * nld + "[맥락 과제] bicond frac=0.50\n[맥락 과제] 시행 3000 중 맥락 켬 %d\n" % (non if b == 11 else 1500)
              + "[반사가중치] good_food_to_motor_l   n=1 w_mean %s (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n" % (refl if b == 13 else "0.0000→0.0000"))
            np.savez_compressed(os.path.join(td, "traces", "E178", "tr_bc_b%d.npz" % b), rows=rows(bad_rule and b == 10, small_frac if b == 12 else 0.10))
            vals = {("none", "off"): 150, ("none", "on"): 50, ("learn", "off"): 150 + eoff, ("learn", "on"): 50 + eon}
            for (ww, c), m in vals.items():
                tag = ("[맥락 평가] ctx=켬\n" if (c == "on" and (on_line or b != 10)) or (c == "off" and off_line and b == 10) else "")
                tl = "" if (notonic_eval and b == 14 and ww == "none" and c == "off") else TL % (tonic_eval if (b == 11 and ww == "learn" and c == "on") else -1000.0)
                r4 = on_rates if (c == "on" and b == 10) else ((0.55, 0.58, 0.58, 0.54) if c == "on" else (0.40, 0.48, 0.48, 0.39))
                dg = diag(r4, ("left",) if (diag_one and b == 12 and ww == "learn" and c == "off") else ("left", "right"))
                w(("logs", "E178", "ev_b%d_%s_%s.log" % (b, ww, c)), tl + LDL + tag + dg + "=> DECOMP mode=all mod=%s acc=0.0\n" % f(m))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("획득", {}, "획득(H099)"), ("요소식", {"eoff": -2000, "eon": -2000}, "요소식(H099-null)"),
                       ("부분 켬만", {"eoff": -500, "eon": 3000}, "부분(H099-partial)"), ("차 0.10 정확 → 보류", {"eoff": -500, "eon": 500}, "보류"),
                       ("끔 발화 > 0", {"sp_off": 3}, "보류(조작검증 실패)"), ("맥락 켬 1,700", {"non": 1700}, "보류(조작검증 실패)"),
                       ("반사 변함", {"refl": "0.0000→1.0000"}, "보류(조작검증 실패)"), ("적재 1줄", {"nld": 1}, "보류(조작검증 실패)"),
                       ("규칙 불일치 1시행", {"bad_rule": True}, "보류(조작검증 실패)"), ("켬 평가 줄 없음", {"on_line": False}, "보류(조작검증 실패)"),
                       ("끔 평가에 맥락 줄", {"off_line": True}, "보류(조작검증 실패)"),
                       ("동작점 학습 값 −500", {"tonic_train": -500.0}, "보류(조작검증 실패)"),
                       ("동작점 평가 값 −2000", {"tonic_eval": -2000.0}, "보류(조작검증 실패)"),
                       ("동작점 평가 줄 없음", {"notonic_eval": True}, "보류(조작검증 실패)"),
                       ("켬 평가 포화(4개 ≥ 0.66)", {"on_rates": (0.6664, 0.6662, 0.6663, 0.6600)}, "보류(조작검증 실패)"),
                       ("켬 평가 하나 0.6599 → 통과", {"on_rates": (0.6664, 0.6662, 0.6663, 0.6599)}, "획득(H099)"),
                       ("진단 줄 한쪽만", {"diag_one": True}, "보류(조작검증 실패)"),
                       ("켬 결정 작음 0.51", {"small_frac": 0.51}, "보류(조작검증 실패)"),
                       ("켬 결정 작음 0.50 → 통과", {"small_frac": 0.50}, "획득(H099)"),
                       ("pick X=none → 결측", {"pick": "X=none"}, "보류(결측"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out and (want != "보류" or out.strip().endswith("독립 판정: 보류"))
    ok_all &= good
    print("%-26s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
