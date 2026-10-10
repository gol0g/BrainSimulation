#!/usr/bin/env python3
"""verify_e176_independent.py 합성 시험(E176 시험을 옮김 — 조작검증 지표만 맥락 집단 발화): 획득·요소식·부분·보류, 조작 실패(끔 발화·맥락 켬 수·반사·적재·규칙 불일치·켬 평가 줄 없음·끔 평가에 맥락 줄), 결측.
실행: python3 scripts/test_verify_e176.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e176_independent as V

LDL = "[E153 종류 입력 적재] k 검증 일치 — x\n"


def rows(bad_rule=False):
    n = 3000
    R = np.zeros((n, 38)); R[:, 2] = np.arange(n) % 2
    ctx = (np.arange(n) // 2) % 2
    R[:, 37] = ctx
    R[:, 6] = np.where(ctx == 1, R[:, 2], 1 - R[:, 2]); R[:, 7] = 1
    if bad_rule:
        R[7, 7] = 0
    R[:, 13] = 1.0; R[:, 21] = (11.0 / 12.0) ** 20
    return R


def run(eoff=-3000, eon=3000, sp_off=0, non=1500, refl="0.0000→0.0000", nld=2, bad_rule=False, on_line=True, off_line=False, miss=False):
    f = lambda x: "%+.4f" % (x / 1e4)
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E176"), ("traces", "E176")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            w(("logs", "E176", "kcctx_b%d.log" % b), "x\n=> KCCTX side=l off=60 on=80 ctx=3 | ctx_i=0.00 | KC 발화 합 good 끔 500 켬 700 · 맥락 단독 40 · 기준선 300 | n_pres=40 | ctx_n=200 w=2.00 p=0.100 level=0.90 맥락 집단 발화 끔 %d 켬 2400\n" % (sp_off if b == 12 else 0))
            if miss and b == 14:
                continue
            w(("logs", "E176", "train_b%d.log" % b), LDL * nld + "[맥락 과제] bicond frac=0.50\n[맥락 과제] 시행 3000 중 맥락 켬 %d\n" % (non if b == 11 else 1500)
              + "[반사가중치] good_food_to_motor_l   n=1 w_mean %s (x)\n[반사가중치] good_food_to_motor_r   n=1 w_mean 0.0000→0.0000 (x)\n" % (refl if b == 13 else "0.0000→0.0000"))
            np.savez_compressed(os.path.join(td, "traces", "E176", "tr_bc_b%d.npz" % b), rows=rows(bad_rule and b == 10))
            vals = {("none", "off"): 150, ("none", "on"): 50, ("learn", "off"): 150 + eoff, ("learn", "on"): 50 + eon}
            for (ww, c), m in vals.items():
                tag = ("[맥락 평가] ctx=켬\n" if (c == "on" and (on_line or b != 10)) or (c == "off" and off_line and b == 10) else "")
                w(("logs", "E176", "ev_b%d_%s_%s.log" % (b, ww, c)), LDL + "=> DECOMP mode=all mod=%s acc=0.0\n" % f(m) + tag)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("획득", {}, "획득(H097)"), ("요소식", {"eoff": -2000, "eon": -2000}, "요소식(H097-null)"),
                       ("부분 켬만", {"eoff": -500, "eon": 3000}, "부분(H097-partial)"), ("차 0.10 정확 → 보류", {"eoff": -500, "eon": 500}, "보류"),
                       ("끔 발화 > 0", {"sp_off": 3}, "보류(조작검증 실패)"), ("맥락 켬 1,700", {"non": 1700}, "보류(조작검증 실패)"),
                       ("반사 변함", {"refl": "0.0000→1.0000"}, "보류(조작검증 실패)"), ("적재 1줄", {"nld": 1}, "보류(조작검증 실패)"),
                       ("규칙 불일치 1시행", {"bad_rule": True}, "보류(조작검증 실패)"), ("켬 평가 줄 없음", {"on_line": False}, "보류(조작검증 실패)"),
                       ("끔 평가에 맥락 줄", {"off_line": True}, "보류(조작검증 실패)"), ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    good = ("독립 판정: %s" % want) in out and (want != "보류" or out.strip().endswith("독립 판정: 보류"))
    ok_all &= good
    print("%-22s → %s %s" % (name, out.strip().splitlines()[-1], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
