#!/usr/bin/env python3
"""verify_e141_independent.py 합성 시험: 답을 아는 원 로그·추적으로 지지·기각·조작검증 실패가 나오는지.
실행: python3 scripts/test_verify_e141.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e141_independent as V


def rows(freeze=True, eda_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    R = np.zeros((500, 27))
    R[:, 7] = (np.arange(500) % 2 == 0)
    R[:, 8] = np.where(R[:, 7] == 1, 3.0, 1.0)
    R[:, 9] = np.where(R[:, 7] == 1, -2.0, -1.0)
    R[:, 12] = R[:, 8:12].sum(1)
    R[:, 13:17] = rng.normal(0, 1000, (500, 4)) * eda_scale
    R[:, 21:25] = (V.RS if freeze else 4.0) * R[:, 13:17]
    return R


def make(td, effects, freeze=True, pre_shift=0):
    os.makedirs(os.path.join(td, "logs", "E141")); os.makedirs(os.path.join(td, "traces", "E141")); os.makedirs(os.path.join(td, "traces", "E139"))
    for i, b in enumerate(sorted(V.E119)):
        pre = V.E119[b][0] + (pre_shift if b == 11 else 0)
        post = pre + effects[i]
        f = lambda x: ("%+.4f" % (x / 1e4))
        open(os.path.join(td, "logs", "E141", "b%d.log" % b), "w", encoding="utf-8").write(
            "잡음 줄\n[사전] 오프셋 -0.020 | 정답률 2.0%% | **변조폭 %s** (양수=반사방향)\n[학습] 5ep 완료, 보상 250회 (탐색 주입 296회)\n[사후] 오프셋 -0.002 | **변조폭 %s**\n" % (f(pre), f(post)))
        R = rows(freeze=freeze); R[:, 7] = (np.arange(500) < 250)   # 보상 250 = 추적 보상 행 250
        np.savez_compressed(os.path.join(td, "traces", "E141", "tr_b%d.npz" % b), rows=R)
        np.savez_compressed(os.path.join(td, "traces", "E139", "tr_b%d.npz" % b), rows=rows(freeze=False, seed=1))


def run(effects, **kw):
    with tempfile.TemporaryDirectory() as td:
        make(td, effects, **kw)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


E = [V.E119[b][1] for b in sorted(V.E119)]
ok_all = True
for name, eff, kw, want in (
        ("지지", [-2000, -1500, -1200, -1300, -1000], {}, "독립 판정: 지지(H064)"),
        ("기각", [x + 40 for x in E], {}, "독립 판정: 기각(H064-null)"),
        ("반대", [x + 200 for x in E], {}, "독립 판정: 반대(H064-rev)"),
        ("경계 평균 −0.10", [-1100, -1000, -1000, -1000, -900], {}, "독립 판정: 지지(H064)"),
        ("M1 실패(무동결 추적)", [-2000, -1500, -1200, -1300, -1000], {"freeze": False}, "독립 판정: 보류(조작검증 실패)"),
        ("M2 실패", [-2000, -1500, -1200, -1300, -1000], {"pre_shift": 30}, "독립 판정: 보류(조작검증 실패)")):
    out = run(eff, **kw)
    good = want in out and "보상 수 로그=추적 일치" in out
    ok_all &= good
    print("%-22s → %s %s" % (name, [l for l in out.splitlines() if l.startswith("독립 판정")], "✓" if good else "✗\n" + out))
try:
    V.i4("0.123"); good = False
except ValueError:
    good = True
ok_all &= good
print("%-22s → %s" % ("4자리 아닌 값 거부", "✓" if good else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
