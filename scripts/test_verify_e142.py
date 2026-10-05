#!/usr/bin/env python3
"""verify_e142_independent.py 합성 시험: 답을 아는 원 로그·추적으로 L2·필요성·부분·조작검증 실패가 나오는지.
실행: python3 scripts/test_verify_e142.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e142_independent as V


def rows(n, freeze, rew):
    rng = np.random.default_rng(n)
    R = np.zeros((n, 27)); R[:rew, 7] = 1
    R[:, 8] = 2.0; R[:, 9] = -1.0; R[:, 12] = R[:, 8:12].sum(1)
    R[:, 13:17] = rng.normal(0, 1000, (n, 4))
    R[:, 21:25] = (V.RS if freeze else 4.0) * R[:, 13:17]
    return R


def make(td, post, refl_bad=None):
    for d in ("logs/E142", "traces/E142"):
        os.makedirs(os.path.join(td, d), exist_ok=True)
    f = lambda x: "%+.4f" % (x / 1e4)
    for a, n in V.NEXP.items():
        for b in V.PRE119:
            pre = V.PRE119[b]
            w2 = "24.9000" if refl_bad == (a, b) else "25.0000"
            open(os.path.join(td, "logs", "E142", "%s_b%d.log" % (a, b)), "w", encoding="utf-8").write(
                "[사전] 오프셋 +0.001 | 정답률 0.0%% | **변조폭 %s** (양수=반사방향)\n[학습] %dep 완료, 보상 150회 (탐색)\n"
                "[반사가중치] good_food_to_motor_l   n=15139 w_mean 25.0000→%s (학습 뇌)\n"
                "[반사가중치] good_food_to_motor_r   n=14818 w_mean 25.0000→25.0000 (학습 뇌)\n[사후] 오프셋 +0.002 | **변조폭 %s**\n"
                % (f(pre), n // 100, w2, f(post[a][b])))
            np.savez_compressed(os.path.join(td, "traces", "E142", "tr_%s_b%d.npz" % (a, b)), rows=rows(n, a != "NF1500", 150))


def run(post, **kw):
    with tempfile.TemporaryDirectory() as td:
        make(td, post, **kw)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


B = list(V.PRE119)
P = lambda f5, f15, nf: {"F500": {b: V.PRE119[b] + f5 for b in B}, "F1500": {b: f15 for b in B}, "NF1500": {b: nf for b in B}}
ok_all = True
for name, post, kw, want, want_nd in (
        ("L2 + 동결 필요", P(-2000, -500, 3000), {}, "독립 판정: L2 달성(H065)", "독립 필요성: 동결이 L2 에 필요"),
        ("L2 + 불필요", P(-2000, -500, -400), {}, "독립 판정: L2 달성(H065)", "독립 필요성: 학습량만으로도 L2"),
        ("L2 경계 m=−0.02", P(-2000, -200, 3000), {}, "독립 판정: L2 달성(H065)", "독립 필요성: 동결이 L2 에 필요"),
        ("부분", P(-1500, 1000, 3000), {}, "독립 판정: 반사를 거스름(H065-partial)", "독립 필요성: 해당 없음"),
        ("효과 없음", P(100, 4000, 4000), {}, "독립 판정: 효과 없음(H065-null)", "독립 필요성: 해당 없음"),
        ("반사 변함(F1500)", P(-2000, -500, 3000), {"refl_bad": ("F1500", 12)}, "독립 판정: 보류(조작검증 실패)", "독립 필요성: 해당 없음"),
        ("반사 변함(NF1500)", P(-2000, -500, 3000), {"refl_bad": ("NF1500", 12)}, "독립 판정: L2 달성(H065)", "독립 필요성: 필요성 미결(NF1500 조작검증 실패)")):
    out = run(post, **kw)
    good = want in out and want_nd in out
    ok_all &= good
    print("%-18s → %s | %s %s" % (name, [l for l in out.splitlines() if l.startswith("독립 판정")][0][7:],
                                  [l for l in out.splitlines() if l.startswith("독립 필요성")][0][8:], "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
