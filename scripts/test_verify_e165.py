#!/usr/bin/env python3
"""verify_e165_independent.py 합성 시험: 견고·빈도 의존·순서·강도 의존·보류, 조작검증 실패(노출 구성·선택성·동결), 결측.
실행: python3 scripts/test_verify_e165.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e165_independent as V

LD = "[E153 종류 입력 적재] /x/kctype.npz 검증 일치 — good_food_eye_l_to_kc_l=1.0\n"


def dev_log(mult, cnt, cyc="0.2957", imean="0.6930", sel="0.8500"):
    return ("[E160 종류 입력 Oja] 4집단 망 안 가소성\n[E165 노출] order=random bad_mult=%s good_l=%s bad_l=%s good_r=%s bad_r=%s n=%d int_min=0.5010 int_max=0.8995 "
            "int_mean=%s cyc_match=%s seed=16\n" % (mult, cnt[0], cnt[1], cnt[2], cnt[3], sum(int(c) for c in cnt), imean, cyc)
            + "=> KCDEVOJA side=l fired=251 sel_med0=0.5671 sel_med=%s frac09=0.4741 goodfrac=0.4781 sum_med=0.9589 sum_q10=0.0390 sum_q90=1.7942 dg_good=21597.6 dg_bad=22500.0 | "
              "side=r fired=237 sel_med0=0.5525 sel_med=0.8000 frac09=0.4346 goodfrac=0.4852 sum_med=0.9305 sum_q10=0.0407 sum_q90=1.7050 dg_good=21325.3 dg_bad=22573.5 | "
              "n=100 eta=0.02 beta=0.3 mmax=32 tau=20 save=/x.npz\n" % sel)


def run(jac=None, r=None, bad=None, miss=False, res_bad=False):
    """jac[(팔, 뇌)] = (좌, 우), r[(팔, 뇌)] = e_F/e_D(기본 2.0), bad[(팔, 뇌)] = dev_log 키워드 덮어쓰기."""
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E165"), ("logs", "E161"), ("traces", "E165")):
            os.makedirs(os.path.join(td, *d))
        w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for b in V.BRAINS:
            w(("logs", "E161", "D_b%d.log" % b), "[사전] x | **변조폭 +0.0200** (y)\n[사후] x | **변조폭 -0.2300**\n")
            for a, (mult, cnt) in V.WANT.items():
                if miss and a == "NU" and b == 20:
                    continue
                w(("logs", "E165", "%s_dev_b%d.log" % (a, b)), dev_log(mult, cnt, **(bad or {}).get((a, b), {})))
                jl, jr = (jac or {}).get((a, b), (0.0, 0.0))
                w(("logs", "E165", "%s_ov_b%d.log" % (a, b)), LD + "=> KCOVERLAP side=l good=54 bad=62 jac=%.4f cos=0.0002 | side=r good=55 bad=63 jac=%.4f cos=0.0160 | n_pres=50\n" % (jl, jr))
                post = 0.0150 + (-0.25 * (r or {}).get((a, b), 2.0))
                w(("logs", "E165", "%s_F_b%d.log" % (a, b)), LD + "[사전] x | **변조폭 +0.0150** (y)\n" + LD + "[사후] x | **변조폭 %+.4f**\n" % post)
                R = np.zeros((500, 37)); R[:, 13] = 1.0; R[:, 14] = 0.5; R[:, 21] = (11.0 / 12.0) ** 20; R[:, 22] = 0.5 * (11.0 / 12.0) ** 20
                if res_bad and a == "NR" and b == 17:
                    R[:, 21] = 1.0
                np.savez_compressed(os.path.join(td, "traces", "E165", "tr_%s_F_b%d.npz" % (a, b)), rows=R)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


allb = lambda a, v: {(a, b): v for b in V.BRAINS}
ok_all = True
for name, kw, want in (("견고", {}, "견고(H088)"),
                       ("빈도 의존(NU 자카드 0.30)", {"jac": allb("NU", (0.30, 0.30))}, "빈도 의존(H088-freq)"),
                       ("빈도 의존(NU r 1.2)", {"r": allb("NU", 1.2)}, "빈도 의존(H088-freq)"),
                       ("순서·강도 의존(NR 실패)", {"jac": allb("NR", (0.30, 0.0))}, "순서·강도 의존(H088-null)"),
                       ("순서·강도 의존(NR r 1.0)", {"r": allb("NR", 1.0)}, "순서·강도 의존(H088-null)"),
                       ("NR 보류(자카드 0.15)", {"jac": allb("NR", (0.15, 0.15))}, "보류"),
                       ("r 1.30 정확 → 성공", {"r": allb("NR", 1.30)}, "견고(H088)"),
                       ("노출 순환 일치 0.41", {"bad": {("NR", 18): {"cyc": "0.4100"}}}, "보류(조작검증 실패)"),
                       ("노출 강도 평균 0.7400", {"bad": {("NU", 19): {"imean": "0.7400"}}}, "보류(조작검증 실패)"),
                       ("선택성 상승 0.08", {"bad": {("NU", 16): {"sel": "0.6471"}}}, "보류(조작검증 실패)"),
                       ("동결 잔차", {"res_bad": True}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want) and (want != "보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-28s → %-34s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
