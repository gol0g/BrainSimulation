#!/usr/bin/env python3
"""e160_pick.py 합성 시험: 조건 칸 중 합 비가 1 에 가장 가까운 칸, 동점(작은 η → 작은 β), sel_med·sum_med 경계, Oja 변화 없음 제외, 결측, 보정 실패, 로그 파싱.
실행: python3 scripts/test_e160_pick.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e160_pick as P


def cell(sel=0.9, sm=1.0, sel_r=None, sm_r=None, dg=100.0):
    return {"l": {"fired": 200, "sel_med": sel, "sum_med": sm, "dg_good": dg, "dg_bad": dg},
            "r": {"fired": 200, "sel_med": sel if sel_r is None else sel_r, "sum_med": sm if sm_r is None else sm_r, "dg_good": dg, "dg_bad": dg}}


def grid(default=None, **over):
    rows = {(e, b): (default or cell(sel=0.5)) for e in P.ETAS for b in P.BETAS}   # 기본: 조건 밖
    for k, v in over.items():
        e, b = k.split("_")
        rows[(e, b)] = v
    return rows


ok_all = True


def chk(name, rows, want):
    global ok_all
    got, why = P.select(rows)
    good = got == want
    ok_all &= good
    print("%-40s 기대 %-16s → %-16s %s" % (name, want, got, "✓" if good else "✗ %s" % why))


chk("한 칸만 조건", grid(**{"0.02_1.0": cell()}), ("0.02", "1.0"))
chk("합 비가 1 에 가까운 칸", grid(**{"0.02_1.0": cell(sm=1.10), "0.08_0.3": cell(sm=0.97)}), ("0.08", "0.3"))
chk("ln 대칭(0.90 vs 1/0.90)", grid(**{"0.02_1.0": cell(sm=0.90), "0.005_3.0": cell(sm=1 / 0.90)}), ("0.005", "3.0"))
chk("동점 → 작은 η", grid(**{"0.08_0.1": cell(sm=1.05), "0.02_3.0": cell(sm=1.05)}), ("0.02", "3.0"))
chk("동점·같은 η → 작은 β", grid(**{"0.02_3.0": cell(sm=1.05), "0.02_0.3": cell(sm=1.05)}), ("0.02", "0.3"))
chk("sel_med 경계 0.80 정확 → 통과", grid(**{"0.02_1.0": cell(sel=0.80)}), ("0.02", "1.0"))
chk("sel_med 0.7999 → 실패", grid(**{"0.02_1.0": cell(sel=0.80, sel_r=0.7999)}), None)
chk("sum_med 경계 0.80·1.25 정확 → 통과", grid(**{"0.02_1.0": cell(sm=0.80, sm_r=1.25)}), ("0.02", "1.0"))
chk("sum_med 1.2501 → 실패", grid(**{"0.02_1.0": cell(sm=1.0, sm_r=1.2501)}), None)
chk("max 양쪽으로 비교(한쪽 나쁨)", grid(**{"0.02_1.0": cell(sm=1.0, sm_r=1.20), "0.08_1.0": cell(sm=1.10, sm_r=1.10)}), ("0.08", "1.0"))
chk("Oja 변화 없음 → 제외", grid(**{"0.02_1.0": cell(dg=0.0), "0.08_1.0": cell(sm=1.2)}), ("0.08", "1.0"))
r = grid(**{"0.02_1.0": cell()}); r[("0.02", "1.0")] = None
chk("결측 → 실패", r, None)
chk("전부 조건 밖 → 실패", grid(), None)


def log(sel_l, sel_r, sm_l, sm_r, dg=50.0):
    return ("[E160 종류 입력 Oja] x\n=> KCDEVOJA side=l fired=210 sel_med0=0.5500 sel_med=%.4f frac09=0.4000 goodfrac=0.5000 sum_med=%.4f sum_q10=0.7000 sum_q90=1.3000 dg_good=%.1f dg_bad=%.1f"
            " | side=r fired=205 sel_med0=0.5500 sel_med=%.4f frac09=0.4000 goodfrac=0.5000 sum_med=%.4f sum_q10=0.7000 sum_q90=1.3000 dg_good=%.1f dg_bad=%.1f"
            " | n=100 eta=0.02 beta=1 mmax=32 tau=20 save=x\n" % (sel_l, sm_l, dg, dg, sel_r, sm_r, dg, dg))


with tempfile.TemporaryDirectory() as td:
    d = os.path.join(td, "logs", "E160", "calib"); os.makedirs(d)
    for e in P.ETAS:
        for b in P.BETAS:
            ok_cell = (e, b) == ("0.02", "1.0")
            open(os.path.join(d, "oja_e%s_b%s_b15.log" % (e, b)), "w", encoding="utf-8").write(
                log(0.85 if ok_cell else 0.6, 0.88 if ok_cell else 0.6, 0.98, 1.03))
    P.EXP = td
    with contextlib.redirect_stdout(io.StringIO()):
        P.main()
    got = open(os.path.join(td, "logs", "E160", "oja_pick.txt"), encoding="utf-8").read().strip()
g = got == "eta=0.02 beta=1.0 (sel_med 0.850/0.880, sum_med 0.980/1.030)"
ok_all &= g
print("%-40s → %s %s" % ("로그 파싱·oja_pick.txt", got, "✓" if g else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
