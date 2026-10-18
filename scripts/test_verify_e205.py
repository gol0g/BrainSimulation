#!/usr/bin/env python3
"""verify_e205_independent 합성 시험 — test_judge_e205 의 가짜 자료(build)로 독립 판정이 판정기와 같은 답을 내는지, 그리고 핵심 조작검증 실패를 잡는지.
실행: python3 scripts/test_verify_e205.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_judge_e205 as TJ  # noqa: E402
import verify_e205_independent as V  # noqa: E402
import judge_e205 as J  # noqa: E402


def both(spec, mods=None):
    root = tempfile.mkdtemp(prefix="tv205_")
    try:
        E = TJ.build(root, spec, mods or {})
        J.EXP = E; V.EXP = E
        b1, b2 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(b1):
            J.main()
        with contextlib.redirect_stdout(b2):
            V.main()
        jv = [ln[len("판정: "):] for ln in b1.getvalue().splitlines() if ln.startswith("판정: ")][-1]
        vv = [ln[len("독립 판정: "):] for ln in b2.getvalue().splitlines() if ln.startswith("독립 판정: ")][-1]
        return jv, vv
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    ok = tot = 0
    B5 = TJ.B5
    cases = [
        ("통과", TJ.spec(), None),
        ("유지만 간섭", TJ.spec(p={b: TJ.P_INTF for b in B5}), None),
        ("체계 확증", TJ.spec(uw=TJ.GOOD), None),
        ("L4 미확증", TJ.spec(gw=TJ.PART), None),
        ("범주 규칙 잃음", TJ.spec(r={b: TJ.ab(-300, -200, 300, 200, C=(0, 0), D=(0, 0)) for b in B5}), None),
        ("반전 고착", TJ.spec(v={b: TJ.ab(-1800, -1300, 1900, 1400) for b in B5}), None),
        ("부분 반전", TJ.spec(v={b: TJ.ab(-500, -300, 500, 300) for b in B5}), None),
        ("전제 미달", TJ.spec(p={b: TJ.ab(-1500, -1000, 1500, 1000, C=(800, 500), D=(-2000, -1500)) for b in B5}), None),
        ("경계 m 차 500", TJ.spec(uw={b: TJ.av_all(1500, 1200) for b in B5}), None),
        ("경계 m 차 499", TJ.spec(uw={b: TJ.av_all(1501, 1200) for b in B5}), None),
        ("경계 유지 몫 등호", TJ.spec(p={b: TJ.ab(-1000, -700, 1000, 700, C=(2000, 1500), D=(-2000, -1500)) for b in B5}), None),
        ("경계 반전 새 사례 749", TJ.spec(v={b: TJ.ab(1500, 749, -1500, -750) for b in B5}), None),
        ("실패: R 게이팅 비", TJ.spec(), {("R", 16): {"ratio": 0.011}}),
        ("실패: V 켬 수", TJ.spec(), {("V", 17): {"ncnt": 2999}}),
        ("실패: 평가 원형", TJ.spec(), {("P_learn", 16): {"proto": "1,2,3,4,5"}}),
        ("실패: 평가 ctx_on", TJ.spec(), {("V_learn", 19): {"ctx_on": 95}}),
        ("실패: UW 게이팅 줄", TJ.spec(), {("UW", 20): {"gate_in_uw": True}}),
        ("실패: V 반전 줄 빠짐", TJ.spec(), {("V", 16): {"no_rev": True}}),
        ("실패: GW 세계", TJ.spec(), {("GW", 16): {"world": 18300}}),
    ]
    for nm, sp, md in cases:
        jv, vv = both(sp, md)
        tot += 1
        good = jv == vv
        ok += good
        print(("OK   " if good else "실패 ") + "%s → 판정기 '%s' · 독립 '%s'" % (nm, jv, vv))
    print("독립 대조 합성 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
