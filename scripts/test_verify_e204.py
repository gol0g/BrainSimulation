#!/usr/bin/env python3
"""verify_e204_independent 합성 시험 — test_judge_e204 의 같은 가짜 자료로 독립 판정이 판정기·사전 규칙과 같은지. 실행: python3 scripts/test_verify_e204.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_judge_e204 as T  # noqa: E402
import verify_e204_independent as V  # noqa: E402


def run(spec, mods=None):
    root = tempfile.mkdtemp(prefix="tv204_")
    try:
        E = T.build(root, spec, mods or {})
        V.EXP = E
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
        v = [ln for ln in buf.getvalue().splitlines() if ln.startswith("독립 판정: ")]
        return v[-1][len("독립 판정: "):] if v else "?"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    S, G, W, F, P = T.spec_of, T.GOOD, T.WEAK, T.FAIL, T.PART
    av = T.av_all
    base = S(G, G, W, W)
    cases = [
        ("H125", base, None, "H125"),
        ("세계 한정", S(G, P, W, W), None, "H125-세계 한정"),
        ("새 세계 게이팅 없이도", S(G, G, G, W), None, "보류(새 세계는 게이팅 없이도 섬)"),
        ("게이팅 무관", S(G, G, W, G), None, "게이팅 무관"),
        ("H125-null", S(F, F, W, W), None, "H125-null"),
        ("보류(G18 부분)", S(P, G, W, W), None, "보류"),
        ("경계 범주 4/5", S({**G, 14: P[14]}, G, W, W), None, "H125"),
        ("경계 범주 3/5", S({**G, 13: P[13], 14: P[14]}, G, W, W), None, "보류"),
        ("경계 m 차 500", S(G, G, W, {b: {**av(2500, 1800), ("X", "A"): (1500, 900)} for b in T.B5}), None, "H125"),
        ("경계 m 차 499", S(G, G, W, {b: {**av(2500, 1800), ("X", "A"): (1501, 900)} for b in T.B5}), None, "게이팅 무관"),
        ("일반화 등호", S({b: av(2000, 1000) for b in T.B5}, G, W, W), None, "H125"),
        ("일반화 미달", S({b: av(2000, 999) for b in T.B5}, G, W, W), None, "보류"),
        ("게이팅 비 0.011", base, {("G18", 12): {"ratio": 0.011}}, "보류(조작검증 실패"),
        ("게이팅 비 = 문턱", base, {("G18", 12): {"ratio": 0.0100}}, "H125"),
        ("구성 줄 1", base, {("GN", 10): {"build": 1}}, "보류(조작검증 실패"),
        ("점검 줄 2", base, {("G18", 11): {"gchk": 2}}, "보류(조작검증 실패"),
        ("활성 0", base, {("G18", 11): {"gact": 0}}, "보류(조작검증 실패"),
        ("요약 시행 2999", base, {("GN", 13): {"gn": 2999}}, "보류(조작검증 실패"),
        ("GN 세계 18300", base, {("GN", 10): {"world": 18300}}, "보류(조작검증 실패"),
        ("UN 배정 시드", base, {("UN", 14): {"trial0": 18500}}, "보류(조작검증 실패"),
        ("GN 평가 세계", base, {("GN", 12): {"eval_world": 18300}}, "보류(조작검증 실패"),
        ("평가 구성 줄 없음", base, {("G18", 10): {"no_eval_build": True}}, "보류(조작검증 실패"),
        ("UN 게이팅 줄", base, {("UN", 11): {"gate_lines_in_un": True}}, "보류(조작검증 실패"),
        ("정규화 잔차", base, {("G18", 13): {"d1": 2e-3}}, "보류(조작검증 실패"),
        ("평균 DA", base, {("GN", 14): {"da_mean": 0.06}}, "보류(조작검증 실패"),
        ("E200 같은 가중치", base, {("G18", 10): {"same_as_e200": True}}, "보류(조작검증 실패"),
        ("결측", base, {("UN", 12): {"missing_train": True}}, "보류(결측"),
    ]
    ok = 0
    for name, spec, mods, want in cases:
        got = run(spec, mods)
        jv = T.run(spec, mods)[0]
        good = got.startswith(want) and (jv.startswith(want) or (want.startswith("보류(") and jv.startswith(want[:6])))
        ok += good
        print(("OK   " if good else "실패 ") + "%s → 독립 %s · 판정기 %s" % (name, got, jv))
    print("합성 %d/%d" % (ok, len(cases)))
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
