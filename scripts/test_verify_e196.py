#!/usr/bin/env python3
"""verify_e196_independent 합성 시험 — test_judge_e196 의 가짜 저장소 생성기를 빌려 독립 대조의 판정을 본다.
실행: python3 scripts/test_verify_e196.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_judge_e196 as TJ  # noqa: E402
import verify_e196_independent as V  # noqa: E402

WANT = {  # 시험 이름 머리 → 독립 판정에 들어 있어야 할 문구
    "T01": "표현 충분·평형 충분", "T02": "표현 충분·상관 평형 부족", "T03": "표현 충분·평형 보류", "T04": "표현·출력 병목",
    "T05": "부분", "T06": "표현 충분·평형 충분", "T07": "부분", "T08": "보류(양성 대조 실패)", "T09": "보류(양성 대조 실패)",
    "T10": "보류(양성 대조 실패)", "T11": "표현 충분·평형 충분", "T12": "독립 판정: 보류", "T13": "보류(결측", "T14": "조작검증 실패",
    "T15": "조작검증 실패", "T16": "조작검증 실패", "T17": "표현 충분·평형 충분", "T18": "조작검증 실패", "T19": "조작검증 실패",
    "T20": "조작검증 실패", "T21": "조작검증 실패", "T22": "조작검증 실패",
}


def main():
    ok = 0
    for name, _, a_sel, a_eq, k_cat, kw in TJ.CASES:
        root = tempfile.mkdtemp(prefix="tv196_")
        try:
            TJ.build(root, a_sel, a_eq, k_cat, brains_override=kw.get("brains_override"), mutate=kw.get("mutate"))
            cwd = os.getcwd()
            os.chdir(root)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    V.main()
            finally:
                os.chdir(cwd)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        got = [ln for ln in buf.getvalue().splitlines() if ln.startswith("독립 판정")][-1]
        want = WANT[name.split()[0]]
        good = want in got
        ok += good
        print("[%s] %s → %s" % ("통과" if good else "실패", name, got))
    print("합성 %d/%d" % (ok, len(TJ.CASES)))
    return 0 if ok == len(TJ.CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
