#!/usr/bin/env python3
"""verify_e198_independent 합성 시험 — test_judge_e198 의 가짜 저장소 생성기를 빌려 독립 대조의 판정을 본다.
실행: python3 scripts/test_verify_e198.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_judge_e198 as TJ  # noqa: E402
import verify_e198_independent as V  # noqa: E402

WANT = {
    "T01": "학습 방향 충분", "T02": "학습 방향 부족", "T03": "독립 판정: 부분", "T04": "학습 방향 충분", "T05": "독립 판정: 부분",
    "T06": "학습 방향 충분", "T07": "독립 판정: 보류", "T08": "조작검증 실패", "T09": "보류(결측", "T10": "보류(결측",
    "T11": "조작검증 실패", "T12": "조작검증 실패", "T13": "조작검증 실패", "T14": "조작검증 실패", "T15": "조작검증 실패",
    "T16": "조작검증 실패", "T17": "학습 방향 충분", "T18": "조작검증 실패", "T19": "조작검증 실패", "T20": "조작검증 실패",
    "T21": "조작검증 실패", "T22": "조작검증 실패", "T23": "보류(결측",
}


def main():
    ok = 0
    for name, _, a1, a2, kw in TJ.CASES:
        root = tempfile.mkdtemp(prefix="tv198_")
        try:
            TJ.build(root, a1, a2, brains_override=kw.get("brains_override"), mutate=kw.get("mutate"))
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
        good = WANT[name[:3]] in got
        ok += good
        print("[%s] %s → %s" % ("통과" if good else "실패", name, got))
    print("합성 %d/%d" % (ok, len(TJ.CASES)))
    return ok == len(TJ.CASES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
