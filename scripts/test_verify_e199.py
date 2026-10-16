#!/usr/bin/env python3
"""verify_e199_independent 합성 시험 — test_judge_e199 의 같은 가짜 자료(24경우)로 독립 판정이 사전 규칙과 같은지 본다.
실행: python3 scripts/test_verify_e199.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_judge_e199 as T  # noqa: E402
import verify_e199_independent as V  # noqa: E402

MAP = {"판정: 경쟁형 규칙으로 L4 학습 — 맥락 의존 범주(H120)": "독립 판정: 경쟁형 규칙으로 L4 학습(H120)",
       "판정: 경쟁형 규칙으로 L4 학습": "독립 판정: 경쟁형 규칙으로 L4 학습(H120)",
       "판정: 획득 실패(H120-null)": "독립 판정: 획득 실패(H120-null)",
       "판정: 부분": "독립 판정: 부분",
       "판정: 맥락 의존(사례 한정)": "독립 판정: 사례 한정",
       "판정: 보류\n": "독립 판정: 보류\n",
       "판정: 보류(결측": "독립 판정: 보류(결측",
       "판정: 보류(조작검증 실패)": "독립 판정: 보류(조작검증 실패",
       "판정: 보류(결측 — 보정 학습률": "독립 판정: 보류(결측 — 보정"}


def run(av, mod, kw):
    root = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        T.build(root, av, mod, **kw)
        os.chdir(root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
        return buf.getvalue()
    finally:
        os.chdir(cwd)
        shutil.rmtree(root)


def main():
    ok = 0
    for name, av, mod, kw, want in T.CASES:
        w = MAP[want]
        out = run(av, mod, kw)
        good = w in out
        ok += good
        last = [ln for ln in out.splitlines() if ln.startswith("독립 판정")]
        print("[%s] %s → %s" % ("통과" if good else "실패", name, last[-1] if last else out.strip()[-200:]))
        if not good:
            print(out)
    print("합성 %d/%d" % (ok, len(T.CASES)))
    return 0 if ok == len(T.CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
