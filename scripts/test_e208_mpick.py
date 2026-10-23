#!/usr/bin/env python3
"""e208_mpick 합성 시험 — 정답을 아는 가짜 무학습 평가 로그(m_{V}_{X|Y}_b15.log, RCEVAL 4줄 × 'motor L/R' 2개)로 동작점 규칙이 사전 문장대로인지.
실행: python3 test_e208_mpick.py (scripts/ 에서)"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e208_mpick as M  # noqa: E402


def write(out, V, m, nval=None, mx=None):
    """V 의 X·Y 로그 — 모든 motor 값 m(맥락 X 만 mx 로 바꿀 수 있음). nval 이면 X 로그의 RCEVAL 줄 수를 바꿈."""
    for X in "XY":
        v = mx if (mx is not None and X == "X") else m
        n = nval if (nval is not None and X == "X") else 4
        with open(os.path.join(out, "m_%s_%s_b15.log" % (V, X)), "w", encoding="utf-8") as f:
            f.write("  [motor 동작점] motor 좌·우 Ioffset 기저값 -1300.0(긴장성 억제)\n")
            for i in range(n):
                c, s = "AB"[i // 2 % 2], ("train", "new")[i % 2]
                f.write("RCEVAL cat=%s set=%s mod=+0.0100 off=+0.0000 acc=50.0 trials=96 feat_on=96 ctx=%s ctx_on=96 | side=left n=240 motor L/R %.6f/%.6f KC L/R 0.005/0.003 "
                        "side=right n=240 motor L/R %.6f/%.6f KC L/R 0.003/0.005\n" % (c, s, X, v, v, v, v))


def run(fn, mode, mr=0.6087, grid=None, extra=None):
    """fn(T) = m_G0(T). grid 의 T 마다 로그를 쓰고 mpick 을 돌린다 → (출력, cal 내용 또는 None)."""
    out = tempfile.mkdtemp(prefix="mp208_")
    try:
        write(out, "ref", mr)
        for T in (grid or [-1300]):
            write(out, M.nm(T), fn(T))
        if extra:
            extra(out)
        cal = os.path.join(out, "calib_tonic.txt")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                M.main(["x", out, cal, mode])
        except SystemExit as e:
            buf.write("SYSTEMEXIT %s" % e)
        return buf.getvalue(), (open(cal, encoding="utf-8").read() if os.path.exists(cal) else None)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main():
    ok = tot = 0

    def chk(name, got, want):
        nonlocal ok, tot
        tot += 1
        good = want(got) if callable(want) else got == want
        ok += bool(good)
        print(("OK   " if good else "실패 ") + "%s → %s" % (name, got))

    lin = lambda a, s: (lambda T: a + s * (T + 1300))  # noqa: E731 — m_G0(T) 직선
    G_UP, G_UPW, G_DN, G_DNW = M.UP, M.UP + M.UPW, M.DOWN, M.DOWN + M.DOWNW
    # first
    o, c = run(lambda T: 0.6087 + 0.0100, "first")
    chk("first: |차| 0.0100 ≤ 0.017 → −1300", (c or "").startswith("T_F=-1300 "), True)
    o, c = run(lambda T: 0.6087 + 0.0170, "first")
    chk("first: |차| 0.0170(경계) → −1300", (c or "").startswith("T_F=-1300 "), True)
    o, c = run(lambda T: 0.5000, "first")
    chk("first: 구동 줄어듦 → NEED_UP·cal 없음", ("NEED_UP" in o and "NEED_DOWN" not in o, c), (True, None))
    o, c = run(lambda T: 0.7000, "first")
    chk("first: 구동 늘어남 → NEED_DOWN·cal 없음", ("NEED_DOWN" in o, c), (True, None))
    # up — 직선 0.50 + 0.0000362·(T+1300): T=+1700 에서 0.6086 → 위 끝이라 NEED_UPWIDE
    o, c = run(lin(0.50, 0.0000362), "up", grid=G_UP)
    chk("up: 최소가 위 끝 +1700 → NEED_UPWIDE·cal 없음", ("NEED_UPWIDE" in o, c), (True, None))
    # up — 직선 0.50 + 0.0000543·(T+1300): T=+700 쯤 교차 → 격자 800(0.6113, 차 0.0026)
    o, c = run(lin(0.50, 0.0000543), "up", grid=G_UP)
    chk("up: 교차 격자 +800 고름", (c or "").split(" ")[0], "T_F=800")
    # up — 오목: 위로 갔다 내려와 m_ref 에 못 닿음(최대 0.58 at +200) → 실패
    o, c = run(lambda T: 0.58 - 0.0000001 * (T - 200) ** 2 / 100, "up", grid=G_UP)
    chk("up: 최소 |차| > 0.017(끝 아님) → 실패·cal 없음", ("[동작점 규칙] 실패" in o, c), (True, None))
    # upwide — 직선 0.50 + 0.0000243·(T+1300): T=+3200 에서 0.6094 → 3200 에서 차 0.0007
    o, c = run(lin(0.50, 0.0000243), "upwide", grid=G_UPW)
    chk("upwide: 넓힌 격자 +3200 고름(차 ≤ 0.017)", (c or "").split(" ")[0], "T_F=3200")
    o, c = run(lin(0.50, 0.0000100), "upwide", grid=G_UPW)
    chk("upwide: 넓혀도 |차| > 0.017 → 실패·cal 없음", ("[동작점 규칙] 실패" in o, c), (True, None))
    # down — 직선 0.70 + 0.0000610·(T+1300): T=−2800 에서 0.6085
    o, c = run(lin(0.70, 0.0000610), "down", grid=G_DN)
    chk("down: 격자 −2800 고름", (c or "").split(" ")[0], "T_F=-2800")
    o, c = run(lin(0.70, 0.0000300), "down", grid=G_DN)
    chk("down: 최소가 아래 끝 −4300 → NEED_DOWNWIDE·cal 없음", ("NEED_DOWNWIDE" in o, c), (True, None))
    o, c = run(lin(0.70, 0.0000200), "downwide", grid=G_DNW)
    chk("downwide: −5800 에서도 차 0.0013 → −5800", (c or "").split(" ")[0], "T_F=-5800")
    # 같음 — 두 후보 |차| 같으면 −1300 에 가까운 쪽
    o, c = run(lambda T: {-1000: 0.6187, -700: 0.5987}.get(T, 0.50), "up", grid=G_UP)
    chk("up: |차| 같음(−1000·−700 둘 다 0.0100) → −1300 에 가까운 −1000", (c or "").split(" ")[0], "T_F=-1000")
    # 값 수 32 아님 → 실패(SystemExit)
    o, c = run(lambda T: 0.6087, "first", extra=lambda out: write(out, "ref", 0.6087, nval=3))
    chk("값 수 28 → 실패", ("SYSTEMEXIT" in o and "motor 값 수 28" in o, c), (True, None))
    # 맥락 X·Y 가 다르면 평균(0.59·0.6274 → 0.6087)
    o, c = run(lambda T: 0.6087, "first", extra=lambda out: write(out, M.nm(-1300), 0.6274, mx=0.5900))
    chk("X·Y 평균: (0.5900 + 0.6274)/2 = 0.6087 → −1300", (c or "").startswith("T_F=-1300 m_ref=0.6087 m_G0=0.6087"), True)
    # 0 을 지나지 않음
    chk("격자에 0 없음", 0 in G_UPW or 0 in G_DNW, False)
    chk("격자 모양(위 −1300…1700·넓힘 2000…3200·아래 −1300…−4300·넓힘 −4600…−5800)",
        (G_UP[0], G_UP[-1], len(G_UP), M.UPW[0], M.UPW[-1], G_DN[-1], len(G_DN), M.DOWNW[0], M.DOWNW[-1]), (-1300, 1700, 11, 2000, 3200, -4300, 11, -4600, -5800))
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
