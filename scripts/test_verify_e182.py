#!/usr/bin/env python3
"""verify_e182_independent.py 합성 시험: 획득·요소식·B만·A만·차 0.10 → 보류, 조작 실패(시행 맞춤 줄·켬 99·A·B 표지·동작점·포화·진단 한쪽·pushed), 결측.
실행: python3 scripts/test_verify_e182.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e182_independent as V  # noqa: E402

TRL = "[맥락 평가 시행] 시행 100 · 맥락 켬 %d회·끔 100회(조향 처리 동안만 켬), 사이 간격 10처리 맥락 끔·무자극, 안정화·오프셋 맥락 없음\n"


def run(eA=-3000, eB=3000, notrial=False, on99=False, noB=False, AhasB=False, tonic=-4000.0, sat=False, diag_one=False, pushed7=False, miss=False):
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E182"))
        for b in V.BRAINS:
            for w in ("learn", "none"):
                for c in ("A", "B"):
                    if miss and b == 14 and w == "none" and c == "B":
                        continue
                    mod = ({"A": 150, "B": 50}[c] + ((eA if c == "A" else eB) if w == "learn" else 0)) / 1e4
                    t = "  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)\n" % (tonic if (b == 12 and w == "none" and c == "A") else -4000.0)
                    if not (notrial and b == 10 and w == "learn" and c == "A"):
                        t += TRL % (99 if (on99 and b == 11 and w == "learn" and c == "B") else 100)
                    if c == "B" and not (noB and b == 13 and w == "learn"):
                        t += "[맥락 평가] ctx=켬 x\n"
                    if c == "A":
                        t += "[맥락 평가 A] ctx2=켬\n" + ("[맥락 평가] ctx=켬 x\n" if (AhasB and b == 14 and w == "none") else "")
                    r4 = (0.6664, 0.6662, 0.6663, 0.66) if (sat and b == 10 and w == "none" and c == "B") else (0.5, 0.46, 0.47, 0.52)
                    t += "[E171 평가 진단] variant=base side=left n=500 motor L/R %.6f/%.6f KC L/R 0.006/0.003\n" % r4[:2]
                    if not (diag_one and b == 11 and w == "none" and c == "A"):
                        t += "[E171 평가 진단] variant=base side=right n=500 motor L/R %.6f/%.6f KC L/R 0.003/0.007\n" % r4[2:]
                    pu = 7 if (pushed7 and b == 12 and w == "learn" and c == "B") else (8 if w == "learn" else 0)
                    t += "=> DECOMP mode=x mod=%+.4f acc=50.0 off=+0.0100 pushed=%d kc_means[x]\n" % (mod, pu)
                    open(os.path.join(td, "logs", "E182", "ev_b%d_%s_%s.log" % (b, w, c)), "w", encoding="utf-8").write(t)
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("획득", {}, "획득(H103)"), ("요소식", {"eA": 2000, "eB": 2500}, "요소식"),
                       ("B만", {"eA": -500, "eB": 3000}, "부분 B만"), ("A만", {"eA": -3000, "eB": 999}, "부분 A만"),
                       ("차 0.10 정확 → 보류", {"eA": -500, "eB": 500}, "보류"),
                       ("시행 맞춤 줄 없음", {"notrial": True}, "보류(조작검증 실패)"), ("켬 99", {"on99": True}, "보류(조작검증 실패)"),
                       ("B 표지 없음", {"noB": True}, "보류(조작검증 실패)"), ("A 평가에 B 표지", {"AhasB": True}, "보류(조작검증 실패)"),
                       ("동작점 −1000", {"tonic": -1000.0}, "보류(조작검증 실패)"), ("포화", {"sat": True}, "보류(조작검증 실패)"),
                       ("진단 한쪽", {"diag_one": True}, "보류(조작검증 실패)"), ("pushed 7", {"pushed7": True}, "보류(조작검증 실패)"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want) and (want != "보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-22s → %-34s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
