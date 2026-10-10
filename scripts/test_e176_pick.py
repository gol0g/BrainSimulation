#!/usr/bin/env python3
"""e176_pick.py 합성 시험: 파싱, 조작검증(맥락 집단 발화 끔 0·켬 > 0), 제약(자카드 합 ≤ 1.60·켬 ≤ 2 × 끔·맥락 단독 ≤ 10 정확 경계), 가장 약한 통과 w, nan, 통과 없음, 결측.
실행: python3 scripts/test_e176_pick.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e176_pick as P


def line(w, l, r, c_off=0, c_on=2400):
    s = lambda d, k: "side=%s off=%d on=%d ctx=%d keep=%d lost=%d conj=%d jac=%s" % (k, d[0], d[1], d[2], 0, 0, 0, d[3])
    return ("=> KCCTX %s | %s | ctx_i=0.00 | KC 발화 합 good 끔 500 켬 700 · 맥락 단독 40 · 기준선 300 | n_pres=40 | ctx_inh_i=0.00 억제 부분집합 발화 끔 0 켬 0"
            " | ctx_ab_i=0.00 연합 결합 발화 끔 0 켬 0 | ctx_n=200 w=%.2f p=0.100 level=0.90 맥락 집단 발화 끔 %d 켬 %d\n" % (s(l, "l"), s(r, "r"), w, c_off, c_on))


ok_all = True


def chk(name, got, want):
    global ok_all
    g = got == want
    ok_all &= g
    print("%-44s 기대 %-14s → %-14s %s" % (name, want, got, "✓" if g else "✗"))


c = P.parse("x\n" + line(4.0, (60, 90, 4, "0.7000"), (61, 100, 10, "0.6500"), c_off=0, c_on=1234))
chk("파싱", (c["l"]["on"], c["r"]["ctx"], c["r"]["jac"], c["w"], c["c_off"], c["c_on"]), (90, 10, "0.6500", 4.0, 0, 1234))
chk("파싱 줄 없음", P.parse("x\n"), None)
base = {"1": ((60, 61, 0, "0.9500"), (60, 60, 0, "0.9600")),
        "2": ((60, 90, 10, "0.8000"), (60, 120, 1, "0.8000")),   # 자카드 1.60·ctx 10·켬 120 = 2×60 정확 → 통과(가장 약한)
        "4": ((60, 100, 3, "0.6000"), (60, 110, 2, "0.6000")),
        "8": ((60, 200, 30, "0.3000"), (60, 200, 30, "0.3000"))}
C = {t: P.parse(line(w, *base[t])) for t, w in P.GRID}
sel, man, cons = P.pick(C)
chk("가장 약한 통과 = 2(경계 정확)", sel, "2")
chk("1 자카드 합 1.91 탈락", cons["1"], False)
chk("8 ctx 30·켬 200 탈락", cons["8"], False)
C2 = dict(C); C2["2"] = P.parse(line(2.0, *base["2"], c_off=5))
chk("2 끔 발화 5 → 조작 실패, 4 선택", P.pick(C2)[0], "4")
C3 = dict(C); C3["2"] = P.parse(line(2.0, *base["2"], c_on=0))
chk("2 켬 발화 0 → 조작 실패", P.pick(C3)[1]["2"], False)
C4 = dict(C); C4["2"] = P.parse(line(2.0, (60, 90, 10, "nan"), (60, 120, 1, "0.5000")))
chk("2 자카드 nan → 탈락, 4 선택", P.pick(C4)[0], "4")
C5 = dict(C); C5["2"] = P.parse(line(2.0, (60, 121, 10, "0.8000"), (60, 120, 1, "0.8000")))
chk("2 켬 121 > 2×60 → 탈락", P.pick(C5)[2]["2"], False)
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E176", "calib"))
    P.EXP = td
    for t, w in P.GRID:
        open(os.path.join(td, "logs", "E176", "calib", "kcctx_w%s_b15.log" % t), "w", encoding="utf-8").write(line(w, *base[t]))
    with contextlib.redirect_stdout(io.StringIO()):
        P.main()
    chk("pick.txt 내용", open(os.path.join(td, "logs", "E176", "pick.txt"), encoding="utf-8").read().strip(), "W=2.0")
    os.remove(os.path.join(td, "logs", "E176", "calib", "kcctx_w8_b15.log")); os.remove(os.path.join(td, "logs", "E176", "pick.txt"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.main()
    chk("결측 → 선택 보류", ("결측" in buf.getvalue(), os.path.exists(os.path.join(td, "logs", "E176", "pick.txt"))), (True, False))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
