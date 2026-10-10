#!/usr/bin/env python3
"""e173_pick.py 합성 시험(정정 1 — 균일 전류 I): 줄 파싱, 조작검증(good KC 발화 합 켬 > 끔), 제약(ctx ≤ 10·on ≤ 2·off — 경계 정확), 가장 큰 통과 I 선택, 통과 없음, 결측.
실행: python3 scripts/test_e173_pick.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e173_pick as P


def line(i, l, r, k_off=500, k_on=700):
    s = lambda d, k: "side=%s off=%d on=%d ctx=%d keep=%d lost=%d conj=%d jac=%.4f" % (k, d[0], d[1], d[2], min(d[0], d[1]), 0, d[3], 0.5)
    return ("=> KCCTX %s | %s | ctx_i=%.2f | KC 발화 합 good 끔 %d 켬 %d · 맥락 단독 40 · 기준선 300 | n_pres=40\n"
            % (s(l, "l"), s(r, "r"), i, k_off, k_on))


ok_all = True


def chk(name, got, want):
    global ok_all
    g = got == want
    ok_all &= g
    print("%-40s 기대 %-12s → %-12s %s" % (name, want, got, "✓" if g else "✗"))


c = P.parse("x\n" + line(4, (60, 90, 4, 20), (55, 100, 10, 30), k_off=480, k_on=650))
chk("파싱 좌 on·우 ctx·발화 합", (c["l"]["on"], c["r"]["ctx"], c["k_off"], c["k_on"], c["i"]), (90, 10, 480, 650, 4.0))
chk("파싱 줄 없음", P.parse("nothing\n"), None)
base = {1: ((60, 62, 0, 1), (60, 61, 0, 1)), 2: ((60, 70, 1, 5), (60, 72, 2, 6)), 4: ((60, 90, 10, 20), (60, 120, 9, 30)),
        8: ((60, 100, 11, 30), (60, 110, 8, 30)), 12: ((60, 121, 5, 40), (60, 100, 3, 30)),
        16: ((60, 150, 30, 50), (60, 150, 30, 50)), 20: ((60, 200, 90, 50), (60, 200, 90, 50))}
C = {i: P.parse(line(i, *base[i])) for i in P.IS}
sel, man, cons = P.pick(C)
chk("가장 큰 통과 I(경계 ctx 10·on 120 = 2×60)", sel, 4)
chk("I8 ctx 11 탈락", cons[8], False)
chk("I12 on 121 > 120 탈락", cons[12], False)
chk("I16·I20 ctx > 10 탈락", (cons[16], cons[20]), (False, False))
# 정정 2 최소 효과: 결합 합 ≥ 10 또는 자카드 합 ≤ 1.80(정확 경계)
mk = lambda cl, cr, jl, jr: {"l": {"conj": cl, "jac": jl}, "r": {"conj": cr, "jac": jr}}
chk("최소 효과: 결합 5+5 = 10 → 통과", P.min_effect(mk(5, 5, "0.9900", "0.9900")), True)
chk("최소 효과: 결합 5+4, 자카드 0.9000+0.9000 → 통과", P.min_effect(mk(5, 4, "0.9000", "0.9000")), True)
chk("최소 효과: 결합 5+4, 자카드 0.9000+0.9001 → 실패", P.min_effect(mk(5, 4, "0.9000", "0.9001")), False)
chk("최소 효과: 자카드 nan → 결합만", P.min_effect(mk(0, 0, "nan", "0.5000")), False)
C2 = dict(C); C2[4] = P.parse(line(4, *base[4], k_off=500, k_on=500))
chk("발화 합 켬 = 끔 → I4 조작 실패, I2 선택", P.pick(C2)[0], 2)
C4 = {i: P.parse(line(i, (60, 200, 50, 5), (60, 200, 50, 5))) for i in P.IS}
chk("통과 없음 → None", P.pick(C4)[0], None)
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E173", "calib"))
    P.EXP = td
    for i in P.IS:
        open(os.path.join(td, "logs", "E173", "calib", "kcctx_i%d_b15.log" % i), "w", encoding="utf-8").write(line(i, *base[i]))
    with contextlib.redirect_stdout(io.StringIO()):
        P.main()
    chk("pick.txt 내용", open(os.path.join(td, "logs", "E173", "pick.txt"), encoding="utf-8").read().strip(), "I=4")
    # 정정 2: I* 가 효과 부족이면 none(효과 부족)
    weak = dict(base); weak[4] = ((60, 61, 0, 2), (60, 61, 0, 2))
    for i in P.IS:
        open(os.path.join(td, "logs", "E173", "calib", "kcctx_i%d_b15.log" % i), "w", encoding="utf-8").write(line(i, *weak[i]).replace("jac=0.5000", "jac=0.9800"))
    with contextlib.redirect_stdout(io.StringIO()):
        P.main()
    chk("I* 효과 부족 → none(효과 부족)", open(os.path.join(td, "logs", "E173", "pick.txt"), encoding="utf-8").read().startswith("I=none(효과 부족"), True)
    os.remove(os.path.join(td, "logs", "E173", "calib", "kcctx_i20_b15.log"))
    os.remove(os.path.join(td, "logs", "E173", "pick.txt"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.main()
    chk("결측 → 선택 보류(pick.txt 없음)", ("결측" in buf.getvalue(), os.path.exists(os.path.join(td, "logs", "E173", "pick.txt"))), (True, False))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
