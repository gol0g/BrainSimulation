#!/usr/bin/env python3
"""e175_pick.py 합성 시험: 파싱, 조작검증(연합 결합 발화 켬 > 끔), 제약(자카드 합 ≤ 1.60 정확 경계·켬 ≤ 2 × 끔 정확 경계·맥락 단독 ≤ 10 정확 경계), 가장 약한 통과 I_ab, 자카드 nan, 통과 없음, 결측.
실행: python3 scripts/test_e175_pick.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e175_pick as P


def line(ia, l, r, a_off=300, a_on=900):
    s = lambda d, k: "side=%s off=%d on=%d ctx=%d keep=%d lost=%d conj=%d jac=%s" % (k, d[0], d[1], d[2], 0, 0, 0, d[3])
    return ("=> KCCTX %s | %s | ctx_i=0.00 | KC 발화 합 good 끔 500 켬 600 · 맥락 단독 40 · 기준선 300 | n_pres=40 | ctx_inh_i=0.00 억제 부분집합 발화 끔 0 켬 0"
            " | ctx_ab_i=%.2f 연합 결합 발화 끔 %d 켬 %d\n" % (s(l, "l"), s(r, "r"), ia, a_off, a_on))


ok_all = True


def chk(name, got, want):
    global ok_all
    g = got == want
    ok_all &= g
    print("%-44s 기대 %-14s → %-14s %s" % (name, want, got, "✓" if g else "✗"))


c = P.parse("x\n" + line(2.0, (60, 70, 2, "0.7000"), (61, 75, 3, "0.7500"), a_off=310, a_on=990))
chk("파싱", (c["l"]["on"], c["r"]["ctx"], c["l"]["jac"], c["ic"], c["i_off"], c["i_on"]), (70, 3, "0.7000", 2.0, 310, 990))
chk("파싱 줄 없음", P.parse("x\n"), None)
base = {"05": ((60, 61, 0, "0.9500"), (60, 60, 0, "0.9600")),      # 자카드 합 1.91
        "10": ((60, 65, 1, "0.8000"), (60, 66, 0, "0.8001")),      # 합 1.6001 > 1.60
        "15": ((60, 70, 10, "0.8000"), (60, 72, 1, "0.8000")),     # 합 1.60·ctx 10 정확 → 통과(가장 약한 통과)
        "20": ((60, 90, 3, "0.6000"), (60, 91, 2, "0.6000")),
        "40": ((60, 121, 0, "0.4000"), (60, 110, 0, "0.4000")),    # 좌 on 121 > 120 → 탈락
        "80": ((60, 200, 30, "0.3000"), (60, 200, 30, "0.3000"))}
C = {t: P.parse(line(ia, *base[t])) for t, ia in P.GRID}
sel, man, cons = P.pick(C)
chk("가장 약한 통과 = 1.5(자카드 합 1.60·ctx 10 정확)", sel, "15")
chk("1.0 자카드 합 1.6001 탈락", cons["10"], False)
chk("4.0 켬 121 > 2×60 탈락", cons["40"], False)
chk("켬 120 = 2×60 정확 → 통과", P.pick({"40": P.parse(line(4.0, (60, 120, 0, "0.4000"), (60, 110, 0, "0.4000")))})[1]["40"], True)
C2 = dict(C); C2["15"] = P.parse(line(1.5, *base["15"], a_off=900, a_on=900))
chk("1.5 연합 결합 발화 켬 = 끔 → 조작 실패, 2.0 선택", P.pick(C2)[0], "20")
C3 = dict(C); C3["15"] = P.parse(line(1.5, (60, 70, 2, "nan"), (60, 72, 1, "0.5000")))
chk("1.5 자카드 nan → 탈락, 2.0 선택", P.pick(C3)[0], "20")
C4 = dict(C2); C4["20"] = P.parse(line(2.0, (60, 90, 11, "0.6000"), (60, 91, 2, "0.6000")))
chk("2.0 맥락 단독 11 탈락(1.5 조작 실패) → 없음", P.pick(C4)[0], None)
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E175", "calib"))
    P.EXP = td
    for t, ia in P.GRID:
        open(os.path.join(td, "logs", "E175", "calib", "kcctx_a%s_b15.log" % t), "w", encoding="utf-8").write(line(ia, *base[t]))
    with contextlib.redirect_stdout(io.StringIO()):
        P.main()
    chk("pick.txt 내용", open(os.path.join(td, "logs", "E175", "pick.txt"), encoding="utf-8").read().strip(), "IAB=1.5")
    os.remove(os.path.join(td, "logs", "E175", "calib", "kcctx_a80_b15.log")); os.remove(os.path.join(td, "logs", "E175", "pick.txt"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.main()
    chk("결측 → 선택 보류", ("결측" in buf.getvalue(), os.path.exists(os.path.join(td, "logs", "E175", "pick.txt"))), (True, False))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
