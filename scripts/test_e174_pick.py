#!/usr/bin/env python3
"""e174_pick.py 합성 시험: 파싱, 조작검증(억제 부분집합 발화 켬 > 끔), 제약(자카드 합 ≤ 1.60 정확 경계·켬 ≥ 0.5 끔 정확 경계·맥락 단독 ≤ 10), 가장 약한 통과 I_c, 자카드 nan, 통과 없음, 결측.
실행: python3 scripts/test_e174_pick.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e174_pick as P


def line(ic, l, r, i_off=300, i_on=900):
    s = lambda d, k: "side=%s off=%d on=%d ctx=%d keep=%d lost=%d conj=%d jac=%s" % (k, d[0], d[1], d[2], 0, 0, 0, d[3])
    return ("=> KCCTX %s | %s | ctx_i=0.00 | KC 발화 합 good 끔 500 켬 450 · 맥락 단독 40 · 기준선 300 | n_pres=40 | ctx_inh_i=%.2f 억제 부분집합 발화 끔 %d 켬 %d\n"
            % (s(l, "l"), s(r, "r"), ic, i_off, i_on))


ok_all = True


def chk(name, got, want):
    global ok_all
    g = got == want
    ok_all &= g
    print("%-44s 기대 %-14s → %-14s %s" % (name, want, got, "✓" if g else "✗"))


c = P.parse("x\n" + line(1.5, (60, 50, 2, "0.7000"), (61, 55, 3, "0.7500"), i_off=310, i_on=990))
chk("파싱", (c["l"]["on"], c["r"]["ctx"], c["l"]["jac"], c["ic"], c["i_off"], c["i_on"]), (50, 3, "0.7000", 1.5, 310, 990))
chk("파싱 줄 없음", P.parse("x\n"), None)
base = {"05": ((60, 59, 0, "0.9500"), (60, 60, 0, "0.9600")),      # 자카드 합 1.91 > 1.60
        "10": ((60, 55, 1, "0.8000"), (60, 56, 0, "0.8001")),      # 합 1.6001 > 1.60 (경계 바로 위)
        "15": ((60, 50, 2, "0.8000"), (60, 52, 1, "0.8000")),      # 합 1.60 정확 → 통과(가장 약한 통과)
        "20": ((60, 40, 3, "0.6000"), (60, 41, 2, "0.6000")),
        "30": ((60, 29, 0, "0.4000"), (60, 30, 0, "0.4000")),      # 좌 on 29 < 30 → 탈락
        "40": ((60, 20, 0, "0.3000"), (60, 20, 0, "0.3000"))}
C = {t: P.parse(line(ic, *base[t])) for t, ic in P.GRID}
sel, man, cons = P.pick(C)
chk("가장 약한 통과 = 1.5(자카드 합 1.60 정확)", sel, "15")
chk("1.0 자카드 합 1.6001 탈락", cons["10"], False)
chk("3.0 켬 29 < 0.5×60 탈락", cons["30"], False)
C2 = dict(C); C2["15"] = P.parse(line(1.5, *base["15"], i_off=900, i_on=900))
chk("1.5 억제 발화 켬 = 끔 → 조작 실패, 2.0 선택", P.pick(C2)[0], "20")
C3 = dict(C); C3["15"] = P.parse(line(1.5, (60, 50, 2, "nan"), (60, 52, 1, "0.5000")))
chk("1.5 자카드 nan → 탈락, 2.0 선택", P.pick(C3)[0], "20")
C4 = dict(C); C4["20"] = P.parse(line(2.0, (60, 40, 11, "0.6000"), (60, 41, 2, "0.6000"))); C4["15"] = C2["15"]
chk("2.0 맥락 단독 11 탈락(1.5 조작 실패) → 없음", P.pick(C4)[0], None)
chk("켬 30 = 0.5×60 정확 → 통과", P.pick({"30": P.parse(line(3.0, (60, 30, 0, "0.4000"), (60, 30, 0, "0.4000")))})[1]["30"], True)
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E174", "calib"))
    P.EXP = td
    for t, ic in P.GRID:
        open(os.path.join(td, "logs", "E174", "calib", "kcctx_c%s_b15.log" % t), "w", encoding="utf-8").write(line(ic, *base[t]))
    with contextlib.redirect_stdout(io.StringIO()):
        P.main()
    chk("pick.txt 내용", open(os.path.join(td, "logs", "E174", "pick.txt"), encoding="utf-8").read().strip(), "IC=1.5")
    os.remove(os.path.join(td, "logs", "E174", "calib", "kcctx_c40_b15.log")); os.remove(os.path.join(td, "logs", "E174", "pick.txt"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.main()
    chk("결측 → 선택 보류", ("결측" in buf.getvalue(), os.path.exists(os.path.join(td, "logs", "E174", "pick.txt"))), (True, False))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
