#!/usr/bin/env python3
"""e175_pick.py 합성 시험(정정 1 — 맥락 = assoc_binding 침묵, 음전류 격자): 파싱(음수), 조작검증(연합 결합 발화 켬 < 끔), 제약(자카드 합 ≤ 1.60 정확 경계·켬 ≥ 0.5 × 끔 정확 경계·
맥락 단독 ≤ 10 정확 경계), 가장 약한(|I| 가장 작은) 통과 I_ab, 자카드 nan, 통과 없음, 결측.
실행: python3 scripts/test_e175_pick.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e175_pick as P


def line(ia, l, r, a_off=20000, a_on=4000):
    s = lambda d, k: "side=%s off=%d on=%d ctx=%d keep=%d lost=%d conj=%d jac=%s" % (k, d[0], d[1], d[2], 0, 0, 0, d[3])
    return ("=> KCCTX %s | %s | ctx_i=0.00 | KC 발화 합 good 끔 500 켬 450 · 맥락 단독 40 · 기준선 300 | n_pres=40 | ctx_inh_i=0.00 억제 부분집합 발화 끔 0 켬 0"
            " | ctx_ab_i=%.2f 연합 결합 발화 끔 %d 켬 %d\n" % (s(l, "l"), s(r, "r"), ia, a_off, a_on))


ok_all = True


def chk(name, got, want):
    global ok_all
    g = got == want
    ok_all &= g
    print("%-46s 기대 %-14s → %-14s %s" % (name, want, got, "✓" if g else "✗"))


c = P.parse("x\n" + line(-40.0, (60, 50, 2, "0.7000"), (61, 55, 3, "0.7500"), a_off=20000, a_on=12))
chk("파싱(음수 I_ab)", (c["l"]["on"], c["r"]["ctx"], c["l"]["jac"], c["ic"], c["i_off"], c["i_on"]), (50, 3, "0.7000", -40.0, 20000, 12))
chk("파싱 줄 없음", P.parse("x\n"), None)
base = {"m5": ((60, 59, 0, "0.9500"), (60, 60, 0, "0.9600")),      # 자카드 합 1.91
        "m10": ((60, 55, 1, "0.8000"), (60, 56, 0, "0.8001")),     # 합 1.6001 > 1.60
        "m20": ((60, 30, 10, "0.8000"), (60, 52, 1, "0.8000")),    # 합 1.60·ctx 10·켬 30 = 0.5×60 정확 → 통과(가장 약한)
        "m40": ((60, 40, 3, "0.6000"), (60, 41, 2, "0.6000")),
        "m80": ((60, 29, 0, "0.4000"), (60, 30, 0, "0.4000")),     # 좌 켬 29 < 30 → 탈락
        "m160": ((60, 10, 0, "0.3000"), (60, 10, 0, "0.3000"))}
C = {t: P.parse(line(ia, *base[t])) for t, ia in P.GRID}
sel, man, cons = P.pick(C)
chk("가장 약한 통과 = −20(자카드 1.60·ctx 10·켬 0.5×끔 정확)", sel, "m20")
chk("−10 자카드 합 1.6001 탈락", cons["m10"], False)
chk("−80 켬 29 < 0.5×60 탈락", cons["m80"], False)
C2 = dict(C); C2["m20"] = P.parse(line(-20.0, *base["m20"], a_off=20000, a_on=20000))
chk("−20 침묵 안 됨(켬 = 끔) → 조작 실패, −40 선택", P.pick(C2)[0], "m40")
C3 = dict(C); C3["m20"] = P.parse(line(-20.0, (60, 50, 2, "nan"), (60, 52, 1, "0.5000")))
chk("−20 자카드 nan → 탈락, −40 선택", P.pick(C3)[0], "m40")
C4 = dict(C2); C4["m40"] = P.parse(line(-40.0, (60, 40, 11, "0.6000"), (60, 41, 2, "0.6000")))
chk("−40 맥락 단독 11 탈락(−20 조작 실패) → 없음", P.pick(C4)[0], None)
C5 = dict(C); C5["m20"] = P.parse(line(-20.0, *base["m20"], a_off=20000, a_on=20001))
chk("−20 켬 > 끔 → 조작 실패", P.pick(C5)[1]["m20"], False)
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E175", "calib"))
    P.EXP = td
    for t, ia in P.GRID:
        open(os.path.join(td, "logs", "E175", "calib", "kcctx_a%s_b15.log" % t), "w", encoding="utf-8").write(line(ia, *base[t]))
    with contextlib.redirect_stdout(io.StringIO()):
        P.main()
    chk("pick.txt 내용", open(os.path.join(td, "logs", "E175", "pick.txt"), encoding="utf-8").read().strip(), "IAB=-20.0")
    os.remove(os.path.join(td, "logs", "E175", "calib", "kcctx_am160_b15.log")); os.remove(os.path.join(td, "logs", "E175", "pick.txt"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.main()
    chk("결측 → 선택 보류", ("결측" in buf.getvalue(), os.path.exists(os.path.join(td, "logs", "E175", "pick.txt"))), (True, False))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
