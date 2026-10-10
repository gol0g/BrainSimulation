#!/usr/bin/env python3
"""e173_pick.py 합성 시험: 줄 파싱, 조작검증(맥락 발화 끔 0·켬 > 0), 제약(ctx ≤ 10·on ≤ 2·off — 경계 정확), 가장 큰 통과 w 선택, 통과 없음, 결측.
실행: python3 scripts/test_e173_pick.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e173_pick as P


def line(w, l, r, sp_off=0, sp_on=900):
    s = lambda d, k: "side=%s off=%d on=%d ctx=%d keep=%d lost=%d conj=%d jac=%.4f" % (k, d[0], d[1], d[2], min(d[0], d[1]), 0, d[3], 0.5)
    return ("=> KCCTX %s | %s | ctx_n=200 w=%.2f p=0.100 level=0.90 연결 l=60000 r=60000 | 맥락 발화 끔 %d 켬 %d | n_pres=40\n"
            % (s(l, "l"), s(r, "r"), w, sp_off, sp_on))


ok_all = True


def chk(name, got, want):
    global ok_all
    g = got == want
    ok_all &= g
    print("%-40s 기대 %-12s → %-12s %s" % (name, want, got, "✓" if g else "✗"))


# 파싱
c = P.parse("x\n" + line(3, (60, 90, 4, 20), (55, 100, 10, 30), sp_off=0, sp_on=1200))
chk("파싱 좌 on·우 ctx·발화", (c["l"]["on"], c["r"]["ctx"], c["sp_on"], c["w"]), (90, 10, 1200, 3.0))
chk("파싱 줄 없음", P.parse("nothing\n"), None)
# 선택: w 1·2·3 통과, 4 ctx 11, 6 on > 2·off → 3
base = {1: ((60, 62, 0, 1), (60, 61, 0, 1)), 2: ((60, 70, 1, 5), (60, 72, 2, 6)), 3: ((60, 90, 10, 20), (60, 120, 9, 30)),
        4: ((60, 100, 11, 30), (60, 110, 8, 30)), 6: ((60, 121, 5, 40), (60, 100, 3, 30))}
C = {w: P.parse(line(w, *base[w])) for w in P.WS}
sel, man, cons = P.pick(C)
chk("가장 큰 통과 w(경계 ctx 10·on 120 = 2×60)", sel, 3)
chk("w4 ctx 11 탈락", cons[4], False)
chk("w6 on 121 > 120 탈락", cons[6], False)
# 조작검증 실패(끔 발화 > 0)인 w 는 제약 통과 못 함
C2 = dict(C); C2[3] = P.parse(line(3, *base[3], sp_off=5))
chk("끔 발화 5 → w3 탈락, w2 선택", P.pick(C2)[0], 2)
C3 = dict(C); C3[3] = P.parse(line(3, *base[3], sp_on=0))
chk("켬 발화 0 → w3 탈락", P.pick(C3)[1][3], False)
# 통과 없음
C4 = {w: P.parse(line(w, (60, 200, 50, 5), (60, 200, 50, 5))) for w in P.WS}
chk("통과 없음 → None", P.pick(C4)[0], None)
# main: 파일 읽기·pick.txt 쓰기·결측
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E173", "calib"))
    P.EXP = td
    for w in P.WS:
        open(os.path.join(td, "logs", "E173", "calib", "kcctx_w%d_b15.log" % w), "w", encoding="utf-8").write(line(w, *base[w]))
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        P.main()
    chk("pick.txt 내용", open(os.path.join(td, "logs", "E173", "pick.txt"), encoding="utf-8").read().strip(), "W=3")
    os.remove(os.path.join(td, "logs", "E173", "calib", "kcctx_w6_b15.log"))
    os.remove(os.path.join(td, "logs", "E173", "pick.txt"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.main()
    chk("결측 → 선택 보류(pick.txt 없음)", ("결측" in buf.getvalue(), os.path.exists(os.path.join(td, "logs", "E173", "pick.txt"))), (True, False))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
