#!/usr/bin/env python3
"""judge_e181.py 합성 시험: 판정 1 회복·손실 유지·부분과 경계(O 0.85 정확·0.8499·0.70 정확·0.7001), 판정 2 경계(−0.0200 정확·−0.0199), 4/5,
조작검증 실패(이식 pushed·적재·배율·동작점 없음·값 다름·진단 없음·포화·R6 기준 효과·R7 반사 발현), 포화 경계 통과, 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e181.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e181 as J  # noqa: E402

RT = (0.40, 0.30, 0.31, 0.41)


def build(O=0.95, w25=500, base=-10000, refl=4000, over=None, drop=None):
    """O = E(W0,R25)/E(W0,R0) — 뇌별 사전이면 그 값. v(none,R0) = 100, v(none,R25) = 100 + refl. v(W0,R0) = 100 + base, v(W0,R25) = v(none,R25) + O·base,
    v(W25,R0) = 100 + 0.6·base, v(W25,R25) = w25(1e-4)."""
    X = {}
    for b in J.BRAINS:
        o = O[b] if isinstance(O, dict) else O
        vals = {"none_R0": 100, "none_R25": 100 + refl, "W0_R0": 100 + base, "W0_R25": 100 + refl + int(round(o * base)),
                "W25_R0": 100 + int(round(0.6 * base)), "W25_R25": (w25[b] if isinstance(w25, dict) else w25)}
        for c in J.CELLS:
            X[(c, b)] = {"mode": "none" if c.startswith("none") else "all", "v": vals[c], "pushed": 0 if c.startswith("none") else 8,
                         "ld": 2, "sc": 2, "tonic": [J.X181, J.X181], "rates": RT}
    for k, f in (over or {}).items():
        f(X[k])
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, w1, w2=None):
    global ok_all
    c, r = J.judge(X)
    if w1 == "결측":
        good, got = (r is None and "결측" in c[0]), c[0]
    else:
        good = r is not None and r["v1"].startswith(w1) and (w2 is None or r["v2"].startswith(w2))
        got = "%s / %s" % (r["v1"], r["v2"]) if r else c[0]
    ok_all &= good
    print("%-36s 기대 %-30s → %-36s %s" % (name, w1 + (" / " + w2 if w2 else ""), got[:36], "✓" if good else "✗ %s" % c))


chk("회복(O 0.95) / 역전 없음(+0.05 — 첫 판 기본값 −0.05 는 역전이라 고침)", build(), "회복(H102)", "역전 없음")
chk("O 0.85 정확 → 회복", build(O=0.85), "회복(H102)")
chk("O 0.8499 → 부분", build(O=0.8499), "부분")
chk("O 0.70 정확 → 손실 유지", build(O=0.70), "손실 유지(H102-null)")
chk("O 0.7001 → 부분", build(O=0.7001), "부분")
chk("손실 유지(O 0.55)", build(O=0.55), "손실 유지(H102-null)")
chk("회복 4/5(한 뇌 0.60)", build(O={10: 0.95, 11: 0.95, 12: 0.60, 13: 0.95, 14: 0.95}), "회복(H102)")
chk("판정 2: −0.0200 정확 → 반사 역전", build(w25=-200), "회복(H102)", "반사 역전")
chk("판정 2: −0.0199 → 역전 없음", build(w25=-199), "회복(H102)", "역전 없음")
chk("판정 2: 4/5 역전", build(w25={10: -300, 11: -300, 12: 100, 13: -300, 14: -300}), "회복(H102)", "반사 역전")
F = "보류(조작검증 실패)"
chk("이식 pushed 7", build(over={("W0_R25", 12): lambda d: d.update(pushed=7)}), F, F)
chk("적재 1", build(over={("none_R0", 11): lambda d: d.update(ld=1)}), F)
chk("배율 1", build(over={("W25_R0", 13): lambda d: d.update(sc=1)}), F)
chk("동작점 줄 없음", build(over={("none_R25", 10): lambda d: d.update(tonic=[])}), F)
chk("동작점 값 −1000", build(over={("W0_R0", 14): lambda d: d.update(tonic=[J.X181, -1000.0])}), F)
chk("진단 없음", build(over={("W25_R25", 12): lambda d: d.update(rates=None)}), F)
chk("포화(4개 ≥ 0.66)", build(over={("W0_R25", 11): lambda d: d.update(rates=(0.6664, 0.6662, 0.6663, 0.6600))}), F)
chk("포화 하나 0.6599 → 통과", build(over={("W0_R25", 11): lambda d: d.update(rates=(0.6664, 0.6662, 0.6663, 0.6599))}), "회복(H102)")
chk("R6 기준 효과 −0.0999", build(base=-999), F)
chk("R7 반사 발현 +0.0999", build(refl=999), F)
chk("R7 반사 발현 +0.1000 정확 → 통과", build(refl=1000), "회복(H102)")
chk("결측(W25_R0 뇌 10)", build(drop=("W25_R0", 10)), "결측")
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E181"))
    open(os.path.join(td, "logs", "E181", "W0_R25_b10.log"), "w", encoding="utf-8").write(
        "  [motor 동작점] motor 좌·우 Ioffset 기저값 -4000.0(긴장성 억제)\n[E153 종류 입력 적재] k 검증 일치\n[E157 종류 입력 배율] x 검증 일치\n"
        "[E153 종류 입력 적재] k 검증 일치\n[E157 종류 입력 배율] x 검증 일치\n"
        "[E171 평가 진단] variant=base side=left n=250 motor L/R 0.400000/0.300000 KC L/R 0.004/0.001\n"
        "[E171 평가 진단] variant=base side=right n=250 motor L/R 0.310000/0.410000 KC L/R 0.001/0.004\n"
        "=> DECOMP mode=all mod=-0.5000 acc=100.0 off=-0.0077 pushed=8 kc_means[x]\n")
    X = J.load(td)
g = X == {("W0_R25", 10): {"mode": "all", "v": -5000, "pushed": 8, "ld": 2, "sc": 2, "tonic": [-4000.0], "rates": (0.40, 0.30, 0.31, 0.41)}}
ok_all &= g
print("%-36s → %s" % ("원 로그 파싱", "✓" if g else "✗ %s" % X))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
