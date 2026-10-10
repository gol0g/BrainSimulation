#!/usr/bin/env python3
"""judge_e170.py 합성 시험: 합성 성공·없음·부분, 경계(ρ 0.80 정확, 1.10×max 정확, 충돌 0.15 정확), 4/5, 조작검증(자극 구성·회귀·적재·전제), 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e170.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e170 as J

GOOD = {k: tuple(J.i4(x) for x in v) for k, v in J.DESIGN.items()}


def build(ea=None, ec=None, eb=-6000, ed=6000, over=None, drop=None):
    """ea[b]·ec[b] = 학습 효과(1e-4). 기본: 가산(ea = eb − ed, ec = eb + ed). 무학습 mod = 100(모든 변형)."""
    X = {"E162": {}}
    for b in J.BRAINS:
        vals = {"base": eb, "bad": ed, "agree": (ea or {}).get(b, eb - ed), "conflict": (ec or {}).get(b, eb + ed)}
        for v in J.VARS:
            X[("none", v, b)] = {"mode": "none", "mod": 100, "pushed": 0, "stim": {k: GOOD[k] for k in GOOD if k[0] == v}}
            X[("AB", v, b)] = {"mode": "all", "mod": 100 + vals[v], "pushed": 8, "stim": {k: GOOD[k] for k in GOOD if k[0] == v}}
        for w in J.WS:
            for v in ("base", "bad"):
                X["E162"][(b, w, v)] = X[(w, v, b)]["mod"]
    for k, f in (over or {}).items():
        f(X[k]) if callable(f) else X.__setitem__(k, f)
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
    print("%-40s 기대 %-28s → %-34s %s" % (name, w1 + (" / " + w2 if w2 else ""), got[:34], "✓" if good else "✗ %s" % c))


allb = lambda v: {b: v for b in J.BRAINS}
chk("가산 합성 + 가산 상쇄", build(), "합성 성공(H093)", "가산 상쇄")
chk("합성 없음(ea = 단독 최대 −0.60)", build(ea=allb(-6000)), "합성 없음(H093-null)")
chk("부분(ea −0.85)", build(ea=allb(-8500)), "부분")
chk("ρ 0.80 정확(−0.96) → 성공", build(ea=allb(-9600)), "합성 성공(H093)")
chk("ρ 0.7999(−0.9599) → 부분", build(ea=allb(-9599)), "부분")
chk("1.10×max 정확(−0.66) → 없음 아님 → 부분", build(ea=allb(-6600)), "부분")
chk("1.10×max 미만(−0.6599) → 없음", build(ea=allb(-6599)), "합성 없음(H093-null)")
chk("반대 부호(+0.30) → 없음", build(ea=allb(3000)), "합성 없음(H093-null)")
chk("4/5 성공(한 뇌 −0.6)", build(ea={**allb(-12000), 12: -6000}), "합성 성공(H093)")
chk("충돌 0.15 정확(차 0.18) → 가산 상쇄", build(ec=allb(1800)), "합성 성공(H093)", "가산 상쇄")
chk("충돌 0.1501 → 비가산", build(ec=allb(1801)), "합성 성공(H093)", "비가산")
F = "보류(조작검증 실패)"
chk("자극 구성 어긋남(agree 오른쪽 bad 0)", build(over={("AB", "agree", 11): lambda d: d["stim"].update({("agree", "right"): (0, 9000, 0, 0, 9000, 9000)})}), F, F)
chk("자극 줄 없음", build(over={("none", "conflict", 13): lambda d: d.update(stim={})}), F)
chk("회귀 어긋남(E162 와 0.0002)", build(over={"E162": lambda d: d.update({(12, "AB", "base"): d[(12, "AB", "base")] + 2})}), F)
chk("적재 pushed 7", build(over={("AB", "bad", 14): lambda d: d.update(pushed=7)}), F)
chk("전제 bad 효과 +0.0999", build(ed=999), F)
chk("결측(none agree 뇌 10)", build(drop=("none", "agree", 10)), "결측")

with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E170"))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    w(("logs", "E170", "ev_b10_AB_agree.log"), "[E170 자극] variant=agree side=left good L/R 0.90/0.00 bad L/R 0.00/0.90 food L/R 0.90/0.90\n"
      "[E170 자극] variant=agree side=right good L/R 0.00/0.90 bad L/R 0.90/0.00 food L/R 0.90/0.90\n[E146 변형] variant=agree vseed=0\n"
      "=> DECOMP mode=all mod=-1.1000 acc=100.0 off=-0.0077 pushed=8 kc_means[l>l=1.0]\n")
    w(("E162.log",), "  e162 b10 AB base: => mod -0.6086\n  e162 b10 AB bad: => mod +0.6558\n  e162 b10 none base: => mod +0.0080\n  e162 b10 none bad: => mod +0.0144\n")
    J.EXP = td
    X = J.load()
g = (X[("AB", "agree", 10)] == {"mode": "all", "mod": -11000, "pushed": 8, "stim": {("agree", "left"): GOOD[("agree", "left")], ("agree", "right"): GOOD[("agree", "right")]}}
     and X["E162"] == {(10, "AB", "base"): -6086, (10, "AB", "bad"): 6558, (10, "none", "base"): 80, (10, "none", "bad"): 144}
     and ("AB", "base", 10) not in X)
ok_all &= g
print("%-40s → %s" % ("원 로그 파싱(자극 줄·DECOMP·E162)", "✓" if g else "✗ %s" % X))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
