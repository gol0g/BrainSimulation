#!/usr/bin/env python3
"""judge_e171.py 합성 시험: KC 보존·감소·보류, 경계(비 0.85·1.15 정확, 평균 0.85), 4/5, 조작검증(n·회귀·적재), 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e171.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e171 as J


def build(kr=None, over=None, drop=None):
    """kr[b] = 일치 조합 KC 비(네 쪽 같은 값, 기본 1.0). 단독 KC 발화율 6000(1e-6)."""
    X = {}
    for b in J.BRAINS:
        f = (kr or {}).get(b, 1.0)
        for w in ("AB", "none"):
            for v in J.VARS:
                k = int(round(6000 * f)) if v == "agree" else 6000
                d = {sd: {"n": 250, "mL": 20000, "mR": 20000, "kL": k, "kR": k} for sd in ("left", "right")}
                X[(w, v, b)] = {"mod": -5000 if w == "AB" else 100, "pushed": 8 if w == "AB" else 0, "d": d}
                X[("E170", w, v, b)] = X[(w, v, b)]["mod"]
    for kk, fn in (over or {}).items():
        fn(X[kk]) if callable(fn) else X.__setitem__(kk, fn)
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, want):
    global ok_all
    c, r = J.judge(X)
    got = r["verdict"] if r else c[0]
    good = (r is None and "결측" in got) if want == "결측" else (got.startswith(want) and (want != "보류" or got == "보류"))
    ok_all &= good
    print("%-34s 기대 %-24s → %-30s %s" % (name, want, got[:30], "✓" if good else "✗ %s" % c))


allb = lambda v: {b: v for b in J.BRAINS}
chk("KC 보존(비 1.0)", build(), "KC 보존(H094")
chk("KC 감소(비 0.6)", build(kr=allb(0.6)), "KC 감소(H094-alt")
chk("비 0.85 정확 → 보존", build(kr=allb(0.85)), "KC 보존(H094")
chk("비 0.8498 → 보존 아님·감소(평균<0.85)", build(kr=allb(0.8498)), "KC 감소(H094-alt")
chk("비 1.15 정확 → 보존", build(kr=allb(1.15)), "KC 보존(H094")
chk("비 1.1502 → 보류(증가)", build(kr=allb(1.1502)), "보류")
chk("4/5 보존(한 뇌 0.6)", build(kr={**allb(1.0), 12: 0.6}), "KC 보존(H094")
chk("3 보존·2 감소 → 보류", build(kr={**allb(1.0), 12: 0.6, 13: 0.6}), "보류")
F = "보류(조작검증 실패)"
chk("진단 n 249", build(over={("AB", "agree", 11): lambda d: d["d"]["left"].update(n=249)}), F)
chk("변조폭 E170 과 0.0002 차", build(over={("none", "bad", 13): lambda d: d.update(mod=d["mod"] + 2)}), F)
chk("적재 7", build(over={("AB", "base", 14): lambda d: d.update(pushed=7)}), F)
chk("결측(E170 none agree b10)", build(drop=("E170", "none", "agree", 10)), "결측")

with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E171"), ("logs", "E170")):
        os.makedirs(os.path.join(td, *d))
    w = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    w(("logs", "E171", "ev_b10_AB_agree.log"), "[E146 변형] variant=agree vseed=0\n=> DECOMP mode=all mod=-0.5916 acc=100.0 off=-0.0077 pushed=8 kc_means[x]\n"
      "[E171 평가 진단] variant=agree side=left n=250 motor L/R 0.010000/0.090000 KC L/R 0.006000/0.005000\n"
      "[E171 평가 진단] variant=agree side=right n=250 motor L/R 0.090000/0.010000 KC L/R 0.005000/0.006000\n")
    w(("logs", "E170", "ev_b10_AB_agree.log"), "=> DECOMP mode=all mod=-0.5916 acc=100.0 off=-0.0077 pushed=8 kc_means[x]\n")
    J.EXP = td
    X = J.load()
g = (X[("AB", "agree", 10)] == {"mod": -5916, "pushed": 8, "d": {"left": {"n": 250, "mL": 10000, "mR": 90000, "kL": 6000, "kR": 5000},
                                                                  "right": {"n": 250, "mL": 90000, "mR": 10000, "kL": 5000, "kR": 6000}}}
     and X[("E170", "AB", "agree", 10)] == -5916 and ("AB", "base", 10) not in X)
ok_all &= g
print("%-34s → %s" % ("원 로그 파싱(진단 줄·DECOMP·E170)", "✓" if g else "✗ %s" % X))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
