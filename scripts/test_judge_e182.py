#!/usr/bin/env python3
"""judge_e182.py 합성 시험: 획득·경계(±0.10 정확)·부분(B만·A만)·요소식·차 0.10 정확, 4/5, 조작검증 실패(시행 맞춤 줄 없음·켬 99·간격 5·A·B 표지·동작점·진단·포화·pushed),
포화 경계 통과(0.6599), 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e182.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e182 as J  # noqa: E402

RT = (0.50, 0.46, 0.47, 0.52)


def build(eA=-3000, eB=3000, per=None, over=None, drop=None):
    X = {}
    for b in J.BRAINS:
        a_, b_ = (per or {}).get(b, (eA, eB))
        base = {"A": 150, "B": 50}
        for (w, c) in J.EVS:
            mod = base[c] + ((a_ if c == "A" else b_) if w == "learn" else 0)
            X[(b, w, c)] = {"mod": mod, "pushed": 8 if w == "learn" else 0, "tonic": [J.X182, J.X182], "rates": RT,
                            "trial": (100, 100, 100, 10), "B": c == "B", "A": c == "A"}
    for k, f in (over or {}).items():
        f(X[k])
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, want):
    global ok_all
    c, r = J.judge(X)
    got = r["verdict"] if r else c[0]
    good = (got.startswith(want) and (want != "보류" or got == "보류")) if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-34s 기대 %-26s → %-34s %s" % (name, want[:26], got[:34], "✓" if good else "✗ %s" % c))


chk("획득", build(), "획득(H103)")
chk("경계 e_A −0.10·e_B +0.10 → 획득", build(eA=-1000, eB=1000), "획득(H103)")
chk("B만", build(eA=-500, eB=3000), "부분 — 한 맥락만(B(같은 쪽)만 5")
chk("A만", build(eA=-3000, eB=999), "부분 — 한 맥락만(B(같은 쪽)만 0·A(교차)만 5")
chk("요소식", build(eA=2000, eB=2500), "요소식")
chk("차 0.10 정확 → 보류", build(eA=-500, eB=500), "보류")
chk("획득 4/5", build(per={14: (-500, 3000)}), "획득(H103)")
F = "보류(조작검증 실패)"
chk("시행 맞춤 줄 없음", build(over={(10, "learn", "A"): lambda d: d.update(trial=None)}), F)
chk("켬 99회", build(over={(11, "none", "B"): lambda d: d.update(trial=(100, 99, 100, 10))}), F)
chk("간격 5", build(over={(12, "learn", "B"): lambda d: d.update(trial=(100, 100, 100, 5))}), F)
chk("B 평가에 B 표지 없음", build(over={(13, "learn", "B"): lambda d: d.update(B=False)}), F)
chk("A 평가에 B 표지", build(over={(14, "none", "A"): lambda d: d.update(B=True)}), F)
chk("동작점 −1000", build(over={(10, "none", "A"): lambda d: d.update(tonic=[-1000.0])}), F)
chk("진단 없음", build(over={(11, "learn", "A"): lambda d: d.update(rates=None)}), F)
chk("포화(4개 ≥ 0.66)", build(over={(12, "none", "B"): lambda d: d.update(rates=(0.6664, 0.6662, 0.6663, 0.66))}), F)
chk("포화 하나 0.6599 → 통과", build(over={(12, "none", "B"): lambda d: d.update(rates=(0.6664, 0.6662, 0.6663, 0.6599))}), "획득(H103)")
chk("pushed 7", build(over={(13, "learn", "A"): lambda d: d.update(pushed=7)}), F)
chk("결측", build(drop=(14, "none", "B")), "결측")
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E182"))
    open(os.path.join(td, "logs", "E182", "ev_b10_learn_A.log"), "w", encoding="utf-8").write(
        "  [motor 동작점] motor 좌·우 Ioffset 기저값 -4000.0(긴장성 억제)\n[맥락 평가 시행] 시행 100 · 맥락 켬 100회·끔 100회(조향 처리 동안만 켬), 사이 간격 10처리 맥락 끔·무자극, 안정화·오프셋 맥락 없음\n"
        "[맥락 평가 A] ctx2=켬 ctx2_n=200\n[E171 평가 진단] variant=base side=left n=500 motor L/R 0.500000/0.460000 KC L/R 0.006/0.003\n"
        "[E171 평가 진단] variant=base side=right n=500 motor L/R 0.470000/0.520000 KC L/R 0.003/0.007\n=> DECOMP mode=all mod=-0.1500 acc=50.0 off=+0.0100 pushed=8 kc_means[x]\n")
    X = J.load(td)
g = X == {(10, "learn", "A"): {"mod": -1500, "pushed": 8, "tonic": [-4000.0], "rates": (0.5, 0.46, 0.47, 0.52), "trial": (100, 100, 100, 10), "B": False, "A": True}}
ok_all &= g
print("%-34s → %s" % ("원 로그 파싱", "✓" if g else "✗ %s" % X))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
