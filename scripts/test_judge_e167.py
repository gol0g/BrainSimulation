#!/usr/bin/env python3
"""judge_e167.py 합성 시험: 남음·손실·부분, 경계(Δ 비 0.80 정확·평균 Δ_N 5.0%p 정확·양수 14/16), 조작검증(발달 재현·적재·대조), 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e167.py (저장소 루트에서)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e167 as J


def build(nl=None, nf=None, hl=900, hf=525, over=None, drop=None):
    """nl[w]·nf[w] = N learn·frozen 새 항목(0.1%p, 두 난수열 같은 값). 기본 N learn 520·frozen 500(Δ_N +2.0%p), H Δ +37.5%p."""
    X = {}
    for w in J.WIRES:
        X[("dv", w)] = {"same": True, "nc": 20000, "ni": 20000}
        for ts in J.TSEEDS:
            for mode, v in (("learn", (nl or {}).get(w, 520)), ("frozen", (nf or {}).get(w, 500))):
                X[("N", mode, w, ts)] = {"train": 600, "novel": v, "nc": 20000, "ni": 20000, "dmax": 0.0, "share_c": 0.027, "share_i": 0.058, "uni": 0.02}
            for mode, v in (("learn", hl), ("frozen", hf)):
                X[("H", mode, w, ts)] = {"train": 900, "novel": v}
    for k, f in (over or {}).items():
        f(X[k]) if callable(f) else X.__setitem__(k, f)
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, want):
    global ok_all
    c, r = J.judge(X)
    got = r["verdict"] if r else c[0]
    good = (r is None and "결측" in got) if want == "결측" else got.startswith(want)
    ok_all &= good
    print("%-44s 기대 %-18s → %-24s %s" % (name, want, got[:24], "✓" if good else "✗ %s" % c))


allw = lambda v: {w: v for w in J.WIRES}
chk("손실(Δ_N +2.0%p)", build(), "손실(H090-null)")
chk("남음(Δ_N = Δ_H)", build(nl=allw(900), nf=allw(525)), "남음(H090)")
chk("부분(Δ_N +20%p)", build(nl=allw(700), nf=allw(500)), "부분")
chk("Δ 비 0.80 정확(+30.0%p) → 남음", build(nl=allw(800), nf=allw(500)), "남음(H090)")
chk("Δ 비 0.7973(+29.9%p) → 부분", build(nl=allw(799), nf=allw(500)), "부분")
chk("평균 Δ_N 5.0%p 정확 → 부분", build(nl=allw(550), nf=allw(500)), "부분")
chk("평균 Δ_N 4.9%p → 손실", build(nl=allw(549), nf=allw(500)), "손실(H090-null)")
# 시험 구성 수정(첫 판: 정수 키를 dict(**) 로 넘겨 TypeError — 판정 코드와 무관)
chk("남음 크기지만 양수 13/16 → 부분", build(nl={**allw(1000), 78: 400, 79: 400, 80: 400}, nf=allw(500)), "부분")
chk("남음 크기·양수 14/16 → 남음", build(nl={**allw(1000), 78: 400, 79: 400}, nf=allw(500)), "남음(H090)")
F = "보류(조작검증 실패)"
chk("MDV 발달 불일치(w85)", build(over={("dv", 85): lambda d: d.update(same=False)}), F)
chk("MLD 장치 대조 차 1e-3", build(over={("N", "learn", 80, 601): lambda d: d.update(dmax=1e-3)}), F)
chk("MLD 후보 수 다름", build(over={("N", "frozen", 81, 600): lambda d: d.update(nc=19999)}), F)
chk("결측(N frozen w93 t601)", build(drop=("N", "frozen", 93, 601)), "결측")
chk("결측(E136 learn w78 t600)", build(drop=("H", "learn", 78, 600)), "결측")

with tempfile.TemporaryDirectory() as td:
    for d in (("logs", "E167"), ("logs", "E136"), ("traces", "E167"), ("traces", "E136")):
        os.makedirs(os.path.join(td, *d))
    w_ = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
    arr = dict(a=np.arange(400), b=np.arange(400) + 50, e=np.arange(400) % 100, i=(np.arange(400) + 50) % 100)
    np.savez_compressed(os.path.join(td, "traces", "E136", "dev_corr_w78.npz"), **arr)
    np.savez_compressed(os.path.join(td, "traces", "E167", "dev_corr_w78.npz"), cpre=np.zeros(20000, int), cpost=np.zeros(20000, int),
                        wc=np.zeros(20000), ipre=np.zeros(20000, int), ipost=np.zeros(20000, int), wi=np.zeros(20000), **arr)
    w_(("logs", "E167", "N_learn_w78_t600.log"), "[KC망안] /x/dev_corr_w78.npz | 흥분 후보 20000개 평균 0.0712 같은 위치 몫 0.0270 | 억제 후보 20000개 크기 평균 0.0578 "
       "같은 위치 몫 0.0580 | 균등 0.0200 | 장치 대조 최대 |차| 0.00e+00\n=> SDLAB diff=cyclic rule=samediff mode=learn seed=78 trialseed=600 train_accL=60.0 train_accR=50.0 "
       "train_lbal=55.0 novel_accL=52.0 novel_accR=50.0 novel_lbal=51.0\n")
    w_(("logs", "E136", "corr_learn_w78_t600.log"), "[KC불러옴] /x | 일치형 같은 위치 317/400 | 불일치형 같은 위치 334/400\n=> SDLAB diff=cyclic rule=samediff mode=learn "
       "seed=78 trialseed=600 train_accL=100.0 train_accR=69.0 train_lbal=84.5 novel_accL=100.0 novel_accR=78.3 novel_lbal=89.1\n")
    w_(("logs", "E136", "corr_frozen_w78_t601.log"), "=> SDLAB diff=cyclic rule=samediff mode=learn seed=78 trialseed=601 train_lbal=50.0 novel_lbal=50.0\n")
    J.EXP = td
    X = J.load()
g = (X[("dv", 78)] == {"same": True, "nc": 20000, "ni": 20000}
     and X[("N", "learn", 78, 600)] == {"train": 550, "novel": 510, "nc": 20000, "ni": 20000, "dmax": 0.0, "share_c": 0.027, "share_i": 0.058, "uni": 0.02}
     and X[("H", "learn", 78, 600)] == {"train": 845, "novel": 891}
     and ("H", "frozen", 78, 601) not in X)      # 모드가 파일 이름과 다르면(learn 줄) 읽지 않는다
ok_all &= g
print("%-44s → %s" % ("원 로그 파싱(실제 줄 형식·모드 대조)", "✓" if g else "✗ %s" % X))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
