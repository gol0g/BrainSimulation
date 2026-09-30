#!/usr/bin/env python3
"""judge_e133 합성 시험(조건 1) + 실제 DEVHEBB 줄(보정 로그) 파싱 + [KC불러옴] 형식.
1차 11/13: 시험 입력의 learn·frozen 훈련 값이 같아 새 항목 50=50 인 런이 learn=frozen 조작검증에 걸림 — 시험 입력 결함, 판정 코드 불변."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e133 as J

W = J.WIRES; TS = J.TSEEDS


def make(c, i, f, csf=0.92, isf=0.02, lm_off=0, same=None):
    D, T = {}, {}
    for w in W:
        D[("corr", w)] = {"seed": w, "env": "corr", "fm": 0.28, "fx": 0.04, "mm": csf, "mx": csf}
        D[("indep", w)] = {"seed": w, "env": "indep", "fm": 0.25, "fx": 0.22, "mm": isf, "mx": isf}
    for k, w in enumerate(W):
        for j, t in enumerate(TS):
            for (e, m_, arr) in (("corr", "learn", c), ("indep", "learn", i), ("corr", "frozen", f)):
                sf = csf if e == "corr" else isf
                T[(e, m_, w, t)] = {"file": ".../dev_%s_w%d.npz" % (e, w), "lm": round(sf * 400) + lm_off, "nm": 400, "lx": round(sf * 400), "nx": 400,
                                    "tl": (80.0 if m_ == "learn" else 60.0) + j, "nl": arr[k]}
    if same:
        T[("corr", "frozen") + same] = dict(T[("corr", "learn") + same])
    return D, T


cases = []
def case(name, DT, want):
    c, r = J.judge(*DT)
    v = r["verdict"] if r else c[0]
    ok = want in v
    cases.append(ok)
    print("%s %-34s → %s" % ("PASS" if ok else "FAIL", name, v))

case("지지", make([85.0] * 16, [55.0] * 16, [52.0] * 16), "지지")
# 배선 13/16 양수 → p=2*P(X≥13|16)=0.0213 → 보류
case("13/16 → 보류", make([85.0] * 13 + [50.0] * 3, [55.0] * 13 + [60.0] * 3, [50.0] * 16), "보류(혼재")
# 14/16 → p=0.0042 → 지지(평균 ≥10)
case("14/16 → 지지", make([85.0] * 14 + [50.0] * 2, [55.0] * 14 + [60.0] * 2, [50.0] * 16), "지지")
case("효과 없음", make([57.0] * 16, [55.0] * 16, [50.0] * 16), "효과 없음")
case("학습 기여 없음", make([85.0] * 16, [55.0] * 16, [84.0] * 16), "보류(혼재")
case("조작: corr 형성 중앙값 0.3", make([85.0] * 16, [55.0] * 16, [52.0] * 16, csf=0.3, isf=0.02), "조작검증")
case("조작: corr<5×indep", make([85.0] * 16, [55.0] * 16, [52.0] * 16, csf=0.45, isf=0.1), "조작검증")
case("조작: 과제 연결 ≠ 발달(2개 차이)", make([85.0] * 16, [55.0] * 16, [52.0] * 16, lm_off=2), "조작검증")
case("반올림 오차 허용(0 차이)", make([85.0] * 16, [55.0] * 16, [52.0] * 16, lm_off=0), "지지")
case("조작: learn=frozen", make([85.0] * 16, [55.0] * 16, [52.0] * 16, same=(50, 601)), "조작검증")
D, T = make([85.0] * 16, [55.0] * 16, [52.0] * 16); del T[("indep", "learn", 55, 600)]
case("결측", (D, T), "결측")
# 실제 DEVHEBB 줄(보정 로그) + 러너 태그
real = open("research/experiments/logs/E133/calib_summary.out", encoding="utf-8").read().split("\n")
dl = [x for x in real if "seed=18 env=corr" in x and "w_fix=4.00" in x][0].strip()
ln = "  hb dev corr w18: => DEVHEBB " + dl + " → x.npz"
m = J.TD.match(ln)
ok = m is not None and m.group(4) == "corr" and float(m.group(7)) == 0.922 and float(m.group(8)) == 0.920
cases.append(ok); print("%s 실제 DEVHEBB 줄 파싱 → %s" % ("PASS" if ok else "FAIL", m.groups() if m else None))
ln2 = ("  hb corr learn w18 t600: => [KC불러옴] /mnt/c/x/traces/E133/dev_corr_w18.npz | 일치형 같은 위치 369/400 | 불일치형 같은 위치 368/400 || "
       "=> SDLAB diff=cyclic rule=samediff mode=learn seed=18 trialseed=600 train_accL=90.0 train_accR=80.0 train_lbal=85.0 novel_accL=80.0 novel_accR=70.0 novel_lbal=75.0")
m2 = J.TT.match(ln2)
ok = m2 is not None and m2.group(6) == "369" and m2.group(11) == "75.0"
cases.append(ok); print("%s [KC불러옴]+SDLAB 형식 파싱" % ("PASS" if ok else "FAIL"))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
