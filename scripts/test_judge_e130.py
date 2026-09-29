#!/usr/bin/env python3
"""judge_e130 합성 시험(조건 1) + 실제 줄 형식 파싱."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e130 as J


def make(nc, ni, nf, cmf=0.56, imf=0.02, same_w=None, envbad=False):
    R = {}
    for i, w in enumerate(J.WIRES):
        for j, t in enumerate(J.TSEEDS):
            R[("corr", "learn", w, t)] = {"env": "corr", "mf": cmf, "xf": 0.5, "tl": 90.0, "nl": nc[2 * i + j]}
            R[("indep", "learn", w, t)] = {"env": "indep" if not envbad else "corr", "mf": imf, "xf": 0.02, "tl": 85.0, "nl": ni[2 * i + j]}
        R[("corr", "frozen", w, 600)] = {"env": "corr", "mf": cmf, "xf": 0.5, "tl": 50.0, "nl": nf[i]}
        if same_w == w:
            R[("corr", "frozen", w, 600)] = dict(R[("corr", "learn", w, 600)])
    return R


cases = []
def case(name, R, want):
    c, r = J.judge(R)
    v = r["verdict"] if r else c[0]
    ok = want in v
    cases.append(ok)
    print("%s %-30s → %s" % ("PASS" if ok else "FAIL", name, v))

case("경험 형성 지지", make([90.0] * 16, [55.0] * 16, [50.0] * 8), "경험 형성 지지")
case("경계 12/16·indep 4/16·frozen 2/8", make([75.0] * 12 + [74.0] * 4, [80.0] * 4 + [50.0] * 12, [76.0] * 2 + [50.0] * 6), "경험 형성 지지")
case("indep 5/16 → 보류", make([90.0] * 16, [80.0] * 5 + [50.0] * 11, [50.0] * 8), "보류(혼재")
case("형성 부족", make([80.0] * 4 + [55.0] * 12, [55.0] * 16, [50.0] * 8), "형성 부족")
case("corr mf 낮음 → 조작검증", make([90.0] * 16, [55.0] * 16, [50.0] * 8, cmf=0.3), "조작검증")
case("indep mf 높음 → 조작검증", make([90.0] * 16, [55.0] * 16, [50.0] * 8, imf=0.1), "조작검증")
case("env 표기 불일치", make([90.0] * 16, [55.0] * 16, [50.0] * 8, envbad=True), "조작검증")
case("learn=frozen", make([90.0] * 16, [55.0] * 16, [50.0] * 8, same_w=12), "조작검증")
R = make([90.0] * 16, [55.0] * 16, [50.0] * 8); del R[("indep", "learn", 14, 601)]
case("결측", R, "결측")
ln = ("  dv corr learn w18 t600: => [KC발달] env=corr rounds=1000 exposures=200 theta=0.15 items=20 | 일치형 같은 위치 225/400 | 불일치형 같은 위치 223/400 | (호스트 계산 0.562/0.557) || "
      "=> SDLAB diff=cyclic rule=samediff mode=learn seed=18 trialseed=600 train_accL=90.0 train_accR=80.0 train_lbal=85.0 novel_accL=80.0 novel_accR=70.0 novel_lbal=75.0")
p = J.parse(ln)
ok = p is not None and p[0] == ("corr", "learn", 18, 600) and abs(p[1]["mf"] - 225 / 400) < 1e-12 and p[1]["nl"] == 75.0 and p[1]["env"] == "corr"
cases.append(ok); print("%s 실제 줄 형식 파싱 → %s" % ("PASS" if ok else "FAIL", p))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
