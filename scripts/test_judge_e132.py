#!/usr/bin/env python3
"""judge_e132 합성 시험(조건 1): 짝 비교·부호검정 경계·조작검증."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e132 as J

P = [(w, t) for w in J.WIRES for t in J.TSEEDS]


def make(c, i, f, cmf=0.56, imf=0.02, envbad=False, same=None):
    R = {}
    for k, (w, t) in enumerate(P):
        R[("corr", "learn", w, t)] = {"env": "corr", "mf": cmf if isinstance(cmf, float) else cmf[w], "tl": 90.0, "nl": c[k]}
        R[("indep", "learn", w, t)] = {"env": "indep" if not envbad else "corr", "mf": imf, "tl": 70.0, "nl": i[k]}
        R[("corr", "frozen", w, t)] = {"env": "corr", "mf": cmf if isinstance(cmf, float) else cmf[w], "tl": 50.0, "nl": f[k]}
        if same == (w, t):
            R[("corr", "frozen", w, t)] = dict(R[("corr", "learn", w, t)])
    return R


cases = []
def case(name, R, want):
    c, r = J.judge(R)
    v = r["verdict"] if r else c[0]
    ok = want in v
    cases.append(ok)
    print("%s %-36s → %s" % ("PASS" if ok else "FAIL", name, v))

case("지지(+30, 32/32)", make([85.0] * 32, [55.0] * 32, [55.0] * 32), "인과 효과 지지")
# 부호검정: 24/32 양수(나머지 음수) → p = 2*P(X>=24|32) ≈ 0.0070 < 0.01
case("지지 경계(24/32 양수, 평균≥10)", make([90.0] * 24 + [50.0] * 8, [55.0] * 24 + [55.5] * 8, [55.0] * 32), "인과 효과 지지")
# 23/32 → p ≈ 0.020 → 보류(0.01~0.05)
case("p 0.02 → 보류", make([90.0] * 23 + [50.0] * 9, [55.0] * 23 + [55.5] * 9, [55.0] * 32), "보류(혼재")
case("효과 없음(평균 +2)", make([57.0] * 32, [55.0] * 32, [50.0] * 32), "효과 없음")
case("학습 기여 없음(frozen 같음 수준)", make([85.0] * 32, [55.0] * 32, [84.0] * 32), "보류(혼재")
case("조작: 중앙값 0.35", make([85.0] * 32, [55.0] * 32, [55.0] * 32, cmf=0.35), "조작검증")
cm = {w: 0.56 for w in J.WIRES}; cm[33] = 0.08
case("조작: 한 배선 corr<5×indep", make([85.0] * 32, [55.0] * 32, [55.0] * 32, cmf=cm), "조작검증")
case("조작: env 표기", make([85.0] * 32, [55.0] * 32, [55.0] * 32, envbad=True), "조작검증")
case("조작: learn=frozen", make([85.0] * 32, [55.0] * 32, [55.0] * 32, same=(35, 601)), "조작검증")
R = make([85.0] * 32, [55.0] * 32, [55.0] * 32); del R[("corr", "frozen", 44, 600)]
case("결측", R, "결측")
p, n, nz = J.sign_p([1] * 24 + [-1] * 8)
ok = abs(p - 0.0070) < 0.0005 and n == 24 and nz == 32
cases.append(ok); print("%s 부호검정 24/32 p=%.5f" % ("PASS" if ok else "FAIL", p))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
