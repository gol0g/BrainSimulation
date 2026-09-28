#!/usr/bin/env python3
"""judge_e123 합성 시험(조건 1) + 실제 줄 파싱."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e123 as J

RUNS = [(w, t) for w in J.WIRES for t in J.TSEEDS]


def make(held_by_n, dw_by_n=None, override=None):
    dw_by_n = dw_by_n or {50: 0.004, 100: 0.007, 200: 0.012, 400: 0.02}
    R = {}
    for n in J.NS:
        for i, (w, t) in enumerate(RUNS):
            h = held_by_n[n][i] if isinstance(held_by_n[n], list) else held_by_n[n]
            R[(n, w, t)] = (h, dw_by_n[n])
    if override:
        R.update(override)
    return R


cases = []
def case(name, R, want):
    c, r = J.judge(R)
    v = r["verdict"] if r else c[0]
    ok = want in v
    cases.append(ok)
    print("%s %-36s → %s" % ("PASS" if ok else "FAIL", name, v))

case("지지(55→70→85→95)", make({50: 55.0, 100: 70.0, 200: 85.0, 400: 95.0}), "지지")
case("경계 gain 15.0 정확히, 200=400", make({50: 80.0 - 0.1, 100: 85.0, 200: 94.9, 400: 94.9}), "지지")
case("gain 14.9 → 보류", make({50: 80.1, 100: 85.0, 200: 90.0, 400: 95.0}), "보류")
case("50에서 포화 → 보류(범위 밖)", make({50: 90.0, 100: 92.0, 200: 94.0, 400: 95.0}), "범위 밖")
case("기각(평탄 60)", make({50: 60.0, 100: 61.0, 200: 59.0, 400: 60.0}), "기각")
case("가중치 비단조 → 조작검증", make({50: 55.0, 100: 70.0, 200: 85.0, 400: 95.0}, {50: 0.01, 100: 0.01, 200: 0.012, 400: 0.02}), "조작검증")
R = make({50: 55.0, 100: 70.0, 200: 85.0, 400: 95.0}); del R[(100, 12, 601)]
case("결측", R, "결측")
R = make({50: 55.0, 100: 70.0, 200: 85.0, 400: 95.0}); R[(200, 11, 600)] = (float("nan"), 0.01)
case("nan", R, "nan")
ln = "  learn n50 w10 t600: => EXGEN mode=learn seed=10 trialseed=600 proto=100.0 train=95.0 d0.10=100.0 d0.20=97.0 d0.30=99.0 d0.40=80.0 | ties train:0 d0.10:0 d0.20:0 d0.30:2 d0.40:3 || dw_l=0.01982 dw_r=0.02263"
m = J.TR23.match(ln)
ok = m is not None and m.groups() == ("50", "10", "600", "97.0", "0.01982", "0.02263")
cases.append(ok); print("%s 실제 줄 파싱(E123 형식) → %s" % ("PASS" if ok else "FAIL", m.groups() if m else None))
R = J.load()
k = [x for x in R if x[0] == 400]
ok = len(k) == 16 and R[(400, 10, 600)] == (97.0, (0.01982 + 0.02263) / 2)
cases.append(ok); print("%s E122 재사용 16런 로드·원 로그 dw → %d런, w10 t600 %s" % ("PASS" if ok else "FAIL", len(k), R.get((400, 10, 600))))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
