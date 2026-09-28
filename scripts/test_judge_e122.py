#!/usr/bin/env python3
"""judge_e122 합성 시험(조건 1) + 실제 경로 검사 로그 줄 파싱."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e122 as J

W, T = J.WIRES, J.TSEEDS


def make(held_learn, held_frozen, train=95.0, same=()):
    R = {}
    for i, w in enumerate(W):
        for j, t in enumerate(T):
            h = held_learn[2 * i + j]
            R[("learn", w, t)] = {"proto": 100.0, "train": train, 0.1: h, 0.2: h, 0.3: h, 0.4: h}
        f = held_frozen[i]
        R[("frozen", w, 600)] = {"proto": 50.0, "train": 50.0, 0.1: f, 0.2: f, 0.3: f, 0.4: f}
        if w in same:
            R[("frozen", w, 600)] = dict(R[("learn", w, 600)])
    return R


cases = []
def case(name, R, want):
    c, r = J.judge(R)
    v = r["verdict"] if r else c[0]
    ok = want in v
    cases.append(ok)
    print("%s %-34s → %s" % ("PASS" if ok else "FAIL", name, v))

case("지지 16/16, frozen 0/8", make([95.0] * 16, [40.0] * 8), "지지")
case("지지 경계 12/16·frozen 2/8", make([80.0] * 12 + [79.9] * 4, [80.0] * 2 + [50.0] * 6), "지지")
case("frozen 3/8 → 보류", make([95.0] * 16, [80.0] * 3 + [50.0] * 5), "보류(혼재")
case("learn 11/16 → 보류", make([90.0] * 11 + [60.0] * 5, [40.0] * 8), "보류(혼재")
case("기각 4/16, train 학습함", make([90.0] * 4 + [55.0] * 12, [40.0] * 8), "기각")
case("train 미학습 → 보류", make([55.0] * 16, [40.0] * 8, train=60.0), "보류(훈련")
case("learn=frozen → 조작검증", make([95.0] * 16, [40.0] * 8, same=(12,)), "보류(조작검증")
R = make([95.0] * 16, [40.0] * 8); del R[("learn", 13, 601)]
case("결측", R, "결측")
R = make([95.0] * 16, [40.0] * 8); R[("learn", 11, 600)][0.2] = float("nan")
case("nan", R, "nan")
# 실제 줄(경로 검사 로그 형식 + 러너 태그)
ln = "  learn w18 t600: => EXGEN mode=learn seed=18 trialseed=600 proto=100.0 train=89.0 d0.10=88.0 d0.20=94.0 d0.30=80.0 d0.40=70.0 | ties train:5 d0.10:1 d0.20:5 d0.30:0 d0.40:0"
p = J.parse_line(ln)
ok = p == (("learn", 18, 600), {"proto": 100.0, "train": 89.0, 0.1: 88.0, 0.2: 94.0, 0.3: 80.0, 0.4: 70.0})
cases.append(ok); print("%s 실제 줄 파싱 → %s" % ("PASS" if ok else "FAIL", p))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
