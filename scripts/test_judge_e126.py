#!/usr/bin/env python3
"""judge_e126 합성 시험(조건 1) + 실제 줄 파싱(SDLAB cyclic 형식)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e126 as J


def rec(tb, nb, ts=None, ns=None):
    ts = tb if ts is None else ts; ns = nb if ns is None else ns
    return {"train_same": ts, "train_diff": 2 * tb - ts, "train_bal": tb, "novel_same": ns, "novel_diff": 2 * nb - ns, "novel_bal": nb}


def make(train_l, novel_l, novel_f, same=()):
    R = {}
    for i, w in enumerate(J.WIRES):
        for j, t in enumerate(J.TSEEDS):
            R[("learn", w, t)] = rec(train_l[2 * i + j], novel_l[2 * i + j])
        R[("frozen", w, 600)] = rec(50.0, novel_f[i])
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

case("획득+전이", make([95.0] * 16, [85.0] * 16, [50.0] * 8), "전이 지지")
case("전이 경계 75·12/16·frozen 2/8", make([80.0] * 16, [75.0] * 12 + [74.5] * 4, [75.0] * 2 + [50.0] * 6), "전이 지지")
case("전이 없음", make([95.0] * 16, [52.0] * 16, [50.0] * 8), "전이 없음")
case("전이 없음 경계 59.5 12/16", make([95.0] * 16, [59.5] * 12 + [70.0] * 4, [50.0] * 8), "전이 없음")
case("획득 혼재 11/16", make([95.0] * 11 + [70.0] * 5, [85.0] * 16, [50.0] * 8), "획득 혼재")
case("획득 불가 4/16", make([95.0] * 4 + [45.0] * 12, [50.0] * 16, [50.0] * 8), "획득 불가")
case("획득 5/16 → 혼재", make([95.0] * 5 + [45.0] * 11, [50.0] * 16, [50.0] * 8), "획득 혼재")
case("혼재", make([95.0] * 16, [70.0] * 16, [50.0] * 8), "보류(혼재")
case("frozen 과다", make([95.0] * 16, [85.0] * 16, [80.0] * 3 + [50.0] * 5), "보류(혼재")
case("learn=frozen", make([95.0] * 16, [85.0] * 16, [50.0] * 8, same=(14,)), "조작검증")
R = make([95.0] * 16, [85.0] * 16, [50.0] * 8); del R[("frozen", 13, 600)]
case("결측", R, "결측")
R = make([95.0] * 16, [85.0] * 16, [50.0] * 8); R[("learn", 11, 601)]["novel_bal"] = float("nan")
case("nan", R, "nan")
ln = "  cy learn w18 t600: => SDLAB diff=cyclic rule=samediff mode=learn seed=18 trialseed=600 train_accL=90.0 train_accR=80.0 train_lbal=85.0 novel_accL=40.0 novel_accR=70.0 novel_lbal=55.0"
p = J.parse_line(ln)
ok = p == (("learn", 18, 600), {"train_same": 90.0, "train_diff": 80.0, "train_bal": 85.0, "novel_same": 40.0, "novel_diff": 70.0, "novel_bal": 55.0})
cases.append(ok); print("%s 실제 줄 형식 파싱 → %s" % ("PASS" if ok else "FAIL", p))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
