#!/usr/bin/env python3
"""judge_e134 합성 시험(조건 1) + 경로 검사 실제 출력 줄 파싱."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e134 as J

W = J.WIRES; TS = J.TSEEDS


def make(v, idc=(0.92, 0.0), shc=(0.0, 0.94), badfile=False):
    """v: {(발달, 과제): 새 항목 값 또는 배선별 목록}"""
    D, T = {}, {}
    for w in W:
        D[("corr", w)] = {"seed": w, "env": "corr", "k": 7, "ms": idc[0], "xs": idc[0], "mk": idc[1], "xk": idc[1]}
        D[("shift", w)] = {"seed": w, "env": "shift", "k": 7, "ms": shc[0], "xs": shc[0], "mk": shc[1], "xk": shc[1]}
    for i, w in enumerate(W):
        for d in ("corr", "shift"):
            for tk in ("tid", "tsh"):
                x = v[(d, tk)]; x = x[i] if isinstance(x, list) else x
                for j, t in enumerate(TS):
                    T[(d, tk, w, t)] = {"file": ".../dev_%s_w%d.npz" % (("shift" if badfile else d), w) if not (badfile and d == "corr") else ".../dev_shift_w%d.npz" % w,
                                        "k": 7, "sh": 7 if tk == "tsh" else 0, "tl": 80.0 + j, "nl": x}
    return D, T


cases = []
def case(name, DT, want):
    c, r = J.judge(*DT)
    vv = r["verdict"] if r else c[0]
    ok = want in vv
    cases.append(ok)
    print("%s %-34s → %s" % ("PASS" if ok else "FAIL", name, vv))

base = {("corr", "tid"): 88.0, ("shift", "tid"): 50.0, ("corr", "tsh"): 50.0, ("shift", "tsh"): 88.0}
case("이중 해리", make(base), "이중 해리 지지")
one = dict(base); one[("corr", "tid")] = 52.0
case("단일 해리(shift 과제만)", make(one), "보류(단일 해리)")
none = {k: 60.0 for k in base}
case("해리 없음", make(none), "해리 없음")
# 경계: 14/16 양수(p=0.0042) → 지지
b2 = dict(base); b2[("shift", "tsh")] = [88.0] * 14 + [45.0] * 2; b2[("corr", "tsh")] = [50.0] * 14 + [50.0] * 2
case("shift 과제 14/16", make(b2), "이중 해리 지지")
b3 = dict(base); b3[("shift", "tsh")] = [88.0] * 13 + [45.0] * 3; b3[("corr", "tsh")] = 50.0
case("shift 과제 13/16(p 0.02) → 단일", make(b3), "보류(단일 해리)")
case("조작: shift 발달 형성 실패", make(base, shc=(0.0, 0.2)), "조작검증")
case("조작: identity 발달에 이동 짝", make(base, idc=(0.92, 0.1)), "조작검증")
case("조작: 파일 불일치", make(base, badfile=True), "조작검증")
D, T = make(base); del T[("shift", "tid", 70, 601)]
case("결측", (D, T), "결측")
# 실제 줄
ps = open("research/experiments/logs/E134/path_summary.out", encoding="utf-8").read()
dsl = [x.strip() for x in ps.split("\n") if x.strip().startswith("seed=18 env=shift")][0]
lg = open("research/experiments/logs/E134/path_dev_shift_w18.log", encoding="utf-8").read()
dh = re.search(r"^=> DEVHEBB .*$", lg, re.M).group(0).split(" → ")[0]
ln = "  ds dev shift w18: " + dh + " || => DEVSHIFT " + dsl
m = J.TD.match(ln)
ok = m is not None and m.group(4) == "shift" and float(m.group(8)) == 0.953
cases.append(ok); print("%s 실제 DEVHEBB+DEVSHIFT 줄 파싱 → %s" % ("PASS" if ok else "FAIL", m.groups()[2:] if m else None))
lq = open("research/experiments/logs/E134/path_loaded_shift_w18.log", encoding="utf-8").read()
k1 = re.search(r"^\[KC불러옴\] .*$", lq, re.M).group(0); k2 = re.search(r"^\[KC불러옴이동\] .*$", lq, re.M).group(0); sd = re.search(r"^=> SDLAB .*$", lq, re.M).group(0)
ln2 = "  ds shift tsh w18 t600: => " + k1 + " || " + k2 + " || " + sd
m2 = J.TT.match(ln2)
ok = m2 is not None and m2.group(6) == "7" and m2.group(7) == "7" and "path_dev_shift_w18.npz" in m2.group(5)
cases.append(ok); print("%s 실제 [KC불러옴]+이동+SDLAB 줄 파싱 → %s" % ("PASS" if ok else "FAIL", m2.groups()[4:] if m2 else None))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
