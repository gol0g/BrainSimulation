#!/usr/bin/env python3
"""exposure_schedule.py 합성 정답 시험: 기본 = E160·E161 고정 순환, 무작위 제시 수·강도 범위·순환 일치율·재현성·전역 난수 불변, 잘못된 인자.
실행: python3 scripts/test_exposure_schedule.py (저장소 루트에서)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "genesis"))
import exposure_schedule as X

ok_all = True


def chk(name, cond, info=""):
    global ok_all
    ok_all &= bool(cond)
    print("%-44s %s %s" % (name, "✓" if cond else "✗", info))


# 1. 기본 = 이전 kcdevoja 순서(seq[rep_i % 4], 4n 회, 강도 0.9)
seq_old = (("good", "left", "l"), ("bad", "left", "l"), ("good", "right", "r"), ("bad", "right", "r"))
d = X.build(100)
chk("기본 = 고정 순환 400회", len(d) == 400 and all(d[i][:3] == seq_old[i % 4] and d[i][3] == 0.9 for i in range(400)))
s = X.summary(d)
chk("기본 요약: 각 100·순환 일치 1.0·강도 0.9", all(s[k] == 100 for k in X.KEYS) and s["cyc_match"] == 1.0 and s["int_min"] == s["int_max"] == 0.9, s)
# 2. 무작위 균등 + 강도 변이
r1 = X.build(100, "random", 1, 0.5, 0.9, seed=16)
s1 = X.summary(r1)
chk("NR 제시 수 각 100", all(s1[k] == 100 for k in X.KEYS) and s1["n"] == 400, s1)
chk("NR 강도 [0.5, 0.9]·평균 0.7±0.03", s1["int_min"] >= 0.5 and s1["int_max"] <= 0.9 and abs(s1["int_mean"] - 0.7) <= 0.03,
    "%.4f %.4f %.4f" % (s1["int_min"], s1["int_max"], s1["int_mean"]))
chk("NR 순환 일치율 ≤ 0.40", s1["cyc_match"] <= 0.40, "%.4f" % s1["cyc_match"])
chk("NR 같은 시드 재현", X.build(100, "random", 1, 0.5, 0.9, seed=16) == r1)
chk("NR 다른 시드 다름", X.build(100, "random", 1, 0.5, 0.9, seed=17) != r1)
# 3. 무작위 불균등(bad 세 배)
r3 = X.build(100, "random", 3, 0.5, 0.9, seed=16)
s3 = X.summary(r3)
chk("NU 제시 수 good 100·bad 300", s3["good_l"] == s3["good_r"] == 100 and s3["bad_l"] == s3["bad_r"] == 300 and s3["n"] == 800, s3)
chk("NU 순환 일치율 ≤ 0.40(기대 ≈0.19)", s3["cyc_match"] <= 0.40, "%.4f" % s3["cyc_match"])
chk("NU 강도 범위", s3["int_min"] >= 0.5 and s3["int_max"] <= 0.9 and abs(s3["int_mean"] - 0.7) <= 0.03, "%.4f" % s3["int_mean"])
# 4. 전역 난수 불변
np.random.seed(5); a = np.random.random()
np.random.seed(5); X.build(100, "random", 3, 0.5, 0.9, seed=99); b = np.random.random()
chk("전역 난수 소비 없음", a == b)
# 5. 무작위 순서·고정 강도(강도만 고정해도 순서는 섞임)
r4 = X.build(100, "random", 1, 0.9, 0.9, seed=16)
chk("무작위·고정 강도: 같은 순서(r1 과)", [x[:3] for x in r4] == [x[:3] for x in r1] and all(x[3] == 0.9 for x in r4))
# 6. 잘못된 인자
for nm, kw in (("cyclic + bad_mult 3", dict(order="cyclic", bad_mult=3)), ("lo > hi", dict(int_lo=0.9, int_hi=0.5)),
               ("order 오타", dict(order="rand")), ("n 0", dict(n=0))):
    try:
        X.build(**dict(dict(n=100), **kw))
        chk("거부: " + nm, False)
    except ValueError:
        chk("거부: " + nm, True)
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
