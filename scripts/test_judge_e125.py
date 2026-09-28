#!/usr/bin/env python3
"""judge_e125 합성 시험(조건 1) + 실제 줄 파싱 + E124 samediff 로드.
1차 10/11: 경계 케이스 frozen 80 이 learn 80 과 튜플까지 같아 조작검증(learn=frozen)에 걸림 — 시험 입력 결함, 판정 코드 불변(frozen 81 로 교체)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e125 as J

KL = [("learn", w, t) for w in J.WIRES for t in J.TSEEDS]
KF = [("frozen", w, 600) for w in J.WIRES]


def make(h1_learn, h1_frozen, sd=50.0, same=()):
    H, S = {}, {}
    for i, k in enumerate(KL):
        v = h1_learn[i]; H[k] = (v, v, v); S[k] = sd
    for i, k in enumerate(KF):
        v = h1_frozen[i]; H[k] = (v, v, v); S[k] = sd
        if k[1] in same:
            H[k] = H[("learn", k[1], 600)]
    return H, S


cases = []
def case(name, HS, want):
    c, r = J.judge(*HS)
    v = r["verdict"] if r else c[0]
    ok = want in v
    cases.append(ok)
    print("%s %-30s → %s" % ("PASS" if ok else "FAIL", name, v))

case("관계 특이(16/16)", make([95.0] * 16, [50.0] * 8), "관계 특이")
case("경계 12/16·frozen 2/8", make([80.0] * 12 + [79.0] * 4, [81.0] * 2 + [50.0] * 6), "관계 특이")
case("관계 무관(4/16)", make([90.0] * 4 + [55.0] * 12, [50.0] * 8), "관계 무관")
case("혼재(8/16)", make([90.0] * 8 + [55.0] * 8, [50.0] * 8), "보류(혼재")
case("frozen 과다(3/8)", make([95.0] * 16, [85.0] * 3 + [50.0] * 5), "보류(혼재")
case("learn=frozen", make([95.0] * 16, [50.0] * 8, same=(13,)), "조작검증")
H, S = make([95.0] * 16, [50.0] * 8); del H[("learn", 12, 601)]
case("결측(half1)", (H, S), "결측")
H, S = make([95.0] * 16, [50.0] * 8); del S[("frozen", 15, 600)]
case("결측(E124)", (H, S), "결측")
H, S = make([95.0] * 16, [50.0] * 8); H[("learn", 11, 600)] = (float("nan"), 1.0, 1.0)
case("nan", (H, S), "nan")
ln = "  h1 learn w18 t600: => SDLAB rule=half1 mode=learn seed=18 trialseed=600 train_accL=90.0 train_accR=70.0 train_lbal=80.0 novel_accL=50.0 novel_accR=50.0 novel_lbal=50.0"
m = J.TR25.match(ln)
ok = m is not None and (m.group(1), m.group(2), m.group(3), m.group(7), m.group(8), m.group(9)) == ("learn", "18", "600", "90.0", "70.0", "80.0")
cases.append(ok); print("%s 실제 줄 형식(SDLAB) 파싱" % ("PASS" if ok else "FAIL"))
H, S = J.load()
ok = len(S) == 24 and S[("learn", 10, 600)] == 50.0
cases.append(ok); print("%s E124 samediff 24런 로드(w10 t600 = %s)" % ("PASS" if ok else "FAIL", S.get(("learn", 10, 600))))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
