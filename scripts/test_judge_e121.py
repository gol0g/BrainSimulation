#!/usr/bin/env python3
"""judge_e121.judge() 합성 시험 — 답을 아는 입력으로 판정 규칙이 E121.md 4절과 같은지 확인(조건 1)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e121 as J

B = J.BRAINS
E1 = [-0.1033, -0.0724, -0.0690, -0.0809, -0.0607]   # E119 실제 효과(배율 1)


def make(e0, pre0=0.01, kb=None, nl_d=0.0, same_post=False, pre1=0.02):
    R, KBW = {}, {}
    for i, b in enumerate(B):
        R[(1, "nolearn", b)] = (pre1, pre1, 0.0)
        R[(1, "learn", b)] = (pre1, round(pre1 + E1[i], 4), round(E1[i], 4))
        R[(0, "nolearn", b)] = (pre0, pre0, nl_d)
        post = pre0 if same_post else round(pre0 + e0[i], 4)
        R[(0, "learn", b)] = (pre0, post, round(post - pre0, 4))
        KBW[b] = kb if kb is not None else (0.0, 4, [0.0, 0.0, 0.0, 0.0])
    return R, KBW


cases = []
def case(name, got, want):
    v = got[1]["verdict"] if got[1] else got[0][0]
    ok = want in v
    cases.append(ok)
    print("%s %-36s → %s" % ("PASS" if ok else "FAIL", name, v))

case("양성", J.judge(*make([x - 0.05 for x in E1])), "양성")
case("음성(±0.01)", J.judge(*make([x + 0.01 for x in E1])), "음성")
case("역방향(+0.05)", J.judge(*make([x + 0.05 for x in E1])), "역방향")
case("개선·미도달(e0 −0.095 대)", J.judge(*make([-0.14, -0.105, -0.099, -0.095, -0.095])), "보류")
# 경계: Δ = −0.03 정확히, e0 = −0.10 정확히 (부동소수)
E1b = E1[:]
case("경계 Δ=−0.03·e0=−0.10", J.judge(*make([-0.1333, -0.1024, -0.10, -0.1109, -0.10])), "양성")
case("음성 경계 |Δ|=0.0299", J.judge(*make([x - 0.0299 for x in E1])), "음성")
case("공통 입력 w≠0", J.judge(*make([x - 0.05 for x in E1], kb=(0.0, 4, [0.0, 2.0, 0.0, 0.0]))), "보류(조작검증")
case("배율 1 로 돌림", J.judge(*make([x - 0.05 for x in E1], kb=(1.0, 4, [2.0] * 4))), "보류(조작검증")
case("공통 입력 집단 0개", J.judge(*make([x - 0.05 for x in E1], kb=(0.0, 0, []))), "보류(조작검증")
case("무학습 변화≠0", J.judge(*make([x - 0.05 for x in E1], nl_d=0.001)), "보류(조작검증")
case("학습=무학습", J.judge(*make([x - 0.05 for x in E1], same_post=True)), "보류(조작검증")
case("사전 같음(조작 무효)", J.judge(*make([x - 0.05 for x in E1], pre0=0.02)), "보류(조작검증")
R, K = make([x - 0.05 for x in E1]); del R[(0, "learn", 13)]
case("결측", J.judge(R, K), "결측")
R, K = make([x - 0.05 for x in E1]); R[(0, "learn", 11)] = (0.01, float("nan"), 0.0)
case("nan", J.judge(R, K), "nan")
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
