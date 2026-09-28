#!/usr/bin/env python3
"""judge_e120.judge() 합성 시험 — 답을 아는 입력으로 판정 규칙이 E120.md 4절과 같은지 확인(조건 1).
1차(19:17 전후, 로그 미보존) 11/15: C·B·drift·혼재 입력이 e(10)=e(5)로 만들어져 조작검증 "학습 사후 5≠10"에 걸림 — 시험 입력 결함, 판정 코드 불변.
2차 14/15: C 입력의 d가 −0.03을 넘지 않았음(계산 착오) — 입력 교체. 실제 e(5)에서는 C가 사실상 불가(E120.md 4절 주)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e120 as J

B = J.BRAINS
PRE = {b: 0.02 for b in B}
BASE = {b: 0.02 for b in B}


def make(e5, e10, e15, rew=(300, 600, 900), eps=(5, 10, 15), pre15=None, nl15=None, nl15d=0.0):
    R, REW = {}, {}
    for i, b in enumerate(B):
        R[(5, "nolearn", b)] = (PRE[b], BASE[b], 0.0)
        R[(15, "nolearn", b)] = (PRE[b], BASE[b] if nl15 is None else nl15, nl15d)
        for n, e in ((5, e5), (10, e10), (15, e15)):
            post = round(BASE[b] + e[i], 4)
            R[(n, "learn", b)] = (PRE[b] if (pre15 is None or n != 15) else pre15, post, round(post - PRE[b], 4))
        for k, n in enumerate((5, 10, 15)):
            REW[(n, b)] = (eps[k], rew[k])
    return R, REW


def V(*a, **k):
    return J.judge(*make(*a, **k))


cases = []
def case(name, got, want):
    ok = want in (got[1]["verdict"] if got[1] else "결측")
    cases.append(ok)
    print("%s %-40s → %s" % ("PASS" if ok else "FAIL", name, got[1]["verdict"] if got[1] else got[0][0]))

E5 = [-0.1033, -0.0724, -0.0690, -0.0809, -0.0607]
case("A 누적·도달", V(E5, [x - 0.03 for x in E5], [x - 0.06 for x in E5]), "A 누적")
case("C 누적·미도달(e5 -0.05 → e15 -0.085)", V([-0.05] * 5, [-0.07] * 5, [-0.085] * 5), "C 누적")
case("B 포화(변화 +-0.01)", V(E5, [x + 0.005 for x in E5], [x + 0.01 for x in E5]), "B 포화")
case("drift", V(E5, [x + 0.02 for x in E5], [x + 0.05 for x in E5]), "drift")
case("혼재", V(E5, [x - 0.01 for x in E5], [-0.20, -0.20, -0.06, -0.05, -0.04]), "보류(혼재)")
# 경계: d = -0.03 정확히(누적 인정), e15 = -0.10 정확히(도달 인정) — 부동소수 경계
case("경계 d=-0.03·e15=-0.10", V([-0.07] * 5, [-0.085] * 5, [-0.10] * 5), "A 누적")
# 경계 바로 안쪽: d = -0.0299 → 누적 아님, |d|<0.03 → 포화
case("경계 d=-0.0299 → 포화", V([-0.07] * 5, [-0.08] * 5, [-0.0999] * 5), "B 포화")
# 조작검증 실패들
case("같은 값(10=5)", V(E5, E5, [x - 0.06 for x in E5]), "보류(조작검증")
case("보상 증가 안 함", V(E5, [x - 0.03 for x in E5], [x - 0.06 for x in E5], rew=(300, 300, 900)), "보류(조작검증")
case("에피소드 수 불일치", V(E5, [x - 0.03 for x in E5], [x - 0.06 for x in E5], eps=(5, 10, 10)), "보류(조작검증")
case("사전 불일치(회귀 실패)", V(E5, [x - 0.03 for x in E5], [x - 0.06 for x in E5], pre15=0.03), "보류(조작검증")
case("무학습15 변화≠0", V(E5, [x - 0.03 for x in E5], [x - 0.06 for x in E5], nl15d=0.001), "보류(조작검증")
case("무학습15 사후≠무학습5", V(E5, [x - 0.03 for x in E5], [x - 0.06 for x in E5], nl15=0.021), "보류(조작검증")
# 결측
R, REW = make(E5, [x - 0.03 for x in E5], [x - 0.06 for x in E5]); del R[(15, "learn", 12)]
g = J.judge(R, REW); ok = g[1] is None and "결측" in g[0][0]; cases.append(ok)
print("%s %-40s → %s" % ("PASS" if ok else "FAIL", "결측 1런", g[0][0][:60]))
R, REW = make(E5, [x - 0.03 for x in E5], [x - 0.06 for x in E5]); R[(10, "learn", 11)] = (0.02, float("nan"), 0.0)
g = J.judge(R, REW); ok = g[1] is None and "nan" in g[0][0]; cases.append(ok)
print("%s %-40s → %s" % ("PASS" if ok else "FAIL", "nan 값", g[0][0][:60]))
print("합성 시험 %d/%d 통과" % (sum(cases), len(cases)))
sys.exit(0 if all(cases) else 1)
