#!/usr/bin/env python3
"""judge_e119 합성 시험(process-request 조건 1): 성공·실패·경계·같은 값·결측·조작검증 실패. 답을 미리 정해 둔다."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e119 as J


def make(eff0, eff25=(0.0, 0.0, 0.0, 0.0, 0.0), base0=0.01, base25=0.40, nl_dmod=0.0):
    """무학습 사후 = 기준선(사전), 학습 사후 = 기준선 + 효과. 효과 0 이면 학습 사후를 1e-4 비켜 '같은 값' 검사를 피한다."""
    R = {}
    for rw, eff, base in ((0, eff0, base0), (25, eff25, base25)):
        for b, e in zip(J.BRAINS, eff):
            R[(rw, "nolearn", b)] = (base, round(base + nl_dmod, 4), nl_dmod)
            post = round(base + e, 4) if e != 0 else round(base + 0.0001, 4)
            R[(rw, "learn", b)] = (base, post, round(post - base, 4))
    return R


CASES = [
    ("성공 4/5", make((-0.2, -0.15, -0.12, -0.11, 0.01)), "양성"),
    ("음성 4/5", make((0.01, -0.02, 0.0, 0.029, 0.2)), "음성"),
    ("경계 −0.10 포함", make((-0.10, -0.10, -0.10, -0.10, 0.0)), "양성"),
    ("부동소수 경계(−0.0877−0.0123)", make((-0.10,) * 4 + (0.0,), base0=0.0123), "양성"),
    ("경계 직전 −0.0999", make((-0.0999,) * 4 + (0.0,)), "보류(혼재"),
    ("|효과| = 0.03 은 음성 아님", make((0.03, -0.03, 0.03, -0.03, 0.0)), "보류(혼재"),
    ("반사 방향 양수", make((0.2,) * 5), "보류(혼재"),
    ("무학습 변화 ≠ 0", make((-0.2,) * 5, nl_dmod=0.001), "보류(조작검증"),
    ("반사 0 미도달(기준선 ≥ 반사 25)", make((-0.2,) * 5, base0=0.45), "보류(조작검증"),
]
fails = 0
for name, R, want in CASES:
    _, res = J.judge(R)
    got = res["verdict"] if res else None
    ok = got is not None and got.startswith(want)
    fails += not ok
    print("%s %-32s → %s" % ("OK " if ok else "BAD", name, got))
# 같은 값: 학습 사후 == 무학습 사후
R = make((-0.2,) * 5)
R[(0, "learn", 10)] = R[(0, "nolearn", 10)]
_, res = J.judge(R)
ok = res["verdict"].startswith("보류(조작검증"); fails += not ok
print("%s %-32s → %s" % ("OK " if ok else "BAD", "학습 사후 = 무학습 사후", res["verdict"]))
# 결측: 한 칸 빠지면 수치 미출력
R = make((-0.2,) * 5); del R[(25, "nolearn", 14)]
chk, res = J.judge(R)
ok = res is None and "결측 1/20" in chk[0]; fails += not ok
print("%s %-32s → %s" % ("OK " if ok else "BAD", "결측 1칸", chk[0]))
# nan
R = make((-0.2,) * 5); R[(0, "learn", 12)] = (0.01, float("nan"), 0.0)
chk, res = J.judge(R)
ok = res is None and "nan" in chk[0]; fails += not ok
print("%s %-32s → %s" % ("OK " if ok else "BAD", "nan 값", chk[0][:40]))
# 파서: 실제 요약 줄 형식
ln = "  rw0 learn b10: => +0.0123 -0.0877 | 정답률 +0.0%p | **변조폭 변화 -0.1000** | 판정: 학습이 조향을 역전 방향으로 이동"
m = J.TR.match(ln)
ok = bool(m) and m.groups() == ("0", "learn", "10", "+0.0123", "-0.0877", "-0.1000"); fails += not ok
print("%s %-32s → %s" % ("OK " if ok else "BAD", "요약 줄 파서", m.groups() if m else None))
print("합성 시험: %d/%d 통과" % (len(CASES) + 4 - fails, len(CASES) + 4))
sys.exit(1 if fails else 0)
