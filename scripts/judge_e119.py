#!/usr/bin/env python3
"""E119 판정 — E119.md 4절(abcd-2026-09-28 C절). 20런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): 변조폭 = mean(조향|good=우) − mean(조향|good=좌), 이식 평가. 양수 = 반사(같은 쪽), 음수 = 정답(교차).
효과 = 학습 사후 변조폭 − 무학습 사후 변조폭(같은 뇌·같은 반사 설정).
"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
RWS = (0, 25)
CONDS = ("learn", "nolearn")
MIN_EFFECT = 0.10      # 최소 의미 효과
NULL_BAND = 0.03       # 관측 최대 재실행 흔들림(E111 B8 b2)
# 요약 줄: "  rw0 learn b10: => +0.0123 +0.0456 | 정답률 ... | **변조폭 변화 +0.0333** | 판정: ..."
TR = re.compile(r"^\s*rw(\d+) (learn|nolearn) b(\d+): => ([-+0-9.]+) ([-+0-9.]+) \|.*?변조폭 변화 ([-+0-9.]+)")


def judge(R):
    """R: {(rw, cond, b): (pre, post, dmod)}. 반환: (측정확인 줄 목록, 결과 dict 또는 None)."""
    checks = []
    missing = [(rw, c, b) for rw in RWS for c in CONDS for b in BRAINS if (rw, c, b) not in R]
    if missing:
        return ["[측정 확인] 결측 %d/20: %s — **판정 보류, 수치 미출력**" % (len(missing), " ".join("rw%d/%s/b%d" % k for k in missing))], None
    bad = [k for k, v in R.items() if any(x != x for x in v)]
    if bad:
        return ["[측정 확인] nan 값: %s — **판정 보류**" % bad], None
    ok = True
    # 조작검증 1: 무학습은 가중치가 안 변하므로 사후 = 사전(변조폭 변화 정확히 0.0000)
    z = [(rw, b) for rw in RWS for b in BRAINS if R[(rw, "nolearn", b)][2] != 0.0]
    checks.append("[측정 확인] 무학습 변조폭 변화 0.0000: %d/10%s" % (10 - len(z), "" if not z else "  ← 실패 %s" % z))
    ok &= not z
    # 조작검증 2: 학습 사후 ≠ 무학습 사후(소수 4자리까지 같으면 학습 무효 의심)
    same = [(rw, b) for rw in RWS for b in BRAINS if R[(rw, "learn", b)][1] == R[(rw, "nolearn", b)][1]]
    checks.append("[측정 확인] 학습 사후 ≠ 무학습 사후: %d/10%s" % (10 - len(same), "" if not same else "  ← 같은 값(학습 무효 의심) %s" % same))
    ok &= not same
    # 조작검증 3: 반사 0 도달 — 같은 뇌 무학습 사전 변조폭이 반사 25보다 작아야(반사 제거)
    reach = [b for b in BRAINS if not (abs(R[(0, "nolearn", b)][0]) < abs(R[(25, "nolearn", b)][0]))]
    checks.append("[측정 확인] 반사 0 기준선 |변조폭| < 반사 25 기준선: %d/5%s" % (5 - len(reach), "" if not reach else "  ← 실패 뇌 %s" % reach))
    ok &= not reach
    # 로그 해상도(소수 4자리)로 반올림 — 부동소수 뺄셈이 경계(−0.10)를 흐리지 않게(−0.0877−0.0123 = −0.0999…)
    eff = {rw: [round(R[(rw, "learn", b)][1] - R[(rw, "nolearn", b)][1], 4) for b in BRAINS] for rw in RWS}
    e0 = eff[0]
    n_pos = sum(x <= -MIN_EFFECT for x in e0)          # 정답(교차) 방향 ≥ 0.10
    n_null = sum(abs(x) < NULL_BAND for x in e0)
    if not ok:
        verdict = "보류(조작검증 실패 — 측정부터)"
    elif n_pos >= 4:
        verdict = "양성 — 반대 반사가 없을 때 전체 모델은 매핑을 학습한다"
    elif n_null >= 4:
        verdict = "음성(검출 안 됨) — 반사 0에서도 ≥%.2f 학습 검출 안 됨(능력 없음 증명 아님)" % MIN_EFFECT
    else:
        verdict = "보류(혼재 또는 반사 방향)"
    n25 = sum(abs(x) < NULL_BAND for x in eff[25])
    return checks, {"eff": eff, "n_pos": n_pos, "n_null": n_null, "n25_null": n25, "ok": ok, "verdict": verdict}


def report(checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for rw in RWS:
        print("반사 %2d 효과(학습−무학습 사후 변조폭, 음수=교차): %s | 평균 %+.4f"
              % (rw, " ".join("b%d %+.4f" % (b, x) for b, x in zip(BRAINS, res["eff"][rw])), sum(res["eff"][rw]) / 5))
    print("반사 0: 효과 ≤ −%.2f %d/5, |효과| < %.2f %d/5" % (MIN_EFFECT, res["n_pos"], NULL_BAND, res["n_null"]))
    print("부지표(판정 불변) 반사 25 |효과| < %.2f: %d/5 (E118 재현 여부)" % (NULL_BAND, res["n25_null"]))
    print("판정: %s" % res["verdict"])


def load():
    R = {}
    try:
        for ln in open(os.path.join(EXP, "E119.log"), encoding="utf-8"):
            m = TR.match(ln)
            if m:
                R[(int(m.group(1)), m.group(2), int(m.group(3)))] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
    except FileNotFoundError:
        pass
    return R


if __name__ == "__main__":
    R = load()
    checks, res = judge(R)
    report(checks, res)
    if res is not None:
        # 부지표: 반사 0 평가 정답률·판정 경로(원 로그)
        acc = re.compile(r"^\[사후\] .*?정답률 ([0-9.]+)%")
        jp = re.compile(r"^\[판정경로\] .*")
        for b in BRAINS:
            p = os.path.join(EXP, "logs/E119/rw0_learn_b%d.log" % b)
            t = open(p, encoding="utf-8").read().splitlines() if os.path.exists(p) else []
            a = [acc.match(x).group(1) for x in t if acc.match(x)]
            j = [x for x in t if jp.match(x)]
            print("  rw0 b%d 사후 정답률(|v|>0.02 기준) %s%% | %s" % (b, a[0] if a else "?", j[0] if j else "[판정경로 줄 없음]"))
    sys.exit(0)
