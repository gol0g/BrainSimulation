#!/usr/bin/env python3
"""E120 판정 — E120.md 4절. 15런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): 변조폭 = mean(조향|good=우) − mean(조향|good=좌), 이식 평가. 양수 = 반사(같은 쪽), 음수 = 정답(교차).
효과 e(n) = 학습 n 에피소드 사후 변조폭 − 무학습 사후 변조폭(같은 뇌, 반사 0). n=5 는 E119 반사 0 칸.
누적 증분 d = e(15) − e(5) (음수 = 학습량이 늘수록 정답 쪽으로 더 감).
"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
MIN_EFFECT = 0.10      # 최소 의미 효과(E119와 같음)
NULL_BAND = 0.03       # 관측 최대 재실행 흔들림
TR19 = re.compile(r"^\s*rw0 (learn|nolearn) b(\d+): => ([-+0-9.]+) ([-+0-9.]+) \|.*?변조폭 변화 ([-+0-9.]+)")
TR20 = re.compile(r"^\s*n(\d+) (learn|nolearn) b(\d+): => ([-+0-9.]+) ([-+0-9.]+) \|.*?변조폭 변화 ([-+0-9.]+)")
KEYS = [(n, c) for n, c in ((5, "learn"), (5, "nolearn"), (10, "learn"), (15, "learn"), (15, "nolearn"))]


def judge(R, REW):
    """R: {(n, cond, b): (pre, post, dmod)}, REW: {(n, b): (ep_done, rewards)} (학습 런).
    반환: (측정확인 줄 목록, 결과 dict 또는 None)."""
    missing = [(n, c, b) for n, c in KEYS for b in BRAINS if (n, c, b) not in R]
    missing += [("rew", n, b) for n in (5, 10, 15) for b in BRAINS if (n, b) not in REW]
    if missing:
        return ["[측정 확인] 결측 %d: %s — **판정 보류, 수치 미출력**" % (len(missing), " ".join(str(k) for k in missing))], None
    bad = [k for k, v in R.items() if any(x != x for x in v)]
    if bad:
        return ["[측정 확인] nan 값: %s — **판정 보류**" % bad], None
    checks, ok = [], True
    # 1. 학습량 도달: 에피소드 완료 수 = n, 보상 5 < 10 < 15
    ep_bad = [(n, b) for n in (5, 10, 15) for b in BRAINS if REW[(n, b)][0] != n]
    inc = [b for b in BRAINS if not (REW[(5, b)][1] < REW[(10, b)][1] < REW[(15, b)][1])]
    checks.append("[측정 확인] 에피소드 수 = 설정: %d/15%s | 보상 횟수 5<10<15: %d/5%s"
                  % (15 - len(ep_bad), "" if not ep_bad else " ← %s" % ep_bad, 5 - len(inc), "" if not inc else " ← 뇌 %s" % inc))
    ok &= not ep_bad and not inc
    # 2. 회귀: 같은 뇌 사전 변조폭이 E119(n=5)와 같다
    pre_bad = [(n, c, b) for n, c in KEYS[2:] for b in BRAINS if R[(n, c, b)][0] != R[(5, "learn", b)][0]]
    checks.append("[측정 확인] 사전 변조폭 = E119 같은 뇌: %d/15%s" % (15 - len(pre_bad), "" if not pre_bad else " ← %s" % pre_bad))
    ok &= not pre_bad
    # 3. 무학습 0 변화, 무학습 15 사후 = 무학습 5 사후
    z = [b for b in BRAINS if R[(15, "nolearn", b)][2] != 0.0 or R[(15, "nolearn", b)][1] != R[(5, "nolearn", b)][1]]
    checks.append("[측정 확인] 무학습15 변화 0.0000·사후 = 무학습5 사후: %d/5%s" % (5 - len(z), "" if not z else " ← 뇌 %s" % z))
    ok &= not z
    # 4. 학습량마다 사후가 다르다
    same = [(b, a, c) for b in BRAINS for a, c in ((5, 10), (10, 15)) if R[(a, "learn", b)][1] == R[(c, "learn", b)][1]]
    checks.append("[측정 확인] 학습 사후 5≠10≠15: %d/10%s" % (10 - len(same), "" if not same else " ← 같은 값(조작 무효 의심) %s" % same))
    ok &= not same
    base = {b: R[(15, "nolearn", b)][1] for b in BRAINS}
    eff = {n: [round(R[(n, "learn", b)][1] - base[b], 4) for b in BRAINS] for n in (5, 10, 15)}
    d = [round(eff[15][i] - eff[5][i], 4) for i in range(5)]
    n_reach = sum(x <= -MIN_EFFECT for x in eff[15])
    n_acc = sum(x <= -NULL_BAND for x in d)
    n_sat = sum(abs(x) < NULL_BAND for x in d)
    n_drift = sum(x >= NULL_BAND for x in d)
    n_mono = sum(eff[5][i] > eff[10][i] > eff[15][i] for i in range(5))
    if not ok:
        verdict = "보류(조작검증 실패 — 측정부터)"
    elif n_reach >= 4 and n_acc >= 4:
        verdict = "A 누적·도달 — H045 지지(학습량 조건부)"
    elif n_acc >= 4:
        verdict = "C 누적·미도달 — 학습은 누적되나 15 에피소드에서 기준 0.10 미달"
    elif n_sat >= 4:
        verdict = "B 포화 — 학습량을 3배로 늘려도 효과 변화 < 0.03"
    elif n_drift >= 4:
        verdict = "drift — 학습량이 늘수록 반사(같은 쪽) 방향으로 감"
    else:
        verdict = "보류(혼재)"
    return checks, {"eff": eff, "d": d, "n_reach": n_reach, "n_acc": n_acc, "n_sat": n_sat,
                    "n_drift": n_drift, "n_mono": n_mono, "ok": ok, "verdict": verdict}


def report(checks, res, REW=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for n in (5, 10, 15):
        print("효과 e(%2d)(학습−무학습 사후 변조폭, 음수=교차): %s | 평균 %+.4f"
              % (n, " ".join("b%d %+.4f" % (b, x) for b, x in zip(BRAINS, res["eff"][n])), sum(res["eff"][n]) / 5))
    print("누적 증분 d = e(15)−e(5) (음수=정답 쪽으로 누적): %s | 평균 %+.4f"
          % (" ".join("b%d %+.4f" % (b, x) for b, x in zip(BRAINS, res["d"])), sum(res["d"]) / 5))
    print("e(15) ≤ −%.2f: %d/5 | d ≤ −%.2f: %d/5 | |d| < %.2f: %d/5 | d ≥ +%.2f: %d/5 | 단조 e5>e10>e15: %d/5"
          % (MIN_EFFECT, res["n_reach"], NULL_BAND, res["n_acc"], NULL_BAND, res["n_sat"], NULL_BAND, res["n_drift"], res["n_mono"]))
    if REW:
        print("보상 횟수(5/10/15 에피소드): %s" % " ".join("b%d %d/%d/%d" % (b, REW[(5, b)][1], REW[(10, b)][1], REW[(15, b)][1]) for b in BRAINS))
    print("판정: %s" % res["verdict"])


RW = re.compile(r"^\[학습\] (\d+)ep 완료, 보상 (\d+)회", re.M)


def load():
    R, REW = {}, {}
    for fn, rx, is19 in (("E119.log", TR19, True), ("E120.log", TR20, False)):
        try:
            for ln in open(os.path.join(EXP, fn), encoding="utf-8"):
                m = rx.match(ln)
                if not m:
                    continue
                g = m.groups()
                if is19:
                    R[(5, g[0], int(g[1]))] = tuple(float(x) for x in g[2:5])
                else:
                    R[(int(g[0]), g[1], int(g[2]))] = tuple(float(x) for x in g[3:6])
        except FileNotFoundError:
            pass
    for b in BRAINS:
        for n, path in ((5, "logs/E119/rw0_learn_b%d.log" % b), (10, "logs/E120/n10_learn_b%d.log" % b), (15, "logs/E120/n15_learn_b%d.log" % b)):
            try:
                m = RW.search(open(os.path.join(EXP, path), encoding="utf-8").read())
                if m:
                    REW[(n, b)] = (int(m.group(1)), int(m.group(2)))
            except FileNotFoundError:
                pass
    return R, REW


if __name__ == "__main__":
    R, REW = load()
    checks, res = judge(R, REW)
    report(checks, res, REW if res else None)
    sys.exit(0)
