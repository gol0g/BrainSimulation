#!/usr/bin/env python3
"""E121 판정 — E121.md 4절. 10런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): 변조폭 = mean(조향|good=우) − mean(조향|good=좌), 이식 평가. 음수 = 정답(교차).
효과 e_s(b) = 학습 사후 − 무학습 사후(같은 뇌, 반사 0, 5 에피소드), s = 좌우 공통 KC 입력 배율(1: E119, 0: E121).
맞바꾸기 차이 Δ(b) = e_0(b) − e_1(b) (음수 = 공통 입력을 빼자 정답 쪽으로 더 커짐).
"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
MIN_EFFECT = 0.10
NULL_BAND = 0.03
TR19 = re.compile(r"^\s*rw0 (learn|nolearn) b(\d+): => ([-+0-9.]+) ([-+0-9.]+) \|.*?변조폭 변화 ([-+0-9.]+)")
TR21 = re.compile(r"^\s*s0 (learn|nolearn) b(\d+): => ([-+0-9.]+) ([-+0-9.]+) \|.*?변조폭 변화 ([-+0-9.]+)")
KB = re.compile(r"^\[KC공통입력\] scale=([0-9.]+) 집단 (\d+)개: (.*)$", re.M)


def judge(R, KBW):
    """R: {(s, cond, b): (pre, post, dmod)}, KBW: {b: (scale, n_groups, [w...])} (E121 학습 런의 공통 입력 가중치)."""
    missing = [(s, c, b) for s in (0, 1) for c in ("learn", "nolearn") for b in BRAINS if (s, c, b) not in R]
    missing += [("kb", b) for b in BRAINS if b not in KBW]
    if missing:
        return ["[측정 확인] 결측 %d: %s — **판정 보류, 수치 미출력**" % (len(missing), " ".join(str(k) for k in missing))], None
    bad = [k for k, v in R.items() if any(x != x for x in v)]
    if bad:
        return ["[측정 확인] nan 값: %s — **판정 보류**" % bad], None
    checks, ok = [], True
    # 1. 조작 도달: 배율 0 런의 공통 입력 가중치가 전부 0, 집단 ≥1
    kb_bad = [b for b in BRAINS if not (KBW[b][0] == 0.0 and KBW[b][1] >= 1 and all(w == 0.0 for w in KBW[b][2]))]
    checks.append("[측정 확인] 배율 0 공통 입력 w=0(전 집단): %d/5%s" % (5 - len(kb_bad), "" if not kb_bad else " ← 뇌 %s" % kb_bad))
    ok &= not kb_bad
    # 2. 무학습 0 변화
    z = [b for b in BRAINS if R[(0, "nolearn", b)][2] != 0.0]
    checks.append("[측정 확인] 배율 0 무학습 변조폭 변화 0.0000: %d/5%s" % (5 - len(z), "" if not z else " ← 뇌 %s" % z))
    ok &= not z
    # 3. 학습 사후 ≠ 무학습 사후
    same = [b for b in BRAINS if R[(0, "learn", b)][1] == R[(0, "nolearn", b)][1]]
    checks.append("[측정 확인] 배율 0 학습 사후 ≠ 무학습 사후: %d/5%s" % (5 - len(same), "" if not same else " ← 같은 값 %s" % same))
    ok &= not same
    # 4. 조작이 뇌를 바꿨는가: 배율 0 사전 ≠ 배율 1 사전(공통 입력이 행동 기준선에 닿는지) — 같으면 조작 무효 의심
    pre_same = [b for b in BRAINS if R[(0, "learn", b)][0] == R[(1, "learn", b)][0]]
    checks.append("[측정 확인] 배율 0 사전 ≠ 배율 1 사전: %d/5%s" % (5 - len(pre_same), "" if not pre_same else " ← 같은 값(조작 무효 의심) %s" % pre_same))
    ok &= not pre_same
    eff = {s: [round(R[(s, "learn", b)][1] - R[(s, "nolearn", b)][1], 4) for b in BRAINS] for s in (0, 1)}
    dl = [round(eff[0][i] - eff[1][i], 4) for i in range(5)]
    n_reach = sum(x <= -MIN_EFFECT for x in eff[0])
    n_better = sum(x <= -NULL_BAND for x in dl)
    n_same = sum(abs(x) < NULL_BAND for x in dl)
    n_worse = sum(x >= NULL_BAND for x in dl)
    if not ok:
        verdict = "보류(조작검증 실패 — 측정부터)"
    elif n_reach >= 4 and n_better >= 4:
        verdict = "양성 — 좌우 공통 KC 입력이 학습 효과 크기를 제한한다(요소 특정)"
    elif n_same >= 4:
        verdict = "음성 — 좌우 공통 KC 입력을 빼도 효과 변화 < 0.03(이 요소 아님)"
    elif n_worse >= 4:
        verdict = "역방향 — 공통 입력을 빼면 효과가 작아짐"
    else:
        verdict = "보류(혼재 또는 개선했으나 0.10 미도달)"
    return checks, {"eff": eff, "dl": dl, "n_reach": n_reach, "n_better": n_better, "n_same": n_same,
                    "n_worse": n_worse, "ok": ok, "verdict": verdict}


def report(checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for s in (1, 0):
        print("배율 %d 효과(학습−무학습 사후 변조폭, 음수=교차): %s | 평균 %+.4f"
              % (s, " ".join("b%d %+.4f" % (b, x) for b, x in zip(BRAINS, res["eff"][s])), sum(res["eff"][s]) / 5))
    print("Δ = e0 − e1 (음수=공통 입력 제거로 정답 쪽 증가): %s | 평균 %+.4f"
          % (" ".join("b%d %+.4f" % (b, x) for b, x in zip(BRAINS, res["dl"])), sum(res["dl"]) / 5))
    print("e0 ≤ −%.2f: %d/5 | Δ ≤ −%.2f: %d/5 | |Δ| < %.2f: %d/5 | Δ ≥ +%.2f: %d/5"
          % (MIN_EFFECT, res["n_reach"], NULL_BAND, res["n_better"], NULL_BAND, res["n_same"], NULL_BAND, res["n_worse"]))
    print("판정: %s" % res["verdict"])


def load():
    R, KBW = {}, {}
    for fn, rx, s in (("E119.log", TR19, 1), ("E121.log", TR21, 0)):
        try:
            for ln in open(os.path.join(EXP, fn), encoding="utf-8"):
                m = rx.match(ln)
                if m:
                    R[(s, m.group(1), int(m.group(2)))] = tuple(float(x) for x in m.group(3, 4, 5))
        except FileNotFoundError:
            pass
    for b in BRAINS:
        try:
            m = KB.search(open(os.path.join(EXP, "logs/E121/s0_learn_b%d.log" % b), encoding="utf-8").read())
            if m:
                ws = [float(x) for x in re.findall(r"w=([-0-9.]+)", m.group(3))]
                KBW[b] = (float(m.group(1)), int(m.group(2)), ws)
        except FileNotFoundError:
            pass
    return R, KBW


if __name__ == "__main__":
    R, KBW = load()
    c, r = judge(R, KBW)
    report(c, r)
    sys.exit(0)
