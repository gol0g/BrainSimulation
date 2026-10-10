#!/usr/bin/env python3
"""E177 탐색(판정 밖) — 쌍조건 학습 중 맥락별 정답 선택률 곡선: 어느 맥락이 지는가.
추적 traces/E177/tr_bc_b{B}.npz 의 rows: 열 2 = 자극 쪽, 6 = 실행 행동(−1 = 행동 창 없음), 7 = 보상, 37 = 맥락(0 끔 · 1 켬).
정답 규칙(judge_e177.bicond_stats 와 같음): 끔 → 실행 ≠ 쪽(교차), 켬 → 실행 = 쪽(같은 쪽).
출력: 뇌마다 맥락별 블록(기본 10 블록 × 300시행) 정답 선택률(행동한 시행 중), 마지막 3 블록 평균.
탐험(epsilon 0.6)이 섞인 실행 기준이라 범위는 대략 0.5(무작위) ~ 0.7(정책 완벽) — 상대 비교용.
실행: python3 scripts/explore_e177_curves.py (저장소 루트) / 합성 검사: --selftest"""
import os
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def curves(rows, nblk=10):
    """{0: [(행동 수, 정답률), ...], 1: [...]} — 시행 순서를 nblk 블록으로 나눠 맥락별로 센다(블록에 행동 0 이면 정답률 nan)."""
    rows = np.asarray(rows, dtype=np.float64)
    n = len(rows)
    edges = np.linspace(0, n, nblk + 1).astype(int)
    out = {0: [], 1: []}
    for i in range(nblk):
        blk = rows[edges[i]:edges[i + 1]]
        act = blk[:, 6] >= 0
        ctx = blk[:, 37] == 1
        corr = np.where(ctx, blk[:, 6] == blk[:, 2], blk[:, 6] != blk[:, 2])
        for c in (0, 1):
            sel = act & (ctx == bool(c))
            k = int(sel.sum())
            out[c].append((k, float(corr[sel].mean()) if k else float("nan")))
    return out


def tail_mean(cv, k=3):
    """마지막 k 블록의 행동 가중 정답률."""
    tot = sum(n for n, _ in cv[-k:])
    return float(sum(n * r for n, r in cv[-k:] if n) / tot) if tot else float("nan")


def selftest():
    rng = np.random.RandomState(0)
    n = 3000
    rows = np.zeros((n, 38))
    rows[:, 2] = rng.randint(0, 2, n)
    rows[:, 37] = rng.randint(0, 2, n)
    t = np.arange(n) / n
    # 끔: 정답 확률 0.5 → 0.9 로 선형 상승, 켬: 정답 확률 0.5 고정. 행동 창 없음 5%.
    p = np.where(rows[:, 37] == 1, 0.5, 0.5 + 0.4 * t)
    good = rng.rand(n) < p
    same = np.where(rows[:, 37] == 1, good, ~good)          # 켬 정답 = 같은 쪽, 끔 정답 = 교차
    rows[:, 6] = np.where(same, rows[:, 2], 1 - rows[:, 2])
    rows[rng.rand(n) < 0.05, 6] = -1
    cv = curves(rows)
    ok = True
    off_first, off_last = cv[0][0][1], tail_mean(cv[0])
    on_last = tail_mean(cv[1])
    exp_off_last = 0.5 + 0.4 * np.mean([0.75, 0.85, 0.95])    # 마지막 3 블록 중앙 시점 기대값 ≈ 0.840
    for name, got, want, tol in (("끔 첫 블록 ≈ 0.52", off_first, 0.52, 0.06), ("끔 마지막 3 블록 ≈ 0.84", off_last, exp_off_last, 0.04),
                                 ("켬 마지막 3 블록 ≈ 0.50", on_last, 0.50, 0.05)):
        good_ = abs(got - want) <= tol
        ok &= good_
        print("  %-24s %.3f (기대 %.3f ± %.2f) %s" % (name, got, want, tol, "✓" if good_ else "✗"))
    nact = sum(k for c in (0, 1) for k, _ in cv[c])
    good_ = nact == int((rows[:, 6] >= 0).sum())
    ok &= good_
    print("  %-24s %d %s" % ("행동 수 합 = 행동 시행", nact, "✓" if good_ else "✗"))
    # 규칙 방향 검사: 켬 시행을 모두 같은 쪽으로 실행하면 켬 정답률 1.0, 끔 시행을 모두 같은 쪽으로 하면 끔 정답률 0.0.
    r2 = rows.copy()
    r2[:, 6] = r2[:, 2]
    cv2 = curves(r2)
    good_ = tail_mean(cv2[1]) == 1.0 and tail_mean(cv2[0]) == 0.0
    ok &= good_
    print("  %-24s 켬 %.1f 끔 %.1f %s" % ("규칙 방향(모두 같은 쪽)", tail_mean(cv2[1]), tail_mean(cv2[0]), "✓" if good_ else "✗"))
    print("합성 검사:", "통과" if ok else "실패")
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    for b in BRAINS:
        f = os.path.join(EXP, "traces", "E177", "tr_bc_b%d.npz" % b)
        if not os.path.exists(f):
            print("b%d: 추적 없음" % b)
            continue
        rows = np.load(f)["rows"]
        cv = curves(rows)
        for c, nm in ((0, "끔(교차)"), (1, "켬(같은 쪽)")):
            print("b%d %-10s %s | 마지막 3 블록 %.3f" % (b, nm, " ".join("%.2f" % r for _, r in cv[c]), tail_mean(cv[c])))


if __name__ == "__main__":
    main()
