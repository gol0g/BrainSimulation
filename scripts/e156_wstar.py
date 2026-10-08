#!/usr/bin/env python3
"""E156 수정 1 보정 판정 — 규칙 logs/E156/criteria_fixed.txt 수정 1(2026-10-09 00:44:40 고정).
격자 W 중 [사전] 이 +0.4397(뇌 15 기본 표현 반사 25) 에 가장 가까운 값(동점이면 작은 W). 그 [사전] 이 ±0.05 밖이면 보정 실패.
경로 확인(조건 2): 적재 검증 줄 ≥ 2, 반사 W→W(좌우), 0시행 이식 [사후] = [사전] ±0.002 — 어긋난 W 는 후보에서 빼고 사유를 출력한다.
1e-4 정수 비교. 실행: python3 scripts/e156_wstar.py (저장소 루트에서) → logs/E156/wstar.txt"""
import os
import re
import sys

EXP = "research/experiments"
GRID = (25, 40, 60, 90, 135, 200, 300)
TARGET = 4397
TOL = 500


def i4(x):
    return int(round(x * 1e4))


def parse(txt):
    pre = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", txt, re.M)
    post = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", txt, re.M)
    refl = re.findall(r"^\[반사가중치\] good_food_to_motor_([lr])\s+n=\d+ w_mean ([-0-9.]+)→([-0-9.]+)", txt, re.M)
    nld = len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", txt, re.M))
    return (float(pre.group(1)) if pre else None, float(post.group(1)) if post else None, refl, nld)


def select(rows):
    """rows: {W: (pre, post, refl, nld)} → (W* 또는 None, 후보 {W: pre}, 사유 목록)"""
    ok, why = {}, []
    for W in GRID:
        if W not in rows:
            why.append("W%d 결측" % W)
            continue
        pre, post, refl, nld = rows[W]
        wv = "%.4f" % W
        bad = []
        if pre is None or post is None:
            bad.append("사전/사후 줄 없음")
        elif abs(i4(post) - i4(pre)) > 20:
            bad.append("0시행 사후 %+.4f ≠ 사전 %+.4f" % (post, pre))
        if nld < 2:
            bad.append("적재 %d줄" % nld)
        if sorted(r[0] for r in refl) != ["l", "r"] or any((a, b) != (wv, wv) for _, a, b in refl):
            bad.append("반사 %s" % (refl,))
        if bad:
            why.append("W%d 제외: %s" % (W, "; ".join(bad)))
        else:
            ok[W] = pre
    if not ok:
        why.append("후보 없음: 보정 실패")
        return None, ok, why
    W = min(ok, key=lambda w: (abs(i4(ok[w]) - TARGET), w))
    if abs(i4(ok[W]) - TARGET) > TOL:
        why.append("최근접 W%d [사전] %+.4f — 목표 +0.4397 ± 0.05 밖: 보정 실패" % (W, ok[W]))
        return None, ok, why
    return W, ok, why


def main():
    rows = {}
    for W in GRID:
        f = os.path.join(EXP, "logs", "E156", "calib", "W%d_b15.log" % W)
        if os.path.exists(f):
            rows[W] = parse(open(f, encoding="utf-8", errors="replace").read())
    W, ok, why = select(rows)
    for w in GRID:
        if w in rows:
            print("W%-3d [사전] %s [사후] %s 적재 %d 반사 %s" % (w, rows[w][0], rows[w][1], rows[w][3], " ".join("%s→%s" % (a, b) for _, a, b in rows[w][2])))
    for y in why:
        print("  " + y)
    line = ("W*=%d 사전 %+.4f (목표 +0.4397 ± 0.05)" % (W, ok[W])) if W is not None else "보정 실패"
    print("=> " + line)
    open(os.path.join(EXP, "logs", "E156", "wstar.txt"), "w", encoding="utf-8").write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
