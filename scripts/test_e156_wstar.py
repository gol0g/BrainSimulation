#!/usr/bin/env python3
"""e156_wstar.py 합성 시험: 최근접 선택·동점(작은 W)·경계 ±0.05 정확·밖이면 실패·0시행 사후≠사전/적재 1줄/반사 변함/결측 제외·로그 파싱.
실행: python3 scripts/test_e156_wstar.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e156_wstar as S


def R(pre, W, post=None, nld=2, refl=None):
    wv = "%.4f" % W
    return (pre, pre if post is None else post, refl if refl is not None else [("l", wv, wv), ("r", wv, wv)], nld)


def grid(pres, over=None):
    rows = {W: R(p, W) for W, p in zip(S.GRID, pres)}
    for W, kw in (over or {}).items():
        rows[W] = R(rows[W][0], W, **kw)
    return rows


ok_all = True


def chk(name, rows, want):
    global ok_all
    W, _, why = S.select(rows)
    good = W == want
    ok_all &= good
    print("%-34s 기대 %-5s → %-5s %s" % (name, want, W, "✓" if good else "✗ %s" % why))


mono = (0.16, 0.25, 0.33, 0.41, 0.47, 0.52, 0.55)
chk("단조 — 최근접 W90(0.41)", grid(mono), 90)
chk("동점(0.4197/0.4597) → 작은 W", grid((0.16, 0.25, 0.33, 0.4197, 0.4597, 0.52, 0.55)), 90)
chk("경계 0.3897(−0.0500) → 통과", grid((0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.3897)), 300)
chk("경계 0.3896(−0.0501) → 실패", grid((0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.3896)), None)
chk("경계 0.4897(+0.0500) → 통과", grid((0.4897, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)), 25)
chk("포화 0.30 → 실패", grid((0.16, 0.22, 0.27, 0.29, 0.30, 0.30, 0.30)), None)
chk("최근접 W 0시행 사후≠사전 → 다음 후보", grid(mono, {90: {"post": 0.3900}}), 135)
chk("최근접 W 적재 1줄 → 다음 후보", grid(mono, {90: {"nld": 1}}), 135)
chk("최근접 W 반사 변함 → 다음 후보", grid(mono, {90: {"refl": [("l", "90.0000", "89.0000"), ("r", "90.0000", "90.0000")]}}), 135)
chk("최근접 W 반사 한쪽만 → 다음 후보", grid(mono, {90: {"refl": [("l", "90.0000", "90.0000")]}}), 135)
r = grid(mono); del r[90]
chk("최근접 W 결측 → 다음 후보", r, 135)
chk("전부 결측 → 실패", {}, None)
# 로그 파싱
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E156", "calib"))
    for W, p in zip(S.GRID, mono):
        open(os.path.join(td, "logs", "E156", "calib", "W%d_b15.log" % W), "w", encoding="utf-8").write(
            "[E153 종류 입력 적재] k 검증 일치 — x\n[사전] 오프셋 +0.000 | 정답률 0.0%% | **변조폭 %+.4f** (양수=반사방향, 음수=역전)\n"
            "[반사가중치] good_food_to_motor_l   n=1 w_mean %.4f→%.4f (학습 뇌; 이식 대상 아님)\n"
            "[반사가중치] good_food_to_motor_r   n=1 w_mean %.4f→%.4f (학습 뇌; 이식 대상 아님)\n"
            "[E153 종류 입력 적재] k 검증 일치 — x\n[사후] 오프셋 +0.000 | 정답률 0.0%% | **변조폭 %+.4f**\n" % (p, W, W, W, W, p))
    S.EXP = td
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        S.main()
    got = open(os.path.join(td, "logs", "E156", "wstar.txt"), encoding="utf-8").read().strip()
g = got == "W*=90 사전 +0.4100 (목표 +0.4397 ± 0.05)"
ok_all &= g
print("%-34s → %s %s" % ("로그 파싱·wstar.txt", got, "✓" if g else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
