#!/usr/bin/env python3
"""e178_pick.py 합성 정답 시험 — 선택 규칙(비포화 ≤ 0.60 · 살아 있음 ≥ 0.05 · 학습 A 끔 ≤ −0.10 · 켬 ≤ −0.05)의 경계값, 전제(X=0 켬 포화 ≥ 0.66),
조작검증(동작점 줄 개수·값, 맥락 평가 줄 위치, 결측), 약한 X 우선 순서. 실행: python3 scripts/test_e178_pick.py (저장소 루트에서)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e178_pick as P  # noqa: E402

n_ok = n_bad = 0


def log(mod, rates, tonic=None, ctx=False, ntonic=2):
    ln = []
    if tonic is not None:
        ln += ["  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)" % tonic] * ntonic
    if ctx:
        ln.append("[맥락 평가] ctx=켬 I=0.00(KC 좌·우 Ioffset) I_inh=0.00(억제 부분집합 0.00) I_ab=0.00(assoc_binding) ctx_n=200 w=4.00 level=0.90(맥락 전용 집단)")
    ln.append("[E171 평가 진단] variant=base side=left n=250 motor L/R %.6f/%.6f KC L/R 0.004603/0.001772" % tuple(rates[:2]))
    ln.append("[E171 평가 진단] variant=base side=right n=250 motor L/R %.6f/%.6f KC L/R 0.001635/0.004875" % tuple(rates[2:]))
    ln.append("=> DECOMP mode=none mod=%+.4f acc=0.0 off=-0.0000 pushed=0 kc_means[]" % mod)
    return "\n".join(ln)


SAT_R = (0.6664, 0.6662, 0.6663, 0.6661)


def grid(spec, override=None):
    """spec[X] = (켬 무학습 motor 4개, 끔 무학습 motor 4개, 학습 A 끔 mod, 학습 A 켬 mod). override[(X, M, C)] = 로그 문자열 또는 None(결측)."""
    D = {}
    for X in P.GRID:
        on_r, off_r, m_off, m_on = spec[X]
        tn = None if X == 0 else X
        T = {("none", "off"): log(0.0080, off_r, tn), ("none", "on"): log(0.0007, on_r, tn, ctx=True),
             ("all", "off"): log(m_off, off_r, tn), ("all", "on"): log(m_on, on_r, tn, ctx=True)}
        for k in list(T):
            if override and (X,) + k in override:
                T[k] = override[(X,) + k]
        D[X] = {k: (P.parse(v, k[1] == "on") if v is not None else None) for k, v in T.items()}
    return D


def base():
    return {0: (SAT_R, (0.57, 0.64, 0.64, 0.55), -0.51, -0.0000),
            -500: ((0.65, 0.64, 0.64, 0.65), (0.50, 0.58, 0.58, 0.49), -0.48, -0.02),
            -1000: ((0.58, 0.55, 0.55, 0.57), (0.30, 0.38, 0.38, 0.29), -0.45, -0.12),
            -2000: ((0.40, 0.33, 0.33, 0.39), (0.08, 0.15, 0.15, 0.07), -0.30, -0.15),
            -4000: ((0.10, 0.05, 0.05, 0.10), (0.00, 0.00, 0.00, 0.00), -0.01, -0.02)}


def chk(name, D, want):
    global n_ok, n_bad
    got = P.pick(D)[0]
    good = got.startswith(want)
    n_ok += good
    n_bad += (not good)
    print("  %-46s → %-46s %s" % (name, got, "✓" if good else "✗ (기대 %s)" % want))


chk("전형: −500 켬 0.65(포화 근처) 탈락 → −1000", grid(base()), "X=-1000")
s = base(); s[-1000] = ((0.6000, 0.55, 0.55, 0.57),) + s[-1000][1:]
chk("경계: 켬 0.6000 → 통과", grid(s), "X=-1000")
s = base(); s[-1000] = ((0.6001, 0.55, 0.55, 0.57),) + s[-1000][1:]
chk("경계: 켬 0.6001 → −1000 탈락 → −2000", grid(s), "X=-2000")
s = base(); s[-1000] = (s[-1000][0], (0.0500, 0.38, 0.38, 0.29)) + s[-1000][2:]
chk("경계: 끔 0.0500 → 통과", grid(s), "X=-1000")
s = base(); s[-1000] = (s[-1000][0], (0.0499, 0.38, 0.38, 0.29)) + s[-1000][2:]
chk("경계: 끔 0.0499 → −1000 탈락 → −2000", grid(s), "X=-2000")
s = base(); s[-1000] = s[-1000][:2] + (-0.1000, s[-1000][3])
chk("경계: 학습 A 끔 −0.1000 → 통과", grid(s), "X=-1000")
s = base(); s[-1000] = s[-1000][:2] + (-0.0999, s[-1000][3])
chk("경계: 학습 A 끔 −0.0999 → 탈락 → −2000", grid(s), "X=-2000")
s = base(); s[-1000] = s[-1000][:3] + (-0.0500,)
chk("경계: 학습 A 켬 −0.0500 → 통과", grid(s), "X=-1000")
s = base(); s[-1000] = s[-1000][:3] + (-0.0499,); s[-2000] = s[-2000][:3] + (-0.0499,)
chk("경계: 학습 A 켬 −0.0499(−1000·−2000) → 없음", grid(s), "X=none")
s = base(); s[0] = ((0.6664, 0.6662, 0.6599, 0.6661),) + s[0][1:]
chk("전제: X=0 켬 하나 0.6599 → 전제 불성립", grid(s), "X=none(전제 불성립")
chk("조작: X=0 로그에 동작점 줄 → 기준 실패", grid(base(), {(0, "none", "on"): log(0.0007, SAT_R, 0.0, ctx=True)}), "X=none(기준")
chk("조작: −1000 동작점 값 −500 → −1000 탈락 → −2000",
    grid(base(), {(-1000, "all", "on"): log(-0.12, (0.58, 0.55, 0.55, 0.57), -500, ctx=True)}), "X=-2000")
chk("조작: 끔 평가에 맥락 평가 줄 → −1000 탈락 → −2000",
    grid(base(), {(-1000, "none", "off"): log(0.0080, (0.30, 0.38, 0.38, 0.29), -1000, ctx=True)}), "X=-2000")
chk("조작: −1000 동작점 줄 0개 → 탈락 → −2000",
    grid(base(), {(-1000, "none", "off"): log(0.0080, (0.30, 0.38, 0.38, 0.29), None)}), "X=-2000")
chk("결측: −1000 로그 하나 없음 → −2000", grid(base(), {(-1000, "all", "off"): None}), "X=-2000")
chk("결측: X=0 로그 없음 → 기준 실패", grid(base(), {(0, "none", "on"): None}), "X=none(기준")
chk("진단 줄 한쪽만 → 그 로그 결측 취급 → −2000",
    grid(base(), {(-1000, "none", "on"): "\n".join(l for l in log(0.0007, (0.58, 0.55, 0.55, 0.57), -1000, ctx=True).splitlines() if "side=right" not in l)}), "X=-2000")
s = base(); s[-500] = ((0.55, 0.55, 0.55, 0.55), (0.40, 0.40, 0.40, 0.40), -0.40, -0.10)
chk("순서: −500·−1000 둘 다 통과 → −500(약한 쪽)", grid(s), "X=-500")
print("test_e178_pick ✓%d ✗%d" % (n_ok, n_bad))
sys.exit(1 if n_bad else 0)
