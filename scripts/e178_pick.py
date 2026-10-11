#!/usr/bin/env python3
"""E178 보정 선택 — 기준 logs/E178/criteria_fixed.txt 규칙. 원 로그 logs/E178/calib/ev_x{X}_{none|all}_{off|on}.log(뇌 15, 맥락 전용 집단 w 4, --eval-diag).
X 격자: 0(기준), −500, −1000, −2000, −4000(motor 좌·우 Ioffset 기저값 — 긴장성 억제).
조작검증(각 X, 모두): 로그 4개에 DECOMP 줄과 진단 줄 좌·우 / '[motor 동작점]' 줄 — X = 0 이면 0개, 아니면 각 로그 ≥ 1개이고 찍힌 기저값 = X / '[맥락 평가]' 줄은 켬 평가에만.
전제(X = 0): 무학습 켬 motor 발화율 8개(자극 좌·우 × motor 좌·우)가 모두 ≥ 0.66(뇌 15 에서도 맥락 켬 포화 — E177 정정 1 과 같은 경계). 전제가 서지 않으면 'X=none(전제 불성립)'.
선택(X ≠ 0, 약한 쪽부터 −500 → −4000): 조작검증 통과 · 무학습 켬 motor 8개 모두 ≤ 0.60 · 무학습 끔 motor 8개 모두 ≥ 0.05 ·
학습 A 끔 변조폭 ≤ −0.10 · 학습 A 켬 변조폭 ≤ −0.05(1e-4 정수 비교) 를 모두 만족하는 첫 X. 없으면 'X=none'. 결과를 logs/E178/pick.txt 에.
실행: python3 scripts/e178_pick.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
GRID = (0, -500, -1000, -2000, -4000)
EVS = (("none", "off"), ("none", "on"), ("all", "off"), ("all", "on"))
MODRE = re.compile(r"^=> DECOMP mode=\w+ mod=([-+0-9.]+)", re.M)
DG = re.compile(r"^\[E171 평가 진단\] variant=\w+ side=(left|right) n=\d+ motor L/R ([0-9.]+)/([0-9.]+) KC L/R ([0-9.]+)/([0-9.]+)", re.M)
TONIC = re.compile(r"^\s*\[motor 동작점\] motor 좌·우 Ioffset 기저값 ([-0-9.]+)", re.M)
SAT, UNSAT, ALIVE = 6600, 6000, 500          # 1e-4 정수: 포화 ≥ 0.66, 비포화 ≤ 0.60, 살아 있음 ≥ 0.05
OFF_MAX, ON_MAX = -1000, -500                # 학습 A 끔 ≤ −0.10, 켬 ≤ −0.05


def i4(x):
    return int(round(float(x) * 1e4))


def parse(t, on):
    """로그 하나 → {'mod', 'rates'(8개 아니면 4개: 좌 자극 L,R, 우 자극 L,R), 'tonic'(찍힌 기저값 목록), 'ctxline'} 또는 None(DECOMP·진단 결측)."""
    if not t:
        return None
    m = MODRE.search(t)
    dg = {d.group(1): (float(d.group(2)), float(d.group(3))) for d in DG.finditer(t)}
    if m is None or set(dg) != {"left", "right"}:
        return None
    return {"mod": float(m.group(1)), "rates": dg["left"] + dg["right"], "tonic": [float(x) for x in TONIC.findall(t)],
            "ctxline": "[맥락 평가]" in t}


def man_ok(X, L):
    """조작검증 — L[(M, C)] = parse 결과."""
    if any(L.get(k) is None for k in EVS):
        return False
    for (M, C), r in L.items():
        if X == 0:
            if r["tonic"]:
                return False
        elif not r["tonic"] or any(abs(t - X) > 1e-6 for t in r["tonic"]):
            return False
        if r["ctxline"] != (C == "on"):
            return False
    return True


def pick(D):
    """D[X][(M, C)] = parse 결과. 반환 (선택 문자열, 전제 bool 또는 None, {X: (조작, 비포화, 살아있음, 끔효과, 켬효과)})."""
    info = {}
    for X in GRID:
        L = D.get(X, {})
        mo = man_ok(X, L)
        if not mo:
            info[X] = (False, None, None, None, None)
            continue
        on_r, off_r = L[("none", "on")]["rates"], L[("none", "off")]["rates"]
        info[X] = (True, all(i4(x) <= UNSAT for x in on_r), all(i4(x) >= ALIVE for x in off_r),
                   i4(L[("all", "off")]["mod"]) <= OFF_MAX, i4(L[("all", "on")]["mod"]) <= ON_MAX)
    L0 = D.get(0, {})
    prem = None
    if info[0][0]:
        prem = all(i4(x) >= SAT for x in L0[("none", "on")]["rates"])
    if prem is None:
        return "X=none(기준 X=0 조작검증·결측 실패)", prem, info
    if not prem:
        return "X=none(전제 불성립 — 뇌 15 X=0 맥락 켬 비포화)", prem, info
    sel = next((X for X in GRID[1:] if info[X][0] and all(info[X][1:])), None)
    return ("X=%d" % sel) if sel is not None else "X=none", prem, info


def load(base=EXP):
    D = {}
    for X in GRID:
        D[X] = {}
        for (M, C) in EVS:
            f = os.path.join(base, "logs", "E178", "calib", "ev_x%d_%s_%s.log" % (X, M, C))
            D[X][(M, C)] = parse(open(f, encoding="utf-8", errors="replace").read(), C == "on") if os.path.exists(f) else None
    return D


def main():
    D = load()
    sel, prem, info = pick(D)
    for X in GRID:
        L = D[X]
        rs = " | ".join("%s %s mod %s motor %s" % (M, C, ("%+.4f" % L[(M, C)]["mod"]) if L[(M, C)] else "결측",
                                                   " ".join("%.4f" % x for x in L[(M, C)]["rates"]) if L[(M, C)] else "-") for (M, C) in EVS)
        print("X=%d 조작 %s 비포화 %s 살아있음 %s 끔효과 %s 켬효과 %s || %s" % ((X,) + tuple("-" if v is None else ("✓" if v else "✗") for v in info[X]) + (rs,)))
    print("전제(X=0 켬 포화): %s" % ("-" if prem is None else ("✓" if prem else "✗")))
    print("선택: %s" % sel)
    with open(os.path.join(EXP, "logs", "E178", "pick.txt"), "w", encoding="utf-8") as fo:
        fo.write(sel + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
