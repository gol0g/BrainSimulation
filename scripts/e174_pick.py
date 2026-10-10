#!/usr/bin/env python3
"""E174 보정 선택 — 기준 logs/E174/criteria_fixed.txt 규칙. 원 로그 logs/E174/calib/kcctx_c{05,10,15,20,30,40}_b15.log 의 '=> KCCTX' 줄(맥락 = KC 억제 뉴런 맥락 부분집합 전류 I_c).
조작검증(각 I_c): 맥락 부분집합 억제 뉴런 발화 합 켬 > 끔. 선택 = 두 쪽 자카드 합 ≤ 1.60(1e-4 정수 ≤ 16000) · 두 쪽 반응 수 켬 ≥ 0.5 × 끔(2·on ≥ off) ·
두 쪽 맥락 단독 반응 ≤ 10 을 모두 만족하는 I_c 중 가장 약한 것 = I_c*. 자카드 nan 이면 그 칸 탈락. 없으면 'none'(본실험 미실행). 결과를 logs/E174/pick.txt 에 'IC=<값>' 으로.
실행: python3 scripts/e174_pick.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
GRID = (("05", 0.5), ("10", 1.0), ("15", 1.5), ("20", 2.0), ("30", 3.0), ("40", 4.0))
SIDE = re.compile(r"side=([lr]) off=(\d+) on=(\d+) ctx=(\d+) keep=(\d+) lost=(\d+) conj=(\d+) jac=([0-9.na]+)")
INH = re.compile(r"ctx_inh_i=([0-9.]+) 억제 부분집합 발화 끔 (\d+) 켬 (\d+)")


def parse(t):
    if not t:
        return None
    ln = next((x for x in t.splitlines() if x.startswith("=> KCCTX")), None)
    if ln is None:
        return None
    sides = {m.group(1): {"off": int(m.group(2)), "on": int(m.group(3)), "ctx": int(m.group(4)), "keep": int(m.group(5)),
                          "lost": int(m.group(6)), "conj": int(m.group(7)), "jac": m.group(8)} for m in SIDE.finditer(ln)}
    ih = INH.search(ln)
    if set(sides) != {"l", "r"} or ih is None:
        return None
    return {"l": sides["l"], "r": sides["r"], "ic": float(ih.group(1)), "i_off": int(ih.group(2)), "i_on": int(ih.group(3))}


def jac4(s):
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else int(round(v * 1e4))


def pick(C):
    """C[tag] = parse 결과. 반환 (선택 tag 또는 None, 조작검증 사전, 제약 사전)."""
    man, cons = {}, {}
    for tag, _ in GRID:
        c = C.get(tag)
        if c is None:
            man[tag] = cons[tag] = None
            continue
        man[tag] = c["i_on"] > c["i_off"]
        jl, jr = jac4(c["l"]["jac"]), jac4(c["r"]["jac"])
        cons[tag] = (man[tag] and jl is not None and jr is not None and jl + jr <= 16000
                     and all(2 * c[s]["on"] >= c[s]["off"] and c[s]["ctx"] <= 10 for s in "lr"))
    sel = next((tag for tag, _ in GRID if cons.get(tag)), None)
    return sel, man, cons


def main():
    C = {}
    for tag, _ in GRID:
        f = os.path.join(EXP, "logs", "E174", "calib", "kcctx_c%s_b15.log" % tag)
        C[tag] = parse(open(f, encoding="utf-8", errors="replace").read()) if os.path.exists(f) else None
    miss = [tag for tag, _ in GRID if C[tag] is None]
    if miss:
        print("[E174 보정] 결측 %s — 선택 보류" % miss)
        return 0
    sel, man, cons = pick(C)
    for tag, ic in GRID:
        c = C[tag]
        print("I_c=%.1f 억제 부분집합 발화 끔 %d 켬 %d(조작 %s) | 좌 off %d on %d ctx %d lost %d conj %d jac %s | 우 off %d on %d ctx %d lost %d conj %d jac %s | 제약 %s"
              % (ic, c["i_off"], c["i_on"], "✓" if man[tag] else "✗", c["l"]["off"], c["l"]["on"], c["l"]["ctx"], c["l"]["lost"], c["l"]["conj"], c["l"]["jac"],
                 c["r"]["off"], c["r"]["on"], c["r"]["ctx"], c["r"]["lost"], c["r"]["conj"], c["r"]["jac"], "통과" if cons[tag] else "-"))
    out = "IC=%s" % (dict(GRID)[sel] if sel is not None else "none")
    open(os.path.join(EXP, "logs", "E174", "pick.txt"), "w", encoding="utf-8").write(out + "\n")
    print("선택: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
