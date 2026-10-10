#!/usr/bin/env python3
"""E176 보정 선택 — 기준 logs/E176/criteria_fixed.txt 규칙. 원 로그 logs/E176/calib/kcctx_w{1,2,4,8}_b15.log 의 '=> KCCTX' 줄(맥락 = 맥락 전용 집단 zz_ctx → KC, 초기 연결 스냅숏).
조작검증(각 w): 맥락 집단 발화 끔 0·켬 > 0. 선택 = 두 쪽 자카드 합 ≤ 1.60(1e-4 정수 ≤ 16000) · 두 쪽 맥락 단독 반응 ≤ 10 · 두 쪽 반응 수 켬 ≤ 2 × 끔 · 조작검증 통과를
모두 만족하는 w 중 가장 약한 것 = w*(자카드 nan 이면 그 칸 탈락). 없으면 'none'(본실험 미실행). 결과를 logs/E176/pick.txt 에 'W=<값>' 으로.
실행: python3 scripts/e176_pick.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
GRID = (("1", 1.0), ("2", 2.0), ("4", 4.0), ("8", 8.0))
SIDE = re.compile(r"side=([lr]) off=(\d+) on=(\d+) ctx=(\d+) keep=(\d+) lost=(\d+) conj=(\d+) jac=([0-9.na]+)")
CTX = re.compile(r"ctx_n=(\d+) w=([0-9.]+) p=([0-9.]+) level=([0-9.]+) 맥락 집단 발화 끔 (\d+) 켬 (\d+)")


def parse(t):
    if not t:
        return None
    ln = next((x for x in t.splitlines() if x.startswith("=> KCCTX")), None)
    if ln is None:
        return None
    sides = {m.group(1): {"off": int(m.group(2)), "on": int(m.group(3)), "ctx": int(m.group(4)), "keep": int(m.group(5)),
                          "lost": int(m.group(6)), "conj": int(m.group(7)), "jac": m.group(8)} for m in SIDE.finditer(ln)}
    cx = CTX.search(ln)
    if set(sides) != {"l", "r"} or cx is None:
        return None
    return {"l": sides["l"], "r": sides["r"], "w": float(cx.group(2)), "c_off": int(cx.group(5)), "c_on": int(cx.group(6))}


def jac4(s):
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else int(round(v * 1e4))


def pick(C):
    man, cons = {}, {}
    for tag, _ in GRID:
        c = C.get(tag)
        if c is None:
            man[tag] = cons[tag] = None
            continue
        man[tag] = c["c_off"] == 0 and c["c_on"] > 0
        jl, jr = jac4(c["l"]["jac"]), jac4(c["r"]["jac"])
        cons[tag] = (man[tag] and jl is not None and jr is not None and jl + jr <= 16000
                     and all(c[s]["on"] <= 2 * c[s]["off"] and c[s]["ctx"] <= 10 for s in "lr"))
    sel = next((tag for tag, _ in GRID if cons.get(tag)), None)
    return sel, man, cons


def main():
    C = {}
    for tag, _ in GRID:
        f = os.path.join(EXP, "logs", "E176", "calib", "kcctx_w%s_b15.log" % tag)
        C[tag] = parse(open(f, encoding="utf-8", errors="replace").read()) if os.path.exists(f) else None
    miss = [tag for tag, _ in GRID if C[tag] is None]
    if miss:
        print("[E176 보정] 결측 %s — 선택 보류" % miss)
        return 0
    sel, man, cons = pick(C)
    for tag, w in GRID:
        c = C[tag]
        print("w=%.0f 맥락 집단 발화 끔 %d 켬 %d(조작 %s) | 좌 off %d on %d ctx %d lost %d conj %d jac %s | 우 off %d on %d ctx %d lost %d conj %d jac %s | 제약 %s"
              % (w, c["c_off"], c["c_on"], "✓" if man[tag] else "✗", c["l"]["off"], c["l"]["on"], c["l"]["ctx"], c["l"]["lost"], c["l"]["conj"], c["l"]["jac"],
                 c["r"]["off"], c["r"]["on"], c["r"]["ctx"], c["r"]["lost"], c["r"]["conj"], c["r"]["jac"], "통과" if cons[tag] else "-"))
    out = "W=%s" % (dict(GRID)[sel] if sel is not None else "none")
    open(os.path.join(EXP, "logs", "E176", "pick.txt"), "w", encoding="utf-8").write(out + "\n")
    print("선택: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
