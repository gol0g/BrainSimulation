#!/usr/bin/env python3
"""E173 보정 선택 — 기준 logs/E173/criteria_fixed.txt 규칙. 원 로그 logs/E173/calib/kcctx_w{1,2,3,4,6}_b15.log 의 '=> KCCTX' 줄.
조작검증(각 w): 맥락 발화 끔 0·켬 > 0. 선택 = 두 쪽 모두 '맥락 단독 반응 KC(ctx) ≤ 10' 이고 '반응 수 켬(on) ≤ 2 × 끔(off)' 을 만족하는 w 중 가장 큰 것(흥분성 맥락 → 가장 작은 자카드).
없으면 'none'(본실험 미실행). 결과를 logs/E173/pick.txt 에 'W=<값>' 으로 쓴다. 정수만 비교.
실행: python3 scripts/e173_pick.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
WS = (1, 2, 3, 4, 6)
SIDE = re.compile(r"side=([lr]) off=(\d+) on=(\d+) ctx=(\d+) keep=(\d+) lost=(\d+) conj=(\d+) jac=([0-9.na]+)")
TAIL = re.compile(r"ctx_n=(\d+) w=([0-9.]+) p=([0-9.]+) level=([0-9.]+) .*맥락 발화 끔 (\d+) 켬 (\d+) \| n_pres=(\d+)")


def parse(t):
    """'=> KCCTX' 줄 하나 → {'l': {...}, 'r': {...}, 'w': float, 'sp_off': int, 'sp_on': int} 또는 None."""
    if not t:
        return None
    ln = next((x for x in t.splitlines() if x.startswith("=> KCCTX")), None)
    if ln is None:
        return None
    sides = {m.group(1): {"off": int(m.group(2)), "on": int(m.group(3)), "ctx": int(m.group(4)), "keep": int(m.group(5)),
                          "lost": int(m.group(6)), "conj": int(m.group(7)), "jac": m.group(8)} for m in SIDE.finditer(ln)}
    tl = TAIL.search(ln)
    if set(sides) != {"l", "r"} or tl is None:
        return None
    return {"l": sides["l"], "r": sides["r"], "w": float(tl.group(2)), "sp_off": int(tl.group(5)), "sp_on": int(tl.group(6))}


def pick(C):
    """C[w] = parse 결과(없으면 None). 반환 (선택 w 또는 None, 조작검증 통과 여부 사전, 제약 통과 여부 사전)."""
    man, cons = {}, {}
    for w in WS:
        c = C.get(w)
        if c is None:
            man[w] = cons[w] = None
            continue
        man[w] = c["sp_off"] == 0 and c["sp_on"] > 0
        cons[w] = man[w] and all(c[s]["ctx"] <= 10 and c[s]["on"] <= 2 * c[s]["off"] for s in "lr")
    ok = [w for w in WS if cons.get(w)]
    return (max(ok) if ok else None), man, cons


def main():
    C = {}
    for w in WS:
        f = os.path.join(EXP, "logs", "E173", "calib", "kcctx_w%d_b15.log" % w)
        C[w] = parse(open(f, encoding="utf-8", errors="replace").read()) if os.path.exists(f) else None
    miss = [w for w in WS if C[w] is None]
    if miss:
        print("[E173 보정] 결측 w=%s — 선택 보류" % miss)
        return 0
    sel, man, cons = pick(C)
    for w in WS:
        c = C[w]
        print("w=%d 맥락 발화 끔 %d 켬 %d(조작 %s) | 좌 off %d on %d ctx %d conj %d jac %s | 우 off %d on %d ctx %d conj %d jac %s | 제약 %s"
              % (w, c["sp_off"], c["sp_on"], "✓" if man[w] else "✗", c["l"]["off"], c["l"]["on"], c["l"]["ctx"], c["l"]["conj"], c["l"]["jac"],
                 c["r"]["off"], c["r"]["on"], c["r"]["ctx"], c["r"]["conj"], c["r"]["jac"], "통과" if cons[w] else "-"))
    out = "W=%s" % (sel if sel is not None else "none")
    open(os.path.join(EXP, "logs", "E173", "pick.txt"), "w", encoding="utf-8").write(out + "\n")
    print("선택: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
