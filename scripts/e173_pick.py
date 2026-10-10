#!/usr/bin/env python3
"""E173 보정 선택 — 기준 logs/E173/criteria_fixed.txt 규칙(정정 1 — 맥락 = KC 좌·우 균일 전류 I, 정정 2 — 격자 확장·최소 효과). 원 로그 logs/E173/calib/kcctx_i{1,2,4,8,12,16,20}_b15.log 의 '=> KCCTX' 줄.
조작검증(각 I): good 제시 KC 발화 합 켬 > 끔(흥분 전류가 KC 에 닿음). 선택 = 두 쪽 모두 '맥락 단독 반응 KC(ctx) ≤ 10' 이고 '반응 수 켬(on) ≤ 2 × 끔(off)' 을 만족하는 I 중 가장 큰 것
(흥분 맥락 → 가장 작은 자카드). 없으면 'none'(본실험 미실행). 정정 2 최소 효과: I* 에서 두 쪽 결합 KC 합 ≥ 10(평균 ≥ 5) 이거나 자카드 합 ≤ 1.80(평균 ≤ 0.90),
못 넘으면 'none(효과 부족)'. 결과를 logs/E173/pick.txt 에 'I=<값>' 으로 쓴다. 정수만 비교(자카드는 1e-4 정수).
실행: python3 scripts/e173_pick.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
IS = (1, 2, 4, 8, 12, 16, 20)
SIDE = re.compile(r"side=([lr]) off=(\d+) on=(\d+) ctx=(\d+) keep=(\d+) lost=(\d+) conj=(\d+) jac=([0-9.na]+)")
TAIL = re.compile(r"ctx_i=([0-9.]+) \| KC 발화 합 good 끔 (\d+) 켬 (\d+) · 맥락 단독 (\d+) · 기준선 (\d+) \| n_pres=(\d+)")


def parse(t):
    """'=> KCCTX' 줄 하나 → {'l': {...}, 'r': {...}, 'i': float, 'k_off': int, 'k_on': int, 'k_ctx': int, 'k_base': int} 또는 None."""
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
    return {"l": sides["l"], "r": sides["r"], "i": float(tl.group(1)), "k_off": int(tl.group(2)), "k_on": int(tl.group(3)),
            "k_ctx": int(tl.group(4)), "k_base": int(tl.group(5))}


def pick(C):
    """C[I] = parse 결과(없으면 None). 반환 (선택 I 또는 None, 조작검증 통과 여부 사전, 제약 통과 여부 사전)."""
    man, cons = {}, {}
    for i in IS:
        c = C.get(i)
        if c is None:
            man[i] = cons[i] = None
            continue
        man[i] = c["k_on"] > c["k_off"]
        cons[i] = man[i] and all(c[s]["ctx"] <= 10 and c[s]["on"] <= 2 * c[s]["off"] for s in "lr")
    ok = [i for i in IS if cons.get(i)]
    return (max(ok) if ok else None), man, cons


def min_effect(c):
    """정정 2: 두 쪽 결합 KC 합 ≥ 10 이거나 자카드 합(1e-4 정수) ≤ 18000. 자카드 nan 이면 자카드 조건 실패."""
    conj = c["l"]["conj"] + c["r"]["conj"]
    try:
        jsum = int(round(float(c["l"]["jac"]) * 1e4)) + int(round(float(c["r"]["jac"]) * 1e4))
        jok = jsum <= 18000
    except ValueError:
        jok = False
    if c["l"]["jac"] in ("nan",) or c["r"]["jac"] in ("nan",):
        jok = False
    return conj >= 10 or jok


def main():
    C = {}
    for i in IS:
        f = os.path.join(EXP, "logs", "E173", "calib", "kcctx_i%d_b15.log" % i)
        C[i] = parse(open(f, encoding="utf-8", errors="replace").read()) if os.path.exists(f) else None
    miss = [i for i in IS if C[i] is None]
    if miss:
        print("[E173 보정] 결측 I=%s — 선택 보류" % miss)
        return 0
    sel, man, cons = pick(C)
    for i in IS:
        c = C[i]
        print("I=%d KC 발화 합 good 끔 %d 켬 %d(조작 %s) 단독 %d 기준선 %d | 좌 off %d on %d ctx %d conj %d jac %s | 우 off %d on %d ctx %d conj %d jac %s | 제약 %s"
              % (i, c["k_off"], c["k_on"], "✓" if man[i] else "✗", c["k_ctx"], c["k_base"], c["l"]["off"], c["l"]["on"], c["l"]["ctx"], c["l"]["conj"],
                 c["l"]["jac"], c["r"]["off"], c["r"]["on"], c["r"]["ctx"], c["r"]["conj"], c["r"]["jac"], "통과" if cons[i] else "-"))
    if sel is None:
        out = "I=none"
    elif not min_effect(C[sel]):
        out = "I=none(효과 부족 — I*=%d 결합 %d·%d 자카드 %s·%s)" % (sel, C[sel]["l"]["conj"], C[sel]["r"]["conj"], C[sel]["l"]["jac"], C[sel]["r"]["jac"])
    else:
        out = "I=%s" % sel
    open(os.path.join(EXP, "logs", "E173", "pick.txt"), "w", encoding="utf-8").write(out + "\n")
    print("선택: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
