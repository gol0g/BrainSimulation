#!/usr/bin/env python3
"""E152 독립 대조 — judge_e152.py 를 쓰지 않고 뇌별 원 로그(logs/E152/b*.log 의 '=> KCOVERLAP3' 줄)와
E149 원 로그(logs/E149/b*_base.log 의 '=> KCOVERLAP' 줄)에서 다시 계산한다. 필드는 key=value 로 읽는다.
기준 수정 1: M2' = 두 쪽 모두 (먹이 반분 자카드 ≥ 0.8 또는 100·F ≤ 15·O) — 원래(M2)·수정(M2') 판정을 둘 다 낸다.
실행: python3 scripts/verify_e152_independent.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)


def kv_line(path, head):
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith(head):
            parts = ln[len(head):].split("|")
            sides = {}
            for p in parts:
                d = dict(tok.split("=", 1) for tok in p.split() if "=" in tok)
                if d.get("side") in ("l", "r"):
                    sides[d["side"]] = d
            return sides
    raise RuntimeError("줄 없음: %s %s" % (path, head))


def q(x):
    """4자리 소수 문자열 → 정수(1e-4). nan → None."""
    if x == "nan":
        return None
    a, b = x.split(".")
    return int(a) * 10000 + int((b + "0000")[:4])


def decide(passed, food, conj):
    if passed < 4:
        return "보류(측정 검증 실패)"
    if food >= 4:
        return "먹이 주도(H075)"
    if conj >= 4:
        return "결합 주도(H075-conj)"
    return "혼합(보류)"


def main():
    tally = {"orig": [0, 0, 0], "amend": [0, 0, 0]}   # 통과, 먹이 주도, 결합 주도
    for b in BRAINS:
        s = kv_line(os.path.join(EXP, "logs", "E152", "b%d.log" % b), "=> KCOVERLAP3 ")
        e = kv_line(os.path.join(EXP, "logs", "E149", "b%d_base.log" % b), "=> KCOVERLAP ")
        O = int(s["l"]["both"]) + int(s["r"]["both"])
        OF = int(s["l"]["both_food"]) + int(s["r"]["both_food"])
        dJ = [None if q(s[k]["jac_gb"]) is None else abs(q(s[k]["jac_gb"]) - q(e[k]["jac"])) for k in "lr"]
        fs = [q(s[k]["food_split_jac"]) for k in "lr"]
        small = [100 * int(s[k]["food"]) <= 15 * int(s[k]["both"]) for k in "lr"]
        m1 = all(d is not None and d <= 500 for d in dJ)
        m2 = all(f is not None and f >= 8000 for f in fs)
        m2a = all((fs[i] is not None and fs[i] >= 8000) or small[i] for i in range(2))
        for key, mm in (("orig", m2), ("amend", m2a)):
            ok = m1 and mm and O >= 20
            tally[key][0] += ok
            tally[key][1] += ok and 10 * OF >= 7 * O
            tally[key][2] += ok and 10 * OF <= 3 * O
        print("b%d O %d OF %d φ %s | ΔJ %s | 먹이 수 %s·%s 반분 %s | M2 %s M2' %s" % (b, O, OF, ("%.3f" % (OF / O)) if O else "nan", dJ,
              s["l"]["food"], s["r"]["food"], fs, "✓" if m2 else "✗", "✓" if m2a else "✗"))
    for key, name in (("orig", "원래 기준"), ("amend", "수정 1 기준")):
        t = tally[key]
        print("%s: 측정 검증 통과 %d/5 | 먹이 주도 %d/5 결합 주도 %d/5 → %s" % (name, t[0], t[1], t[2], decide(*t)))
    print("독립 판정(원래): %s" % decide(*tally["orig"]))
    print("독립 판정: %s" % decide(*tally["amend"]))


if __name__ == "__main__":
    sys.exit(main())
