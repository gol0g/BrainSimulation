#!/usr/bin/env python3
"""E152 독립 대조 — judge_e152.py 를 쓰지 않고 뇌별 원 로그(logs/E152/b*.log 의 '=> KCOVERLAP3' 줄)와
E149 원 로그(logs/E149/b*_base.log 의 '=> KCOVERLAP' 줄)에서 다시 계산한다. 필드는 key=value 로 읽는다.
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


def main():
    passed = food = conj = 0
    for b in BRAINS:
        s = kv_line(os.path.join(EXP, "logs", "E152", "b%d.log" % b), "=> KCOVERLAP3 ")
        e = kv_line(os.path.join(EXP, "logs", "E149", "b%d_base.log" % b), "=> KCOVERLAP ")
        O = int(s["l"]["both"]) + int(s["r"]["both"])
        OF = int(s["l"]["both_food"]) + int(s["r"]["both_food"])
        dJ = [None if q(s[k]["jac_gb"]) is None else abs(q(s[k]["jac_gb"]) - q(e[k]["jac"])) for k in "lr"]
        fs = [q(s[k]["food_split_jac"]) for k in "lr"]
        ok = all(d is not None and d <= 500 for d in dJ) and all(f is not None and f >= 8000 for f in fs) and O >= 20
        passed += ok
        food += ok and 10 * OF >= 7 * O
        conj += ok and 10 * OF <= 3 * O
        print("b%d O %d OF %d φ %s | ΔJ %s | 먹이 반분 %s | %s" % (b, O, OF, ("%.3f" % (OF / O)) if O else "nan", dJ, fs, "통과" if ok else "검증 실패"))
    print("측정 검증 통과 %d/5 | 먹이 주도 %d/5 결합 주도 %d/5" % (passed, food, conj))
    if passed < 4:
        v = "보류(측정 검증 실패)"
    elif food >= 4:
        v = "먹이 주도(H075)"
    elif conj >= 4:
        v = "결합 주도(H075-conj)"
    else:
        v = "혼합(보류)"
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
