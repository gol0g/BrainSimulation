#!/usr/bin/env python3
"""E159 판정 — 기준 logs/E159/criteria_fixed.txt(2026-10-09 14:35:05, 수정 1 14:35:46). 6칸 × 뇌 5개가 다 모이기 전에는 수치를 출력하지 않는다.
원 로그(logs/E159/{칸}_b{B}.log)의 '=> DECOMP ... mod= ... pushed=' 줄과 적재·배율 검증 줄을 읽는다.
E(W,R) = v(W,R) − v(none,R). O = E(W0,R25)/E(W0,R0), C = E(W25,R0)/E(W0,R0). 1e-4 정수: O ≤ 0.70 ⇔ 10·E(W0,R25) ≥ 7·E(W0,R0).
실행: python3 scripts/judge_e159.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
CELLS = ("none_R0", "none_R25", "W0_R0", "W0_R25", "W25_R0", "W25_R25")
R1 = {10: 0.3519, 11: 0.3617, 12: 0.3513, 13: 0.3338, 14: 0.3714}     # E157 r25 Fk [사전]
R2 = {10: 0.0169, 11: 0.0088, 12: 0.0197, 13: 0.0326, 14: 0.0223}     # E157 learn Fk [사전]
R3 = {10: -0.4316, 11: -0.4306, 12: -0.4337, 13: -0.4114, 14: -0.4473}  # E157 learn Fk [사후]
R4 = {10: 0.0528, 11: 0.0673, 12: 0.0596, 13: 0.0501, 14: 0.0748}     # E158 F500 [사후]
CATS = ("출력(H082)", "내용(H082-content)", "둘 다(H082-both)", "둘 다 아님(H082-neither)")
DL = re.compile(r"^=> DECOMP mode=(\w+) mod=([-+0-9.]+) .*pushed=(\d+)", re.M)


def i4(x):
    return int(round(x * 1e4))


def parse(txt):
    m = DL.search(txt)
    if not m:
        return None
    return {"mode": m.group(1), "v": i4(float(m.group(2))), "pushed": int(m.group(3)),
            "ld": len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", txt, re.M)),
            "sc": len(re.findall(r"^\[E157 종류 입력 배율\].*검증 일치", txt, re.M))}


def load():
    X = {}
    for b in BRAINS:
        for c in CELLS:
            f = os.path.join(EXP, "logs", "E159", "%s_b%d.log" % (c, b))
            if os.path.exists(f):
                p = parse(open(f, encoding="utf-8", errors="replace").read())
                if p is not None:
                    X[(c, b)] = p
    return X


def judge(X):
    miss = [(c, b) for b in BRAINS for c in CELLS if (c, b) not in X]
    if miss:
        return ["[측정 확인] 결측 %d칸(예: %s) — **판정 보류, 수치 미출력**" % (len(miss), miss[:4])], None
    v = {k: X[k]["v"] for k in X}
    r1 = sum(abs(v[("none_R25", b)] - i4(R1[b])) <= 20 for b in BRAINS)
    r2 = sum(abs(v[("none_R0", b)] - i4(R2[b])) <= 20 for b in BRAINS)
    r3 = sum(abs(v[("W0_R0", b)] - i4(R3[b])) <= 20 for b in BRAINS)
    r4 = sum(abs(v[("W25_R25", b)] - i4(R4[b])) <= 20 for b in BRAINS)
    r5 = sum(all(X[(c, b)]["pushed"] == (0 if c.startswith("none") else 8) and X[(c, b)]["ld"] >= 2 and X[(c, b)]["sc"] >= 2 for c in CELLS) for b in BRAINS)
    E = {}
    for b in BRAINS:
        for w in ("W0", "W25"):
            for r in ("R0", "R25"):
                E[(w, r, b)] = v[("%s_%s" % (w, r), b)] - v[("none_%s" % r, b)]
    r6 = sum(E[("W0", "R0", b)] <= -1000 for b in BRAINS)
    ok = r1 == r2 == r3 == r4 == r5 == r6 == 5
    checks = ["[조작검증] R1 무학습 반사25 재현 %d/5 · R2 무학습 반사0 재현 %d/5 · R3 W0 반사0 재현 %d/5 · R4 W25 반사25 재현 %d/5 · R5 이식·적재·배율 %d/5 · R6 기준 효과 %d/5 %s"
              % (r1, r2, r3, r4, r5, r6, "통과" if ok else "실패")]
    cat = {}
    for b in BRAINS:
        base = E[("W0", "R0", b)]
        o_small = 10 * E[("W0", "R25", b)] >= 7 * base     # O ≤ 0.70
        c_small = 10 * E[("W25", "R0", b)] >= 7 * base     # C ≤ 0.70
        cat[b] = CATS[0] if (o_small and not c_small) else CATS[1] if (c_small and not o_small) else CATS[2] if (o_small and c_small) else CATS[3]
    cnt = {c: sum(cat[b] == c for b in BRAINS) for c in CATS}
    top = [c for c in CATS if cnt[c] >= 4]
    verdict = "보류(조작검증 실패)" if not ok else (top[0] if top else "보류")
    return checks, {"E": E, "cat": cat, "cnt": cnt, "verdict": verdict, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    E = res["E"]
    for b in BRAINS:
        base = E[("W0", "R0", b)]
        print("b%d v none R0 %+.4f R25 %+.4f | E W0 R0 %+.4f R25 %+.4f | E W25 R0 %+.4f R25 %+.4f | O %.2f C %.2f | 관측 상한 비 %.2f O·C %.2f W25/W0(반사25) %.2f | %s"
              % (b, X[("none_R0", b)]["v"] / 1e4, X[("none_R25", b)]["v"] / 1e4, base / 1e4, E[("W0", "R25", b)] / 1e4,
                 E[("W25", "R0", b)] / 1e4, E[("W25", "R25", b)] / 1e4, E[("W0", "R25", b)] / base, E[("W25", "R0", b)] / base,
                 E[("W25", "R25", b)] / base, (E[("W0", "R25", b)] / base) * (E[("W25", "R0", b)] / base),
                 (E[("W25", "R25", b)] / E[("W0", "R25", b)]) if E[("W0", "R25", b)] else float("nan"), res["cat"][b]))
    print("범주: " + " · ".join("%s %d/5" % (c, res["cnt"][c]) for c in CATS))
    print("판정: %s" % res["verdict"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
