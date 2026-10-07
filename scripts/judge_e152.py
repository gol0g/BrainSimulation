#!/usr/bin/env python3
"""E152 판정 — 기준 logs/E152/criteria_fixed.txt(경로 검사·본측정 전 고정). 측정 5줄이 다 모이기 전에는 수치를 출력하지 않는다.
요약 줄: "  e152 b10: => KCOVERLAP3 side=l good=.. bad=.. food=.. both=.. both_food=.. food_in=.. jac_gb=.. food_split_jac=.. | side=r ... | ..."
E149 기준 J: E149.log "  e149 b10 base: => KCOVERLAP side=l good=.. bad=.. jac=.. ... | side=r ... jac=.. ...".
φ = (both_food_l + both_food_r)/(both_l + both_r). 정수 비교(1e-4·개수).
기준 수정 1(본측정 전, criteria_fixed.txt 뒤쪽): M2' = 두 쪽 모두 (먹이 반분 자카드 ≥ 0.8 또는 100·F ≤ 15·O). 원래 기준(M2)·수정 기준(M2') 판정을 둘 다 낸다."""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
F3 = r"side=%s good=(\d+) bad=(\d+) food=(\d+) both=(\d+) both_food=(\d+) food_in=(\d+) jac_gb=([0-9.]+|nan) food_split_jac=([0-9.]+|nan)"
L152 = re.compile(r"^\s*e152 b(\d+): => KCOVERLAP3 " + F3 % "l" + r" \| " + F3 % "r")
L149 = re.compile(r"^\s*e149 b(\d+) base: => KCOVERLAP side=l good=\d+ bad=\d+ jac=([0-9.]+) .*\| side=r good=\d+ bad=\d+ jac=([0-9.]+) ")


def i4(x):
    return None if x == "nan" else int(round(float(x) * 1e4))


def parse152(m):
    out = {"b": int(m.group(1))}
    for k, off in (("l", 2), ("r", 10)):
        g = [m.group(off + i) for i in range(8)]
        out[k] = {"G": int(g[0]), "B": int(g[1]), "F": int(g[2]), "O": int(g[3]), "OF": int(g[4]), "Fin": int(g[5]), "J": i4(g[6]), "FS": i4(g[7])}
    return out


def verdict(rows, key):
    ok_m = sum(r["m1"] and r[key] and r["m3"] for r in rows.values())
    food = sum(r["m1"] and r[key] and r["m3"] and 10 * r["OF"] >= 7 * r["O"] for r in rows.values())
    conj = sum(r["m1"] and r[key] and r["m3"] and 10 * r["OF"] <= 3 * r["O"] for r in rows.values())
    if ok_m < 4:
        v = "보류(측정 검증 실패)"
    elif food >= 4:
        v = "먹이 주도(H075) — 겹침 KC 대부분이 먹이 단독에도 반응: 공통 입력 단독 구동"
    elif conj >= 4:
        v = "결합 주도(H075-conj) — 겹침 KC 대부분이 먹이 단독에는 반응 안 함: 공통 + 종류 입력 결합"
    else:
        v = "혼합(보류)"
    return ok_m, food, conj, v


def judge(M, J149):
    miss = [b for b in BRAINS if b not in M or b not in J149]
    if miss:
        return ["[측정 확인] 결측 %d — **판정 보류, 수치 미출력**" % len(miss)], None
    rows = {}
    for b in BRAINS:
        d = M[b]
        m1 = all(d[s]["J"] is not None and abs(d[s]["J"] - J149[b][s]) <= 500 for s in "lr")
        m2 = all(d[s]["FS"] is not None and d[s]["FS"] >= 8000 for s in "lr")
        m2a = all((d[s]["FS"] is not None and d[s]["FS"] >= 8000) or 100 * d[s]["F"] <= 15 * d[s]["O"] for s in "lr")
        O = d["l"]["O"] + d["r"]["O"]; OF = d["l"]["OF"] + d["r"]["OF"]
        rows[b] = {"m1": m1, "m2": m2, "m2a": m2a, "m3": O >= 20, "O": O, "OF": OF, "phi": OF / O if O else float("nan"),
                   "J": (d["l"]["J"], d["r"]["J"]), "J149": (J149[b]["l"], J149[b]["r"]), "FS": (d["l"]["FS"], d["r"]["FS"]),
                   "F": d["l"]["F"] + d["r"]["F"], "Fin": d["l"]["Fin"] + d["r"]["Fin"], "G": d["l"]["G"] + d["r"]["G"], "B": d["l"]["B"] + d["r"]["B"]}
    ok0, food0, conj0, v0 = verdict(rows, "m2")
    ok1, food1, conj1, v1 = verdict(rows, "m2a")
    checks = ["[측정 검증] 원래 기준(M1·M2·M3) 통과 %d/5 · 수정 1 기준(M1·M2'·M3) 통과 %d/5" % (ok0, ok1)]
    return checks, {"rows": rows, "food": food1, "conj": conj1, "ok_m": ok1, "verdict": v1,
                    "food0": food0, "conj0": conj0, "ok_m0": ok0, "verdict0": v0}


def report(checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        r = res["rows"][b]
        print("b%d: 겹침 O %d 중 먹이 단독도 반응 %d → φ %.2f | J %.4f/%.4f (E149 %.4f/%.4f) | 먹이 반응 %d(그중 G∪B %d) 반분 %.4f/%.4f | G %d B %d | M1 %s M2 %s M3 %s"
              % (b, r["O"], r["OF"], r["phi"], r["J"][0] / 1e4, r["J"][1] / 1e4, r["J149"][0] / 1e4, r["J149"][1] / 1e4,
                 r["F"], r["Fin"], (r["FS"][0] or 0) / 1e4, (r["FS"][1] or 0) / 1e4, r["G"], r["B"],
                 "✓" if r["m1"] else "✗", ("✓" if r["m2"] else "✗") + "(M2' " + ("✓" if r["m2a"] else "✗") + ")", "✓" if r["m3"] else "✗"))
    print("원래 기준: 먹이 주도 %d/5 · 결합 주도 %d/5 → 판정(원래): %s" % (res["food0"], res["conj0"], res["verdict0"]))
    print("수정 1 기준: 먹이 주도(φ ≥ 0.7) %d/5 · 결합 주도(φ ≤ 0.3) %d/5" % (res["food"], res["conj"]))
    print("판정: %s" % res["verdict"])


def load():
    M, J149 = {}, {}
    try:
        for ln in open(os.path.join(EXP, "E152.log"), encoding="utf-8", errors="replace"):
            m = L152.match(ln)
            if m:
                d = parse152(m); M[d["b"]] = d
    except FileNotFoundError:
        pass
    try:
        for ln in open(os.path.join(EXP, "E149.log"), encoding="utf-8", errors="replace"):
            m = L149.match(ln)
            if m:
                J149[int(m.group(1))] = {"l": i4(m.group(2)), "r": i4(m.group(3))}
    except FileNotFoundError:
        pass
    return M, J149


if __name__ == "__main__":
    c, r = judge(*load())
    report(c, r)
    sys.exit(0)
