#!/usr/bin/env python3
"""E157 k 보정 선택 — 규칙 logs/E157/criteria_fixed.txt(2026-10-09 01:06:04 고정).
총수 = kcrate '제시 스파이크' kc_l + kc_r. Fk 목표 48,856(뇌 15 기본), 격자 0.70~0.90; Dk 목표 61,313(뇌 15 형성), 격자 1.10~1.50.
목표에 가장 가까운 격자값(동점이면 1 에 가까운 값), 그 총수가 목표 ±10% 밖이면 보정 실패.
경로 확인(조건 2): k=1 재현(기본 48,856·형성 61,313 정확히 — E156 진단), 후보 로그의 배율 검증 줄 ≥ 2·Fk 적재 줄 ≥ 2 — 어긋난 후보는 빼고 사유 출력.
실행: python3 scripts/e157_kstar.py (저장소 루트에서) → logs/E157/kstar.txt"""
import os
import re
import sys

EXP = "research/experiments"
TARGET = {"Fk": 48856, "Dk": 61313}
GRID = {"Fk": ("0.70", "0.75", "0.80", "0.85", "0.90"), "Dk": ("1.10", "1.20", "1.30", "1.40", "1.50")}
REPRO = {"D_k1.00": 48856, "F_k1.00": 61313}


def parse(txt):
    sp = {m.group(1): int(m.group(2)) for m in re.finditer(r"^=> KCRATE kc_([lr]) .*?제시 스파이크 (\d+) ", txt, re.M)}
    tot = sp["l"] + sp["r"] if set(sp) == {"l", "r"} else None
    nld = len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", txt, re.M))
    nsc = len(re.findall(r"^\[E157 종류 입력 배율\].*검증 일치", txt, re.M))
    return tot, nld, nsc


def select(arm, rows):
    """rows: {k문자열: (총수, 적재, 배율)} → (k 또는 None, 총수, 사유 목록)"""
    ok, why = {}, []
    for k in GRID[arm]:
        if k not in rows:
            why.append("%s k=%s 결측" % (arm, k))
            continue
        tot, nld, nsc = rows[k]
        bad = []
        if tot is None:
            bad.append("발화 줄 없음")
        if nsc < 2:
            bad.append("배율 %d줄" % nsc)
        if arm == "Fk" and nld < 2:
            bad.append("적재 %d줄" % nld)
        if arm == "Dk" and nld != 0:
            bad.append("기본인데 적재 %d줄" % nld)
        if bad:
            why.append("%s k=%s 제외: %s" % (arm, k, "; ".join(bad)))
        else:
            ok[k] = tot
    if not ok:
        why.append("%s 후보 없음: 보정 실패" % arm)
        return None, None, why
    tg = TARGET[arm]
    k = min(ok, key=lambda x: (abs(ok[x] - tg), abs(float(x) - 1.0)))
    if 10 * abs(ok[k] - tg) > tg:
        why.append("%s 최근접 k=%s 총수 %d — 목표 %d ±10%% 밖: 보정 실패" % (arm, k, ok[k], tg))
        return None, ok[k], why
    return k, ok[k], why


def main():
    rows = {"Fk": {}, "Dk": {}}
    rep = {}
    d = os.path.join(EXP, "logs", "E157", "calib")
    for arm in rows:
        for k in GRID[arm]:
            f = os.path.join(d, "kcrate_%s_k%s_b15.log" % (arm, k))
            if os.path.exists(f):
                rows[arm][k] = parse(open(f, encoding="utf-8", errors="replace").read())
    for tag, want in REPRO.items():
        f = os.path.join(d, "kcrate_%s_b15.log" % tag)
        got = parse(open(f, encoding="utf-8", errors="replace").read())[0] if os.path.exists(f) else None
        rep[tag] = (got, want)
    repro_ok = all(g == w for g, w in rep.values())
    for tag, (g, w) in rep.items():
        print("재현 %s 총수 %s (E156 진단 %d) %s" % (tag, g, w, "일치" if g == w else "불일치"))
    out = []
    for arm in ("Fk", "Dk"):
        for k in GRID[arm]:
            if k in rows[arm]:
                t, nld, nsc = rows[arm][k]
                print("%s k=%s 총수 %s (목표 %d) 적재 %d 배율 %d" % (arm, k, t, TARGET[arm], nld, nsc))
        k, t, why = select(arm, rows[arm])
        for y in why:
            print("  " + y)
        out.append((arm, k, t))
    if not repro_ok:
        line = "보정 실패(경로 — k=1 재현 불일치)"
    elif any(k is None for _, k, _ in out):
        line = "보정 실패(" + ", ".join("%s %s" % (a, "없음" if k is None else "k=" + k) for a, k, _ in out) + ")"
    else:
        line = " | ".join("%s k=%s 총수 %d (목표 %d, 차 %+.1f%%)" % (a, k, t, TARGET[a], 100.0 * (t - TARGET[a]) / TARGET[a]) for a, k, t in out)
    print("=> " + line)
    open(os.path.join(EXP, "logs", "E157", "kstar.txt"), "w", encoding="utf-8").write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
