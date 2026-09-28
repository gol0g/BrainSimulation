#!/usr/bin/env python3
"""E122 판정 — E122.md 4절. 24런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): 정답률(%) = 평가 100시행(탐색·학습 없음) 중 범주 규칙(A→L, B→R)대로 고른 비율. 동점은 무작위 선택.
held = 훈련에 한 번도 안 나온 평가 사례(d_test = 0.20) 정답률, train = 훈련 사례 정답률, proto = 원형(훈련에 없음) 정답률.
성공 = held ≥ 80%.
"""
import os
import re
import sys

EXP = "research/experiments"
WIRES = tuple(range(10, 18))
TSEEDS = (600, 601)
OK_THR = 80.0
TR = re.compile(r"^\s*(learn|frozen) w(\d+) t(\d+): => EXGEN mode=(\w+) seed=(\d+) trialseed=(\d+) proto=([0-9.]+) train=([0-9.]+) (.*?) \|")


def judge(R):
    """R: {(mode, wire, tseed): {"proto":, "train":, 0.1:, 0.2:, 0.3:, 0.4:}}"""
    need = [("learn", w, t) for w in WIRES for t in TSEEDS] + [("frozen", w, 600) for w in WIRES]
    missing = [k for k in need if k not in R]
    if missing:
        return ["[측정 확인] 결측 %d/24: %s — **판정 보류, 수치 미출력**" % (len(missing), missing)], None
    bad = [k for k in need if any(v != v for v in R[k].values()) or 0.2 not in R[k]]
    if bad:
        return ["[측정 확인] nan 또는 d0.20 누락: %s — **판정 보류**" % bad], None
    checks, ok = [], True
    # 조작검증 1: learn 과 frozen 이 다르다(같은 배선·난수열 600에서 held·train·proto 전부 같으면 학습 무효 의심)
    same = [w for w in WIRES if all(R[("learn", w, 600)][k] == R[("frozen", w, 600)][k] for k in ("proto", "train", 0.2))]
    checks.append("[측정 확인] learn ≠ frozen(배선별, t600): %d/8%s" % (8 - len(same), "" if not same else " ← 같은 값 %s" % same))
    ok &= not same
    # 조작검증 2: 훈련 사례를 학습했는가(과제 난이도 확인) — learn train ≥ 80% 런 수
    n_train = sum(R[("learn", w, t)]["train"] >= OK_THR for w in WIRES for t in TSEEDS)
    checks.append("[측정 확인] learn 훈련 사례 정답률 ≥ 80%%: %d/16" % n_train)
    n_learn = sum(R[("learn", w, t)][0.2] >= OK_THR for w in WIRES for t in TSEEDS)
    n_frozen = sum(R[("frozen", w, 600)][0.2] >= OK_THR for w in WIRES)
    if not ok:
        verdict = "보류(조작검증 실패 — 측정부터)"
    elif n_learn >= 12 and n_frozen <= 2:
        verdict = "지지 — 미학습 사례를 범주대로 분류한다(유사도 기반 범주 일반화)"
    elif n_learn <= 4 and n_train >= 12:
        verdict = "기각 — 훈련 사례는 학습했으나 미학습 사례로 일반화하지 못한다"
    elif n_train < 12:
        verdict = "보류(훈련 사례 자체를 학습 못 함 — 과제 난이도·측정부터)"
    else:
        verdict = "보류(혼재 또는 frozen 성공 과다)"
    return checks, {"n_learn": n_learn, "n_frozen": n_frozen, "n_train": n_train, "ok": ok, "verdict": verdict}


def report(checks, res, R=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        L = [R[("learn", w, t)] for t in TSEEDS]; F = R[("frozen", w, 600)]
        print("w%d learn held(d0.20) %s | train %s | proto %s || frozen held %.0f train %.0f proto %.0f"
              % (w, "/".join("%.0f" % x[0.2] for x in L), "/".join("%.0f" % x["train"] for x in L),
                 "/".join("%.0f" % x["proto"] for x in L), F[0.2], F["train"], F["proto"]))
    lv = sorted(k for k in R[("learn", WIRES[0], 600)] if isinstance(k, float))
    for mode, keys in (("learn", [("learn", w, t) for w in WIRES for t in TSEEDS]), ("frozen", [("frozen", w, 600) for w in WIRES])):
        print("%s 평균 정답률(%%): proto %.1f train %.1f %s" % (mode, sum(R[k]["proto"] for k in keys) / len(keys),
              sum(R[k]["train"] for k in keys) / len(keys), " ".join("d%.2f %.1f" % (d, sum(R[k][d] for k in keys) / len(keys)) for d in lv)))
    print("성공(held d0.20 ≥ 80%%): learn %d/16, frozen %d/8 | learn 훈련 사례 ≥80%%: %d/16" % (res["n_learn"], res["n_frozen"], res["n_train"]))
    print("판정: %s" % res["verdict"])


def parse_line(ln):
    m = TR.match(ln)
    if not m:
        return None
    d = {"proto": float(m.group(7)), "train": float(m.group(8))}
    for lv, v in re.findall(r"d([0-9.]+)=([0-9.]+)", m.group(9)):
        d[round(float(lv), 2)] = float(v)
    return (m.group(1), int(m.group(2)), int(m.group(3))), d


def load():
    R = {}
    try:
        for ln in open(os.path.join(EXP, "E122.log"), encoding="utf-8"):
            p = parse_line(ln)
            if p:
                R[p[0]] = p[1]
    except FileNotFoundError:
        pass
    return R


if __name__ == "__main__":
    R = load()
    c, r = judge(R)
    report(c, r, R)
    sys.exit(0)
