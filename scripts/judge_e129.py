#!/usr/bin/env python3
"""E129 판정 — E129.md 4절. 비교기 배선(--kc-wiring comparator) 균형 같음/다름(cyclic). 24런이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): 라벨 균형 정답률 = (L 정답률 + R 정답률)/2 (samediff 규칙이라 같음/다름 균형과 같은 양), 평가 100시행(같음·다름 무작위 반반,
탐색·학습 없음, 동점 무작위). train = 훈련 8자극(같음 4 + 순환 다름 4), novel = 처음 보는 항목(4~7) 쌍. 우연 수준 50%.
"""
import os
import re
import sys

EXP = "research/experiments"
WIRES = tuple(range(10, 18))
TSEEDS = (600, 601)
NOVEL_OK = 75.0     # 관계 전이 성공
NOVEL_NULL = 60.0   # 전이 없음 상한
TRAIN_OK = 80.0     # 훈련 쌍 획득
TR = re.compile(r"^\s*cp (learn|frozen) w(\d+) t(\d+): => SDLAB diff=cyclic rule=samediff mode=(\w+) seed=(\d+) trialseed=(\d+) "
                r"train_accL=([0-9.]+) train_accR=([0-9.]+) train_lbal=([0-9.]+) novel_accL=([0-9.]+) novel_accR=([0-9.]+) novel_lbal=([0-9.]+)")


def judge(R):
    """R: {(mode, w, t): {"train_same","train_diff","train_bal","novel_same","novel_diff","novel_bal"}}"""
    need = [("learn", w, t) for w in WIRES for t in TSEEDS] + [("frozen", w, 600) for w in WIRES]
    missing = [k for k in need if k not in R]
    if missing:
        return ["[측정 확인] 결측 %d/24: %s — **판정 보류, 수치 미출력**" % (len(missing), missing[:6])], None
    bad = [k for k in need if any(v != v for v in R[k].values())]
    if bad:
        return ["[측정 확인] nan: %s — **판정 보류**" % bad], None
    checks, ok = [], True
    same = [w for w in WIRES if R[("learn", w, 600)] == R[("frozen", w, 600)]]
    checks.append("[측정 확인] learn ≠ frozen(배선별, t600): %d/8%s" % (8 - len(same), "" if not same else " ← 같은 값 %s" % same))
    ok &= not same
    L = [R[("learn", w, t)] for w in WIRES for t in TSEEDS]
    n_train = sum(x["train_bal"] >= TRAIN_OK for x in L)
    n_novel = sum(x["novel_bal"] >= NOVEL_OK for x in L)
    n_null = sum(x["novel_bal"] < NOVEL_NULL for x in L)
    n_frozen = sum(R[("frozen", w, 600)]["novel_bal"] >= NOVEL_OK for w in WIRES)
    checks.append("[1차 — 획득] learn 훈련 쌍 균형 정답률 ≥ %.0f%%: %d/16 (≥12 획득, ≤4 획득 불가)" % (TRAIN_OK, n_train))
    if not ok:
        verdict = "보류(조작검증 실패 — 측정부터)"
    elif n_train <= 4:
        verdict = "불획득 — 비교기 배선에서도 같음/다름 훈련 자극을 학습 못 함"
    elif n_train < 12:
        verdict = "보류(훈련 쌍 획득 혼재 — 전이 시험이 성립하지 않음)"
    elif n_novel >= 12 and n_frozen <= 2:
        verdict = "획득 + 전이 지지 — 비교기 특징으로 처음 보는 항목에 같음/다름 규칙을 적용한다"
    elif n_null >= 12:
        verdict = "획득 + 전이 없음 — 비교기 배선에서 훈련 자극은 획득하나 새 항목 전이 없음"
    else:
        verdict = "보류(혼재 또는 frozen 성공 과다)"
    return checks, {"n_train": n_train, "n_novel": n_novel, "n_null": n_null, "n_frozen": n_frozen, "ok": ok, "verdict": verdict}


def report(checks, res, R=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        L = [R[("learn", w, t)] for t in TSEEDS]; F = R[("frozen", w, 600)]
        print("w%d learn train_bal %s novel_bal %s (novel 같음/다름 %s) || frozen train %.0f novel %.0f"
              % (w, "/".join("%.0f" % x["train_bal"] for x in L), "/".join("%.0f" % x["novel_bal"] for x in L),
                 " ".join("%.0f/%.0f" % (x["novel_same"], x["novel_diff"]) for x in L), F["train_bal"], F["novel_bal"]))
    for mode, keys in (("learn", [("learn", w, t) for w in WIRES for t in TSEEDS]), ("frozen", [("frozen", w, 600) for w in WIRES])):
        print("%s 평균(%%): train 같음 %.1f 다름 %.1f 균형 %.1f | novel 같음 %.1f 다름 %.1f 균형 %.1f" % ((mode,) + tuple(
            sum(R[k][f] for k in keys) / len(keys) for f in ("train_same", "train_diff", "train_bal", "novel_same", "novel_diff", "novel_bal"))))
    print("learn novel_bal ≥ %.0f%%: %d/16, < %.0f%%: %d/16 | frozen novel_bal ≥ %.0f%%: %d/8 | learn train_bal ≥ %.0f%%: %d/16"
          % (NOVEL_OK, res["n_novel"], NOVEL_NULL, res["n_null"], NOVEL_OK, res["n_frozen"], TRAIN_OK, res["n_train"]))
    print("판정: %s" % res["verdict"])


def parse_line(ln):
    m = TR.match(ln)
    if not m:
        return None
    v = [float(x) for x in m.groups()[6:12]]
    return (m.group(1), int(m.group(2)), int(m.group(3))), dict(zip(
        ("train_same", "train_diff", "train_bal", "novel_same", "novel_diff", "novel_bal"), v))   # same=L, diff=R


def load():
    R = {}
    try:
        for ln in open(os.path.join(EXP, "E129.log"), encoding="utf-8"):
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
