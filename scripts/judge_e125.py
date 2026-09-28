#!/usr/bin/env python3
"""E125 판정 — E125.md 4절. half1 24런(learn 16 + frozen 8)이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): 라벨 균형 정답률 lbal(%) = (정답 L 인 시행 정답률 + 정답 R 인 시행 정답률)/2,
훈련 항목(0~3) 16자극, 평가 100시행(탐색·학습 없음, 동점 무작위). samediff 는 E124 SDGEN train_bal(라벨 = 같음/다름이므로 같은 양).
"""
import os
import re
import sys

EXP = "research/experiments"
WIRES = tuple(range(10, 18))
TSEEDS = (600, 601)
OK = 80.0
TR25 = re.compile(r"^\s*h1 (learn|frozen) w(\d+) t(\d+): => SDLAB rule=half1 mode=(\w+) seed=(\d+) trialseed=(\d+) "
                  r"train_accL=([0-9.]+) train_accR=([0-9.]+) train_lbal=([0-9.]+)")
TR24 = re.compile(r"^\s*(learn|frozen) w(\d+) t(\d+): => SDGEN .*? train_bal=([0-9.]+) ")


def judge(H, S):
    """H: {(mode, w, t): (accL, accR, lbal)} half1, S: {(mode, w, t): train_bal} samediff(E124)"""
    need = [("learn", w, t) for w in WIRES for t in TSEEDS] + [("frozen", w, 600) for w in WIRES]
    miss = [k for k in need if k not in H] + [("sd",) + k for k in need if k not in S]
    if miss:
        return ["[측정 확인] 결측 %d: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    bad = [k for k in need if any(x != x for x in H[k])]
    if bad:
        return ["[측정 확인] nan: %s — **판정 보류**" % bad], None
    checks, ok = [], True
    same = [w for w in WIRES if H[("learn", w, 600)] == H[("frozen", w, 600)]]
    checks.append("[측정 확인] half1 learn ≠ frozen(배선별, t600): %d/8%s" % (8 - len(same), "" if not same else " ← 같은 값 %s" % same))
    ok &= not same
    L = [("learn", w, t) for w in WIRES for t in TSEEDS]
    n_h1 = sum(H[k][2] >= OK for k in L)
    n_sd = sum(S[k] >= OK for k in L)
    n_fr = sum(H[("frozen", w, 600)][2] >= OK for w in WIRES)
    diff = [H[k][2] - S[k] for k in L]
    if not ok:
        verdict = "보류(조작검증 실패 — 측정부터)"
    elif n_h1 >= 12 and n_fr <= 2:
        verdict = "관계 특이 장애 — 같은 16자극의 선형 분리 규칙은 획득(같음/다름만 실패)"
    elif n_h1 <= 4:
        verdict = "관계 무관 장애 — 같은 자극의 선형 분리 규칙도 획득 못 함(용량·겹침·학습 체제 쪽)"
    else:
        verdict = "보류(혼재 또는 frozen 성공 과다)"
    return checks, {"n_h1": n_h1, "n_sd": n_sd, "n_fr": n_fr, "diff": diff, "ok": ok, "verdict": verdict}


def report(checks, res, H=None, S=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        print("w%d half1 learn lbal %s (L/R %s) | samediff(E124) %s || half1 frozen %.0f"
              % (w, "/".join("%.0f" % H[("learn", w, t)][2] for t in TSEEDS),
                 " ".join("%.0f/%.0f" % H[("learn", w, t)][:2] for t in TSEEDS),
                 "/".join("%.0f" % S[("learn", w, t)] for t in TSEEDS), H[("frozen", w, 600)][2]))
    L = [("learn", w, t) for w in WIRES for t in TSEEDS]
    print("평균 lbal(%%): half1 learn %.1f, samediff learn %.1f, half1 frozen %.1f | 짝 차이(half1−samediff) 평균 %+.1f"
          % (sum(H[k][2] for k in L) / 16, sum(S[k] for k in L) / 16, sum(H[("frozen", w, 600)][2] for w in WIRES) / 8, sum(res["diff"]) / 16))
    print("성공(≥80%%): half1 learn %d/16, samediff learn %d/16, half1 frozen %d/8" % (res["n_h1"], res["n_sd"], res["n_fr"]))
    print("판정: %s" % res["verdict"])


def load():
    H, S = {}, {}
    for fn, rx in (("E125.log", TR25), ("E124.log", TR24)):
        try:
            for ln in open(os.path.join(EXP, fn), encoding="utf-8"):
                m = rx.match(ln)
                if not m:
                    continue
                k = (m.group(1), int(m.group(2)), int(m.group(3)))
                if rx is TR25:
                    H[k] = (float(m.group(7)), float(m.group(8)), float(m.group(9)))
                else:
                    S[k] = float(m.group(4))
        except FileNotFoundError:
            pass
    return H, S


if __name__ == "__main__":
    H, S = load()
    c, r = judge(H, S)
    report(c, r, H, S)
    sys.exit(0)
