#!/usr/bin/env python3
"""E123 판정 — E123.md 4절. 64런(48 새 + E122 learn 16)이 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): held = 미학습 사례(d 0.20) 평가 정답률(%, 100시행, 탐색·학습 없음, 동점 무작위).
dw = 학습 후 KC→출력 가중치 |Δg| 평균(초기 0.5 대비, 두 출력 평균) — 학습량 조작 검증용.
"""
import os
import re
import sys

EXP = "research/experiments"
WIRES = tuple(range(10, 18))
TSEEDS = (600, 601)
NS = (50, 100, 200, 400)
GAIN = 15.0     # held(400) − held(50) 최소 증가(%p) — E122 난수열 간 차이(≤5%p)의 3배
SAT = 80.0
TR23 = re.compile(r"^\s*learn n(\d+) w(\d+) t(\d+): => EXGEN .*? d0\.20=([0-9.]+) .*?\|\| dw_l=([0-9.]+) dw_r=([0-9.]+)")
TR22 = re.compile(r"^\s*learn w(\d+) t(\d+): => EXGEN .*? d0\.20=([0-9.]+) ")
DW = re.compile(r"^  kc_out_([lr]): .*?\|Δ\|평균 ([0-9.]+)", re.M)


def judge(R):
    """R: {(n, w, t): (held, dw)}"""
    need = [(n, w, t) for n in NS for w in WIRES for t in TSEEDS]
    missing = [k for k in need if k not in R]
    if missing:
        return ["[측정 확인] 결측 %d/64: %s — **판정 보류, 수치 미출력**" % (len(missing), missing[:6])], None
    bad = [k for k in need if any(x != x for x in R[k])]
    if bad:
        return ["[측정 확인] nan: %s — **판정 보류**" % bad], None
    checks, ok = [], True
    runs = [(w, t) for w in WIRES for t in TSEEDS]
    # 조작검증: 가중치 변화량이 학습량에 따라 증가(런별 50<100<200<400)
    mono_dw = sum(R[(50, w, t)][1] < R[(100, w, t)][1] < R[(200, w, t)][1] < R[(400, w, t)][1] for w, t in runs)
    checks.append("[측정 확인] |Δg| 50<100<200<400 런별: %d/16" % mono_dw)
    ok &= mono_dw >= 12
    mean = {n: sum(R[(n, w, t)][0] for w, t in runs) / 16 for n in NS}
    group_mono = mean[50] < mean[100] < mean[200] <= mean[400]
    n_gain = sum(R[(400, w, t)][0] - R[(50, w, t)][0] >= GAIN for w, t in runs)
    n_sat50 = sum(R[(50, w, t)][0] >= SAT for w, t in runs)
    if not ok:
        verdict = "보류(조작검증 실패 — 학습량 조작이 가중치에 반영 안 됨)"
    elif group_mono and n_gain >= 12:
        verdict = "지지 — 미학습 사례 정답률이 학습량에 따라 증가(누적 학습)"
    elif n_sat50 >= 12:
        verdict = "보류(범위 밖 — 50시행에서 이미 ≥80%, 이 범위로 용량-반응 판정 불가)"
    elif n_gain <= 4 and not group_mono:
        verdict = "기각 — 학습량과 무관(일회성 변화)"
    else:
        verdict = "보류(혼재)"
    return checks, {"mean": mean, "group_mono": group_mono, "n_gain": n_gain, "n_sat50": n_sat50,
                    "mono_dw": mono_dw, "ok": ok, "verdict": verdict}


def report(checks, res, R=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for w in WIRES:
        print("w%d held(d0.20) n50/100/200/400: %s" % (w, "  ".join(
            "t%d %s" % (t, "/".join("%.0f" % R[(n, w, t)][0] for n in NS)) for t in TSEEDS)))
    print("평균 held(%%): %s | 단조 50<100<200≤400: %s" % (" ".join("n%d %.1f" % (n, res["mean"][n]) for n in NS), res["group_mono"]))
    print("held(400)−held(50) ≥ %.0f%%p: %d/16 | held(50) ≥ 80%%: %d/16" % (GAIN, res["n_gain"], res["n_sat50"]))
    print("판정: %s" % res["verdict"])


def load():
    R = {}
    try:
        for ln in open(os.path.join(EXP, "E123.log"), encoding="utf-8"):
            m = TR23.match(ln)
            if m:
                R[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = (float(m.group(4)), (float(m.group(5)) + float(m.group(6))) / 2)
    except FileNotFoundError:
        pass
    try:
        for ln in open(os.path.join(EXP, "E122.log"), encoding="utf-8"):
            m = TR22.match(ln)
            if m:
                w, t = int(m.group(1)), int(m.group(2))
                raw = open(os.path.join(EXP, "logs/E122/learn_w%d_t%d.log" % (w, t)), encoding="utf-8").read()
                d = dict(DW.findall(raw))
                R[(400, w, t)] = (float(m.group(3)), (float(d["l"]) + float(d["r"])) / 2)
    except FileNotFoundError:
        pass
    return R


if __name__ == "__main__":
    R = load()
    c, r = judge(R)
    report(c, r, R)
    sys.exit(0)
