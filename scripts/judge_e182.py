#!/usr/bin/env python3
"""E182 판정 — 대칭 맥락 쌍조건(E179 가중치)을 학습과 같은 시간 구조(시행마다 조향 처리 동안만 맥락 켬, 사이 10처리 끔, 안정화·오프셋 맥락 없음)로 평가하면
두 규칙이 모두 기준을 넘는가. 평가만. 기준 logs/E182/criteria_fixed.txt. 평가 20개(뇌 10~14 × 학습·무학습 × 맥락 A·B)가 다 모이기 전에는 수치를 출력하지 않는다.
e_A = m(learn A) − m(none A), e_B = m(learn B) − m(none B). 1e-4 정수.
판정(E179 규칙, 적용 순서): 조작검증 실패 → 보류. 획득(H103) e_A ≤ −0.10 이면서 e_B ≥ +0.10 ≥ 4/5 → 요소식 |e_B − e_A| < 0.10 ≥ 4/5 → 부분(B만·A만) ≥ 4/5 → 그 밖 보류.
조작검증(20개 모두): '[맥락 평가 시행]' 줄 — 시행 100 · 맥락 켬 100회·끔 100회 · 간격 10처리 / A 평가에만 '[맥락 평가 A]'·B 평가에만 '[맥락 평가] ctx=켬' /
'[motor 동작점]' 줄 ≥ 1·값 = −4000 / 진단 좌·우·비포화(motor 4개가 모두 ≥ 0.66 아님) / pushed 학습 8·무학습 0.
실행: python3 scripts/judge_e182.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
EVS = (("learn", "A"), ("learn", "B"), ("none", "A"), ("none", "B"))
X182 = -4000.0
SAT = 6600
DL = re.compile(r"^=> DECOMP mode=(\w+) mod=([-+0-9.]+) .*?pushed=(\d+)", re.M)
TONIC = re.compile(r"^\s*\[motor 동작점\] motor 좌·우 Ioffset 기저값 ([-0-9.]+)", re.M)
DG = re.compile(r"^\[E171 평가 진단\] variant=\w+ side=(left|right) n=\d+ motor L/R ([0-9.]+)/([0-9.]+)", re.M)
TR = re.compile(r"^\[맥락 평가 시행\] 시행 (\d+) · 맥락 켬 (\d+)회·끔 (\d+)회\(조향 처리 동안만 켬\), 사이 간격 (\d+)처리", re.M)


def i4(x):
    return int(round(float(x) * 1e4))


def parse(t):
    m = DL.search(t)
    if not m:
        return None
    dg = {d.group(1): (float(d.group(2)), float(d.group(3))) for d in DG.finditer(t)}
    tr = TR.search(t)
    return {"mod": i4(m.group(2)), "pushed": int(m.group(3)), "tonic": [float(x) for x in TONIC.findall(t)],
            "rates": (dg["left"] + dg["right"]) if set(dg) == {"left", "right"} else None,
            "trial": tuple(int(tr.group(k)) for k in (1, 2, 3, 4)) if tr else None,
            "B": "[맥락 평가] ctx=켬" in t, "A": "[맥락 평가 A]" in t}


def load(base=EXP):
    X = {}
    for b in BRAINS:
        for (w, c) in EVS:
            f = os.path.join(base, "logs", "E182", "ev_b%d_%s_%s.log" % (b, w, c))
            if os.path.exists(f):
                p = parse(open(f, encoding="utf-8", errors="replace").read())
                if p is not None:
                    X[(b, w, c)] = p
    return X


def judge(X):
    miss = [(b, w, c) for b in BRAINS for (w, c) in EVS if (b, w, c) not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % (miss[:6],)], None
    keys = [(b, w, c) for b in BRAINS for (w, c) in EVS]
    mtr = sum(X[k]["trial"] == (100, 100, 100, 10) for k in keys)
    mmk = sum(X[k]["B"] == (k[2] == "B") and X[k]["A"] == (k[2] == "A") for k in keys)
    mto = sum(len(X[k]["tonic"]) >= 1 and all(abs(t - X182) <= 1e-6 for t in X[k]["tonic"]) for k in keys)
    mun = sum(X[k]["rates"] is not None and not all(i4(x) >= SAT for x in X[k]["rates"]) for k in keys)
    mpu = sum(X[k]["pushed"] == (8 if k[1] == "learn" else 0) for k in keys)
    ok = mtr == mmk == mto == mun == mpu == 20
    checks = ["[조작검증] 시행 맞춤 줄(100·100·100·10) %d/20 · A·B 표지 %d/20 · 동작점(−4000) %d/20 · 진단·비포화 %d/20 · 이식(pushed) %d/20 → %s"
              % (mtr, mmk, mto, mun, mpu, "통과" if ok else "실패")]
    eA = {b: X[(b, "learn", "A")]["mod"] - X[(b, "none", "A")]["mod"] for b in BRAINS}
    eB = {b: X[(b, "learn", "B")]["mod"] - X[(b, "none", "B")]["mod"] for b in BRAINS}
    acq = [b for b in BRAINS if eA[b] <= -1000 and eB[b] >= 1000]
    elem = [b for b in BRAINS if abs(eB[b] - eA[b]) < 1000]
    p_B = [b for b in BRAINS if eB[b] >= 1000 and eA[b] > -1000]
    p_A = [b for b in BRAINS if eA[b] <= -1000 and eB[b] < 1000]
    if not ok:
        v = "보류(조작검증 실패)"
    elif len(acq) >= 4:
        v = "획득(H103) — 학습과 같은 시간 구조로 재면 같은 자극의 정답을 맥락에 따라 반대로 낸다(쌍조건)"
    elif len(elem) >= 4:
        v = "요소식 — 맥락을 무시하고 두 규칙이 겹친다"
    elif len(p_B) + len(p_A) >= 4:
        v = "부분 — 한 맥락만(B(같은 쪽)만 %d·A(교차)만 %d)" % (len(p_B), len(p_A))
    else:
        v = "보류"
    return checks, {"eA": eA, "eB": eB, "acq": acq, "elem": elem, "p_B": p_B, "p_A": p_A, "ok": ok, "verdict": v}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d e_A %+.4f e_B %+.4f (차 %+.4f) | m 학습 A %+.4f B %+.4f · 무학습 A %+.4f B %+.4f"
              % (b, res["eA"][b] / 1e4, res["eB"][b] / 1e4, (res["eB"][b] - res["eA"][b]) / 1e4,
                 X[(b, "learn", "A")]["mod"] / 1e4, X[(b, "learn", "B")]["mod"] / 1e4, X[(b, "none", "A")]["mod"] / 1e4, X[(b, "none", "B")]["mod"] / 1e4))
    print("획득 %d/5 | 요소식 %d/5 | 부분 B만 %d/5·A만 %d/5" % (len(res["acq"]), len(res["elem"]), len(res["p_B"]), len(res["p_A"])))
    print("판정: %s" % res["verdict"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
