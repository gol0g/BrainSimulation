#!/usr/bin/env python3
"""E188 판정 — 원시 특징 범주 규칙의 반전(K96 의 범주판, 앞 능력 보존). 규칙은 research/experiments/logs/E188/criteria_fixed.txt.
입력(뇌 10~14): 학습 logs/E188/train_b{B}.log(catrule 1,500 → 반전 1,500), 평가 logs/E188/ev_b{B}_{learn|none}.log(RCEVAL 4줄 train·new).
반전 전 기준 R = E184 U(같은 뇌·같은 시드 — 앞 1,500시행이 같은 시행열): logs/E184/ev_b{B}_U_{learn|none}.log.
지표(1e-4): e'_{c,s} = m(학습) − m(무학습)(반전 뒤), R_{c,s}(E184 U), Δ = e' − R. 반전 뒤 정답: A 같은 쪽(+), B 교차(−).
뇌 단위: 반전 획득 = e'_A,train ≥ +1000 이면서 e'_B,train ≤ −1000; 반전 일반화 = e'_A,new ≥ +500·e'_B,new ≤ −500·2·e'_A,new ≥ e'_A,train·2·e'_B,new ≤ e'_B,train;
  부분 반전 = 반전 획득 아님 이면서 Δ_A,train ≥ +1000·Δ_B,train ≤ −1000; 고착 = |Δ_A,train| < 500 이면서 |Δ_B,train| < 500.
판정(우선순위): 조작검증 실패 → 보류 / 반전 성공(H109) 반전 획득·일반화 ≥ 4/5 / 반전(사례 한정) 반전 획득·일반화 아님 ≥ 4/5
  / 부분 반전 ≥ 4/5 / 고착(H109-null) ≥ 4/5 / 그 밖 보류.
실행: python3 scripts/judge_e188.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
TRIALS = 96


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_eval(path):
    rc, pushed, kt, fw, sat, bad = {}, None, 0, 0, False, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCEVAL "):
                kv = {}
                for x in ln.split(" | ")[0].split()[1:]:
                    k, v = x.split("=", 1); kv[k] = v
                rc[(kv["cat"], kv["set"])] = q4(kv["mod"])
                if int(kv["trials"]) != TRIALS or int(kv["feat_on"]) != TRIALS:
                    bad = True
                for seg in ln.split(" | ")[1:]:
                    for part in seg.split("motor L/R ")[1:]:
                        a, b = part.split()[0].split("/")
                        if q4(a) >= 6600 and q4(b) >= 6600:
                            sat = True
            elif ln.startswith("=> DECOMP ") and "rceval" in ln:
                pushed = int(ln.split("pushed=")[1].split()[0])
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                kt += 1
            elif ln.startswith("[E184 특징 가중치 적재]"):
                fw += 1
    return rc, pushed, kt, fw, sat, bad


def read_train(path):
    r = {"start": None, "nrev": None, "cat": None, "chk": [], "done": False, "kt": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("[범주 반전] 시행 "):
                r["start"] = int(ln.split("시행 ")[1].split()[0])
            elif ln.startswith("[범주 반전] 반전 시행 "):
                r["nrev"] = (int(ln.split("반전 시행 ")[1].split()[0]), int(ln.split("(시행 ")[1].split()[0]), int(ln.split("전체 ")[1].split(")")[0]))
            elif ln.startswith("[범주 과제] 시행 "):
                p = ln.split()
                r["cat"] = (int(p[3]), int(ln.split("범주 B(같은 쪽) ")[1].split()[0]))
            elif ln.startswith("[범주 과제 점검]"):
                r["chk"].append((int(ln.split("행동 창 끝 특징 발화 ")[1].split()[0]), int(ln.split("시행 끝 ")[1].split()[0])))
            elif ln.startswith("[학습] "):
                r["done"] = True
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                r["kt"] += 1
    return r


def effects(lp, np_):
    L, N = read_eval(lp), read_eval(np_)
    return L, N


def main():
    ok, why, rows = True, [], {}
    for b in BRAINS:
        try:
            tr = read_train(os.path.join(EXP, "logs", "E188", "train_b%d.log" % b))
            L = read_eval(os.path.join(EXP, "logs", "E188", "ev_b%d_learn.log" % b))
            N = read_eval(os.path.join(EXP, "logs", "E188", "ev_b%d_none.log" % b))
            RL = read_eval(os.path.join(EXP, "logs", "E184", "ev_b%d_U_learn.log" % b))
            RN = read_eval(os.path.join(EXP, "logs", "E184", "ev_b%d_U_none.log" % b))
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        if not (tr["start"] == 1500 and tr["nrev"] == (1500, 1500, 3000) and tr["cat"] and tr["cat"][0] == 3000 and 1200 <= tr["cat"][1] <= 1800
                and tr["done"] and tr["kt"] >= 1):
            ok = False; why.append("b%d 학습 줄(%s %s %s)" % (b, tr["start"], tr["nrev"], tr["cat"]))
        if len(tr["chk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["chk"]):
            ok = False; why.append("b%d 특징 점검 %s" % (b, tr["chk"]))
        for nm, (rc, pushed, kt, fw, sat, bad) in (("learn", L), ("none", N)):
            if len(rc) != 4 or bad or sat or kt < 1 or fw != 0 or pushed != (8 if nm == "learn" else 0):
                ok = False; why.append("b%d %s 평가" % (b, nm))
        if len(RL[0]) != 4 or len(RN[0]) != 4:
            ok = False; why.append("b%d 반전 전 기준(E184) 결측" % b)
        if ok:
            e = {k: L[0][k] - N[0][k] for k in L[0]}
            R = {k: RL[0][k] - RN[0][k] for k in RL[0]}
            rows[b] = (e, R)
    if not ok:
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    n = {"succ": 0, "memo": 0, "part": 0, "stuck": 0}
    for b in BRAINS:
        e, R = rows[b]
        aT, aN, bT, bN = e[("A", "train")], e[("A", "new")], e[("B", "train")], e[("B", "new")]
        dA, dB = aT - R[("A", "train")], bT - R[("B", "train")]
        acq = aT >= 1000 and bT <= -1000
        gen = aN >= 500 and bN <= -500 and 2 * aN >= aT and 2 * bN <= bT
        n["succ"] += acq and gen
        n["memo"] += acq and not gen
        n["part"] += (not acq) and dA >= 1000 and dB <= -1000
        n["stuck"] += abs(dA) < 500 and abs(dB) < 500
        print("b%d 반전 뒤 e(1e-4) A %+d/%+d B %+d/%+d | 반전 전(E184) A %+d B %+d | Δ A %+d B %+d"
              % (b, aT, aN, bT, bN, R[("A", "train")], R[("B", "train")], dA, dB))
    print("조작검증 통과 | 반전 성공 %d/5 · 사례 한정 %d/5 · 부분 반전 %d/5 · 고착 %d/5" % (n["succ"], n["memo"], n["part"], n["stuck"]))
    v = ("반전 성공(H109)" if n["succ"] >= 4 else "반전(사례 한정)" if n["memo"] >= 4 else "부분 반전" if n["part"] >= 4
         else "고착(H109-null)" if n["stuck"] >= 4 else "보류")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
