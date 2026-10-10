#!/usr/bin/env python3
"""E171 판정 — 조합 비가산(K103)이 KC 반응 수준인가 motor 출력 수준인가(측정). 기준 logs/E171/criteria_fixed.txt. 30 평가가 다 모이기 전에는 수치를 출력하지 않는다.
평가: 뇌 10~14, E162 AB·none × base·bad·agree, --eval-diag(쪽별 평균 motor·KC 발화율, 읽기 전용). E170 과 같은 인자.
KC 비(AB, 1e-6 정수 비교): 일치 조합의 각 KC 쪽 발화율 / 그 쪽 성분 단독 발화율 —
  r1 = KC_L(agree, side L) / KC_L(base, L), r2 = KC_R(agree, L) / KC_R(bad, R), r3 = KC_R(agree, R) / KC_R(base, R), r4 = KC_L(agree, R) / KC_L(bad, L).
판정: KC 보존(H094 — 비가산은 출력 쪽) = 네 비 모두 0.85~1.15 인 뇌 ≥ 4/5. KC 감소(H094-alt — KC 수준 경쟁) = 네 비 평균 < 0.85 인 뇌 ≥ 4/5. 그 밖 보류.
조작검증(하나라도 실패면 보류): 진단 줄 쪽마다 n = 250(30 평가 × 2), 변조폭이 E170 같은 평가와 같음(±0.0001, 30/30 — 진단은 읽기 전용), pushed AB 8·none 0.
부지표(판정 밖): motor 수준 가산 ρ_m = 우세 motor 학습분(agree) / (good 단독 + bad 단독 학습분), 비우세 motor 발화율 바닥.
실행: python3 scripts/judge_e171.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
VARS = ("base", "bad", "agree")
DL = re.compile(r"^\[E171 평가 진단\] variant=(\w+) side=(\w+) n=(\d+) motor L/R ([0-9.na]+)/([0-9.na]+) KC L/R ([0-9.na]+)/([0-9.na]+)", re.M)


def i4(x):
    return int(round(float(x) * 1e4))


def i6(x):
    return int(round(float(x) * 1e6))


def rd(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def ev(t, var):
    m = re.search(r"^=> DECOMP mode=(\w+) mod=([-+]?\d+\.\d+) .*?pushed=(\d+)", t, re.M) if t else None
    if not m:
        return None
    d = {}
    for g in DL.finditer(t):
        if g.group(1) == var and "nan" not in g.group(4, 5, 6, 7):
            d[g.group(2)] = {"n": int(g.group(3)), "mL": i6(g.group(4)), "mR": i6(g.group(5)), "kL": i6(g.group(6)), "kR": i6(g.group(7))}
    if set(d) != {"left", "right"}:
        return None
    return {"mod": i4(m.group(2)), "pushed": int(m.group(3)), "d": d}


def load():
    X = {}
    for b in BRAINS:
        for w in ("AB", "none"):
            for v in VARS:
                r = ev(rd("logs", "E171", "ev_b%d_%s_%s.log" % (b, w, v)), v)
                if r:
                    X[(w, v, b)] = r
                t = rd("logs", "E170", "ev_b%d_%s_%s.log" % (b, w, v))
                m = re.search(r"^=> DECOMP mode=\w+ mod=([-+]?\d+\.\d+)", t, re.M) if t else None
                if m:
                    X[("E170", w, v, b)] = i4(m.group(1))
    return X


def ratios(X, b):
    D = {v: X[("AB", v, b)]["d"] for v in VARS}
    pairs = ((D["agree"]["left"]["kL"], D["base"]["left"]["kL"]), (D["agree"]["left"]["kR"], D["bad"]["right"]["kR"]),
             (D["agree"]["right"]["kR"], D["base"]["right"]["kR"]), (D["agree"]["right"]["kL"], D["bad"]["left"]["kL"]))
    return pairs


def judge(X):
    need = [(w, v, b) for w in ("AB", "none") for v in VARS for b in BRAINS] + [("E170", w, v, b) for w in ("AB", "none") for v in VARS for b in BRAINS]
    miss = [k for k in need if k not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    mn = sum(all(X[(w, v, b)]["d"][sd]["n"] == 250 for sd in ("left", "right")) for w in ("AB", "none") for v in VARS for b in BRAINS)
    mr = sum(abs(X[(w, v, b)]["mod"] - X[("E170", w, v, b)]) <= 1 for w in ("AB", "none") for v in VARS for b in BRAINS)
    mp = sum(X[(w, v, b)]["pushed"] == (8 if w == "AB" else 0) for w in ("AB", "none") for v in VARS for b in BRAINS)
    ok = mn == mr == mp == 30
    checks = ["[조작검증] 진단 줄 n=250 %d/30 · 변조폭 = E170(진단 읽기 전용) %d/30 · 적재 %d/30 %s" % (mn, mr, mp, "통과" if ok else "실패")]
    keep, red = [], []
    for b in BRAINS:
        pr = ratios(X, b)
        if all(den > 0 and 100 * num >= 85 * den and 100 * num <= 115 * den for num, den in pr):
            keep.append(b)
        if all(den > 0 for num, den in pr) and sum(num / den for num, den in pr) / 4.0 < 0.85:
            red.append(b)
    v = "KC 보존(H094 — 비가산은 출력 쪽)" if len(keep) >= 4 else ("KC 감소(H094-alt — KC 수준 경쟁)" if len(red) >= 4 else "보류")
    if not ok:
        v = "보류(조작검증 실패)"
    return checks, {"keep": keep, "red": red, "verdict": v, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        pr = ratios(X, b)
        A = {v: X[("AB", v, b)]["d"] for v in VARS}; N = {v: X[("none", v, b)]["d"] for v in VARS}
        # 우세 motor(일치 side L → motor R, side R → motor L) 학습분과 가산 예측
        dl = lambda W, v, sd, k: W[v][sd][k]
        agR = (dl(A, "agree", "left", "mR") - dl(N, "agree", "left", "mR")); sgR = (dl(A, "base", "left", "mR") - dl(N, "base", "left", "mR")) + (dl(A, "bad", "right", "mR") - dl(N, "bad", "right", "mR"))
        agL = (dl(A, "agree", "right", "mL") - dl(N, "agree", "right", "mL")); sgL = (dl(A, "base", "right", "mL") - dl(N, "base", "right", "mL")) + (dl(A, "bad", "left", "mL") - dl(N, "bad", "left", "mL"))
        print("b%d KC 비 %.3f %.3f %.3f %.3f | 우세 motor 학습분 일치/단독 합: R %.4f/%.4f(ρ_m %.2f) L %.4f/%.4f(ρ_m %.2f) | 비우세 motor(학습) 일치 L %.4f R %.4f · 단독 base L %.4f bad R %.4f"
              % (b, *(num / den if den else float("nan") for num, den in pr), agR / 1e6, sgR / 1e6, agR / sgR if sgR else float("nan"),
                 agL / 1e6, sgL / 1e6, agL / sgL if sgL else float("nan"), A["agree"]["left"]["mL"] / 1e6, A["agree"]["right"]["mR"] / 1e6,
                 A["base"]["left"]["mL"] / 1e6, A["bad"]["right"]["mL"] / 1e6))
    print("KC 보존(네 비 0.85~1.15) %d/5 · KC 감소(평균 < 0.85) %d/5" % (len(res["keep"]), len(res["red"])))
    print("판정: %s" % res["verdict"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
