#!/usr/bin/env python3
"""E137 판정 — 기준 logs/E137/criteria_fixed.txt(실행 전 고정). 발달 16 + 학습 128 + 무학습 32 가 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): novel_lbal = 처음 보는 항목(4~7) 라벨 균형 정답률(%), train_lbal = 훈련 항목(0~3) 라벨 균형 정답률(%).
N_T(w) = 배선 w 의 학습 T시행 새 항목 정답률(난수열 600·601 평균), F_T(w) = 무학습 T시행 새 항목 정답률(난수열 600). 독립 단위 = 배선.
|Δg| = 학습 전후 KC→출력 가중치 절대 변화 평균((kc_out_l + kc_out_r)/2), 블록 = 100시행 블록 정답률 줄 수, 보상 = 보상 계열 파일 줄 수.
sf = 가지치기 후 같은 위치 짝 비율, 발달 DEVHEBB 줄(일치형·불일치형·두 유형 평균).
"""
import os
import re
import sys
from math import comb

EXP = "research/experiments"
TRACE = os.path.join(EXP, "traces", "E137")
WIRES = tuple(range(94, 110))
TSEEDS = (600, 601)
DOSES = (100, 200, 400, 800)
FDOSES = (100, 800)
TD = re.compile(r"^\s*e137 dev w(\d+): => DEVHEBB seed=(\d+) env=(\w+) .*?발화율\(KC·노출당\) 일치형 ([0-9.]+) 불일치형 ([0-9.]+) \| "
                r"가지치기 후 같은 위치: 일치형 ([0-9.]+) 불일치형 ([0-9.]+)")
TT = re.compile(r"^\s*e137 (learn|frozen) w(\d+) T(\d+) t(\d+): => \[KC불러옴\] (\S+) \| 일치형 같은 위치 (\d+)/(\d+) \| 불일치형 같은 위치 (\d+)/(\d+)"
                r" \|\| => SDLAB diff=cyclic rule=samediff mode=(\w+) seed=(\d+) trialseed=(\d+) .*?train_lbal=([0-9.]+) .*?novel_lbal=([0-9.]+)"
                r" \|\| 블록 (\d+) \|\| 보상 (\S+) \|\| dl ([0-9.]+) dr ([0-9.]+)")


def sign_p(d):
    n = sum(x > 0 for x in d); nz = sum(x != 0 for x in d)
    if nz == 0:
        return 1.0, n, nz
    k = max(n, nz - n)
    return min(1.0, 2 * sum(comb(nz, i) for i in range(k, nz + 1)) / 2 ** nz), n, nz


def med(xs):
    s = sorted(xs); n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def judge(D, T, RW=None):
    needL = [("learn", w, d, s) for w in WIRES for d in DOSES for s in TSEEDS]
    needF = [("frozen", w, d, 600) for w in WIRES for d in FDOSES]
    miss = [("dev", w) for w in WIRES if w not in D] + [k for k in needL + needF if k not in T]
    if miss:
        return ["[측정 확인] 결측 %d/176: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    checks, ok = [], True
    # M1 발달 형성
    mm = [D[w]["mm"] for w in WIRES]; mx = [D[w]["mx"] for w in WIRES]; sf = [(a + b) / 2 for a, b in zip(mm, mx)]
    envbad = [w for w in WIRES if D[w]["env"] != "corr" or D[w]["seed"] != w]
    m1 = med(sf) >= 0.40 and med(mm) >= 0.40 and med(mx) >= 0.40 and not envbad
    checks.append("[M1 발달 형성] 같은 위치 중앙값 두 유형 평균 %.3f · 일치형 %.3f · 불일치형 %.3f (각 ≥0.40), env·seed 표기 %d/16 → %s"
                  % (med(sf), med(mm), med(mx), 16 - len(envbad), "통과" if m1 else "실패"))
    ok &= m1
    # M2 과제 연결 = 발달 결과 (DEVHEBB 비율은 소수 3자리 → 개수 ±0.2 오차, 1 미만 차이만 허용)
    allk = needL + needF
    fbad = [k for k in allk if abs(D[k[1]]["mm"] * T[k]["nm"] - T[k]["lm"]) >= 1.0 or abs(D[k[1]]["mx"] * T[k]["nx"] - T[k]["lx"]) >= 1.0
            or ("dev_corr_w%d.npz" % k[1]) not in T[k]["file"] or T[k]["mode"] != k[0] or T[k]["seed"] != k[1] or T[k]["ts"] != k[3]]
    checks.append("[M2 과제 연결 = 발달 결과, mode·seed·trialseed] %d/160 → %s" % (160 - len(fbad), "통과" if not fbad else "실패 %s" % fbad[:4]))
    ok &= not fbad
    # M3 시행 수 실현
    bbad = [k for k in allk if T[k]["blocks"] != k[2] // 100]
    rbad = [k for k in needL if T[k]["rw"] != str(k[2])]
    checks.append("[M3 시행 수] 블록 줄 = T/100: %d/160, 학습 보상 계열 = T줄: %d/128 → %s"
                  % (160 - len(bbad), 128 - len(rbad), "통과" if not bbad and not rbad else "실패 %s" % (bbad + rbad)[:4]))
    ok &= not bbad and not rbad
    # M4 용량이 시냅스에 닿음
    dg = {d: sum((T[("learn", w, d, s)]["dl"] + T[("learn", w, d, s)]["dr"]) / 2 for w in WIRES for s in TSEEDS) / 32 for d in DOSES}
    inc = all(dg[a] < dg[b] for a, b in zip(DOSES, DOSES[1:]))
    fz = [k for k in needF if T[k]["dl"] != 0.0 or T[k]["dr"] != 0.0]
    checks.append("[M4 |Δg| 평균] %s, 무학습 |Δg|=0: %d/32 → %s"
                  % (" < ".join("T%d %.5f" % (d, dg[d]) for d in DOSES), 32 - len(fz), "통과" if inc and not fz else "실패"))
    ok &= inc and not fz
    # M5 시행 수는 학습으로만 작용
    fd = [round(abs(T[("frozen", w, 100, 600)]["nl"] - T[("frozen", w, 800, 600)]["nl"]), 6) for w in WIRES]
    m5 = max(fd) <= 5.0
    checks.append("[M5 무학습 T100 vs T800 새 항목] 최대 차 %.1f%%p (≤5), 소수점까지 같음 %d/16 → %s"
                  % (max(fd), sum(x == 0 for x in fd), "통과" if m5 else "실패"))
    ok &= m5
    # (보고만) 중첩
    if RW is not None:
        nest = [(w, s, d) for w in WIRES for s in TSEEDS for d in DOSES[1:]
                if (w, 100, s) in RW and (w, d, s) in RW and RW[(w, d, s)][:100] == RW[(w, 100, s)]]
        checks.append("[보고: 중첩] T100 보상 계열 = 긴 런의 앞 100시행: %d/96" % len(nest))
    # 주 판정
    N = {d: [sum(T[("learn", w, d, s)]["nl"] for s in TSEEDS) / 2 for w in WIRES] for d in DOSES}
    TRn = {d: [sum(T[("learn", w, d, s)]["tl"] for s in TSEEDS) / 2 for w in WIRES] for d in DOSES}
    F = {d: [T[("frozen", w, d, 600)]["nl"] for w in WIRES] for d in FDOSES}
    # 경계값 비교는 소수 6자리 반올림 후(입력은 소수 1자리 정답률 — 부동소수 합 오차로 정확히 +10 이 9.999… 가 되지 않게)
    mN = {d: round(sum(N[d]) / 16, 6) for d in DOSES}
    dD = [round(a - b, 6) for a, b in zip(N[800], N[100])]; p_D, n_D, nz_D = sign_p(dD); m_D = round(sum(dD) / 16, 6)
    dG = [round(a - b, 6) for a, b in zip(N[800], F[800])]; p_G, n_G, nz_G = sign_p(dG); m_G = round(sum(dG) / 16, 6)
    s1 = m_D >= 10 and p_D < 0.01
    s2 = all(mN[d] > mN[100] for d in (200, 400, 800)) and mN[800] >= max(mN[200], mN[400]) - 5
    s3 = m_G >= 10 and p_G < 0.01
    if not ok:
        verdict = "보류(조작검증 실패)"
    elif s1 and s2 and s3:
        verdict = "지지(H060) — 처음 보는 항목 같음/다름 정답률이 보상 학습량에 따라 증가한다(헌장 개념 조건 4, 관계 수준)"
    elif not s3:
        verdict = "보류(전이 재현 실패 — 800시행 학습−무학습 기준 미달)"
    elif mN[100] >= 80:
        verdict = "보류(천장 — 100시행에서 이미 새 항목 평균 ≥80%)"
    elif m_D < 5 or p_D >= 0.05:
        verdict = "용량 무관(H060-flat) — 800시행 전이는 있으나 100→800 증가 검출 안 됨"
    else:
        verdict = "보류(혼재)"
    gap = {d: round(sum(a - b for a, b in zip(TRn[d], N[d])) / 16, 6) for d in DOSES}
    lag = (gap[100] >= 10 or gap[200] >= 10) and gap[800] < 5
    return checks, {"N": N, "TRn": TRn, "F": F, "mN": mN, "m_D": m_D, "p_D": p_D, "n_D": n_D, "nz_D": nz_D, "m_G": m_G, "p_G": p_G,
                    "n_G": n_G, "nz_G": nz_G, "s1": s1, "s2": s2, "s3": s3, "gap": gap, "lag": lag, "dg": dg, "ok": ok, "verdict": verdict}


def report(checks, res, D=None, T=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for i, w in enumerate(WIRES):
        print("w%d 발달 같은 위치 %.2f/%.2f | 새 항목 T100 %.1f T200 %.1f T400 %.1f T800 %.1f | 무학습 %.1f/%.1f" % (
            w, D[w]["mm"], D[w]["mx"], res["N"][100][i], res["N"][200][i], res["N"][400][i], res["N"][800][i], res["F"][100][i], res["F"][800][i]))
    for d in DOSES:
        ks = [("learn", w, d, s) for w in WIRES for s in TSEEDS]
        print("T%d 학습: 새 항목 평균 %.1f%% 훈련 %.1f%% (훈련−새 %+.1f%%p) | 새 항목 ≥75%%: %d/32"
              % (d, res["mN"][d], sum(res["TRn"][d]) / 16, res["gap"][d], sum(T[k]["nl"] >= 75 for k in ks)))
    for d in FDOSES:
        print("T%d 무학습: 새 항목 평균 %.1f%%" % (d, sum(res["F"][d]) / 16))
    print("배선 단위 N800−N100: 평균 %+.1f%%p, 양수 %d/%d, 부호검정 양측 p=%.5f → S1 %s" % (res["m_D"], res["n_D"], res["nz_D"], res["p_D"], res["s1"]))
    print("배선 평균 단조: N100 %.1f < N200 %.1f, N400 %.1f, N800 %.1f; N800 ≥ max(N200,N400)−5 → S2 %s"
          % (res["mN"][100], res["mN"][200], res["mN"][400], res["mN"][800], res["s2"]))
    print("배선 단위 N800−무학습800: 평균 %+.1f%%p, 양수 %d/%d, 부호검정 양측 p=%.5f → S3 %s" % (res["m_G"], res["n_G"], res["nz_G"], res["p_G"], res["s3"]))
    print("부지표 H060-lag(새 항목이 훈련보다 늦게 오름): %s" % res["lag"])
    print("판정: %s" % res["verdict"])


def load():
    D, T, RW = {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E137.log"), encoding="utf-8"):
            m = TD.match(ln)
            if m:
                g = m.groups()
                D[int(g[0])] = {"seed": int(g[1]), "env": g[2], "fm": float(g[3]), "fx": float(g[4]), "mm": float(g[5]), "mx": float(g[6])}
                continue
            m = TT.match(ln)
            if m:
                g = m.groups()
                T[(g[0], int(g[1]), int(g[2]), int(g[3]))] = {
                    "file": g[4], "lm": int(g[5]), "nm": int(g[6]), "lx": int(g[7]), "nx": int(g[8]), "mode": g[9], "seed": int(g[10]),
                    "ts": int(g[11]), "tl": float(g[12]), "nl": float(g[13]), "blocks": int(g[14]), "rw": g[15], "dl": float(g[16]), "dr": float(g[17])}
    except FileNotFoundError:
        pass
    for w in WIRES:
        for d in DOSES:
            for s in TSEEDS:
                f = os.path.join(TRACE, "rw_w%d_T%d_t%d.txt" % (w, d, s))
                if os.path.exists(f):
                    RW[(w, d, s)] = [x.strip() for x in open(f, encoding="utf-8") if x.strip()]
    return D, T, RW


if __name__ == "__main__":
    D, T, RW = load()
    c, r = judge(D, T, RW)
    report(c, r, D, T)
    sys.exit(0)
