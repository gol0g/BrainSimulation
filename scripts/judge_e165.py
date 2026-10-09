#!/usr/bin/env python3
"""E165 판정 — 망 안 형성의 노출 통계 일반성(외부 검토 권고 1). 기준 logs/E165/criteria_fixed.txt. 팔·뇌 10칸의 dev·ov·F 가 다 모이기 전에는 수치를 출력하지 않는다.
팔 NR = 무작위 순서 + 제시마다 강도 U[0.5, 0.9], good·bad × 좌·우 각 100. 팔 NU = NR + bad 세 배(good 좌·우 각 100, bad 각 300). 뇌 16~20.
원 로그 logs/E165/{NR,NU}_{dev,ov,F}_b*.log, 추적 traces/E165/tr_{NR,NU}_F_b*.npz. e_D = 같은 뇌 E161 D 원 로그(logs/E161/D_b*.log).
팔별(E161 규칙): 성공 = 자카드 ≤ 0.10(양쪽) 이고 r = e_F/e_D ≥ 1.30(⇔ 10·E_F ≤ 13·E_D, e_D < 0) 인 뇌 ≥ 4/5, 분리만 = 성공 아님·자카드 ≤ 0.10 ≥ 4/5,
실패 = 자카드 > 0.25(어느 한쪽) ≥ 4/5, 그 밖 보류. 1e-4 정수.
종합: NR·NU 성공 → 견고(H088). NR 성공·NU 아님 → 빈도 의존(H088-freq). NR 실패·분리만 → 순서·강도 의존(H088-null). NR 보류 → 보류.
조작검증(팔·뇌 10/10 — 하나라도 실패면 보류): MX 노출 구성(order=random, 배수·제시 수, 강도 최소 ≥ 0.5·최대 ≤ 0.9·평균 0.7±0.03, 순환 일치 ≤ 0.40),
MO Oja 줄 ≥ 1·4집단 Σ|Δg| > 0, MS' 양쪽 sel_med − sel_med0 ≥ 0.10, ML 적재(ov ≥ 1, F ≥ 2), M1 동결 잔차 ≤ 1e-3, M1b 흔적 생존 ≥ 0.9,
M3 추적 500시행·도파민 전 ≤ 1e-3. 뇌별 MD: E161 기준 e_D ≤ −0.10(5/5).
부지표(판정 밖): 반응 KC 수(good·bad, 쪽별)와 같은 뇌 E161 고정 순환 대비, e_F 의 E161 고정 순환 형성 학습 대비 비, 선택성 ≥ 0.80, 보상 수.
실행: python3 scripts/judge_e165.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
ARMS = {"NR": (1, (100, 100, 100, 100)), "NU": (3, (100, 300, 100, 300))}   # 팔: (bad 배수, 제시 수 good_l·bad_l·good_r·bad_r)
R_STAR = (1.0 - 1.0 / 12.0) ** 20
SIDE = re.compile(r"side=([lr]) fired=(\d+) sel_med0=([0-9.na]+) sel_med=([0-9.na]+) frac09=([0-9.na]+) goodfrac=([0-9.na]+) "
                  r"sum_med=([0-9.na]+) sum_q10=([0-9.na]+) sum_q90=([0-9.na]+) dg_good=([0-9.na]+) dg_bad=([0-9.na]+)")
XLINE = re.compile(r"^\[E165 노출\] order=(\w+) bad_mult=(\d+) good_l=(\d+) bad_l=(\d+) good_r=(\d+) bad_r=(\d+) n=(\d+) "
                   r"int_min=([0-9.]+) int_max=([0-9.]+) int_mean=([0-9.]+) cyc_match=([0-9.]+) seed=(\d+)", re.M)


def i4(x):
    return int(round(x * 1e4))


def rd(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def dev(t):
    ln = next((x for x in t.splitlines() if x.startswith("=> KCDEVOJA ")), None) if t else None
    if ln is None:
        return None
    d = {m.group(1): {"fired": int(m.group(2)), "sel_med0": float(m.group(3)), "sel_med": float(m.group(4)), "goodfrac": float(m.group(6)),
                      "sum_med": float(m.group(7)), "dg_good": float(m.group(10)), "dg_bad": float(m.group(11))} for m in SIDE.finditer(ln)}
    return d if set(d) == {"l", "r"} else None


def expo(t):
    m = XLINE.search(t) if t else None
    if not m:
        return None
    return {"order": m.group(1), "mult": int(m.group(2)), "cnt": tuple(int(m.group(k)) for k in (3, 4, 5, 6)), "n": int(m.group(7)),
            "imin": i4(float(m.group(8))), "imax": i4(float(m.group(9))), "imean": i4(float(m.group(10))), "cyc": i4(float(m.group(11)))}


def jac(t):
    m = re.search(r"^=> KCOVERLAP side=l good=(\d+) bad=(\d+) jac=([0-9.]+) .*\| side=r good=(\d+) bad=(\d+) jac=([0-9.]+) ", t, re.M) if t else None
    return {"jl": i4(float(m.group(3))), "jr": i4(float(m.group(6))), "n": (int(m.group(1)), int(m.group(2)), int(m.group(4)), int(m.group(5)))} if m else None


def mods(t):
    a = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    b = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    return (i4(float(a.group(1))), i4(float(b.group(1)))) if (a and b) else None


def nload(t):
    return len(re.findall(r"^\[E153 종류 입력 적재\].*검증 일치", t, re.M)) if t else 0


def stats(rows):
    eda, eend = rows[:, 13:17], rows[:, 21:25]
    return {"n": len(rows), "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12)),
            "res": float(np.abs(eend - R_STAR * eda).sum() / max(np.abs(eda).sum(), 1e-12)),
            "alive": float(np.mean((np.abs(rows[:, 13]) + np.abs(rows[:, 14])) > 1.0))}


def lrn(t):
    m = mods(t)
    if not m:
        return None
    r = re.search(r"보상 (\d+)회", t)
    return {"pre": m[0], "post": m[1], "ld": nload(t), "rew": int(r.group(1)) if r else None}


def load():
    X = {}
    for b in BRAINS:
        t = rd("logs", "E161", "D_b%d.log" % b)
        if lrn(t):
            X[("D", b)] = lrn(t)
        t = rd("logs", "E161", "F_b%d.log" % b)            # 부지표: 같은 뇌 고정 순환 형성 학습
        if lrn(t):
            X[("C", b)] = lrn(t)
        t = rd("logs", "E161", "ov_b%d.log" % b)
        if jac(t):
            X[("Cov", b)] = jac(t)
        for a in ARMS:
            t = rd("logs", "E165", "%s_dev_b%d.log" % (a, b))
            if dev(t) and expo(t):
                X[("dev", a, b)] = dict(dev(t), oja=len(re.findall(r"^\[E160 종류 입력 Oja\]", t, re.M)))
                X[("x", a, b)] = expo(t)
            t = rd("logs", "E165", "%s_ov_b%d.log" % (a, b))
            if jac(t):
                X[("ov", a, b)] = dict(jac(t), ld=nload(t))
            t = rd("logs", "E165", "%s_F_b%d.log" % (a, b))
            if lrn(t):
                X[("F", a, b)] = lrn(t)
            f = os.path.join(EXP, "traces", "E165", "tr_%s_F_b%d.npz" % (a, b))
            if os.path.exists(f):
                X[("st", a, b)] = stats(np.load(f)["rows"])
    return X


def mx_ok(x, a):
    mult, cnt = ARMS[a]
    return (x["order"] == "random" and x["mult"] == mult and x["cnt"] == cnt and x["n"] == sum(cnt) and x["imin"] >= 5000 and x["imax"] <= 9000
            and abs(x["imean"] - 7000) <= 300 and x["cyc"] <= 4000)


def arm_level(X, a, eD):
    jl = {b: X[("ov", a, b)]["jl"] for b in BRAINS}; jr = {b: X[("ov", a, b)]["jr"] for b in BRAINS}
    eF = {b: X[("F", a, b)]["post"] - X[("F", a, b)]["pre"] for b in BRAINS}
    sep = [b for b in BRAINS if jl[b] <= 1000 and jr[b] <= 1000]
    succ = [b for b in sep if 10 * eF[b] <= 13 * eD[b]]
    null = [b for b in BRAINS if jl[b] > 2500 or jr[b] > 2500]
    lv = "성공" if len(succ) >= 4 else ("분리만" if len(sep) >= 4 else ("실패" if len(null) >= 4 else "보류"))
    return {"lv": lv, "sep": sep, "succ": succ, "null": null, "eF": eF}


def judge(X):
    need = [("D", b) for b in BRAINS] + [(k, a, b) for a in ARMS for b in BRAINS for k in ("dev", "x", "ov", "F", "st")]
    miss = [k for k in need if k not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    eD = {b: X[("D", b)]["post"] - X[("D", b)]["pre"] for b in BRAINS}
    c = {k: 0 for k in ("MX", "MO", "MS", "ML", "M1", "M1b", "M3")}
    for a in ARMS:
        for b in BRAINS:
            d, st = X[("dev", a, b)], X[("st", a, b)]
            c["MX"] += mx_ok(X[("x", a, b)], a)
            c["MO"] += d["oja"] >= 1 and all(d[s][g] > 0 for s in "lr" for g in ("dg_good", "dg_bad"))
            c["MS"] += all(i4(d[s]["sel_med"]) - i4(d[s]["sel_med0"]) >= 1000 for s in "lr")
            c["ML"] += X[("ov", a, b)]["ld"] >= 1 and X[("F", a, b)]["ld"] >= 2
            c["M1"] += st["res"] <= 1e-3
            c["M1b"] += st["alive"] >= 0.9
            c["M3"] += st["n"] == 500 and st["pre_ratio"] <= 1e-3
    md = sum(eD[b] <= -1000 for b in BRAINS)
    ok = all(v == 10 for v in c.values()) and md == 5
    checks = ["[조작검증] MX 노출 구성 %d/10 · MO Oja 변화 %d/10 · MS' 선택성 상승 ≥0.10 %d/10 · ML 적재 %d/10 · M1 동결 %d/10 · M1b 흔적 생존 %d/10 · "
              "M3 추적 %d/10 · MD 기준 효과(E161 D) %d/5 %s" % (c["MX"], c["MO"], c["MS"], c["ML"], c["M1"], c["M1b"], c["M3"], md, "통과" if ok else "실패")]
    L = {a: arm_level(X, a, eD) for a in ARMS}
    nr, nu = L["NR"]["lv"], L["NU"]["lv"]
    if not ok:
        v = "보류(조작검증 실패)"
    elif nr == "성공" and nu == "성공":
        v = "견고(H088) — 무작위 순서·강도 변이(NR)와 bad 세 배 빈도(NU) 모두에서 망 안 형성이 분리·학습 이점을 낸다"
    elif nr == "성공":
        v = "빈도 의존(H088-freq) — 순서·강도 변이에는 견고하나 good 희소(1:3)에서는 %s" % nu
    elif nr in ("실패", "분리만"):
        v = "순서·강도 의존(H088-null) — 무작위 순서·강도 변이(NR)에서 %s" % nr
    else:
        v = "보류"
    return checks, {"eD": eD, "L": L, "ok": ok, "verdict": v}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for a in ARMS:
        for b in BRAINS:
            d, o, x = X[("dev", a, b)], X[("ov", a, b)], X[("x", a, b)]
            eF = res["L"][a]["eF"][b]; eD = res["eD"][b]
            C = X.get(("C", b)); Co = X.get(("Cov", b))
            cr = ("%.2f" % (eF / (C["post"] - C["pre"]))) if C else "-"
            nn = ("good %d/%d bad %d/%d (순환 %d/%d·%d/%d)" % (o["n"][0], o["n"][2], o["n"][1], o["n"][3], Co["n"][0], Co["n"][2], Co["n"][1], Co["n"][3])
                  if Co else "good %d/%d bad %d/%d" % (o["n"][0], o["n"][2], o["n"][1], o["n"][3]))
            print("%s b%d 노출 %s 강도 %.4f~%.4f 평균 %.4f 순환 %.4f | 선택성 %.3f→%.3f / %.3f→%.3f goodfrac %.3f/%.3f | 자카드 %.4f/%.4f %s | "
                  "e %+.4f (기본 %+.4f, r %.2f, 순환 형성 대비 %s) 보상 %s"
                  % (a, b, "·".join(str(k) for k in x["cnt"]), x["imin"] / 1e4, x["imax"] / 1e4, x["imean"] / 1e4, x["cyc"] / 1e4,
                     d["l"]["sel_med0"], d["l"]["sel_med"], d["r"]["sel_med0"], d["r"]["sel_med"], d["l"]["goodfrac"], d["r"]["goodfrac"],
                     o["jl"] / 1e4, o["jr"] / 1e4, nn, eF / 1e4, eD / 1e4, eF / eD, cr, X[("F", a, b)]["rew"]))
        L = res["L"][a]
        print("%s: %s (분리 %d/5 · 성공 %d/5 · 실패 %d/5)" % (a, L["lv"], len(L["sep"]), len(L["succ"]), len(L["null"])))
    print("판정: %s" % res["verdict"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
