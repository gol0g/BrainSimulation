#!/usr/bin/env python3
"""E168 판정 — 보상 창 흔적의 망 안 차단(도파민 뉴런 → KC 억제 뉴런). 기준 logs/E168/criteria_fixed.txt. 10런이 다 모이기 전에는 수치를 출력하지 않는다.
팔 FI = 형성 표현 + 동결 없음 + rw-da-reset + 연결 W*, FR = 같은 것에서 연결 없음. 기준 e_F = 같은 뇌 E161 F(형성 + 동결). 뇌 16~20. 1e-4 정수 산술.
판정 1: 대체 성공(H091) = q_I = e_FI/e_F ≥ 0.80(⇔ 10·E_FI ≤ 8·E_F) ≥ 4/5, 대체 실패(H091-null) = q_I ≤ 0.50(⇔ 2·E_FI ≥ E_F) ≥ 4/5, 그 밖 부분.
판정 2(부): 연결 효과 = q_I − q_R ≥ 0.20(⇔ 5·(E_FI − E_FR) ≤ E_F) ≥ 4/5, 효과 없음 = |q_I − q_R| < 0.10(⇔ 10·|E_FI − E_FR| < |E_F|) ≥ 4/5, 그 밖 중간.
조작검증(하나라도 실패면 보류): MC 연결 줄 FI 1·FR 0(10/10), MK 보상 시행 보상 창 KC 발화율 FI ≤ FR 의 10% 이고 결정 단계 FI ≥ FR 의 90%(5/5, 1e-6 정수),
MD '[구현 점검] 첫 보상 창 끝 도파민 뉴런 I_input … → 0.0'(10/10), MP [사전] FI = FR = E161 F ±0.002(10/10), MT 추적 500·도파민 전 변화 ≤ 1e-3(10/10). 전제 e_F ≤ −0.10(5/5).
실행: python3 scripts/judge_e168.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
KL = re.compile(r"^\[E168 KC 발화\] 결정 단계\(3처리 끝\) 평균 ([0-9.na]+) n=(\d+) \| 보상 창 보상 시행 평균 ([0-9.na]+) n=(\d+) \| "
                r"보상 창 처벌 시행 평균 ([0-9.na]+) n=(\d+) \| da_kc_inh=([0-9.]+)", re.M)


def i4(x):
    return int(round(float(x) * 1e4))


def i6(x):
    return int(round(float(x) * 1e6))


def rd(*p):
    f = os.path.join(EXP, *p)
    return open(f, encoding="utf-8", errors="replace").read() if os.path.exists(f) else None


def lrn(t):
    a = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    b = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d+)", t, re.M) if t else None
    if not (a and b):
        return None
    r = re.search(r"보상 (\d+)회", t)
    k = KL.search(t)
    da = re.search(r"^\[구현 점검\] 첫 보상 창 끝 도파민 뉴런 I_input ([0-9.]+) → ([0-9.]+)", t, re.M)
    return {"pre": i4(a.group(1)), "post": i4(b.group(1)), "rew": int(r.group(1)) if r else None,
            "conn": len(re.findall(r"^  \[E168 도파민→KC억제\]", t, re.M)),
            "kc": {"dec": i6(k.group(1)), "rw": i6(k.group(3)), "pun": i6(k.group(5)), "w": float(k.group(7))} if k and "nan" not in (k.group(1), k.group(3)) else None,
            "da0": (float(da.group(2)) == 0.0) if da else False}


def stats(rows):
    return {"n": len(rows), "pre_ratio": float(np.abs(rows[:, 17:21].sum()) / max(np.abs(rows[:, 12].sum()), 1e-12))}


def load():
    X = {}
    for b in BRAINS:
        for a in ("FI", "FR"):
            t = rd("logs", "E168", "%s_b%d.log" % (a, b))
            if lrn(t):
                X[(a, b)] = lrn(t)
            f = os.path.join(EXP, "traces", "E168", "tr_%s_b%d.npz" % (a, b))
            if os.path.exists(f):
                X[("s", a, b)] = stats(np.load(f)["rows"])
        t = rd("logs", "E161", "F_b%d.log" % b)
        if lrn(t):
            X[("F", b)] = lrn(t)
    return X


def judge(X):
    need = [(a, b) for a in ("FI", "FR", "F") for b in BRAINS] + [("s", a, b) for a in ("FI", "FR") for b in BRAINS]
    miss = [k for k in need if k not in X]
    if miss:
        return ["[측정 확인] 결측 %s — **판정 보류, 수치 미출력**" % miss[:6]], None
    mc = sum(X[("FI", b)]["conn"] == 1 for b in BRAINS) + sum(X[("FR", b)]["conn"] == 0 for b in BRAINS)
    mk = sum(X[("FI", b)]["kc"] is not None and X[("FR", b)]["kc"] is not None
             and 10 * X[("FI", b)]["kc"]["rw"] <= X[("FR", b)]["kc"]["rw"] and 10 * X[("FI", b)]["kc"]["dec"] >= 9 * X[("FR", b)]["kc"]["dec"] for b in BRAINS)
    md = sum(X[(a, b)]["da0"] for a in ("FI", "FR") for b in BRAINS)
    mp = sum(abs(X[(a, b)]["pre"] - X[("F", b)]["pre"]) <= 20 for a in ("FI", "FR") for b in BRAINS)
    mt = sum(X[("s", a, b)]["n"] == 500 and X[("s", a, b)]["pre_ratio"] <= 1e-3 for a in ("FI", "FR") for b in BRAINS)
    eF = {b: X[("F", b)]["post"] - X[("F", b)]["pre"] for b in BRAINS}
    pc = sum(eF[b] <= -1000 for b in BRAINS)
    ok = mc == 10 and mk == 5 and md == 10 and mp == 10 and mt == 10 and pc == 5
    checks = ["[조작검증] MC 연결 줄 %d/10 · MK 보상 창 KC ≤10%%·결정 ≥90%% %d/5 · MD I_input → 0 %d/10 · MP [사전] 재현 %d/10 · MT 추적 %d/10 · 전제 e_F ≤ −0.10 %d/5 %s"
              % (mc, mk, md, mp, mt, pc, "통과" if ok else "실패")]
    eI = {b: X[("FI", b)]["post"] - X[("FI", b)]["pre"] for b in BRAINS}
    eR = {b: X[("FR", b)]["post"] - X[("FR", b)]["pre"] for b in BRAINS}
    succ = [b for b in BRAINS if 10 * eI[b] <= 8 * eF[b]]
    fail = [b for b in BRAINS if 2 * eI[b] >= eF[b]]
    eff = [b for b in BRAINS if 5 * (eI[b] - eR[b]) <= eF[b]]
    none = [b for b in BRAINS if 10 * abs(eI[b] - eR[b]) < abs(eF[b])]
    v1 = "대체 성공(H091)" if len(succ) >= 4 else ("대체 실패(H091-null)" if len(fail) >= 4 else "부분")
    v2 = "연결 효과" if len(eff) >= 4 else ("효과 없음" if len(none) >= 4 else "중간")
    if not ok:
        v1 = v2 = "보류(조작검증 실패)"
    return checks, {"eI": eI, "eR": eR, "eF": eF, "succ": succ, "fail": fail, "eff": eff, "none": none, "v1": v1, "v2": v2, "ok": ok}


def report(X, checks, res):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        I, Rr = X[("FI", b)], X[("FR", b)]
        print("b%d FI e %+.4f q_I %.3f 보상 %s | FR e %+.4f q_R %.3f 보상 %s | 동결 e_F %+.4f | KC 결정 %.6f/%.6f 보상 창(보상) %.6f/%.6f(처벌 %.6f/%.6f)"
              % (b, res["eI"][b] / 1e4, res["eI"][b] / res["eF"][b], I["rew"], res["eR"][b] / 1e4, res["eR"][b] / res["eF"][b], Rr["rew"], res["eF"][b] / 1e4,
                 I["kc"]["dec"] / 1e6, Rr["kc"]["dec"] / 1e6, I["kc"]["rw"] / 1e6, Rr["kc"]["rw"] / 1e6, I["kc"]["pun"] / 1e6, Rr["kc"]["pun"] / 1e6))
    print("q_I ≥ 0.80 %d/5 · q_I ≤ 0.50 %d/5 | q_I − q_R ≥ 0.20 %d/5 · |q_I − q_R| < 0.10 %d/5"
          % (len(res["succ"]), len(res["fail"]), len(res["eff"]), len(res["none"])))
    print("판정 1: %s" % res["v1"])
    print("판정 2: %s" % res["v2"])


if __name__ == "__main__":
    X = load()
    c, r = judge(X)
    report(X, c, r)
    sys.exit(0)
