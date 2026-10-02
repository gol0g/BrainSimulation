#!/usr/bin/env python3
"""E138 판정 — 기준 logs/E138/criteria_fixed.txt(실행 전 고정, 23:20:43 수정 포함). 뇌 5 × (kcrate 1 + 이식 평가 6)가 다 모이기 전에는 수치를 출력하지 않는다.

단위(규약 P19): mod = 이식 평가 변조폭 = mean(조향|good=우) − mean(조향|good=좌)(음수 = 정답 교차 방향).
e = mod_kc_only − mod_none(KC→motor 학습 효과), e_so = mod_kcselonly − mod_none, R = mod_none − mod_kcpop(집단 교차 권한),
R_sel = mod_none − mod_kcsel(선택 KC 만 교차 권한). ρ = R_sel/|e|, κ = −e_so/|e|. 독립 단위 = 뇌.
"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
MODES = ("none", "all", "kc_only", "kcpop", "kcsel", "kcselonly")
PRE = {10: 0.0195, 11: 0.0150, 12: 0.0262, 13: 0.0320, 14: 0.0165}      # E119 [사전]
POST = {10: -0.0838, 11: -0.0574, 12: -0.0428, 13: -0.0489, 14: -0.0442}  # E119 [사후]
NUM = r"([-+]?(?:[0-9.]+|nan))"
TRATE = re.compile(r"^\s*e138(f?) b(\d+) kcrate: => KCRATE kc_l \| (.*?) \|\| KCRATE kc_r \| (.*)$")
PR = re.compile(r"좌선택 (\d+) 우선택 (\d+) 비선택 (\d+) 무활동 (\d+) \| 희석 " + NUM + r" \| ≥1스파이크 좌전용 (\d+) 우전용 (\d+) 공유 (\d+) 무반응 (\d+)"
                r" \| 여유합 " + NUM + r" 몫 좌선택 " + NUM + r" 우선택 " + NUM + r" 비선택 " + NUM + r" \| θ0\.3 선택 (\d+) θ0\.7 선택 (\d+) \| 제시 스파이크 (\d+)")
TEVAL = re.compile(r"^\s*e138(f?) b(\d+) (none|all|kc_only|kcpop|kcsel|kcselonly): => DECOMP mode=(\w+) mod=([-+0-9.]+)")
FIXMODES = ("kcsel", "kcselonly")   # 2026-10-03 수리 재실행(e138f)이 대체하는 분류 의존 모드(+ kcrate)


def parse_pop(s):
    m = PR.search(s)
    if not m:
        return None
    g = m.groups()
    return {"nL": int(g[0]), "nR": int(g[1]), "nNS": int(g[2]), "n0": int(g[3]), "D": float(g[4]), "g1": tuple(int(x) for x in g[5:9]),
            "M": float(g[9]), "shL": float(g[10]), "shR": float(g[11]), "shNS": float(g[12]), "n03": int(g[13]), "n07": int(g[14]), "sp": int(g[15])}


def judge(RT, EV):
    miss = [("kcrate", b) for b in BRAINS if b not in RT] + [(m, b) for b in BRAINS for m in MODES if (b, m) not in EV]
    if miss:
        return ["[측정 확인] 결측 %d/35: %s — **판정 보류, 수치 미출력**" % (len(miss), miss[:6])], None
    checks, ok = [], True
    bad_mode = [(b, m) for (b, m), v in EV.items() if v["mode"] != m]
    c1 = [b for b in BRAINS if abs(EV[(b, "none")]["mod"] - PRE[b]) <= 0.005 + 1e-9 and abs(EV[(b, "all")]["mod"] - POST[b]) <= 0.005 + 1e-9]
    checks.append("[C1 재현] none·all 이 E119 [사전]·[사후] ±0.005: %d/5 %s" % (len(c1), "통과" if len(c1) == 5 and not bad_mode else "실패"))
    ok &= len(c1) == 5 and not bad_mode
    Rv = {b: round(EV[(b, "none")]["mod"] - EV[(b, "kcpop")]["mod"], 6) for b in BRAINS}
    c2 = [b for b in BRAINS if Rv[b] >= 0.40]
    checks.append("[C2 권한 재현] R = none − kcpop ≥ 0.40: %d/5 (%s) %s" % (len(c2), " ".join("%.3f" % Rv[b] for b in BRAINS), "통과" if len(c2) == 5 else "실패"))
    ok &= len(c2) == 5
    checks.append("[C3 이식 정확 일치] 이식 모드 DECOMP 줄 존재 = TE.verify 통과(실패 시 예외로 줄 없음): 25/25 통과")
    latl = sum(RT[b]["l"]["nL"] > RT[b]["l"]["nR"] for b in BRAINS); latr = sum(RT[b]["r"]["nR"] > RT[b]["r"]["nL"] for b in BRAINS)
    checks.append("[C4 편측성] kc_l 좌선택 > 우선택 %d/5, kc_r 우선택 > 좌선택 %d/5 (각 ≥4) %s" % (latl, latr, "통과" if latl >= 4 and latr >= 4 else "실패"))
    ok &= latl >= 4 and latr >= 4
    c5 = [b for b in BRAINS if all(RT[b][p]["sp"] > 0 and RT[b][p]["nL"] + RT[b][p]["nR"] >= 1 for p in "lr")]
    checks.append("[C5 측정 유효] 제시 스파이크 > 0·선택 KC ≥1(각 집단): %d/5 %s" % (len(c5), "통과" if len(c5) == 5 else "실패"))
    ok &= len(c5) == 5
    e = {b: round(EV[(b, "kc_only")]["mod"] - EV[(b, "none")]["mod"], 6) for b in BRAINS}
    eso = {b: round(EV[(b, "kcselonly")]["mod"] - EV[(b, "none")]["mod"], 6) for b in BRAINS}
    Rs = {b: round(EV[(b, "none")]["mod"] - EV[(b, "kcsel")]["mod"], 6) for b in BRAINS}
    valid = [b for b in BRAINS if e[b] < -0.005]          # KC 경로 학습 효과가 정답 방향으로 있어야 비율이 뜻을 가진다
    rho = {b: (round(Rs[b] / abs(e[b]), 6) if b in valid else float("nan")) for b in BRAINS}
    kap = {b: (round(-eso[b] / abs(e[b]), 6) if b in valid else float("nan")) for b in BRAINS}
    n_rep = sum(1 for b in valid if rho[b] <= 2.5); n_lrn = sum(1 for b in valid if rho[b] >= 4)
    n_cm = sum(1 for b in valid if kap[b] >= 1.5); n_ncm = sum(1 for b in valid if kap[b] <= 1.1)
    if not ok:
        q1 = q2 = "보류(조작검증 실패)"
    else:
        q1 = "표현 상한 — 선택 KC 만의 권한이 학습 효과와 비슷하다" if n_rep >= 4 else ("학습 상한 — 선택 KC 의 권한을 학습이 다 쓰지 못한다" if n_lrn >= 4 else "보류(Q1)")
        q2 = "공통 모드 억제 — 비선택 KC 학습값을 되돌리면 효과가 커진다" if n_cm >= 4 else ("공통 모드 억제 없음" if n_ncm >= 4 else "보류(Q2)")
    return checks, {"e": e, "eso": eso, "R": Rv, "Rs": Rs, "rho": rho, "kap": kap, "valid": valid, "n_rep": n_rep, "n_lrn": n_lrn,
                    "n_cm": n_cm, "n_ncm": n_ncm, "ok": ok, "q1": q1, "q2": q2}


def report(checks, res, RT=None, EV=None):
    for c in checks:
        print(c)
    if res is None:
        return
    for b in BRAINS:
        print("b%d mod: none %+.4f all %+.4f kc_only %+.4f kcpop %+.4f kcsel %+.4f kcselonly %+.4f | e %+.4f e_so %+.4f R %.4f R_sel %.4f | ρ %.2f κ %.2f"
              % (b, *(EV[(b, m)]["mod"] for m in MODES), res["e"][b], res["eso"][b], res["R"][b], res["Rs"][b], res["rho"][b], res["kap"][b]))
        for p in "lr":
            q = RT[b][p]
            print("   kc_%s 좌선택 %d 우선택 %d 비선택 %d 무활동 %d | 희석 %.3f | ≥1스파이크 공유 %d | 여유 몫 좌 %.3f 우 %.3f 비선택 %.3f"
                  % (p, q["nL"], q["nR"], q["nNS"], q["n0"], q["D"], q["g1"][2], q["shL"], q["shR"], q["shNS"]))
    print("Q1(ρ = R_sel/|e|): ρ ≤ 2.5 %d/5, ρ ≥ 4 %d/5 (유효 뇌 %d) → %s" % (res["n_rep"], res["n_lrn"], len(res["valid"]), res["q1"]))
    print("Q2(κ = −e_so/|e|): κ ≥ 1.5 %d/5, κ ≤ 1.1 %d/5 → %s" % (res["n_cm"], res["n_ncm"], res["q2"]))


def load(use_fix=True):
    """1차(태그 e138)와 수리 재실행(태그 e138f, 2026-10-03 — kc_selectivity eps 경계 수리 뒤 kcrate·kcsel·kcselonly)을 읽는다.
    수리 재실행이 5뇌 × 3모드 모두 있으면 그 값으로 대체한다(판정 규칙 불변, criteria_fixed.txt [수리] 줄)."""
    RT, EV, RTf, EVf = {}, {}, {}, {}
    try:
        for ln in open(os.path.join(EXP, "E138.log"), encoding="utf-8"):
            m = TRATE.match(ln)
            if m:
                pl, pr = parse_pop(m.group(3)), parse_pop(m.group(4))
                if pl and pr:
                    (RTf if m.group(1) else RT)[int(m.group(2))] = {"l": pl, "r": pr}
                continue
            m = TEVAL.match(ln)
            if m:
                (EVf if m.group(1) else EV)[(int(m.group(2)), m.group(3))] = {"mode": m.group(4), "mod": float(m.group(5))}
    except FileNotFoundError:
        pass
    complete = all(b in RTf for b in BRAINS) and all((b, mm) in EVf for b in BRAINS for mm in FIXMODES)
    if use_fix and complete:
        return dict(RTf), {**EV, **{k: v for k, v in EVf.items() if k[1] in FIXMODES}}, (RT, EV)
    return RT, EV, None


if __name__ == "__main__":
    RT, EV, first = load()
    if first is not None:
        c1, r1 = judge(*first)
        print("[1차(eps 경계 결함 포함) 참고] Q1 %s / Q2 %s" % ((r1["q1"], r1["q2"]) if r1 else ("결측", "결측")))
        print("[수리 재실행 e138f 사용] kcrate·kcsel·kcselonly 15/15 — 아래가 판정")
    c, r = judge(RT, EV)
    report(c, r, RT, EV)
    sys.exit(0)
