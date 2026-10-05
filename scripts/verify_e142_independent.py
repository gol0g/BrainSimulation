#!/usr/bin/env python3
"""E142 독립 대조 — judge_e142.py 를 쓰지 않고 런별 원 로그(logs/E142/{팔}_b*.log)와 추적에서 다시 계산한다.
- 변조폭·보상은 원 로그의 [사전]·[사후]·[학습] 줄에서 1e-4 정수로(요약 E142.log 를 쓰지 않음).
- 반사 불변은 원 로그 [반사가중치] good_food_to_motor 줄의 두 값 문자열이 같은지(25.0000) 직접.
- 동결 잔차는 시행·역할별 비 |e_end/e_da − r*| 의 최대(|e_da| ≥ 1 인 칸)와 합 기준 둘 다.
- 판정은 criteria_fixed.txt 문장(수정 1 포함)을 이 파일 안에서 정수 비교로 다시 구현.
실행: python3 scripts/verify_e142_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
PRE119 = {10: 4148, 11: 4195, 12: 3954, 13: 3773, 14: 4248}
RS = (11.0 / 12.0) ** 20
NEXP = {"F500": 500, "F1500": 1500, "NF1500": 1500}


def i4(s):
    m = re.fullmatch(r"([-+]?)(\d+)\.(\d{4})", s)
    if not m:
        raise ValueError("4자리 값 아님: %r" % s)
    v = int(m.group(2)) * 10000 + int(m.group(3))
    return -v if m.group(1) == "-" else v


def raw(arm, b):
    out = {"pre": None, "post": None, "rew": None, "refl": {}}
    for ln in open(os.path.join(EXP, "logs", "E142", "%s_b%d.log" % (arm, b)), encoding="utf-8"):
        if ln.startswith("[사전]"):
            out["pre"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[사후]"):
            out["post"] = i4(re.search(r"변조폭 ([-+]?\d+\.\d{4})", ln).group(1))
        elif ln.startswith("[학습]"):
            out["rew"] = int(re.search(r"보상 (\d+)회", ln).group(1))
        elif ln.startswith("[반사가중치] good_food_to_motor_"):
            side = ln.split("good_food_to_motor_")[1][0]
            w = re.search(r"w_mean (\S+)→(\S+)", ln)
            out["refl"][side] = (w.group(1), w.group(2))
    return out


def tr(arm, b):
    R = np.load(os.path.join(EXP, "traces", "E142", "tr_%s_b%d.npz" % (arm, b)))["rows"]
    eda, eend = R[:, 13:17], R[:, 21:25]
    big = np.abs(eda) >= 1.0
    ratio_dev = float(np.max(np.abs(eend[big] / eda[big] - RS))) if big.any() else float("nan")
    return {"n": R.shape[0], "res_sum": float(np.abs(eend - RS * eda).sum() / np.abs(eda).sum()), "ratio_dev": ratio_dev,
            "alive": float(np.count_nonzero((np.abs(R[:, 13]) + np.abs(R[:, 14])) > 1.0) / R.shape[0]),
            "pre": float(abs(R[:, 17:21].sum()) / abs(R[:, 12].sum())), "rew_rows": int((R[:, 7] == 1).sum())}


def main():
    D = {a: {b: (raw(a, b), tr(a, b)) for b in PRE119} for a in NEXP}
    ok_f, ok_nf, lines = True, True, []
    for a in NEXP:
        c = {"동결": 0, "되돌림": 0, "출발점": 0, "추적": 0, "반사": 0, "보상일치": 0}
        for b, (r, t) in D[a].items():
            frozen = t["res_sum"] <= 1e-3
            c["동결"] += (not frozen) if a == "NF1500" else frozen
            c["되돌림"] += t["alive"] >= 0.9
            c["출발점"] += abs(r["pre"] - PRE119[b]) <= 20
            c["추적"] += (t["n"] == NEXP[a] and t["pre"] <= 1e-3)
            c["반사"] += (set(r["refl"]) == {"l", "r"} and all(v == ("25.0000", "25.0000") for v in r["refl"].values()))
            c["보상일치"] += r["rew"] == t["rew_rows"]
        need = ("동결", "출발점", "추적", "반사", "보상일치") if a == "NF1500" else tuple(c)
        good = all(c[k] == 5 for k in need)
        if a == "NF1500":
            ok_nf = good
        else:
            ok_f &= good
        lines.append("%-6s %s → %s" % (a, " ".join("%s %d/5" % (k, c[k]) for k in need), "통과" if good else "실패"))
    print("뇌  팔      [사전]  [사후]    e     보상 | 잔차 합·비편차최대 | 흔적살아있음")
    for a in NEXP:
        for b, (r, t) in D[a].items():
            print("b%d %-6s %+6d %+7d %+6d %4d | %.1e %.1e | %.3f" % (b, a, r["pre"], r["post"], r["post"] - r["pre"], r["rew"], t["res_sum"], t["ratio_dev"], t["alive"]))
    for ln in lines:
        print(ln)
    e5 = {b: D["F500"][b][0]["post"] - D["F500"][b][0]["pre"] for b in PRE119}
    m15 = {b: D["F1500"][b][0]["post"] for b in PRE119}
    mnf = {b: D["NF1500"][b][0]["post"] for b in PRE119}
    l2 = sum(m15[b] <= -200 for b in PRE119)
    opp = sum(e5[b] <= -1000 for b in PRE119)
    nul = sum(abs(e5[b]) < 300 for b in PRE119)
    if not ok_f:
        v = "보류(조작검증 실패)"
    elif l2 >= 4:
        v = "L2 달성(H065)"
    elif opp == 5:
        v = "반사를 거스름(H065-partial)"
    elif nul >= 4:
        v = "효과 없음(H065-null)"
    else:
        v = "보류"
    if not v.startswith("L2"):
        nd = "해당 없음(L2 아님)"
    elif not ok_nf:
        nd = "필요성 미결(NF1500 조작검증 실패)"
    elif sum(mnf[b] > -200 for b in PRE119) >= 4:
        nd = "동결이 L2 에 필요(이 학습량에서)"
    elif sum(mnf[b] <= -200 for b in PRE119) >= 4:
        nd = "학습량만으로도 L2 — 동결 불필요"
    else:
        nd = "필요성 미결"
    print("F1500 m ≤ −0.02 %d/5 | F500 e ≤ −0.10 %d/5 | F500 |e| < 0.03 %d/5" % (l2, opp, nul))
    print("독립 판정: %s" % v)
    print("독립 필요성: %s" % nd)


if __name__ == "__main__":
    sys.exit(main())
