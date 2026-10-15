#!/usr/bin/env python3
"""E197 판정 — L4 판독 형식: 부호 판독(sgnR)으로 범주 × 맥락 XOR 이 행동에 서는가. 규칙은 research/experiments/logs/E197/criteria_fixed.txt.
입력(뇌 10~14): 평가 logs/E197/ev_b{B}_{none|sel|sgnR|sgnA}_{X|Y}.log(RCEVAL 4줄 — 범주 A·B × train·new, ctx=X|Y),
  지수 파일 research/experiments/traces/E196/sxi_b{B}.npz(E196 −4900 측정), 회귀 기준 logs/E196/ev_b{B}_{none|sel}_{X|Y}.log.
지표(1e-4 정수): e_{팔,x,c,s} = m(팔) − m(none). XOR 방향 a = −e(X·A, Y·B 교차 칸), +e(X·B, Y·A 같은 쪽 칸).
조작검증(하나라도 어기면 보류): E196 과 같은 평가 줄 검사(judge_e196.read_eval — RCEVAL 4줄·trials 96·feat_on 96·ctx 표지·ctx_on 96, 종류 검증 줄 ≥ 1,
  특징 가중치 적재 없음, 포화 없음), pushed = 0(none)·4(팔), 팔 로그의 '[E196 오라클]' 줄: 팔·θ 0.50·파일·쪽마다 교차/같은 쪽/대상 수 = 지수 파일에서 다시 센 값
  (sel: 반응·|SXI| ≥ θ, sgnR: 반응·SXI 부호, sgnA: 발화 있는 KC(tot > 0)·SXI 부호), '장치 검증 일치' 줄;
  none 회귀(E196 none)·sel 회귀(E196 sel): 뇌마다 8값(맥락 2 × 범주 2 × 세트 2) 모두 |차| ≤ 2;
  조작 무효 의심: sgnR 효과 8값이 모두 |e| ≤ 2 인 뇌가 있으면 보류.
뇌 단위(sgnR·sgnA·sel): 획득 칸 n = #{a_train ≥ 1000}; 획득 = n 4; 부분 = 2 ≤ n ≤ 3; 실패 = n ≤ 1.
종합(우선순위): 조작검증 실패 → 보류 / sgnR 획득 ≥ 4/5 → 표현 충분(부호 판독, H118) / sgnR 실패 ≥ 4/5 → 표현·출력 병목(H118-null)
  / sgnR 부분 ≥ 4/5 → 부분 / 그 밖 보류. (보고) sgnA·sel.
실행: python3 scripts/judge_e197.py (저장소 루트에서)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_e196 import CELLS, THETA, read_eval  # noqa: E402 — E196 에서 합성 22/22·실제 로그로 검증한 파서

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = ("sel", "sgnR", "sgnA")
REG = ("none", "sel")


def expect_counts(npz_path, arm):
    Z = np.load(npz_path)
    out = {}
    for sd in "lr":
        s = np.round(Z["sxi_" + sd], 9)
        el = (np.round(Z["tot_" + sd], 9) > 0) if arm == "sgnA" else Z["resp_" + sd].astype(bool)
        if arm == "sel":
            cr, sm = el & (s >= THETA), el & (s <= -THETA)
        else:
            cr, sm = el & (s > 0), el & (s < 0)
        out["cross_" + sd] = int(cr.sum())
        out["same_" + sd] = int(sm.sum())
        out["resp_" + sd] = int(el.sum())
    return out


def orc_ok(orc, arm, npz_path):
    if orc is None:
        return False, "오라클 줄 없음"
    try:
        if orc.get("arm") != arm or abs(float(orc.get("theta", "nan")) - THETA) > 1e-9 or os.path.basename(orc.get("file", "")) != os.path.basename(npz_path):
            return False, "팔·θ·파일 %s" % orc
        for k, v in expect_counts(npz_path, arm).items():
            if int(orc.get(k, -1)) != v:
                return False, "%s %s≠%d" % (k, orc.get(k), v)
    except (OSError, KeyError, ValueError) as e:
        return False, "%s" % type(e).__name__
    return True, ""


def main():
    ok, why, E = True, [], {}
    for b in BRAINS:
        npz = os.path.join(EXP, "traces", "E196", "sxi_b%d.npz" % b)
        try:
            ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E197", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("none",) + ARMS for x in "XY"}
            ref = {(w, x): read_eval(os.path.join(EXP, "logs", "E196", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in REG for x in "XY"}
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        m = {}
        for (w, x), r in ev.items():
            good = (set(r["rc"]) == {(c, s) for c in "AB" for s in ("train", "new")} and not r["bad"] and not r["sat"] and r["kt"] >= 1
                    and r["fw"] == 0 and r["pushed"] == (0 if w == "none" else 4))
            note = ""
            if w != "none":
                oo, note = orc_ok(r["orc"], w, npz)
                good = good and oo and r["ver"] >= 1
            if not good:
                ok = False
                why.append("b%d %s %s 평가(줄 %d·나쁨 %s·포화 %s·종류 %d·특징 %d·pushed %s·검증 %d %s)"
                           % (b, w, x, len(r["rc"]), r["bad"], r["sat"], r["kt"], r["fw"], r["pushed"], r["ver"], note))
                continue
            for (c, s), v in r["rc"].items():
                m[(w, x, c, s)] = v
        for w in REG:
            for x in "XY":
                for c in "AB":
                    for s in ("train", "new"):
                        if (w, x, c, s) in m and (c, s) in ref[(w, x)]["rc"] and abs(m[(w, x, c, s)] - ref[(w, x)]["rc"][(c, s)]) > 2:
                            ok = False
                            why.append("b%d %s 회귀 %s %s %s %d vs E196 %d" % (b, w, x, c, s, m[(w, x, c, s)], ref[(w, x)]["rc"][(c, s)]))
        if len(m) == 32:
            E[b] = {(w, x, c, s): m[(w, x, c, s)] - m[("none", x, c, s)] for w in ARMS for x in "XY" for c in "AB" for s in ("train", "new")}
            if all(abs(E[b][("sgnR", x, c, s)]) <= 2 for x in "XY" for c in "AB" for s in ("train", "new")):
                ok = False
                why.append("b%d sgnR 효과 8값 모두 |e| ≤ 2 — 조작 무효 의심" % b)
    if not ok or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    cnt = {w + k: 0 for w in ARMS for k in ("_acq", "_part", "_fail")}
    for b in BRAINS:
        e = E[b]
        line = "b%d" % b
        for w in ("sgnR", "sgnA", "sel"):
            a = {(x, c, s): sg * e[(w, x, c, s)] for x, c, sg in CELLS for s in ("train", "new")}
            n = sum(a[(x, c, "train")] >= 1000 for x, c, _ in CELLS)
            cnt[w + "_acq"] += n == 4
            cnt[w + "_part"] += 2 <= n <= 3
            cnt[w + "_fail"] += n <= 1
            line += " | %s X·A %+d/%+d X·B %+d/%+d Y·A %+d/%+d Y·B %+d/%+d 획득 칸 %d" % (
                w, a[("X", "A", "train")], a[("X", "A", "new")], a[("X", "B", "train")], a[("X", "B", "new")],
                a[("Y", "A", "train")], a[("Y", "A", "new")], a[("Y", "B", "train")], a[("Y", "B", "new")], n)
        print(line)
    print("조작검증 통과(none·sel 회귀 뇌마다 8값 포함) | sgnR 획득 %d · 부분 %d · 실패 %d | sgnA 획득 %d · 부분 %d · 실패 %d | sel 획득 %d · 부분 %d · 실패 %d"
          % tuple(cnt[w + k] for w in ("sgnR", "sgnA", "sel") for k in ("_acq", "_part", "_fail")))
    if cnt["sgnR_acq"] >= 4:
        v = "표현 충분(부호 판독, H118)"
    elif cnt["sgnR_fail"] >= 4:
        v = "표현·출력 병목(H118-null)"
    elif cnt["sgnR_part"] >= 4:
        v = "부분"
    else:
        v = "보류"
    print("판정: " + v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
