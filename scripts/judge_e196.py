#!/usr/bin/env python3
"""E196 판정 — L4 병목 위치: 이상 KC→motor 가중치(오라클)로 범주 × 맥락 XOR 이 행동에 서는가(K74 의 L4 판). 규칙은 research/experiments/logs/E196/criteria_fixed.txt.
입력(뇌 10~14): 평가 logs/E196/ev_b{B}_{none|cat|sel|eq}_{X|Y}.log(RCEVAL 4줄 — 범주 A·B × train·new, ctx=X|Y),
  지수 파일 research/experiments/traces/E196/sxi_b{B}.npz(scripts/kcxor_build.py — E195 S1 측정), 회귀 기준 logs/E193/ev_b{B}_none_{X|Y}.log.
지표(1e-4 정수): e_{팔,x,c,s} = m(팔) − m(none). XOR 방향 a = −e(X·A, Y·B 교차 칸), +e(X·B, Y·A 같은 쪽 칸). 범주 방향(양성 대조 O_cat) k = −e(A), +e(B) — 두 맥락.
조작검증(하나라도 어기면 보류): RCEVAL 4줄·trials 96·feat_on 96·ctx 표지·ctx_on 96, 종류 검증 줄 ≥ 1, '[E184 특징 가중치 적재]' 없음, 포화 칸(한 쪽 두 motor ≥ 0.66) 없음,
  pushed = 0(none)·4(팔), 팔 로그의 '[E196 오라클]' 줄: 팔·θ 0.50·파일 = 지수 파일·선택 수(쪽마다 교차·같은 쪽)와 반응 수 = 파일에서 다시 센 값, '장치 검증 일치' 줄;
  none 회귀: E196 none 과 E193 none 이 16값 모두 |차| ≤ 2(1e-4).
뇌 단위(팔 sel·eq): 획득 칸 n = #{a_train ≥ 1000}; 획득 = n 4; 부분 = 2 ≤ n ≤ 3; 실패 = n ≤ 1; 일반화(보고) = 네 칸 a_new ≥ 500·2·a_new ≥ a_train.
양성 대조(뇌 단위): O_cat 범주 방향 노출 네 값(X·A, X·B, Y·A, Y·B) 모두 ≥ 1000.
종합(우선순위): 조작검증 실패 → 보류 / 양성 대조 통과 < 4/5 → 보류(양성 대조 실패) /
  sel 획득 ≥ 4/5 → [eq 획득 ≥ 4/5 → 표현 충분·평형 충분(학습 동역학 병목) / eq 실패 ≥ 4/5 → 표현 충분·상관 평형 부족(학습 규칙 병목) / 그 밖 → 표현 충분·평형 보류]
  / sel 실패 ≥ 4/5 → 표현·출력 병목(H117-null) / sel 부분 ≥ 4/5 → 부분 / 그 밖 보류.
(보고) 같은 뇌 E193 학습(learn) 칸 효과.
실행: python3 scripts/judge_e196.py (저장소 루트에서)"""
import os
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
TRIALS = 96
ARMS = ("cat", "sel", "eq")
CELLS = (("X", "A", -1), ("X", "B", +1), ("Y", "A", +1), ("Y", "B", -1))
CATDIR = {"A": -1, "B": +1}
THETA = 0.5


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_eval(path, ctx):
    r = {"rc": {}, "pushed": None, "kt": 0, "fw": 0, "sat": False, "bad": False, "orc": None, "ver": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCEVAL "):
                kv = {}
                for x in ln.split(" | ")[0].split()[1:]:
                    k, v = x.split("=", 1)
                    kv[k] = v
                r["rc"][(kv["cat"], kv["set"])] = q4(kv["mod"])
                if int(kv["trials"]) != TRIALS or int(kv["feat_on"]) != TRIALS or kv.get("ctx") != ctx or int(kv.get("ctx_on", "-1")) != TRIALS:
                    r["bad"] = True
                for seg in ln.split(" | ")[1:]:
                    for part in seg.split("motor L/R ")[1:]:
                        a, b = part.split()[0].split("/")
                        if q4(a) >= 6600 and q4(b) >= 6600:
                            r["sat"] = True
            elif ln.startswith("=> DECOMP ") and "rceval" in ln:
                r["pushed"] = int(ln.split("pushed=")[1].split()[0])
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                r["kt"] += 1
            elif ln.startswith("[E184 특징 가중치 적재]"):
                r["fw"] += 1
            elif ln.startswith("[E196 오라클] "):
                kv = dict(x.split("=", 1) for x in ln.split()[2:] if "=" in x)
                r["orc"] = kv
            elif ln.startswith("[E196 오라클 장치 검증] 일치"):
                r["ver"] += 1
    return r


def expect_counts(npz_path, arm):
    Z = np.load(npz_path)
    out = {}
    for sd in "lr":
        rp = Z["resp_" + sd].astype(bool)
        if arm == "cat":
            s = Z["csi_" + sd]
        else:
            s = Z["sxi_" + sd]
        out["cross_" + sd] = int(np.sum(rp & (np.round(s, 9) >= THETA)))
        out["same_" + sd] = int(np.sum(rp & (np.round(s, 9) <= -THETA)))
        out["resp_" + sd] = int(rp.sum())
    return out


def orc_ok(orc, arm, npz_path):
    if orc is None:
        return False, "오라클 줄 없음"
    try:
        if orc.get("arm") != arm or abs(float(orc.get("theta", "nan")) - THETA) > 1e-9 or os.path.basename(orc.get("file", "")) != os.path.basename(npz_path):
            return False, "팔·θ·파일 %s" % orc
        ex = expect_counts(npz_path, arm)
        for k, v in ex.items():
            if int(orc.get(k, -1)) != v:
                return False, "%s %s≠%d" % (k, orc.get(k), v)
    except (OSError, KeyError, ValueError) as e:
        return False, "%s" % type(e).__name__
    return True, ""


def main():
    ok, why, E, NONE = True, [], {}, {}
    for b in BRAINS:
        npz = os.path.join(EXP, "traces", "E196", "sxi_b%d.npz" % b)
        try:
            ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E196", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("none",) + ARMS for x in "XY"}
            ref = {x: read_eval(os.path.join(EXP, "logs", "E193", "ev_b%d_none_%s.log" % (b, x)), x) for x in "XY"}
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
        for x in "XY":
            for c in "AB":
                for s in ("train", "new"):
                    if ("none", x, c, s) in m and (c, s) in ref[x]["rc"] and abs(m[("none", x, c, s)] - ref[x]["rc"][(c, s)]) > 2:
                        ok = False
                        why.append("b%d none 회귀 %s %s %s %d vs E193 %d" % (b, x, c, s, m[("none", x, c, s)], ref[x]["rc"][(c, s)]))
        if len(m) == 32:
            E[b] = {(w, x, c, s): m[(w, x, c, s)] - m[("none", x, c, s)] for w in ARMS for x in "XY" for c in "AB" for s in ("train", "new")}
    if not ok or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    cnt = {"pc": 0, "sel_acq": 0, "sel_part": 0, "sel_fail": 0, "eq_acq": 0, "eq_part": 0, "eq_fail": 0}
    for b in BRAINS:
        e = E[b]
        k = {(x, c): CATDIR[c] * e[("cat", x, c, "train")] for x in "XY" for c in "AB"}
        pc = all(v >= 1000 for v in k.values())
        cnt["pc"] += pc
        line = "b%d 양성 대조(범주 방향, 노출) X·A %+d X·B %+d Y·A %+d Y·B %+d → %s" % (b, k[("X", "A")], k[("X", "B")], k[("Y", "A")], k[("Y", "B")], "통과" if pc else "미달")
        for w in ("sel", "eq"):
            a = {(x, c, s): sg * e[(w, x, c, s)] for x, c, sg in CELLS for s in ("train", "new")}
            n = sum(a[(x, c, "train")] >= 1000 for x, c, _ in CELLS)
            gen = all(a[(x, c, "new")] >= 500 and 2 * a[(x, c, "new")] >= a[(x, c, "train")] for x, c, _ in CELLS)
            cnt[w + "_acq"] += n == 4
            cnt[w + "_part"] += 2 <= n <= 3
            cnt[w + "_fail"] += n <= 1
            line += " | %s X·A %+d/%+d X·B %+d/%+d Y·A %+d/%+d Y·B %+d/%+d 획득 칸 %d%s" % (
                w, a[("X", "A", "train")], a[("X", "A", "new")], a[("X", "B", "train")], a[("X", "B", "new")],
                a[("Y", "A", "train")], a[("Y", "A", "new")], a[("Y", "B", "train")], a[("Y", "B", "new")], n, " 일반화" if gen else "")
        print(line)
    print("조작검증 통과(none 회귀 16/16 포함) | 양성 대조 %d/5 | sel 획득 %d · 부분 %d · 실패 %d | eq 획득 %d · 부분 %d · 실패 %d"
          % (cnt["pc"], cnt["sel_acq"], cnt["sel_part"], cnt["sel_fail"], cnt["eq_acq"], cnt["eq_part"], cnt["eq_fail"]))
    if cnt["pc"] < 4:
        v = "보류(양성 대조 실패 — 측정→평가 대응 미확인)"
    elif cnt["sel_acq"] >= 4:
        if cnt["eq_acq"] >= 4:
            v = "표현 충분·평형 충분 — 학습 동역학 병목(H117)"
        elif cnt["eq_fail"] >= 4:
            v = "표현 충분·상관 평형 부족 — 학습 규칙 병목(H117)"
        else:
            v = "표현 충분·평형 보류(H117)"
    elif cnt["sel_fail"] >= 4:
        v = "표현·출력 병목(H117-null)"
    elif cnt["sel_part"] >= 4:
        v = "부분"
    else:
        v = "보류"
    print("판정: " + v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
