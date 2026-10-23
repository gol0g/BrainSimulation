#!/usr/bin/env python3
"""E207 사후 기술(탐색 — 판정 밖): 맥락 X 의 범주 KC 겹침이 '네 범주 모두에 드는 공통 KC(공통 핵)' 에서 오는가 — E207.md 6절 J(A,B) 보고의 연장.
본실험 결과를 보기 전(경로 검사 뇌 15 만 본 뒤, 2026-10-22) 정한 정의:
  범주 집합 = RCKC 줄의 노출(train) 사례 절반 이상에서 반응한 KC(rawcat.cat_set·posthoc_kc_rewrite.form_sets 와 같은 규칙), 쪽마다.
  공통 핵 = A ∩ B ∩ C ∩ D. 핵 뺀 J(X,Y) = J(X∖핵, Y∖핵). 범주 특이 겹침이면 J(A,C) 가 핵을 빼도 남고 J(A,B) 보다 크다; 공통 핵이면 핵을 빼면 둘 다 낮아진다.
  학습이 바꾼 KC(E205 U·E206 F 의 w_R·w_P, |d_R| ≥ 50 = R 맞춤, |d_P − d_R| ≥ 50 = P 바뀜 — posthoc_kc_rewrite 와 같은 정의) 중 핵 안 몫.
입력: logs/E207/meas_{U|F}_{none|X}_b{B}.log(RCKC tag=pre·post), traces/E176/snap/conn_b{B}.npz, traces/E205|E206/w_R·w_P_b{B}.npz.
실행: python3 scripts/posthoc_e207_core.py (저장소 루트에서) [θ]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import posthoc_kc_rewrite as PK  # noqa: E402

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
WDIR = {"U": "E205", "F": "E206"}


def jac(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def core_stats(cs):
    """cs = {범주: 집합}(한 쪽) → 공통 핵과 핵 포함·제외 J."""
    A, B, C, D = (cs.get(k, set()) for k in "ABCD")
    core = A & B & C & D
    ex = {k: v - core for k, v in (("A", A), ("B", B), ("C", C), ("D", D))}
    return {"core": core, "nA": len(A), "J_AC": jac(A, C), "J_BD": jac(B, D), "J_AB": jac(A, B),
            "Jx_AC": jac(ex["A"], ex["C"]), "Jx_BD": jac(ex["B"], ex["D"]), "Jx_AB": jac(ex["A"], ex["B"])}


def learn_in_core(dR, dP, both, core, theta):
    import numpy as np
    tuned = set(np.nonzero(both & (np.abs(dR) >= theta))[0].tolist())
    changed = set(np.nonzero(both & (np.abs(dP - dR) >= theta))[0].tolist())
    return len(tuned), len(tuned & core), len(changed), len(changed & core)


def main(argv):
    import numpy as np
    theta = float(argv[1]) if len(argv) > 1 else 50.0
    PK.EXP = EXP
    L = os.path.join(EXP, "logs", "E207")
    print("E207 사후 기술(탐색 — 판정 밖): 공통 핵(A∩B∩C∩D)과 핵 뺀 겹침, 학습이 바꾼 KC 중 핵 안 몫 — 두 쪽 합(J 는 쪽 평균), θ %.0f" % theta)
    for b in BRAINS:
        S = np.load(os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b))
        for rep in ("U", "F"):
            R = np.load(os.path.join(EXP, "traces", WDIR[rep], "w_R_b%d.npz" % b)); P = np.load(os.path.join(EXP, "traces", WDIR[rep], "w_P_b%d.npz" % b))
            pref = {sd: (PK.kc_pref(R, S, sd), PK.kc_pref(P, S, sd)) for sd in "lr"}
            parts = []
            for cond in ("none", "X"):
                for tag in ("pre", "post"):
                    sets = PK.form_sets(os.path.join(L, "meas_%s_%s_b%d.log" % (rep, cond, b)), tag)
                    acc = {"core": 0, "nA": 0, "t": 0, "tc": 0, "c": 0, "cc": 0}
                    js = {k: [] for k in ("J_AC", "J_BD", "J_AB", "Jx_AC", "Jx_BD", "Jx_AB")}
                    for sd in "lr":
                        st = core_stats(sets.get(sd, {}))
                        acc["core"] += len(st["core"]); acc["nA"] += st["nA"]
                        for k in js:
                            js[k].append(st[k])
                        (dR, bR), (dP, bP) = pref[sd]
                        t, tc, c, cc = learn_in_core(dR, dP, bR & bP, st["core"], theta)
                        acc["t"] += t; acc["tc"] += tc; acc["c"] += c; acc["cc"] += cc
                    m = {k: sum(v) / len(v) for k, v in js.items()}
                    parts.append("%s·%s 핵 %d(|A| %d) J(A,C) %.3f→%.3f · J(B,D) %.3f→%.3f · J(A,B) %.3f→%.3f · R 맞춤 %d 중 핵 %.3f · P 바뀜 %d 중 핵 %.3f"
                                 % (cond, "첫" if tag == "pre" else "둘째", acc["core"], acc["nA"], m["J_AC"], m["Jx_AC"], m["J_BD"], m["Jx_BD"], m["J_AB"], m["Jx_AB"],
                                    acc["t"], acc["tc"] / acc["t"] if acc["t"] else float("nan"), acc["c"], acc["cc"] / acc["c"] if acc["c"] else float("nan")))
            print("b%d %s: %s" % (b, rep, " | ".join(parts)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
