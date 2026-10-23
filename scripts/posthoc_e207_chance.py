# E207 사후 기술 보강(탐색 — 판정 밖, 결과 본 뒤): 학습 KC 중 집합 안 몫의 우연 기준선 = |집합 ∩ 대상 KC| / |대상 KC|(대상 = 두 motor 모두 시냅스 있는 KC — learn_in_core 와 같은 분모 정의)
import os, sys
sys.path.insert(0, "scripts")
import numpy as np
import posthoc_e207_core as C
import posthoc_kc_rewrite as PK
PK.EXP = C.EXP

def chance(both, st):
    el = set(np.nonzero(both)[0].tolist())
    return len(el & st), len(el)

# 합성 정답
_b = np.zeros(10, bool); _b[[1, 2, 3, 4]] = True
assert chance(_b, {2, 3, 9}) == (2, 4)
_dR = np.array([0, 60, 0, -70, 0, 0, 0, 0, 0, 0.]); _dP = np.array([0, 60, 80, -70, 0, 0, 0, 0, 0, 0.])
assert C.learn_in_core(_dR, _dP, _b, {1, 2}, 50.0) == (2, 1, 1, 1)
print("합성 2/2")

L = os.path.join(C.EXP, "logs", "E207")
for b in C.BRAINS:
    S = np.load(os.path.join(C.EXP, "traces", "E176", "snap", "conn_b%d.npz" % b))
    for rep in ("U", "F"):
        R = np.load(os.path.join(C.EXP, "traces", C.WDIR[rep], "w_R_b%d.npz" % b)); P = np.load(os.path.join(C.EXP, "traces", C.WDIR[rep], "w_P_b%d.npz" % b))
        pref = {sd: (PK.kc_pref(R, S, sd), PK.kc_pref(P, S, sd)) for sd in "lr"}
        out = []
        for cond in ("none", "X"):
            for tag in ("pre", "post"):
                sets = PK.form_sets(os.path.join(L, "meas_%s_%s_b%d.log" % (rep, cond, b)), tag)
                a = dict(el=0, core=0, ab=0, cd=0, t=0, tc=0, c=0, cc=0, tab=0, ccd=0)
                for sd in "lr":
                    cs = sets.get(sd, {})
                    st = C.core_stats(cs)
                    (dR, bR), (dP, bP) = pref[sd]
                    both = bR & bP
                    ab = cs.get("A", set()) | cs.get("B", set()); cd = cs.get("C", set()) | cs.get("D", set())
                    k, n = chance(both, st["core"]); a["core"] += k; a["el"] += n
                    a["ab"] += chance(both, ab)[0]; a["cd"] += chance(both, cd)[0]
                    t, tc, c, cc = C.learn_in_core(dR, dP, both, st["core"], 50.0)
                    a["t"] += t; a["tc"] += tc; a["c"] += c; a["cc"] += cc
                    tuned = set(np.nonzero(both & (np.abs(dR) >= 50))[0].tolist()); changed = set(np.nonzero(both & (np.abs(dP - dR) >= 50))[0].tolist())
                    a["tab"] += len(tuned & ab); a["ccd"] += len(changed & cd)
                el = a["el"]
                out.append("%s·%s 대상 %d | 핵 우연 %.3f · R 핵 %.3f(×%.1f) · P 핵 %.3f(×%.1f) | A∪B 우연 %.3f · R A∪B %.3f(×%.1f) | C∪D 우연 %.3f · P C∪D %.3f(×%.1f)" % (
                    cond, "첫" if tag == "pre" else "둘째", el,
                    a["core"] / el, a["tc"] / a["t"], (a["tc"] / a["t"]) / (a["core"] / el) if a["core"] else float("nan"),
                    a["cc"] / a["c"], (a["cc"] / a["c"]) / (a["core"] / el) if a["core"] else float("nan"),
                    a["ab"] / el, a["tab"] / a["t"], (a["tab"] / a["t"]) / (a["ab"] / el),
                    a["cd"] / el, a["ccd"] / a["c"], (a["ccd"] / a["c"]) / (a["cd"] / el)))
        print("b%d %s: %s" % (b, rep, " || ".join(out)))
