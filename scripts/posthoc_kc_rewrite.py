#!/usr/bin/env python3
"""E205·E206 사후 기술(탐색 — 판정 밖): 상충 둘째 쌍 학습(P 의 뒤 1,500시행)이 첫 쌍 학습(R)이 맞춘 KC 를 다시 쓰는가 — 형성 전 U(E205) 대 형성 F(E206), 같은 뇌.
KC 별 선호 d = (KC → motor_l 평균 가중치) − (KC → motor_r 평균 가중치)(경쟁형 규칙 — KC 별 두 합 300 보존, 처음 0). R 과 P 는 앞 1,500시행이 같은 시드라 P 의 둘째 쌍 효과 ≈ d_P − d_R.
지표(쪽마다 — 두 motor 모두에 시냅스가 있는 KC 만):
  R 맞춤 KC = |d_R| ≥ θ, P 바뀐 KC = |d_P − d_R| ≥ θ (θ 기본 50 — 범위 ±300 의 1/6).
  다시 씀 몫 = |R 맞춤 ∩ P 바뀜| / |R 맞춤|, 뒤집힘 몫 = R 맞춤 중 d_P·d_R < 0 이고 |d_P| ≥ θ, 몫 중앙 = R 맞춤의 d_P/d_R 중앙값(1 유지 · 0 지움 · < 0 뒤집힘),
  P 바뀐 KC 중 R 맞춤 몫 = |R 맞춤 ∩ P 바뀜| / |P 바뀜| (둘째 쌍이 첫 쌍 KC 를 쓰는 정도).
읽기: 형성으로 KC 집합이 갈렸다면 F 에서 '다시 씀 몫' 이 U 보다 크게 작아야 한다. 둘째 쌍 제시 동안(맥락 X 켬) 활동한 KC 만 바뀌므로, F 에서도 R 맞춤 KC 가 바뀌면 그 KC 가 학습 조건에서 C·D 에 반응한다는 뜻(기능적 겹침).
실행: python3 scripts/posthoc_kc_rewrite.py (저장소 루트에서) [θ]"""
import os
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
N_KC = 3000


def kc_pref(W, S, side, n_kc=N_KC):
    """KC 별 d = mean(w→motor_l) − mean(w→motor_r), 두 motor 모두 시냅스가 있는 KC 표지."""
    a, b = "kc_%s_to_motor_l" % side, "kc_%s_to_motor_r" % side
    pa, pb = np.asarray(S[a + "__pre"], dtype=np.int64), np.asarray(S[b + "__pre"], dtype=np.int64)
    ca, cb = np.bincount(pa, minlength=n_kc)[:n_kc], np.bincount(pb, minlength=n_kc)[:n_kc]
    ma = np.bincount(pa, weights=np.asarray(W[a], dtype=np.float64), minlength=n_kc)[:n_kc] / np.maximum(ca, 1)
    mb = np.bincount(pb, weights=np.asarray(W[b], dtype=np.float64), minlength=n_kc)[:n_kc] / np.maximum(cb, 1)
    return ma - mb, (ca > 0) & (cb > 0)


def rewrite_stats(dR, dP, both, theta):
    tuned = both & (np.abs(dR) >= theta)
    changed = both & (np.abs(dP - dR) >= theta)
    nt, nc, ni = int(tuned.sum()), int(changed.sum()), int((tuned & changed).sum())
    flip = int((tuned & (dP * dR < 0) & (np.abs(dP) >= theta)).sum())
    ratio = float(np.median(dP[tuned] / dR[tuned])) if nt else float("nan")
    return {"tuned": nt, "changed": nc, "inter": ni,
            "rewrite": ni / nt if nt else float("nan"), "flip": flip / nt if nt else float("nan"),
            "ratio_med": ratio, "changed_from_tuned": ni / nc if nc else float("nan")}


def brain_stats(exp_dir, b, theta):
    S = np.load(os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b))
    R = np.load(os.path.join(EXP, "traces", exp_dir, "w_R_b%d.npz" % b))
    P = np.load(os.path.join(EXP, "traces", exp_dir, "w_P_b%d.npz" % b))
    out = {}
    for sd in "lr":
        dR, bothR = kc_pref(R, S, sd)
        dP, bothP = kc_pref(P, S, sd)
        out[sd] = rewrite_stats(dR, dP, bothR & bothP, theta)
    return out


def merge(o):
    """두 쪽 합산(개수 더함, 몫 다시 계산 — 중앙값은 쪽 평균)."""
    t = sum(o[s]["tuned"] for s in "lr"); c = sum(o[s]["changed"] for s in "lr"); i = sum(o[s]["inter"] for s in "lr")
    f = sum(o[s]["flip"] * o[s]["tuned"] for s in "lr")
    return {"tuned": t, "changed": c, "inter": i, "rewrite": i / t if t else float("nan"), "flip": f / t if t else float("nan"),
            "ratio_med": float(np.nanmean([o[s]["ratio_med"] for s in "lr"])), "changed_from_tuned": i / c if c else float("nan")}


def form_sets(path, tag):
    """형성 로그 RCKC 줄(맥락 없이 잰 반응 KC) → {쪽: {범주: 범주 KC 집합(노출 사례 절반 이상 — rawcat.cat_set 과 같은 규칙)}}."""
    ex = {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if not ln.startswith("RCKC tag=%s " % tag):
            continue
        kv = dict(x.split("=", 1) for x in ln.split()[1:])
        if kv["set"] != "train":
            continue
        ids = set(int(x) for x in kv["ids"].split(",") if x) if kv["ids"] else set()
        ex.setdefault(kv["side"], {}).setdefault(kv["cat"], []).append(ids)
    out = {}
    for sd, cats in ex.items():
        out[sd] = {}
        for c, lst in cats.items():
            need = max(1, -(-len(lst) // 2))
            cnt = {}
            for st in lst:
                for x in st:
                    cnt[x] = cnt.get(x, 0) + 1
            out[sd][c] = {x for x, n in cnt.items() if n >= need}
    return out


def set_stats(dR, dP, both, sets, theta):
    """맥락 없이 잰 범주 집합과 학습이 바꾼 KC 의 관계 — R 맞춤 중 A∪B 몫, 겹침 KC(A∪B)∩(C∪D) 수, P·R 함께 바뀐 KC 중 겹침 KC 몫·C∪D 몫."""
    tuned = set(np.nonzero(both & (np.abs(dR) >= theta))[0].tolist())
    changed = set(np.nonzero(both & (np.abs(dP - dR) >= theta))[0].tolist())
    ab = sets.get("A", set()) | sets.get("B", set()); cd = sets.get("C", set()) | sets.get("D", set())
    shared = ab & cd
    inter = tuned & changed
    return {"tuned": len(tuned), "tuned_in_ab": len(tuned & ab), "shared": len(shared), "ab": len(ab), "cd": len(cd),
            "inter": len(inter), "inter_in_shared": len(inter & shared), "inter_in_cd": len(inter & cd), "changed": len(changed), "changed_in_cd": len(changed & cd)}


def set_part(theta):
    print("\n(둘째) 맥락 없이 잰 범주 KC 집합(E206 형성 로그 — pre = U, post = F)과 학습이 바꾼 KC — 두 쪽 합")
    for b in BRAINS:
        S = np.load(os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b))
        form = os.path.join(EXP, "logs", "E206", "form_b%d.log" % b)
        for rep, ed, tag in (("U", "E205", "pre"), ("F", "E206", "post")):
            R = np.load(os.path.join(EXP, "traces", ed, "w_R_b%d.npz" % b)); Pw = np.load(os.path.join(EXP, "traces", ed, "w_P_b%d.npz" % b))
            sets = form_sets(form, tag)
            acc = {}
            for sd in "lr":
                dR, bR = kc_pref(R, S, sd); dP, bP = kc_pref(Pw, S, sd)
                st = set_stats(dR, dP, bR & bP, sets.get(sd, {}), theta)
                for k, v in st.items():
                    acc[k] = acc.get(k, 0) + v
            f = lambda a, b_: (a / b_) if b_ else float("nan")  # noqa: E731
            print("b%d %s: 범주 집합 A∪B %d · C∪D %d · 겹침(A∪B)∩(C∪D) %d | R 맞춤 %d 중 A∪B 안 %.3f | R·P 함께 바뀐 %d 중 겹침 KC %.3f · C∪D 안 %.3f | P 바뀜 %d 중 C∪D 안 %.3f"
                  % (b, rep, acc["ab"], acc["cd"], acc["shared"], acc["tuned"], f(acc["tuned_in_ab"], acc["tuned"]), acc["inter"], f(acc["inter_in_shared"], acc["inter"]),
                     f(acc["inter_in_cd"], acc["inter"]), acc["changed"], f(acc["changed_in_cd"], acc["changed"])))


def main(argv):
    theta = float(argv[1]) if len(argv) > 1 else 50.0
    print("θ %.0f — 다시 씀 몫 = R 맞춤 KC 중 P 에서 바뀐 몫 · 뒤집힘 몫 · d_P/d_R 중앙 · P 바뀐 KC 중 R 맞춤 몫 (두 쪽 합)" % theta)
    rows = {}
    for b in BRAINS:
        for rep, ed in (("U", "E205"), ("F", "E206")):
            m = merge(brain_stats(ed, b, theta))
            rows[(b, rep)] = m
            print("b%d %s: R 맞춤 %d · P 바뀜 %d · 겹침 %d | 다시 씀 %.3f · 뒤집힘 %.3f · d_P/d_R 중앙 %+.3f · P 바뀐 KC 중 R 맞춤 %.3f"
                  % (b, rep, m["tuned"], m["changed"], m["inter"], m["rewrite"], m["flip"], m["ratio_med"], m["changed_from_tuned"]))
    for k in ("rewrite", "flip", "ratio_med", "changed_from_tuned"):
        d = [rows[(b, "F")][k] - rows[(b, "U")][k] for b in BRAINS]
        print("F − U %s: %s" % (k, " ".join("%+.3f" % x for x in d)))
    set_part(theta)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
