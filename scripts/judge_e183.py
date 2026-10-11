#!/usr/bin/env python3
"""E183 판정 — 원시 특징 범주 형성(측정). 규칙은 research/experiments/logs/E183/criteria_fixed.txt.
입력: research/experiments/logs/E183/rc_b{B}_{O|Z}.log (O = Oja 노출 eta 0.02, Z = 같은 노출 eta 0 대조), 뇌 10~14.
지표(backend/genesis/rawcat.py 의 separation·selectivity):
  쪽 sd 마다 S_new = 새 사례 x 의 J(K(x), K_자기) − J(K(x), K_다른) 평균(K_범주 = 노출 사례 절반 이상에 반응한 KC), S_train = 노출 사례 자신.
  뇌 단위: G_new = 쪽 평균(S_new[O post] − S_new[Z post]), G_train 같은 식, J_AB = max_쪽 J(K_A, K_B)[O post].
경계 비교는 모두 1e-4 단위 정수 반올림 뒤(q(x) = round(x·10000) — 부동소수 경계 오차 방지).
판정(우선순위): 조작검증 실패 → 보류(조작검증 실패) / 붕괴(null-3) J_AB ≥ 0.8 ≥ 4/5 / 지지(H104) G_new ≥ 0.05 이면서 (G_train ≤ 0 또는 G_new ≥ 0.5·G_train) ≥ 4/5
  / 암기(null-2) G_train ≥ 0.05 이면서 G_new < 0.5·G_train ≥ 4/5 / 역효과 G_new ≤ −0.05 ≥ 4/5 / 이득 없음(null-1) G_new < 0.05 ≥ 4/5 / 그 밖 보류.
실행: python3 scripts/judge_e183.py (저장소 루트에서)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "genesis"))
import rawcat as RC  # noqa: E402

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
DELTA = 0.05


def q(x):
    return int(round(x * 10000))


QD = q(DELTA)   # 500
NTR = 6
NNEW = 6
REPS = 3


def parse(path):
    sets, chk, rcw, done = {}, {}, {}, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith("RCKC "):
                kv = dict(x.split("=", 1) for x in ln.split()[1:])
                ids = set(int(v) for v in kv["ids"].split(",") if v != "")
                if len(ids) != int(kv["n"]):
                    raise ValueError("RCKC n 불일치")
                sets.setdefault(kv["tag"], {}).setdefault(kv["side"], {}).setdefault(kv["set"], {}).setdefault(kv["cat"], {})[int(kv["i"])] = ids
            elif ln.startswith("RCCHK "):
                kv = dict(x.split("=", 1) for x in ln.split()[1:])
                chk[kv["tag"]] = kv
            elif ln.startswith("RCW "):
                kv = dict(x.split("=", 1) for x in ln.split()[1:])
                rcw[kv["side"]] = kv
            elif ln.startswith("=> RAWCAT "):
                done = True
    return sets, chk, rcw, done


def as_lists(s, side):
    """sets[tag][side] → {'train': {'A': [..6], 'B': [..]}, 'new': {...}} (번호 순)."""
    out = {}
    for st, m in (("train", NTR), ("new", NNEW)):
        out[st] = {}
        for c in "AB":
            d = s[side][st][c]
            if sorted(d) != list(range(m)):
                raise ValueError("사례 번호 결측 %s %s %s" % (side, st, c))
            out[st][c] = [d[i] for i in range(m)]
    return out


def chk_ok(chk, tags):
    for t in tags:
        c = chk.get(t)
        if c is None:
            return False, "RCCHK %s 없음" % t
        fa, fi = int(c["feat_active_spikes"]), int(c["feat_inactive_spikes"])
        if not (float(c["feat_active_frac"]) >= 0.90 and fi <= 0.01 * fa and int(c["feat_other_side"]) == 0
                and int(c["goodbad"]) == 0 and int(c["g_same"]) == 1 and int(c["npres"]) == 2 * 2 * (NTR + NNEW) * REPS):
            return False, "RCCHK %s 기준 밖(%s)" % (t, " ".join("%s=%s" % kv for kv in sorted(c.items())))
    return True, ""


def sval(lst, side_sets):
    return RC.separation(side_sets["train"], side_sets[lst])


def mean_k(side_sets):
    v = [len(x) for st in ("train", "new") for c in "AB" for x in side_sets[st][c]]
    return sum(v) / float(len(v))


def main():
    rows, okall, why = [], True, []
    for b in BRAINS:
        L = {}
        for arm in ("O", "Z"):
            p = os.path.join(EXP, "logs", "E183", "rc_b%d_%s.log" % (b, arm))
            try:
                L[arm] = parse(p)
            except (OSError, ValueError, KeyError) as ex:
                print("판정: 보류(결측 — b%d %s %s)" % (b, arm, type(ex).__name__))
                return 0
            if not L[arm][3]:
                print("판정: 보류(결측 — b%d %s 완료 줄 없음)" % (b, arm))
                return 0
        r = {"b": b}
        try:
            for arm in ("O", "Z"):
                ok, w = chk_ok(L[arm][1], ("pre", "pre2", "post"))
                if not ok:
                    okall = False; why.append("b%d %s %s" % (b, arm, w))
            # 노출 조작: O 는 바뀌고 Z 는 그대로
            for sd in "lr":
                o_, z_ = L["O"][2].get(sd), L["Z"][2].get(sd)
                if o_ is None or z_ is None or not (float(o_["dg_abs"]) > 0 and float(z_["dg_abs"]) == 0 and float(z_["frac_changed"]) == 0):
                    okall = False; why.append("b%d 노출 조작(쪽 %s)" % (b, sd))
            sides = {arm: {tag: {sd: as_lists(L[arm][0][tag], sd) for sd in "lr"} for tag in ("pre", "pre2", "post")} for arm in ("O", "Z")}
        except (KeyError, ValueError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        # 짝 검사: O·Z 형성 전 집합이 거의 같아야(평균 자카드 ≥ 0.95)
        js = [RC.jac(sides["O"]["pre"][sd][st][c][i], sides["Z"]["pre"][sd][st][c][i])
              for sd in "lr" for st in ("train", "new") for c in "AB" for i in range(NTR if st == "train" else NNEW)]
        r["pair_j"] = sum(js) / len(js)
        r["pair_exact"] = sum(1 for x in js if x == 1.0)
        if q(r["pair_j"]) < 9500:
            okall = False; why.append("b%d 짝(형성 전 O·Z) 평균 자카드 %.4f < 0.95" % (b, r["pair_j"]))
        # 측정 반복성: |S_new(pre2) − S_new(pre)| 쪽 평균 < 0.025 (두 팔 모두)
        for arm in ("O", "Z"):
            rep = sum(abs(sval("new", sides[arm]["pre2"][sd])[1] - sval("new", sides[arm]["pre"][sd])[1]) for sd in "lr") / 2.0
            r["rep_" + arm] = rep
            if q(rep) >= QD // 2:
                okall = False; why.append("b%d %s 측정 반복성 %.4f ≥ 0.025" % (b, arm, rep))
        # 반응 크기(형성 전) 하한
        kpre = min(mean_k(sides[arm]["pre"][sd]) for arm in ("O", "Z") for sd in "lr")
        r["kpre"] = kpre
        if kpre < 10:
            okall = False; why.append("b%d 형성 전 KC 반응 평균 %.1f < 10" % (b, kpre))
        sn = {arm: [sval("new", sides[arm]["post"][sd])[1] for sd in "lr"] for arm in ("O", "Z")}
        stn = {arm: [sval("train", sides[arm]["post"][sd])[1] for sd in "lr"] for arm in ("O", "Z")}
        s0 = [sval("new", sides["O"]["pre"][sd])[1] for sd in "lr"]
        r["G_new"] = sum(sn["O"][i] - sn["Z"][i] for i in range(2)) / 2.0
        r["G_train"] = sum(stn["O"][i] - stn["Z"][i] for i in range(2)) / 2.0
        r["J_AB"] = max(sval("new", sides["O"]["post"][sd])[4] for sd in "lr")
        r["S_new_pre"] = sum(s0) / 2.0
        r["S_new_O"] = sum(sn["O"]) / 2.0
        r["S_new_Z"] = sum(sn["Z"]) / 2.0
        r["sel"] = (sum(RC.selectivity(sides["O"]["pre"][sd]["train"])[0] for sd in "lr") / 2.0,
                    sum(RC.selectivity(sides["O"]["post"][sd]["train"])[0] for sd in "lr") / 2.0)
        r["K"] = (sum(mean_k(sides["O"]["pre"][sd]) for sd in "lr") / 2.0, sum(mean_k(sides["O"]["post"][sd]) for sd in "lr") / 2.0)
        r["wsel"] = tuple(float(L["O"][2][sd][k]) for sd in "lr" for k in ("wsel0", "wsel1"))
        rows.append(r)
    for r in rows:
        print("b%d G_new %+.4f G_train %+.4f J_AB %.4f | S_new 전 %.4f · O 후 %.4f · Z 후 %.4f | 선택성 %.4f→%.4f | K %.1f→%.1f | wsel 좌 %.4f→%.4f 우 %.4f→%.4f | 짝 %.4f(정확 %d/48) 반복성 O %.4f Z %.4f"
              % (r["b"], r["G_new"], r["G_train"], r["J_AB"], r["S_new_pre"], r["S_new_O"], r["S_new_Z"], r["sel"][0], r["sel"][1],
                 r["K"][0], r["K"][1], r["wsel"][0], r["wsel"][1], r["wsel"][2], r["wsel"][3], r["pair_j"], r["pair_exact"], r["rep_O"], r["rep_Z"]))
    if not okall:
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    for r in rows:
        r["qn"], r["qt"], r["qj"] = q(r["G_new"]), q(r["G_train"]), q(r["J_AB"])
    n3 = sum(r["qj"] >= 8000 for r in rows)
    sup = sum(r["qn"] >= QD and (r["qt"] <= 0 or 2 * r["qn"] >= r["qt"]) for r in rows)
    mem = sum(r["qt"] >= QD and 2 * r["qn"] < r["qt"] for r in rows)
    harm = sum(r["qn"] <= -QD for r in rows)
    n1 = sum(r["qn"] < QD for r in rows)
    print("조작검증 통과 | 붕괴 %d/5 · 지지 %d/5 · 암기 %d/5 · 역효과 %d/5 · 이득 없음 %d/5" % (n3, sup, mem, harm, n1))
    v = ("붕괴(H104-null-3)" if n3 >= 4 else "지지(H104)" if sup >= 4 else "암기(H104-null-2)" if mem >= 4
         else "역효과" if harm >= 4 else "이득 없음(H104-null-1)" if n1 >= 4 else "보류")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
