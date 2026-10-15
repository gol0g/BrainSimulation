#!/usr/bin/env python3
"""(E195 사후 탐색) XOR 순도의 잡음 바닥과 편향 보정 — rcxor.purity_rows 의 |상호작용| 은 시행 잡음으로 위로 치우친다.
KC 마다 노출(train) 제시를 반복(rep) r 별로 칸 평균 → 상호작용 I_r = (AX+BY) − (AY+BX) (사례 효과는 같은 사례가 X·Y 에 모두 나와 상쇄).
  관측 순도  = Σ_resp |Ī| / Σ_resp tot                          (rcxor.purity_rows 와 같은 값 — 회귀 확인)
  잡음 바닥  = Σ_resp √(2/π)·σ̂_Ī / Σ_resp tot,  σ̂_Ī = s(I_r)/√R/c4(R)   (참 상호작용이 0 일 때 기대 순도)
  에너지 순도 ρ = √(Σ I² / Σ tot²) — 전체 KC(반응 선택 없음), tot² 가중 순도² 평균의 제곱근(원 지표와 가중·집합이 다른 같은 개념)
    날 ρ_raw 는 Ī·tot 그대로(잡음으로 위로 치우침), 보정(ρ² 에서 불편 — 개별 값은 0 근처에서 잡음이 커 음수 가능, 평균·차이는 ρ²_corr 로) ρ_corr 는 반복 쌍 교차곱 평균 P_I = Σ_{r<r'} I_r·I_r' / C(R,2)·
    P_T(같은 꼴, tot) 로 집단 합에서 편향 제거: ρ_corr = √(max(0, Σ P_I) / Σ P_T)  (E[P] = 참값², 반복 사이 잡음 독립 가정)
반응 KC(resp)·tot 는 rcxor 와 같게(전체 칸 평균 4칸 평균 ≥ 0.25). 두 쪽 평균.
실행: python3 scripts/purity_floor.py --selftest  /  python3 scripts/purity_floor.py <RCKX 로그> [...]"""
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rcxor  # noqa: E402

CELLS = ("AX", "AY", "BX", "BY")


def c4(n):
    return math.sqrt(2.0 / (n - 1)) * math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0)


def floor_rows(rows):
    """rows: rcxor.parse 의 한 쪽 행 [(set, cat, ctx, i, rep, vec, k)] → dict(obs, floor, rho2_raw, rho_raw, rho2_corr, rho_corr, n_resp, R, resp)."""
    tr = [r for r in rows if r[0] == "train"]
    reps = sorted({r[4] for r in tr})
    R = len(reps)
    if R < 2:
        raise ValueError("반복 %d < 2" % R)
    full = {}
    for row in tr:
        full.setdefault(row[1] + row[2], []).append(row[5])
    if set(full) != set(CELLS):
        raise ValueError("칸 부족 %s" % sorted(full))
    m = {}
    for rp in reps:
        for c in CELLS:
            v = [r[5] for r in tr if r[4] == rp and r[1] + r[2] == c]
            if not v:
                raise ValueError("칸 없음 rep %d %s" % (rp, c))
            m[(rp, c)] = np.mean(v, axis=0)
    cnt = {c: len(full[c]) for c in CELLS}
    if len(set(cnt.values())) != 1 or any(sum(1 for r in tr if r[4] == rp and r[1] + r[2] == c) * R != cnt[c]
                                          for rp in reps for c in CELLS):
        raise ValueError("반복·칸 균형 아님 %s" % cnt)
    fm = {c: np.mean(full[c], axis=0) for c in CELLS}                       # rcxor.purity_rows 와 같은 계산
    tot = fm["AX"] + fm["AY"] + fm["BX"] + fm["BY"]
    Ibar = (fm["AX"] + fm["BY"]) - (fm["AY"] + fm["BX"])
    resp = tot / 4.0 >= 0.25
    I = np.array([(m[(rp, "AX")] + m[(rp, "BY")]) - (m[(rp, "AY")] + m[(rp, "BX")]) for rp in reps])   # R × n_kc
    T = np.array([m[(rp, "AX")] + m[(rp, "AY")] + m[(rp, "BX")] + m[(rp, "BY")] for rp in reps])
    if np.max(np.abs(I.mean(axis=0) - Ibar)) > 1e-9:
        raise ValueError("반복 평균 상호작용 ≠ 전체 상호작용")
    sig = I.std(axis=0, ddof=1) / math.sqrt(R) / c4(R)
    PI = np.zeros_like(Ibar)
    PT = np.zeros_like(Ibar)
    npair = 0
    for a in range(R):
        for b in range(a + 1, R):
            PI += I[a] * I[b]
            PT += T[a] * T[b]
            npair += 1
    PI /= npair
    PT /= npair
    out = {"n_resp": int(resp.sum()), "R": R, "resp": resp}
    if not resp.any():
        out.update(obs=float("nan"), floor=float("nan"), rho2_raw=float("nan"), rho_raw=float("nan"), rho2_corr=float("nan"), rho_corr=float("nan"))
        return out
    den = np.sum(tot[resp])
    out["obs"] = float(np.sum(np.abs(Ibar[resp])) / den)
    out["floor"] = float(np.sum(math.sqrt(2.0 / math.pi) * sig[resp]) / den)
    # 에너지 순도는 반응 선택 없이 전체 KC(tot² 가중이라 조용한 KC 기여는 작다) — 같은 자료로 고른 반응 집합은 선택 잡음이
    # 상호작용 잡음과 상관해 교차곱을 치우치게 한다(합성 시험에서 확인)
    out["rho2_raw"] = float(np.sum(Ibar ** 2) / np.sum(tot ** 2))
    out["rho2_corr"] = float(np.sum(PI) / np.sum(PT))          # 자르지 않은 불편 추정(음수 가능 — 평균·차이는 이것으로)
    out["rho_raw"] = float(math.sqrt(out["rho2_raw"]))
    out["rho_corr"] = float(math.sqrt(max(0.0, out["rho2_corr"])))
    return out


def floor_log(path, n_kc=rcxor.N_KC):
    return {sd: floor_rows(rows) for sd, rows in sorted(rcxor.parse(path, n_kc).items())}


def _synth(seed, n_kc, frac_conj, scale, n_ex=6, reps=3):
    """합성 노출 행: 덧셈 KC(범주·맥락 주효과, 상호작용 0) + 결합 KC(한 칸 전용). λ 에 scale 을 곱해 희소도만 바꾼다.
    반환 rows, 기대 칸 평균 E(사례 효과 포함 — 추정 대상)."""
    rs = np.random.RandomState(seed)
    base = rs.uniform(0.0, 0.3, n_kc)
    lam = {c: base.copy() for c in CELLS}
    kind = rs.rand(n_kc)
    for k in range(n_kc):
        if kind[k] < frac_conj:
            lam[CELLS[rs.randint(4)]][k] += rs.uniform(1.0, 3.0)
        else:
            a, b = rs.uniform(0, 2.0, 2)
            ca, cx = rs.randint(2), rs.randint(2)
            for c in CELLS:
                lam[c][k] += a * (c[0] == "AB"[ca]) + b * (c[1] == "XY"[cx])
    lam = {c: v * scale for c, v in lam.items()}
    exk = {(c, i): rs.uniform(0.7, 1.3, n_kc) for c in "AB" for i in range(n_ex)}   # 사례 효과(곱) — 같은 사례가 X·Y 에 공통
    rows = []
    for rp in range(reps):
        for cat in "AB":
            for i in range(n_ex):
                for ctx in "XY":
                    rows.append(("train", cat, ctx, i, rp, rs.poisson(lam[cat + ctx] * exk[(cat, i)]).astype(float), -1))
    ex_mean = {c: np.mean([exk[(c, i)] for i in range(n_ex)], axis=0) for c in "AB"}
    E = {c: lam[c] * ex_mean[c[0]] for c in CELLS}
    return rows, E


def _truth(E, resp):
    """참값: (원 지표 참 순도 Σ_resp|I|/Σ_resp tot — 추정기와 같은 반응 집합, 에너지 순도 참 ρ — 전체 KC)."""
    tot = E["AX"] + E["AY"] + E["BX"] + E["BY"]
    inter = E["AX"] + E["BY"] - E["AY"] - E["BX"]
    return (float(np.sum(np.abs(inter[resp])) / np.sum(tot[resp])),
            float(math.sqrt(np.sum(inter ** 2) / np.sum(tot ** 2))))


def _selftest():
    ok = 0
    n = 0

    def chk(name, cond, msg=""):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print("  [%s] %s %s" % ("통과" if cond else "실패", name, msg))

    # 1. 회귀: 관측 순도 = rcxor.purity_rows (여러 시드·희소도)
    worst = 0.0
    for sd, sc in ((1, 1.0), (7, 0.3), (8, 0.15)):
        rows, _ = _synth(sd, 600, 0.1, sc)
        worst = max(worst, abs(floor_rows(rows)["obs"] - rcxor.purity_rows(rows)[0]))
    chk("관측 = rcxor.purity_rows(3 경우)", worst < 1e-12, "최대 차 %.2e" % worst)
    # 2·3. 불편성(시드 8개 평균, ρ² 영역): 덧셈만·결합 15% × scale 1·0.3·0.15 — ρ²_corr 평균 ≈ 참(3 SE + 0.0002), ρ²_raw 는 위로 치우침,
    #      원 지표 바닥 ≈ 관측(덧셈만, ±15%), 희소할수록 원 지표 치우침·바닥 증가
    summ = {}
    for frac in (0.0, 0.15):
        for sc in (1.0, 0.3, 0.15):
            acc = {"c": [], "r": [], "t": [], "obs": [], "tobs": [], "floor": []}
            for sd in range(8):
                rows, E = _synth(100 + sd, 3000, frac, sc)
                r = floor_rows(rows)
                t_obs, t_rho = _truth(E, r["resp"])
                acc["c"].append(r["rho2_corr"]); acc["r"].append(r["rho2_raw"]); acc["t"].append(t_rho ** 2)
                acc["obs"].append(r["obs"]); acc["tobs"].append(t_obs); acc["floor"].append(r["floor"])
            mc, mr, mt = np.mean(acc["c"]), np.mean(acc["r"]), np.mean(acc["t"])
            se = np.std(acc["c"], ddof=1) / math.sqrt(len(acc["c"]))
            summ[(frac, sc)] = {k: float(np.mean(v)) for k, v in acc.items()}
            chk("결합 %d%% scale %.2f ρ²_corr 평균 ≈ 참(3 SE)" % (frac * 100, sc), abs(mc - mt) < 3 * se + 0.0002,
                "ρ²_corr %.5f(±%.5f) 참 %.5f ρ²_raw %.5f" % (mc, se, mt, mr))
            chk("결합 %d%% scale %.2f ρ²_raw 위로 치우침" % (frac * 100, sc), mr > mt + 0.003)
            if frac == 0.0:
                fo = np.mean(acc["floor"]) / np.mean(acc["obs"])
                chk("덧셈만 scale %.2f 바닥 ≈ 관측" % sc, abs(fo - 1.0) < 0.15,
                    "바닥 %.4f 관측 %.4f 참 %.4f" % (np.mean(acc["floor"]), np.mean(acc["obs"]), np.mean(acc["tobs"])))
    bias = {sc: summ[(0.15, sc)]["obs"] - summ[(0.15, sc)]["tobs"] for sc in (1.0, 0.3, 0.15)}
    chk("희소할수록 원 지표 치우침 증가", bias[0.15] > bias[1.0] + 0.01, "치우침 %+.4f → %+.4f" % (bias[1.0], bias[0.15]))
    chk("희소할수록 바닥 상승", summ[(0.15, 0.15)]["floor"] > summ[(0.15, 1.0)]["floor"] + 0.01,
        "바닥 %.4f → %.4f" % (summ[(0.15, 1.0)]["floor"], summ[(0.15, 0.15)]["floor"]))
    # 4. 결합 비율 ↑ (같은 scale) → ρ_corr 가 참 ρ 의 상승을 따른다
    rows1, E1 = _synth(4, 3000, 0.05, 1.0)
    rows2, E2 = _synth(4, 3000, 0.30, 1.0)
    x, y = floor_rows(rows1), floor_rows(rows2)
    t1, t2 = _truth(E1, x["resp"])[1] ** 2, _truth(E2, y["resp"])[1] ** 2
    chk("결합 비율 ↑ → ρ²_corr ↑(참 상승의 ±25%)", abs((y["rho2_corr"] - x["rho2_corr"]) - (t2 - t1)) < 0.25 * (t2 - t1),
        "ρ²_corr %.5f → %.5f (참 %.5f → %.5f)" % (x["rho2_corr"], y["rho2_corr"], t1, t2))
    # 5. 결측·불균형: 반복 1 → 오류, 칸 하나 없음 → 오류, 반복 하나에서 칸 결측 → 오류
    rows, _ = _synth(5, 100, 0.1, 1.0, reps=1)
    try:
        floor_rows(rows)
        chk("반복 1 거부", False)
    except ValueError:
        chk("반복 1 거부", True)
    rows, _ = _synth(6, 100, 0.1, 1.0)
    try:
        floor_rows([r for r in rows if r[1] + r[2] != "BY"])
        chk("칸 결측 거부", False)
    except ValueError:
        chk("칸 결측 거부", True)
    try:
        floor_rows([r for r in rows if not (r[4] == 0 and r[1] + r[2] == "BY")])
        chk("반복 하나 칸 결측 거부", False)
    except ValueError:
        chk("반복 하나 칸 결측 거부", True)
    print("합성 %d/%d" % (ok, n))
    return ok == n


def main(argv):
    if argv[:1] == ["--selftest"]:
        sys.exit(0 if _selftest() else 1)
    for p in argv:
        res = floor_log(p)
        av = {k: float(np.mean([res[sd][k] for sd in res])) for k in ("obs", "floor", "rho2_raw", "rho2_corr")}
        print("%s | 관측 %.4f · 바닥 %.4f | ρ²_raw %.5f · ρ²_corr %.5f | %s" % (
            p.rsplit("/", 1)[-1], av["obs"], av["floor"], av["rho2_raw"], av["rho2_corr"],
            " ".join("%s:resp %d R %d" % (sd, res[sd]["n_resp"], res[sd]["R"]) for sd in sorted(res))))


if __name__ == "__main__":
    main(sys.argv[1:])
