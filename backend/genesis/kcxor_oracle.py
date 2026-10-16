"""E196: L4 이상 가중치(오라클) — KC 별 선택 지수와 KC→motor 이상 가중치(순수 함수, GeNN 불사용 — 합성 정답 검사용).
지수(노출(train) 제시 칸 평균 m_AX·m_AY·m_BX·m_BY, 합 tot):
  SXI = ((AX + BY) − (AY + BX)) / tot  — + = 교차 칸(X·A, Y·B: 범주 × 맥락 XOR 의 정답 교차) 선택, − = 같은 쪽 칸(X·B, Y·A) 선택
       (rcxor.purity_rows 의 순도 = |SXI|)
  CSI = ((AX + AY) − (BX + BY)) / tot  — + = 범주 A 선택(범주 규칙 'A 교차·B 같은 쪽' 의 교차), − = B 선택
  반응 = tot/4 ≥ 0.25(rcxor 와 같음). 경계 비교 전 소수 9자리 반올림(E138 교훈).
가중치(KC 쪽 pop, motor 쪽 motor; 교차 = pop ≠ motor), 지수 v(cat 은 CSI, sel·eq 는 SXI):
  cat·sel: 반응·v ≥ θ → 교차 wmax·같은 쪽 0 / 반응·v ≤ −θ → 같은 쪽 wmax·교차 0 / 그 밖 init        (E138 kcsel 의 범주·L4 판)
  eq     : 반응 → 교차 wmax·(1 + v)/2 · 같은 쪽 wmax·(1 − v)/2 / 비반응 init
           (SXI 비례 상관 판독 — 대칭 도파민·무작위 선택 가정에서 rstdp_model 연성 상한식의 평형 g* = wmax·P/(P + D) 근사)
  sgnR   : (E197) 반응 KC 전부 — SXI > 0 → 교차 wmax·같은 쪽 0 / SXI < 0 → 반대 / SXI = 0·비반응 init
           (부호 판독 — 경성 상한(가산형) R-STDP 의 고정점: 상관 부호 쪽 경계로 포화)
  sgnA   : (E197) sgnR 과 같은 규칙을 발화 있는 KC 전부(tot > 0)에 — 경성 상한 규칙의 긴 시간 극한
  lsgnR  : (E198) 지수 대신 학습 차 Δ_k = mean(w_교차[k]) − mean(w_같은쪽[k])(학습 가중치, KC 별 묶음 — learned_diff)의 부호로 sgnR 규칙, 대상 = 반응 KC
  lsgnA  : (E198) 같은 규칙, 대상 = Δ ≠ 0 인 KC 전부(학습 중 흔적이 한 번이라도 굳은 KC)
KC 마다 두 motor 로 가는 값의 합은 모든 팔에서 wmax(= init 150 × 2 일 때 이전과 같은 공통 구동).
실행: python3 kcxor_oracle.py  (합성 자체 시험)"""
import os
import sys

import numpy as np

CELLS = ("AX", "AY", "BX", "BY")
ARMS = ("cat", "sel", "eq", "sgnR", "sgnA", "lsgnR", "lsgnA")
SIGN_ARMS = ("sgnR", "sgnA", "lsgnR", "lsgnA")
LEARN_ARMS = ("lsgnR", "lsgnA")


def sxi_rows(rows, n_kc):
    """rows: [(set, cat, ctx, i, rep, vec, k)] 한 쪽 → dict(sxi, csi, resp, tot, npc). 노출(train) 제시만."""
    tr = [r for r in rows if r[0] == "train"]
    cell = {}
    for r in tr:
        cell.setdefault(r[1] + r[2], []).append(r[5])
    if set(cell) != set(CELLS):
        raise ValueError("칸 부족 %s" % sorted(cell))
    n = {c: len(cell[c]) for c in CELLS}
    if len(set(n.values())) != 1:
        raise ValueError("칸 제시 수 불균형 %s" % n)
    m = {c: np.mean(cell[c], axis=0) for c in CELLS}
    if m["AX"].size != n_kc:
        raise ValueError("KC 수 %d ≠ %d" % (m["AX"].size, n_kc))
    tot = m["AX"] + m["AY"] + m["BX"] + m["BY"]
    den = np.maximum(tot, 1e-12)
    sxi = np.round(np.where(tot > 0, ((m["AX"] + m["BY"]) - (m["AY"] + m["BX"])) / den, 0.0), 9)
    csi = np.round(np.where(tot > 0, ((m["AX"] + m["AY"]) - (m["BX"] + m["BY"])) / den, 0.0), 9)
    resp = np.round(tot / 4.0, 9) >= 0.25
    return {"sxi": sxi, "csi": csi, "resp": resp, "tot": tot, "npc": n["AX"]}


def oracle_weights(pre, index, resp, pop, motor, wmax, init, arm, theta=0.5):
    """KC→motor 한 집단의 시냅스 값(float32). pre = 시냅스별 전시냅스 KC 인덱스, index = KC 별 지수(cat 은 CSI, sel·eq 는 SXI)."""
    if arm not in ARMS:
        raise ValueError(arm)
    pre = np.asarray(pre, dtype=np.int64)
    v = np.round(np.asarray(index, dtype=np.float64), 9)[pre]
    rp = np.asarray(resp, dtype=bool)[pre]
    crossed = pop != motor
    out = np.full(pre.size, init, dtype=np.float64)
    if arm in ("cat", "sel") or arm in SIGN_ARMS:
        if arm in SIGN_ARMS:     # resp = elig(arm, …) — 대상 KC
            hi = rp & (v > 0)
            lo = rp & (v < 0)
        else:
            hi = rp & (v >= theta)
            lo = rp & (v <= -theta)
        out[hi] = wmax if crossed else 0.0
        out[lo] = 0.0 if crossed else wmax
    else:
        sign = 1.0 if crossed else -1.0
        out[rp] = wmax * (1.0 + sign * v[rp]) / 2.0
    return np.clip(out, 0.0, wmax).astype(np.float32)


def elig(arm, resp, tot, delta=None):
    """(E197) 팔별 대상 KC 마스크: sgnA = 발화 있는 KC(tot 9자리 반올림 > 0), (E198) lsgnA = Δ(9자리 반올림) ≠ 0, 그 밖 = 반응 KC."""
    if arm == "sgnA":
        return np.round(np.asarray(tot, dtype=np.float64), 9) > 0
    if arm == "lsgnA":
        if delta is None:
            raise ValueError("lsgnA 는 delta 가 필요하다")
        return np.round(np.asarray(delta, dtype=np.float64), 9) != 0
    return np.asarray(resp, dtype=bool)


def learned_diff(pre_c, w_c, pre_s, w_s, n_kc):
    """(E198) KC 별 학습 차 Δ_k = mean(w_교차 over k 의 시냅스) − mean(w_같은쪽 over k 의 시냅스), 9자리 반올림. 한쪽이라도 시냅스가 없는 KC 는 0.
    pre_* = 시냅스별 전시냅스 KC 인덱스(장치 연결 또는 스냅숏 __pre — 행 묶음만 쓰므로 행 안 순서와 무관), w_* = 같은 순서의 학습 가중치."""
    pre_c = np.asarray(pre_c, dtype=np.int64); pre_s = np.asarray(pre_s, dtype=np.int64)
    w_c = np.asarray(w_c, dtype=np.float64); w_s = np.asarray(w_s, dtype=np.float64)
    if pre_c.size != w_c.size or pre_s.size != w_s.size:
        raise ValueError("pre·w 크기 불일치")
    nc = np.bincount(pre_c, minlength=n_kc); ns = np.bincount(pre_s, minlength=n_kc)
    if nc.size != n_kc or ns.size != n_kc:
        raise ValueError("KC 인덱스 범위 밖")
    mc = np.bincount(pre_c, weights=w_c, minlength=n_kc) / np.maximum(nc, 1)
    ms = np.bincount(pre_s, weights=w_s, minlength=n_kc) / np.maximum(ns, 1)
    return np.round(np.where((nc > 0) & (ns > 0), mc - ms, 0.0), 9)


def agree(delta, sxi, mask):
    """(E198 보고) 대상 KC 중 sign(Δ) = sign(SXI) 인 수(둘 다 0 아님)."""
    d = np.round(np.asarray(delta, dtype=np.float64), 9); s = np.round(np.asarray(sxi, dtype=np.float64), 9); m = np.asarray(mask, dtype=bool)
    return int(np.sum(m & (d != 0) & (s != 0) & (np.sign(d) == np.sign(s))))


def counts(index, resp, theta=0.5, sign=False):
    """오라클 점검 줄 수: (교차 선택, 같은 쪽 선택, 대상). sign=True(부호 판독) 이면 지수 > 0·< 0, resp 는 elig() 마스크."""
    v = np.round(np.asarray(index, dtype=np.float64), 9)
    r = np.asarray(resp, dtype=bool)
    if sign:
        return int(np.sum(r & (v > 0))), int(np.sum(r & (v < 0))), int(r.sum())
    return int(np.sum(r & (v >= theta))), int(np.sum(r & (v <= -theta))), int(r.sum())


def _selftest():
    ok = 0
    n = 0

    def chk(name, cond, msg=""):
        nonlocal ok, n
        n += 1
        ok += bool(cond)
        print("  [%s] %s %s" % ("통과" if cond else "실패", name, msg))

    # KC0 교차 전용(AX·BY), KC1 같은 쪽 전용(AY·BX), KC2 범주 A(AX·AY), KC3 맥락 X(AX·BX), KC4 AX 만, KC5 무반응,
    # KC6 교차 3·같은 쪽 1(SXI 정확히 0.5), KC7 4칸 평균 정확히 0.25(반응 경계), KC8 범주 B(BX·BY)
    nk = 9
    cv = {"AX": [2, 0, 1, 1, 2, 0, 2, 1, 0], "BY": [2, 0, 0, 0, 0, 0, 1, 0, 1],
          "AY": [0, 2, 1, 0, 0, 0, 1, 0, 0], "BX": [0, 2, 0, 1, 0, 0, 0, 0, 1]}
    rows = []
    for c in CELLS:
        for rep in range(3):
            rows.append(("train", c[0], c[1], 0, rep, np.array(cv[c], dtype=float), -1))
            rows.append(("new", c[0], c[1], 0, rep, np.full(nk, 9.0), -1))
    d = sxi_rows(rows, nk)
    chk("SXI 정답표", np.allclose(d["sxi"], [1, -1, 0, 0, 1, 0, 0.5, 1, 0]), str(np.round(d["sxi"], 3)))
    chk("CSI 정답표", np.allclose(d["csi"], [0, 0, 1, 0, 1, 0, 0.5, 1, -1]), str(np.round(d["csi"], 3)))
    chk("반응 정답표", list(d["resp"]) == [True, True, True, True, True, False, True, True, True], str(d["resp"].astype(int)))
    chk("새 사례 무시·칸당 제시 3", d["npc"] == 3 and d["tot"][0] == 4.0)
    pre = np.arange(nk)
    sr = oracle_weights(pre, d["sxi"], d["resp"], "l", "r", 300.0, 150.0, "sel")
    sl = oracle_weights(pre, d["sxi"], d["resp"], "l", "l", 300.0, 150.0, "sel")
    chk("sel 교차", list(sr) == [300, 0, 150, 150, 300, 150, 300, 300, 150], str(sr))
    chk("sel 같은 쪽", list(sl) == [0, 300, 150, 150, 0, 150, 0, 0, 150], str(sl))
    cr = oracle_weights(pre, d["csi"], d["resp"], "l", "r", 300.0, 150.0, "cat")
    chk("cat 교차(A 선택 → 교차)", list(cr) == [150, 150, 300, 150, 300, 150, 300, 300, 0], str(cr))
    er = oracle_weights(pre, d["sxi"], d["resp"], "l", "r", 300.0, 150.0, "eq")
    el = oracle_weights(pre, d["sxi"], d["resp"], "l", "l", 300.0, 150.0, "eq")
    chk("eq 교차", np.allclose(er, [300, 0, 150, 150, 300, 150, 225, 300, 150]), str(er))
    chk("eq 같은 쪽", np.allclose(el, [0, 300, 150, 150, 0, 150, 75, 0, 150]), str(el))
    tot_ok = True
    for arm, idx in (("cat", d["csi"]), ("sel", d["sxi"]), ("eq", d["sxi"])):
        a = oracle_weights(pre, idx, d["resp"], "l", "r", 300.0, 150.0, arm)
        b = oracle_weights(pre, idx, d["resp"], "l", "l", 300.0, 150.0, arm)
        tot_ok &= bool(np.allclose(a + b, 300.0))
    chk("세 팔 모두 KC 마다 두 motor 합 300", tot_ok)
    chk("kc_r → motor_l 이 교차", list(oracle_weights(pre, d["sxi"], d["resp"], "r", "l", 300.0, 150.0, "sel")) == list(sr))
    pre2 = np.array([6, 0, 6, 1, 5, 0])
    chk("시냅스 단위 매핑", list(oracle_weights(pre2, d["sxi"], d["resp"], "l", "r", 300.0, 150.0, "sel")) == [300, 300, 300, 0, 150, 300])
    chk("θ 경계(0.5 선택·0.499 아님)", list(oracle_weights(np.arange(4), np.array([0.5, 0.499, -0.5, -0.499]), np.ones(4, bool), "l", "r", 300.0, 150.0, "sel")) == [300, 150, 0, 150])
    chk("점검 수(sel)", counts(d["sxi"], d["resp"]) == (4, 1, 8), str(counts(d["sxi"], d["resp"])))
    chk("점검 수(cat)", counts(d["csi"], d["resp"]) == (4, 1, 8), str(counts(d["csi"], d["resp"])))
    # (E197) 부호 판독: 약한 선택(|SXI| < θ)도 부호대로 300/0, 0 은 init, sgnA 는 비반응이라도 발화 있으면 포함
    iv = np.array([0.2, -0.1, 0.0, 0.7, -0.6, 0.3])
    rv = np.array([True, True, True, True, True, False])
    tv = np.array([4.0, 2.0, 3.0, 4.0, 4.0, 0.5])
    p6 = np.arange(6)
    gR = oracle_weights(p6, iv, elig("sgnR", rv, tv), "l", "r", 300.0, 150.0, "sgnR")
    gRs = oracle_weights(p6, iv, elig("sgnR", rv, tv), "l", "l", 300.0, 150.0, "sgnR")
    gA = oracle_weights(p6, iv, elig("sgnA", rv, tv), "l", "r", 300.0, 150.0, "sgnA")
    chk("sgnR 교차(약한 선택 포함·0 은 init·비반응 init)", list(gR) == [300, 0, 150, 300, 0, 150], str(gR))
    chk("sgnR 같은 쪽", list(gRs) == [0, 300, 150, 0, 300, 150], str(gRs))
    chk("sgnA 교차(발화 있는 비반응 포함)", list(gA) == [300, 0, 150, 300, 0, 300], str(gA))
    chk("sel 은 약한 선택 제외(대조)", list(oracle_weights(p6, iv, rv, "l", "r", 300.0, 150.0, "sel")) == [150, 150, 150, 300, 0, 150])
    chk("sgnA 대상: tot 0·1e-12 제외", list(elig("sgnA", rv, np.array([0.0, 1e-12, 0.25, 1, 1, 1]))) == [False, False, True, True, True, True])
    chk("점검 수 sgnR", counts(iv, elig("sgnR", rv, tv), sign=True) == (2, 2, 5), str(counts(iv, elig("sgnR", rv, tv), sign=True)))
    chk("점검 수 sgnA", counts(iv, elig("sgnA", rv, tv), sign=True) == (3, 2, 6), str(counts(iv, elig("sgnA", rv, tv), sign=True)))
    s_ok = True
    for arm in ("sgnR", "sgnA"):
        e = elig(arm, rv, tv)
        s_ok &= bool(np.allclose(oracle_weights(p6, iv, e, "r", "l", 300.0, 150.0, arm) + oracle_weights(p6, iv, e, "r", "r", 300.0, 150.0, arm), 300.0))
    chk("부호 팔 KC 마다 두 motor 합 300·kc_r 교차 = motor_l", s_ok and list(oracle_weights(p6, iv, elig("sgnR", rv, tv), "r", "l", 300.0, 150.0, "sgnR")) == list(gR))
    # (E198) 학습 차 Δ: KC0 교차 쪽 학습 큼(+), KC1 같은 쪽 큼(−), KC2 같음(0), KC3 교차 시냅스 없음(0), 행 안 순서 섞어도 같음
    pc = np.array([0, 0, 1, 1, 2, 2]); wc = np.array([200., 220., 140., 150., 150., 150.])
    ps = np.array([0, 1, 1, 2, 3, 3]); ws = np.array([150., 160., 170., 150., 100., 120.])
    dl = learned_diff(pc, wc, ps, ws, 4)
    chk("Δ 정답표(KC1: 교차 (140+150)/2 − 같은 쪽 (160+170)/2 = −20)", list(dl) == [60.0, -20.0, 0.0, 0.0], str(dl))
    perm = np.array([5, 3, 1, 0, 4, 2])
    chk("Δ 행 안 순서 무관", list(learned_diff(pc[perm], wc[perm], ps, ws, 4)) == list(dl))
    rL = np.array([True, True, False, True])
    gl = oracle_weights(np.arange(4), dl, elig("lsgnR", rL, None, dl), "l", "r", 300.0, 150.0, "lsgnR")
    chk("lsgnR 교차(Δ 부호·반응 대상)", list(gl) == [300, 0, 150, 150], str(gl))
    chk("lsgnA 대상 = Δ ≠ 0", list(elig("lsgnA", rL, None, dl)) == [True, True, False, False])
    chk("lsgn 점검 수", counts(dl, elig("lsgnR", rL, None, dl), sign=True) == (1, 1, 3), str(counts(dl, elig("lsgnR", rL, None, dl), sign=True)))
    chk("부호 일치 수", agree(dl, np.array([0.3, 0.2, -0.1, 0.5]), np.array([True, True, True, True])) == 1)
    for nm, fn in (("lsgnA delta 없음 거부", lambda: elig("lsgnA", rL, None)),
                   ("Δ pre·w 크기 불일치 거부", lambda: learned_diff(pc, wc[:-1], ps, ws, 4)),
                   ("Δ KC 범위 밖 거부", lambda: learned_diff(pc, wc, ps, ws, 3))):
        try:
            fn()
            chk(nm, False)
        except ValueError:
            chk(nm, True)
    for nm, fn in (("칸 결측 거부", lambda: sxi_rows([r for r in rows if r[1] + r[2] != "BY"], nk)),
                   ("칸 불균형 거부", lambda: sxi_rows(rows[:-2], nk)),
                   ("KC 수 불일치 거부", lambda: sxi_rows(rows, nk + 1)),
                   ("알 수 없는 팔 거부", lambda: oracle_weights(pre, d["sxi"], d["resp"], "l", "r", 300.0, 150.0, "foo"))):
        try:
            fn()
            chk(nm, False)
        except ValueError:
            chk(nm, True)
    # rcxor 순도와 같은 정의(|SXI| 활동 가중 평균 = purity_rows)
    sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
    sys.path.insert(0, sd)
    try:
        import rcxor
        rs = np.random.RandomState(3)
        rows3 = []
        for c in CELLS:
            lam = rs.uniform(0, 1.5, 200)
            for i in range(6):
                for rep in range(3):
                    rows3.append(("train", c[0], c[1], i, rep, rs.poisson(lam).astype(float), -1))
        d3 = sxi_rows(rows3, 200)
        w_ours = float(np.sum(np.abs(d3["sxi"][d3["resp"]]) * d3["tot"][d3["resp"]]) / np.sum(d3["tot"][d3["resp"]]))
        w_rc = rcxor.purity_rows(rows3)[0]
        chk("rcxor 순도 정의와 일치", abs(w_ours - w_rc) < 1e-8, "%.9f vs %.9f" % (w_ours, w_rc))
    except ImportError:
        chk("rcxor 순도 정의와 일치(rcxor 없음)", False)
    print("합성 %d/%d" % (ok, n))
    return ok == n


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
