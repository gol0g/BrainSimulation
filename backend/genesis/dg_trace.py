#!/usr/bin/env python3
"""E202 계측(읽기 전용): 시행마다 보상 창의 KC→motor 가중치 변화(Δg)와 도파민 직전 자격흔적(e)을 KC 별·경로별로 모은다.

- before(brain): 도파민 방출 직전 — 4경로 g·e 를 장치에서 당긴다(pull 만, push 없음).
- after(brain, meta): 보상 창 뒤 도파민 0 직후(경로·KC 정규화 전) — g 를 다시 당겨 Δg = g1 − g0.
  KC 별 경로 평균(np.bincount)으로 x = Δg(교차) − Δg(같은 쪽)을 만들어, 시행을
  (실행 motor 가 자극 쪽 S / 반대 C) × (보상 + / 무보상 −) 네 범주로 KC 마다 누적한다.
  같은 범주별로 도파민 직전 e 의 KC 별 경로 평균(교차 ec·같은 쪽 es)도 누적한다.
- 범주 c 의 도파민 합 W[c]·시행 수 N[c], 시행별 요약 행(rows)을 남기고 save(path) 로 npz 저장.
어떤 KC 무리 G 에 대해서도 Σ_{t∈c} mean_{k∈G} x = mean_{k∈G} acc_x[c][k] 이므로(합과 평균의 교환) 무리는 분석 때 고른다.
E202 정정 1(결과 전): 같은 누적을 자극 쪽(l·r)별로도 나눠 둔다(xs·ecs·ess·Ws·Ns) — 자극 반대쪽 KC 의 활동이 반대 범주로 섞이는 크기를 잴 수 있게.
KC 별 정규화는 KC 마다 두 경로에 같은 값을 빼므로 x 의 누적은 학습 Δ(교차 − 같은 쪽)와 같아야 한다(경계 자르기만 오차).
"""
import numpy as np

CLASSES = ("S+", "S-", "C+", "C-")
POPS = ("kc_l_to_motor_l", "kc_l_to_motor_r", "kc_r_to_motor_l", "kc_r_to_motor_r")
ROW_COLS = ("trial", "stim_r", "ex_r", "correct", "same", "ctx_x", "cat_b", "da", "cls")


def cls_of(stim, ex, correct):
    """stim·ex 는 'l'/'r'(또는 'left'/'right'), correct 는 bool → 범주 이름."""
    return ("S" if ex[0] == stim[0] else "C") + ("+" if correct else "-")


class DgTrace:
    def __init__(self, n_kc):
        self.n_kc = int(n_kc)
        self.pre = None
        self.cnt = {}
        self.acc = {q: {c: {sd: np.zeros(self.n_kc) for sd in "lr"} for c in CLASSES} for q in ("x", "ec", "es")}
        self.W = {c: 0.0 for c in CLASSES}
        self.N = {c: 0 for c in CLASSES}
        # 자극 쪽별 분할(정정 1): accs[q][c][stim][KC 쪽], Ws[c][stim], Ns[c][stim]
        self.accs = {q: {c: {st: {sd: np.zeros(self.n_kc) for sd in "lr"} for st in "lr"} for c in CLASSES} for q in ("x", "ec", "es")}
        self.Ws = {c: {st: 0.0 for st in "lr"} for c in CLASSES}
        self.Ns = {c: {st: 0 for st in "lr"} for c in CLASSES}
        self.tot_x = {sd: np.zeros(self.n_kc) for sd in "lr"}
        self.rows = []
        self.g0 = self.e0 = None
        self.da = None

    # ── 장치 접근(읽기만) ───────────────────────────────────
    def prepare(self, brain):
        self.pre = {}
        for nm in POPS:
            s = getattr(brain, nm)
            s.pull_connectivity_from_device()
            self.pre[nm] = np.asarray(s.get_sparse_pre_inds(), dtype=np.int64)
            self.cnt[nm] = np.bincount(self.pre[nm], minlength=self.n_kc).astype(np.float64)

    def _pull(self, brain, var):
        out = {}
        for nm in POPS:
            s = getattr(brain, nm)
            s.vars[var].pull_from_device()
            v = np.asarray(s.vars[var].values, dtype=np.float64).ravel().copy()
            if v.size != self.pre[nm].size:
                raise RuntimeError("E202 계측: %s.%s 길이 %d ≠ 연결 %d" % (nm, var, v.size, self.pre[nm].size))
            out[nm] = v
        return out

    def kc_mean(self, nm, v):
        return np.bincount(self.pre[nm], weights=v, minlength=self.n_kc) / np.maximum(self.cnt[nm], 1.0)

    # ── 시행마다 ───────────────────────────────────────────
    def before(self, brain):
        if self.pre is None:
            self.prepare(brain)
        self.g0 = self._pull(brain, "g")
        self.e0 = self._pull(brain, "e")
        self.da = None

    def after(self, brain, meta):
        """meta: trial·side(자극 쪽)·ex(실행 쪽)·correct·same(정답 같은 쪽)·ctx('X'/'Y'/None)·cat('A'/'B'/None). DA 는 self.da(방출 직후 값)."""
        if self.g0 is None or self.da is None:
            raise RuntimeError("E202 계측: before()·DA 기록 없이 after() 호출")
        g1 = self._pull(brain, "g")
        c = cls_of(meta["side"], meta["ex"], bool(meta["correct"]))
        st = meta["side"][0]
        self.W[c] += float(self.da)
        self.N[c] += 1
        self.Ws[c][st] += float(self.da)
        self.Ns[c][st] += 1
        for kn in "lr":
            oc = "r" if kn == "l" else "l"
            cr, sa = "kc_%s_to_motor_%s" % (kn, oc), "kc_%s_to_motor_%s" % (kn, kn)
            x = self.kc_mean(cr, g1[cr] - self.g0[cr]) - self.kc_mean(sa, g1[sa] - self.g0[sa])
            both = (self.cnt[cr] > 0) & (self.cnt[sa] > 0)
            x[~both] = 0.0
            ec = np.where(both, self.kc_mean(cr, self.e0[cr]), 0.0)
            es = np.where(both, self.kc_mean(sa, self.e0[sa]), 0.0)
            self.acc["x"][c][kn] += x
            self.acc["ec"][c][kn] += ec
            self.acc["es"][c][kn] += es
            self.accs["x"][c][st][kn] += x
            self.accs["ec"][c][st][kn] += ec
            self.accs["es"][c][st][kn] += es
            self.tot_x[kn] += x
        self.rows.append([int(meta["trial"]), int(meta["side"][0] == "r"), int(meta["ex"][0] == "r"), int(bool(meta["correct"])),
                          int(bool(meta["same"])), int(meta.get("ctx") == "X"), int(meta.get("cat") in ("B", "D")),
                          float(self.da), CLASSES.index(c)])
        self.g0 = self.e0 = None
        self.da = None
        return c

    # ── 저장·요약 ──────────────────────────────────────────
    def arrays(self):
        out = {"rows": np.asarray(self.rows, dtype=np.float64).reshape(-1, len(ROW_COLS)), "row_cols": np.asarray(ROW_COLS),
               "classes": np.asarray(CLASSES), "W": np.asarray([self.W[c] for c in CLASSES]), "N": np.asarray([self.N[c] for c in CLASSES])}
        for q in ("x", "ec", "es"):
            for c in CLASSES:
                for sd in "lr":
                    out["%s_%s_%s" % (q, c.replace("+", "p").replace("-", "m"), sd)] = self.acc[q][c][sd]
        for sd in "lr":
            out["tot_x_" + sd] = self.tot_x[sd]
        # 정정 1: 자극 쪽별 — 키 '<q>s_<범주>_<자극 쪽>_<KC 쪽>', Ws·Ns 는 (범주 4 × 자극 쪽 l·r)
        for q in ("x", "ec", "es"):
            for c in CLASSES:
                for st in "lr":
                    for sd in "lr":
                        out["%ss_%s_%s_%s" % (q, c.replace("+", "p").replace("-", "m"), st, sd)] = self.accs[q][c][st][sd]
        out["Ws"] = np.asarray([[self.Ws[c][st] for st in "lr"] for c in CLASSES])
        out["Ns"] = np.asarray([[self.Ns[c][st] for st in "lr"] for c in CLASSES])
        return out

    def save(self, path):
        np.savez(path, **self.arrays())

    def summary(self):
        n = sum(self.N.values())
        return "시행 %d · 범주 %s · 도파민 합 %s" % (n, " ".join("%s %d" % (c, self.N[c]) for c in CLASSES),
                                                " ".join("%s %+.2f" % (c, self.W[c]) for c in CLASSES))


# ── 합성 시험 ───────────────────────────────────────────────
class _Var:
    def __init__(self, v):
        self.values = np.asarray(v, dtype=np.float32)
        self.pulls = 0

    def pull_from_device(self):
        self.pulls += 1

    def push_to_device(self):
        raise AssertionError("계측은 push 하면 안 된다")


class _Pop:
    def __init__(self, pre, g, e):
        self._pre = np.asarray(pre, dtype=np.int64)
        self.vars = {"g": _Var(g), "e": _Var(e)}

    def pull_connectivity_from_device(self):
        pass

    def get_sparse_pre_inds(self):
        return self._pre


class _Brain:
    pass


def _selftest():
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        if cond:
            ok[0] += 1
        else:
            print("  실패:", name)

    rng = np.random.RandomState(3)
    n = 6
    br = _Brain()
    pre = {}
    for nm in POPS:
        p = np.repeat(np.arange(n), 3)            # KC 마다 시냅스 3개
        p = p[rng.permutation(p.size)]
        if nm == "kc_r_to_motor_l":
            p = p[p != 5]                          # KC 5 는 r→l 연결 없음(x 0 이어야)
        pre[nm] = p
        setattr(br, nm, _Pop(p, np.full(p.size, 150.0), np.zeros(p.size)))
    T = DgTrace(n)
    # 시행 1: 자극 왼쪽, 실행 왼쪽(S), 보상(+) DA +0.7 — l→l(같은 쪽) 시냅스 +1·l→r(교차) +0.25, e 는 같은 쪽 2·교차 0.5
    T.before(br)
    # before 는 g·e 를 당긴다 — 그 뒤 장치 값을 바꿔 보상 창을 흉내낸다
    br.kc_l_to_motor_l.vars["g"].values = br.kc_l_to_motor_l.vars["g"].values + 1.0
    br.kc_l_to_motor_r.vars["g"].values = br.kc_l_to_motor_r.vars["g"].values + 0.25
    T.da = 0.7
    c1 = T.after(br, {"trial": 0, "side": "left", "ex": "left", "correct": True, "same": True, "ctx": "Y", "cat": "A"})
    chk("범주 S+", c1 == "S+")
    chk("KC l 쪽 x = 0.25 − 1 = −0.75", np.allclose(T.acc["x"]["S+"]["l"], -0.75))
    chk("KC r 쪽 x 0", np.allclose(T.acc["x"]["S+"]["r"], 0.0))
    chk("W S+ = 0.7", abs(T.W["S+"] - 0.7) < 1e-12 and T.N["S+"] == 1)
    # 시행 2: 자극 오른쪽, 실행 왼쪽(C), 무보상(−) DA −0.8 — r→l(교차) −0.5 (KC 5 는 연결 없음), e 도 미리 넣어 둠
    for nm in POPS:
        getattr(br, nm).vars["e"].values = np.full(pre[nm].size, 1.0 if nm == "kc_r_to_motor_l" else 0.2, dtype=np.float32)
    T.before(br)
    br.kc_r_to_motor_l.vars["g"].values = br.kc_r_to_motor_l.vars["g"].values - 0.5
    T.da = -0.8
    c2 = T.after(br, {"trial": 1, "side": "right", "ex": "left", "correct": False, "same": False, "ctx": "X", "cat": "B"})
    chk("범주 C−", c2 == "C-")
    xr = T.acc["x"]["C-"]["r"]
    chk("KC r 0~4 x = −0.5", np.allclose(xr[:5], -0.5))
    chk("KC r 5 x = 0(연결 없음)", xr[5] == 0.0)
    chk("ec(교차 = r→l) 1.0·es(같은 쪽 = r→r) 0.2", np.allclose(T.acc["ec"]["C-"]["r"][:5], 1.0) and np.allclose(T.acc["es"]["C-"]["r"][:5], 0.2))
    chk("ec KC 5 0", T.acc["ec"]["C-"]["r"][5] == 0.0)
    chk("총 x = 범주 합", all(np.allclose(T.tot_x[sd], sum(T.acc["x"][c][sd] for c in CLASSES)) for sd in "lr"))
    chk("W C− = −0.8", abs(T.W["C-"] + 0.8) < 1e-12)
    a = T.arrays()
    chk("rows 2행 9열", a["rows"].shape == (2, 9))
    chk("rows 범주 번호", list(a["rows"][:, 8]) == [0.0, 3.0])
    chk("rows 맥락 X·범주 B", list(a["rows"][1, 5:7]) == [1.0, 1.0])
    chk("저장 키", "x_Sp_l" in a and "x_Cm_r" in a and "ec_Cm_r" in a and "tot_x_l" in a)
    # 정정 1: 자극 쪽별 분할 — 시행 1 은 자극 왼쪽, 시행 2 는 자극 오른쪽
    chk("분할 키·모양", "xs_Sp_l_l" in a and a["Ws"].shape == (4, 2) and a["Ns"].shape == (4, 2))
    chk("시행 1 → xs_Sp_l_l = −0.75, 자극 오른쪽 칸 0", np.allclose(a["xs_Sp_l_l"], -0.75) and np.allclose(a["xs_Sp_r_l"], 0.0))
    chk("시행 2 → xs_Cm_r_r 0~4 = −0.5, 자극 왼쪽 칸 0", np.allclose(a["xs_Cm_r_r"][:5], -0.5) and np.allclose(a["xs_Cm_l_r"], 0.0))
    chk("Ws: S+ 왼쪽 0.7·C− 오른쪽 −0.8", abs(a["Ws"][0, 0] - 0.7) < 1e-12 and abs(a["Ws"][3, 1] + 0.8) < 1e-12 and abs(a["Ws"].sum() + 0.1) < 1e-12)
    chk("Ns 합 = N", int(a["Ns"].sum()) == 2 and list(a["Ns"].sum(axis=1)) == list(a["N"]))
    chk("분할 합 = 합계(모든 키)", all(np.allclose(a["%ss_%s_l_%s" % (q, k, sd)] + a["%ss_%s_r_%s" % (q, k, sd)], a["%s_%s_%s" % (q, k, sd)])
                                    for q in ("x", "ec", "es") for k in ("Sp", "Sm", "Cp", "Cm") for sd in "lr"))
    # 읽기 전용: push 하면 _Var 가 오류를 낸다 — 위 호출이 통과했으면 push 없음
    chk("pull 횟수(시행마다 g 2회 + e 1회)", br.kc_l_to_motor_l.vars["g"].pulls == 4 and br.kc_l_to_motor_l.vars["e"].pulls == 2)
    # 순서 오류는 거부
    try:
        T.after(br, {"trial": 2, "side": "left", "ex": "left", "correct": True, "same": True})
        chk("before 없이 after 거부", False)
    except RuntimeError:
        chk("before 없이 after 거부", True)
    # 길이 불일치 거부
    br2 = _Brain()
    for nm in POPS:
        setattr(br2, nm, _Pop(pre[nm], np.full(pre[nm].size, 150.0), np.zeros(pre[nm].size)))
    br2.kc_l_to_motor_l.vars["g"].values = np.zeros(2, dtype=np.float32)
    T2 = DgTrace(n)
    try:
        T2.before(br2)
        chk("길이 불일치 거부", False)
    except RuntimeError:
        chk("길이 불일치 거부", True)
    print("dg_trace 합성 %d/%d" % tuple(ok))
    return ok[0] == ok[1]


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
