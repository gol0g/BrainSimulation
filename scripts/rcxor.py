#!/usr/bin/env python3
"""E192: 범주 × 맥락 결합 부호 해독 — RCKX 줄(자극 쪽 KC 제시별 스파이크 수)로 이상적 선형 해독기(능형 회귀)의 정답률.
쪽(l·r)마다: 노출 사례 제시로 맞추고(λ = 노출 사례 하나 빼기 교차 검증으로 LAMBDAS 중), 새 사례 제시로 시험.
표지: XOR(정답 같은 쪽 = X·B 또는 Y·A → +1, 교차 = X·A 또는 Y·B → −1), 범주(B +1·A −1), 맥락(Y +1·X −1).
(보고) 결합 KC 수: 노출 제시 평균 스파이크로 조합 4칸(AX·AY·BX·BY) 상호작용 |(AX + BY) − (AY + BX)|/2 ≥ 0.5 이면서 4칸 평균 ≥ 0.25 인 KC.
순수 numpy. 실행: python3 scripts/rcxor.py <RCKX 로그> [...]  /  모듈로 decode_log(path) → {side: {"xor":acc,"cat":acc,"ctx":acc,"lam":{...},"conj":n,"n_train":..,"n_test":..}}"""
import sys

import numpy as np

LAMBDAS = (1.0, 10.0, 100.0, 1000.0)
N_KC = 3000


def parse(path, n_kc=N_KC):
    """RCKX 줄 → {side: [(set, cat, ctx, i, rep, vec)]}"""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.startswith("RCKX "):
                continue
            head, ids = ln.rstrip("\n").split(" ids=")
            kv = dict(x.split("=", 1) for x in head.split()[1:])
            v = np.zeros(n_kc)
            if ids:
                for tok in ids.split(","):
                    k, c = tok.split(":")
                    v[int(k)] = float(c)
            out.setdefault(kv["side"], []).append((kv["set"], kv["cat"], kv["ctx"], int(kv["i"]), int(kv["rep"]), v, int(kv.get("k", -1))))
    return out


def prev_cats(path):
    """(보고) 제시 순서상 바로 앞 제시의 범주 — 파일 순서(= 실행 순서) 그대로, 쪽 무관 전역 직전. {(side, rep, k): 앞 범주 또는 None}"""
    prev, out = None, {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.startswith("RCKX "):
                continue
            kv = dict(x.split("=", 1) for x in ln.split(" ids=")[0].split()[1:])
            out[(kv["side"], int(kv["rep"]), int(kv.get("k", -1)), kv["set"], kv["cat"], kv["ctx"], int(kv["i"]))] = prev
            prev = kv["cat"]
    return out


def labels(rows, kind):
    if kind == "prev":
        return np.array([1.0 if r[-1] == "B" else -1.0 for r in rows])
    if kind == "xor":
        return np.array([1.0 if (r[1] == "B") == (r[2] == "X") else -1.0 for r in rows])
    if kind == "cat":
        return np.array([1.0 if r[1] == "B" else -1.0 for r in rows])
    return np.array([1.0 if r[2] == "Y" else -1.0 for r in rows])


def ridge_fit(X, y, lam):
    """중심화 능형 회귀(쌍대 형식 — 표본 수 << 차원). (w, b) 반환."""
    mx = X.mean(axis=0)
    my = y.mean()
    Xc = X - mx
    K = Xc @ Xc.T
    a = np.linalg.solve(K + lam * np.eye(K.shape[0]), y - my)
    w = Xc.T @ a
    return w, my - mx @ w


def acc(X, y, w, b):
    p = X @ w + b
    return float(np.mean(np.sign(p) == y)) if len(y) else float("nan")


def kfit(G, tr, y, lam):
    """그람 행렬 G(전체 표본) 위 중심화 능형 회귀 — 학습 표본 번호 tr. 1ᵀa = 0 이라 예측 = ȳ + G[t, tr]·a − mean(G[tr, tr]·a)."""
    n = len(tr)
    K = G[np.ix_(tr, tr)]
    H = np.eye(n) - 1.0 / n
    a = np.linalg.solve(H @ K @ H + lam * np.eye(n), y - y.mean())
    return a, y.mean(), float(np.mean(K @ a))


def kpredict(G, te, tr, a, my, off):
    return my + G[np.ix_(te, tr)] @ a - off


def choose_lambda(G, tr, y, groups):
    """노출 사례(범주·번호) 하나 빼기 교차 검증 정답률이 가장 높은 λ(같으면 큰 λ — 보수적). tr·y·groups 는 학습 표본 순서."""
    tr = np.asarray(tr)
    best, best_l = -1.0, None
    for lam in LAMBDAS:
        hit = n = 0
        for g in sorted(set(groups)):
            te = np.array([gg == g for gg in groups])
            a, my, off = kfit(G, tr[~te], y[~te], lam)
            hit += int(np.sum(np.sign(kpredict(G, tr[te], tr[~te], a, my, off)) == y[te])); n += int(te.sum())
        acc_ = hit / float(n)
        if acc_ >= best:
            best, best_l = acc_, lam
    return best_l, best


def conj_count(rows):
    cell = {}
    for row in rows:
        cell.setdefault(row[1] + row[2], []).append(row[5])
    if set(cell) != {"AX", "AY", "BX", "BY"}:
        return -1
    m = {k: np.mean(v, axis=0) for k, v in cell.items()}
    inter = np.abs((m["AX"] + m["BY"]) - (m["AY"] + m["BX"])) / 2.0
    mean4 = (m["AX"] + m["AY"] + m["BX"] + m["BY"]) / 4.0
    return int(np.sum((inter >= 0.5) & (mean4 >= 0.25)))


def decode_rows(rows):
    tr = [r for r in rows if r[0] == "train"]
    te = [r for r in rows if r[0] == "new"]
    Xall = np.array([r[5] for r in tr + te], dtype=np.float64)
    G = Xall @ Xall.T                                   # 그람 행렬 한 번(표본 수 × 표본 수)
    itr = np.arange(len(tr)); ite = np.arange(len(tr), len(tr) + len(te))
    groups = [(r[1], r[3]) for r in tr]
    res = {"lam": {}, "cv": {}, "n_train": len(tr), "n_test": len(te), "conj": conj_count(tr)}
    kinds = ("xor", "cat", "ctx") + (("prev",) if all(len(r) >= 8 and r[-1] in ("A", "B") for r in tr + te) else ())
    for kind in kinds:
        ytr, yte = labels(tr, kind), labels(te, kind)
        lam, cv = choose_lambda(G, itr, ytr, groups)
        a, my, off = kfit(G, itr, ytr, lam)
        res[kind] = float(np.mean(np.sign(kpredict(G, ite, itr, a, my, off)) == yte)) if len(te) else float("nan")
        res["lam"][kind] = lam
        res["cv"][kind] = cv
    return res


def decode_log(path, n_kc=N_KC):
    pc = prev_cats(path)
    out = {}
    for sd, rows in sorted(parse(path, n_kc).items()):
        res = decode_rows(rows)
        rr = [r + (pc.get((sd, r[4], r[6], r[0], r[1], r[2], r[3])),) for r in rows]
        rr = [r for r in rr if r[-1] in ("A", "B")]
        if rr:
            res["prev"] = decode_rows(rr)["prev"]
        out[sd] = res
    return out


def _synthetic(kind, seed, n_kc=400, n_ex=6, reps=3):
    """합성 RCKX 행: kind = additive(범주·맥락 독립 성분) / conj(조합 전용 단위) / noise."""
    rs = np.random.RandomState(seed)
    rows = []
    base = rs.uniform(0.0, 0.2, n_kc)
    cat_u = {c: rs.rand(n_kc) < 0.1 for c in "AB"}
    ctx_u = {x: rs.rand(n_kc) < 0.1 for x in "XY"}
    comb_u = {(c, x): rs.rand(n_kc) < 0.05 for c in "AB" for x in "XY"}
    for st in ("train", "new"):
        ex = {c: [rs.rand(n_kc) < 0.03 for _ in range(n_ex)] for c in "AB"}   # 사례 고유 잡음 단위
        for c in "AB":
            for i in range(n_ex):
                for x in "XY":
                    for r in range(reps):
                        lam = base.copy() + 1.0 * ex[c][i]
                        if kind == "additive":
                            lam = lam + 2.0 * cat_u[c] + 2.0 * ctx_u[x]
                        elif kind == "conj":
                            lam = lam + 2.0 * cat_u[c] + 2.0 * ctx_u[x] + 3.0 * comb_u[(c, x)]
                        rows.append((st, c, x, i, r, rs.poisson(lam).astype(float)))
    return rows


def _synth_log(order, hist, seed, n_kc=400, n_ex=6, reps=3):
    """합성 RCKX 로그(l 쪽만): 덧셈 부호 + hist × 직전 제시 범주 단위. order = fixed(사례마다 AX·AY·BX·BY) / shuffled."""
    import os
    import tempfile
    rs = np.random.RandomState(seed)
    base = rs.uniform(0.0, 0.2, n_kc)
    cat_u = {c: rs.rand(n_kc) < 0.1 for c in "AB"}
    ctx_u = {x: rs.rand(n_kc) < 0.1 for x in "XY"}
    hist_u = {c: rs.rand(n_kc) < 0.1 for c in "AB"}
    ex = {(st, c): [rs.rand(n_kc) < 0.03 for _ in range(n_ex)] for st in ("train", "new") for c in "AB"}
    lines, prev = [], None
    for r in range(reps):
        seq = [(st, i, c, x) for st in ("train", "new") for i in range(n_ex) for c in "AB" for x in "XY"]
        if order == "shuffled":
            seq = [seq[j] for j in np.random.RandomState(seed + 100 + r).permutation(len(seq))]
        for k, (st, i, c, x) in enumerate(seq):
            lam = base + 1.0 * ex[(st, c)][i] + 2.0 * cat_u[c] + 2.0 * ctx_u[x] + (hist * hist_u[prev] if prev else 0.0)
            v = rs.poisson(lam)
            nz = np.nonzero(v)[0]
            lines.append("RCKX rep=%d side=l k=%d set=%s cat=%s ctx=%s i=%d n=%d ids=%s" % (r, k, st, c, x, i, nz.size, ",".join("%d:%d" % (j, v[j]) for j in nz)))
            prev = c
    fd, pth = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return pth


def _selftest():
    import os
    ok = True
    for kind, want in (("additive", {"xor": (0.30, 0.70), "cat": (0.9, 1.01), "ctx": (0.9, 1.01)}),
                       ("conj", {"xor": (0.9, 1.01), "cat": (0.9, 1.01), "ctx": (0.9, 1.01)}),
                       ("noise", {"xor": (0.30, 0.70), "cat": (0.30, 0.70), "ctx": (0.30, 0.70)})):
        r = decode_rows(_synthetic(kind, 7))
        good = all(lo <= r[k] < hi for k, (lo, hi) in want.items())
        ok &= good
        print("  %-8s XOR %.3f 범주 %.3f 맥락 %.3f 결합 KC %d λ %s %s" % (kind, r["xor"], r["cat"], r["ctx"], r["conj"], r["lam"], "✓" if good else "✗"))
    # 표지 정의 점검: X·B·Y·A = +1, X·A·Y·B = −1
    rows = [("train", c, x, 0, 0, np.zeros(3)) for c in "AB" for x in "XY"]
    good = list(labels(rows, "xor")) == [-1.0, 1.0, 1.0, -1.0] and list(labels(rows, "cat")) == [-1.0, -1.0, 1.0, 1.0] and list(labels(rows, "ctx")) == [-1.0, 1.0, -1.0, 1.0]
    ok &= good
    print("  표지 정의(AX −·AY +·BX +·BY −) %s" % ("✓" if good else "✗"))
    # 능형 회귀: 선형 분리 가능 자료 정답률 1, 결합 KC 수 계산(조합 전용 단위 수와 같은 크기)
    rs = np.random.RandomState(3)
    X = rs.randn(40, 10); y = np.sign(X[:, 0] + 0.1)
    w, b = ridge_fit(X, y, 1.0)
    good = acc(X, y, w, b) >= 0.95
    ok &= good
    print("  능형 회귀 분리 가능 자료 %s" % ("✓" if good else "✗"))
    # 커널 형식(그람 행렬) 예측 = 직접 계산(중심화 쌍대) 예측
    Xa = rs.randn(30, 50); ya = np.sign(rs.randn(30)); tri = np.arange(20); tei = np.arange(20, 30)
    w, b = ridge_fit(Xa[tri], ya[tri], 10.0)
    a_, my_, off_ = kfit(Xa @ Xa.T, tri, ya[tri], 10.0)
    good = np.allclose(Xa[tei] @ w + b, kpredict(Xa @ Xa.T, tei, tri, a_, my_, off_), atol=1e-9)
    ok &= good
    print("  커널 형식 = 직접 계산 %s" % ("✓" if good else "✗"))
    # 이력 교란 진단: 직전 범주 성분이 있는 덧셈 부호 — 고정 순서(AX·AY·BX·BY)면 XOR 이 직전 범주로 읽히고, 섞으면 XOR 은 우연·직전 범주는 읽힘
    for order, hist, want in (("fixed", 2.5, {"xor": (0.8, 1.01), "prev": (0.8, 1.01)}),
                              ("shuffled", 2.5, {"xor": (0.3, 0.7), "prev": (0.8, 1.01)}),
                              ("shuffled", 0.0, {"xor": (0.3, 0.7), "prev": (0.3, 0.7)})):
        _pth = _synth_log(order, hist, 11)
        r = decode_log(_pth, 400)["l"]
        os.unlink(_pth)
        good = all(lo <= r[k] < hi for k, (lo, hi) in want.items())
        ok &= good
        print("  이력 %-8s hist %.1f → XOR %.3f 직전 범주 %.3f %s" % (order, hist, r["xor"], r["prev"], "✓" if good else "✗"))
    # 파싱: RCKX 줄 왕복
    import os
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as fh:
        fh.write("RCKX rep=0 side=l set=train cat=A ctx=X i=0 n=2 ids=5:2,7:1\nRCKX rep=1 side=r set=new cat=B ctx=Y i=3 n=0 ids=\nother line\n")
        pth = fh.name
    pr = parse(pth, 10)
    os.unlink(pth)
    good = (len(pr["l"]) == 1 and pr["l"][0][:5] == ("train", "A", "X", 0, 0) and pr["l"][0][5][5] == 2 and pr["l"][0][5][7] == 1 and pr["l"][0][5].sum() == 3
            and len(pr["r"]) == 1 and pr["r"][0][5].sum() == 0)
    ok &= good
    print("  RCKX 파싱 %s" % ("✓" if good else "✗"))
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for pth in sys.argv[1:]:
            for sd, r in decode_log(pth).items():
                print("%s side=%s XOR %.3f 범주 %.3f 맥락 %.3f 결합 KC %d (λ %s, 시험 %d)" % (pth, sd, r["xor"], r["cat"], r["ctx"], r["conj"], r["lam"], r["n_test"]))
    else:
        r = _selftest()
        print("rcxor 자체 시험: %s" % ("통과" if r else "실패"))
        sys.exit(0 if r else 1)
