"""E183: 원시 특징 범주 — 사례 세계(원형·왜곡 사례)와 KC 집합 지표. 순수 numpy(GPU 무관), 과제 스크립트와 판정 스크립트가 같이 쓴다.

사례 세계: 특징 집단 n(기본 200) 중 원형 A·B 가 각각 k(40) 개 활성, 둘이 shared(10) 개 공유.
사례 = 원형 활성 k 개 중 d(12) 개를 끄고, 두 원형 모두에 없는 배경 단위에서 d 개를 켠 것 — 범주 표지는 없다(같은 집단, 패턴만 다름).
노출(형성) 사례 범주당 n_train, 새 사례 범주당 n_new — 서로 다른 고정 난수열(같은 세계를 모든 뇌가 공유).

KC 집합 지표: K(x) = 사례 x 제시 동안 1회 이상 발화한 KC 번호 집합.
K_범주 = 그 범주 노출 사례 중 절반 이상(>= ceil(n/2))에 반응한 KC.
분리 s(x) = J(K(x), K_자기) − J(K(x), K_다른). 새 사례 평균 S_new, 노출 사례 평균 S_train(노출 사례는 K_범주 계산에 자기도 들어감 — 비교용).
"""
import math

import numpy as np

WORLD_SEED = 18300


def make_world(n=200, k=40, shared=10, d=12, n_train=6, n_new=6, seed=WORLD_SEED, d_far=20, n_far=6):
    if not (0 <= shared <= k and 2 * k - shared <= n and 0 <= d <= k):
        raise ValueError("사례 세계 인자 범위 밖")
    rs = np.random.RandomState(seed)
    perm = rs.permutation(n)
    a = np.sort(perm[:k])
    b_own = perm[k:k + (k - shared)]
    b_sh = np.sort(rs.choice(a, shared, replace=False))
    b = np.sort(np.concatenate([b_sh, b_own]))
    bg = np.setdiff1d(np.arange(n), np.union1d(a, b))
    protos = {"A": a, "B": b}

    def exemplars(c, m, s, dd=d):
        r = np.random.RandomState(s)
        out = []
        for _ in range(m):
            keep = np.sort(r.choice(protos[c], k - dd, replace=False))
            add = np.sort(r.choice(bg, dd, replace=False))
            out.append(np.sort(np.concatenate([keep, add])))
        return out

    train = {c: exemplars(c, n_train, seed + 1 + i) for i, c in enumerate(("A", "B"))}
    new = {c: exemplars(c, n_new, seed + 11 + i) for i, c in enumerate(("A", "B"))}
    # E186: 원형 자체(학습·노출에 없음, 왜곡 0)와 먼 사례(왜곡 d_far, 따로 고정한 난수열) — train·new 의 난수열은 그대로(회귀 검사)
    proto = {c: [np.array(protos[c])] for c in ("A", "B")}
    if not (0 <= d_far <= k and d_far <= len(bg)):
        raise ValueError("d_far 범위 밖")
    far = {c: exemplars(c, n_far, seed + 21 + i, d_far) for i, c in enumerate(("A", "B"))}
    return {"n": n, "k": k, "shared": shared, "d": d, "d_far": d_far, "protos": protos, "bg": bg, "train": train, "new": new,
            "proto": proto, "far": far}


def pattern(world, units, level=1.0):
    """특징 집단 전류 패턴(0/level) — units 만 켬."""
    v = np.zeros(world["n"], dtype=np.float32)
    v[np.asarray(units, dtype=int)] = level
    return v


def jac(a, b):
    a = set(a); b = set(b)
    u = a | b
    return (len(a & b) / len(u)) if u else 0.0


def cat_set(sets, frac=0.5):
    """노출 사례 KC 집합 목록 → 절반 이상(>= ceil(frac·m))에 나온 KC."""
    m = len(sets)
    need = max(1, int(math.ceil(frac * m)))
    cnt = {}
    for s in sets:
        for x in set(s):
            cnt[x] = cnt.get(x, 0) + 1
    return set(x for x, c in cnt.items() if c >= need)


def separation(train_sets, test_sets):
    """train_sets/test_sets: {'A': [KC 집합...], 'B': [...]} → (사례별 s 목록 {'A':[...],'B':[...]}, 평균 S, 정분류 수, 전체 수, J(K_A,K_B))."""
    KA = cat_set(train_sets["A"]); KB = cat_set(train_sets["B"])
    K = {"A": KA, "B": KB}
    out = {}
    tot = 0.0; n = 0; ok = 0
    for c, o in (("A", "B"), ("B", "A")):
        out[c] = []
        for s in test_sets[c]:
            v = jac(s, K[c]) - jac(s, K[o])
            out[c].append(v)
            tot += v; n += 1; ok += v > 0
    return out, (tot / n if n else 0.0), ok, n, jac(KA, KB)


def selectivity(train_sets):
    """KC 별 범주 선택성 |rA − rB|/(rA + rB) (rA = A 노출 사례 중 반응 비율), 반응한 KC 의 중앙값."""
    mA = len(train_sets["A"]); mB = len(train_sets["B"])
    ca = {}; cb = {}
    for s in train_sets["A"]:
        for x in set(s):
            ca[x] = ca.get(x, 0) + 1
    for s in train_sets["B"]:
        for x in set(s):
            cb[x] = cb.get(x, 0) + 1
    vals = []
    for x in set(ca) | set(cb):
        ra = ca.get(x, 0) / mA; rb = cb.get(x, 0) / mB
        vals.append(abs(ra - rb) / (ra + rb))
    return (float(np.median(vals)) if vals else 0.0), len(vals)


def _selftest():
    ok = True
    w = make_world()
    a, b = w["protos"]["A"], w["protos"]["B"]
    c1 = len(a) == 40 and len(b) == 40 and len(np.intersect1d(a, b)) == 10
    c2 = all(len(x) == 40 and len(np.intersect1d(x, w["protos"][c])) == 28 and len(np.intersect1d(x, w["bg"])) == 12
             for c in "AB" for x in w["train"][c] + w["new"][c])
    tr = set(tuple(x) for c in "AB" for x in w["train"][c]); nw = set(tuple(x) for c in "AB" for x in w["new"][c])
    c3 = len(tr & nw) == 0 and len(tr) == 12 and len(nw) == 12
    w2 = make_world()
    c4 = all(np.array_equal(w["train"][c][i], w2["train"][c][i]) and np.array_equal(w["new"][c][i], w2["new"][c][i]) for c in "AB" for i in range(6))
    p = pattern(w, w["train"]["A"][0], 0.7)
    c5 = p.shape == (200,) and abs(p.sum() - 28.0) < 1e-4 and set(np.nonzero(p)[0]) == set(w["train"]["A"][0].tolist())
    # 합성 KC 집합: 완벽 분리(범주마다 고유 KC) → s = 1·정분류 전부·J(K_A,K_B)=0
    trs = {"A": [set(range(0, 50)) for _ in range(6)], "B": [set(range(100, 150)) for _ in range(6)]}
    tes = {"A": [set(range(0, 50)) for _ in range(6)], "B": [set(range(100, 150)) for _ in range(6)]}
    _, S, okn, nn, jab = separation(trs, tes)
    c6 = abs(S - 1.0) < 1e-12 and okn == 12 and nn == 12 and jab == 0.0
    # 같은 집합(붕괴) → s = 0, 정분류 0, J = 1
    trs2 = {"A": [set(range(0, 50))] * 6, "B": [set(range(0, 50))] * 6}
    _, S2, ok2, _, jab2 = separation(trs2, trs2)
    c7 = abs(S2) < 1e-12 and ok2 == 0 and jab2 == 1.0
    # 반쪽 겹침: K_A = 0..49, K_B = 25..74, x = 0..49(A) → J 자기 1, J 다른 25/75 → s = 2/3
    trs3 = {"A": [set(range(0, 50))] * 6, "B": [set(range(25, 75))] * 6}
    tes3 = {"A": [set(range(0, 50))], "B": [set(range(25, 75))]}
    o3, S3, _, _, jab3 = separation(trs3, tes3)
    c8 = abs(o3["A"][0] - (1 - 25 / 75)) < 1e-12 and abs(jab3 - 25 / 75) < 1e-12
    # 절반 이상 규칙: 6 사례 중 3 회 나온 KC 포함, 2 회 제외
    c9 = cat_set([{1, 2}, {1, 2}, {1, 3}, {4}, {4}, {5}]) == {1}
    c9b = cat_set([{1}, {1}, {1}, {2}, {2}, {9}]) == {1}
    # 선택성: A 에만 반응 KC 1, 둘 다 같은 비율 KC 2 → 중앙 (1 + 0)/2 = 0.5
    sel, nsel = selectivity({"A": [{1, 2}] * 6, "B": [{2}] * 6})
    c10 = abs(sel - 0.5) < 1e-12 and nsel == 2
    # E186: 원형 집합 = 원형 그대로 1개, 먼 사례 = 원형 20 + 배경 20, 노출·새·먼 사례 서로 다름
    c11 = all(len(w["proto"][c]) == 1 and np.array_equal(w["proto"][c][0], w["protos"][c]) for c in "AB")
    c12 = all(len(x) == 40 and len(np.intersect1d(x, w["protos"][c])) == 20 and len(np.intersect1d(x, w["bg"])) == 20 for c in "AB" for x in w["far"][c])
    fr = set(tuple(x) for c in "AB" for x in w["far"][c])
    c13 = len(fr) == 12 and not (fr & tr) and not (fr & nw)
    for name, c in (("원형 크기·공유", c1), ("사례 구성 28+12", c2), ("노출·새 사례 서로 다름", c3), ("결정론", c4), ("전류 패턴", c5),
                    ("완벽 분리", c6), ("붕괴", c7), ("반쪽 겹침 s", c8), ("절반 규칙", c9 and c9b), ("선택성", c10),
                    ("원형 집합", c11), ("먼 사례 20+20", c12), ("먼 사례 서로 다름", c13)):
        print("  %-18s %s" % (name, "✓" if c else "✗"))
        ok &= bool(c)
    return ok


if __name__ == "__main__":
    import sys
    r = _selftest()
    print("rawcat 자체 시험: %s" % ("통과" if r else "실패"))
    sys.exit(0 if r else 1)
