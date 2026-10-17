#!/usr/bin/env python3
"""E199: KC→motor 경쟁 정규화 — KC 별 두 motor(motor_l·motor_r) 평균 가중치 합을 목표값(2·init)으로 되돌리는 감산 정규화.
KC k 의 모든 시냅스(두 motor 집단)에서 같은 값 c_k 를 빼고 경계 [w_min, w_max] 로 자른다 — c_k 는 S_k(c) = mean(clip(g_a[k] − c)) + mean(clip(g_b[k] − c)) = target 의 해
(S_k 는 c 에 대해 단조 감소 → 이분법; 경계에 닿은 시냅스가 있어도 합을 정확히 맞추는 투영). 두 집단 중 한쪽에 시냅스가 없는 KC 는 바꾸지 않는다.
의미: 두 motor 를 함께 올리는 공통 모드 표류(C36 의 전 시냅스 w_max 붕괴 원인)를 지우고 두 motor 의 차만 남긴다 — 경성 상한 갱신과 함께 부호 고정점(300/0)으로 경쟁.
합성 자체 시험: python3 kc_comp.py"""
import numpy as np

ITERS = 40


def project(pre_a, g_a, pre_b, g_b, n_kc, target, w_min, w_max, iters=ITERS):
    """반환 (새 g_a, 새 g_b, 정규화 전 max|S − target|, 정규화 뒤 max|S − target|) — max 는 두 집단에 시냅스가 있는 KC 에서."""
    pre_a = np.asarray(pre_a, dtype=np.int64); pre_b = np.asarray(pre_b, dtype=np.int64)
    g_a = np.asarray(g_a, dtype=np.float64); g_b = np.asarray(g_b, dtype=np.float64)
    if pre_a.size != g_a.size or pre_b.size != g_b.size:
        raise ValueError("pre·g 크기 불일치")
    na = np.bincount(pre_a, minlength=n_kc); nb = np.bincount(pre_b, minlength=n_kc)
    if na.size != n_kc or nb.size != n_kc:
        raise ValueError("KC 인덱스 범위 밖")
    if not (w_min <= target / 2.0 <= w_max):
        raise ValueError("목표 합이 경계 밖")
    both = (na > 0) & (nb > 0)
    ia = 1.0 / np.maximum(na, 1); ib = 1.0 / np.maximum(nb, 1)

    def ssum(c, ma=None, mb=None):
        # ma·mb: 계산할 시냅스 부분집합(필요한 KC 의 것만 — 속도), None 이면 전부
        pa_, ga_ = (pre_a, g_a) if ma is None else (pre_a[ma], g_a[ma])
        pb_, gb_ = (pre_b, g_b) if mb is None else (pre_b[mb], g_b[mb])
        return (np.bincount(pa_, weights=np.clip(ga_ - c[pa_], w_min, w_max), minlength=n_kc) * ia
                + np.bincount(pb_, weights=np.clip(gb_ - c[pb_], w_min, w_max), minlength=n_kc) * ib)

    span = float(w_max - w_min)
    s0 = ssum(np.zeros(n_kc))
    # 1차: 경계에 안 닿으면 c = (S − target)/2 가 정확한 해(두 집단 평균에서 같은 값을 뺌)
    c = np.where(both & (np.abs(s0 - target) > 1e-9), 0.5 * (s0 - target), 0.0)   # 이미 목표 합인 KC 는 그대로
    need = both & (np.abs(ssum(c) - target) > 1e-7)       # 경계 자르기로 합이 어긋난 KC 만 다시 푼다
    if need.any():
        # S(c) 는 조각별 선형·단조 감소 — 기울기 −(경계 안 시냅스 비율 a + b) 의 뉴턴 단계(구간 [lo, hi] 로 보호, 밖이면 이분법)
        ma, mb = need[pre_a], need[pre_b]
        pa_, ga_, pb_, gb_ = pre_a[ma], g_a[ma], pre_b[mb], g_b[mb]
        lo = np.full(n_kc, -span); hi = np.full(n_kc, span)
        act = need.copy()
        for _ in range(iters):
            xa = ga_ - c[pa_]; xb = gb_ - c[pb_]
            s = (np.bincount(pa_, weights=np.clip(xa, w_min, w_max), minlength=n_kc) * ia
                 + np.bincount(pb_, weights=np.clip(xb, w_min, w_max), minlength=n_kc) * ib)
            r = s - target
            act &= np.abs(r) > 1e-7
            if not act.any():
                break
            lo = np.where(act & (r > 0), c, lo); hi = np.where(act & (r <= 0), c, hi)   # 합이 크면 c 를 키운다
            fr = (np.bincount(pa_, weights=((xa > w_min) & (xa < w_max)).astype(float), minlength=n_kc) * ia
                  + np.bincount(pb_, weights=((xb > w_min) & (xb < w_max)).astype(float), minlength=n_kc) * ib)
            cn = np.where(fr > 0, c + r / np.maximum(fr, 1e-12), 0.5 * (lo + hi))
            cn = np.where((cn <= lo) | (cn >= hi), 0.5 * (lo + hi), cn)
            c = np.where(act, cn, c)
    ga = np.clip(g_a - c[pre_a], w_min, w_max); gb = np.clip(g_b - c[pre_b], w_min, w_max)
    s1 = np.bincount(pre_a, weights=ga, minlength=n_kc) * ia + np.bincount(pre_b, weights=gb, minlength=n_kc) * ib
    d0 = float(np.max(np.abs(s0 - target)[both])) if both.any() else 0.0
    d1 = float(np.max(np.abs(s1 - target)[both])) if both.any() else 0.0
    return ga, gb, d0, d1


def prepare(brain, sides="lr"):
    """KC→motor 4집단의 전시냅스 KC 인덱스(연결은 고정 — 한 번만 가져온다)."""
    pre = {}
    for kn in sides:
        for m in "lr":
            nm = "kc_%s_to_motor_%s" % (kn, m)
            s = getattr(brain, nm)
            s.pull_connectivity_from_device()
            pre[nm] = np.asarray(s.get_sparse_pre_inds(), dtype=np.int64)
    return pre


def apply(brain, pre, n_kc, target, w_min, w_max):
    """장치 가중치에 정규화를 적용. 반환 {쪽: (정규화 전 max|S−target|, 뒤 max|S−target|)}."""
    out = {}
    for kn in "lr":
        a, b = "kc_%s_to_motor_l" % kn, "kc_%s_to_motor_r" % kn
        sa, sb = getattr(brain, a), getattr(brain, b)
        sa.vars["g"].pull_from_device(); sb.vars["g"].pull_from_device()
        ga = np.asarray(sa.vars["g"].values, dtype=np.float64).ravel()
        gb = np.asarray(sb.vars["g"].values, dtype=np.float64).ravel()
        na_, nb_, d0, d1 = project(pre[a], ga, pre[b], gb, n_kc, target, w_min, w_max)
        sa.vars["g"].values = na_; sa.vars["g"].push_to_device()
        sb.vars["g"].values = nb_; sb.vars["g"].push_to_device()
        out[kn] = (d0, d1)
    return out


def center_path(g, target, w_min, w_max, iters=ITERS):
    """(E201) 한 경로(KC 반구 → motor 집단)의 평균 가중치를 target 으로 — 모든 시냅스에서 같은 s 를 빼고 [w_min, w_max] 로 자른다.
    mean(clip(g − s)) 는 s 에 대해 단조 감소 → 1차 해 s = mean − target 에서 뉴턴(구간 보호, 밖이면 이분법). 반환 (새 g, 전 |mean − target|, 뒤 |mean − target|)."""
    g = np.asarray(g, dtype=np.float64)
    if not (w_min <= target <= w_max):
        raise ValueError("목표 평균이 경계 밖")
    m0 = float(g.mean())
    if abs(m0 - target) <= 1e-9:
        return g.copy(), abs(m0 - target), abs(m0 - target)
    span = float(w_max - w_min)
    lo, hi = -span, span
    s = m0 - target
    for _ in range(iters):
        x = g - s
        r = float(np.clip(x, w_min, w_max).mean()) - target
        if abs(r) <= 1e-9:
            break
        if r > 0:
            lo = s
        else:
            hi = s
        fr = float(np.mean((x > w_min) & (x < w_max)))
        sn = s + r / fr if fr > 0 else 0.5 * (lo + hi)
        if not (lo < sn < hi):
            sn = 0.5 * (lo + hi)
        s = sn
    out = np.clip(g - s, w_min, w_max)
    return out, abs(m0 - target), abs(float(out.mean()) - target)


def apply_paths(brain, target, w_min, w_max):
    """(E201) 장치 가중치의 KC→motor 4경로 평균을 target 으로(경로 항상성). 그다음 KC 별 합 보존은 apply 가 한다. 반환 {경로: (전, 뒤 |mean − target|)}."""
    out = {}
    for kn in "lr":
        for m in "lr":
            nm = "kc_%s_to_motor_%s" % (kn, m)
            s = getattr(brain, nm)
            s.vars["g"].pull_from_device()
            g = np.asarray(s.vars["g"].values, dtype=np.float64).ravel()
            x, d0, d1 = center_path(g, target, w_min, w_max)
            s.vars["g"].values = x
            s.vars["g"].push_to_device()
            out[nm] = (d0, d1)
    return out


def kc_sums(pre_a, g_a, pre_b, g_b, n_kc):
    """(판정·보고용) KC 별 mean(g_a) + mean(g_b) 와 두 집단 모두 시냅스가 있는 KC 마스크."""
    pre_a = np.asarray(pre_a, dtype=np.int64); pre_b = np.asarray(pre_b, dtype=np.int64)
    na = np.bincount(pre_a, minlength=n_kc); nb = np.bincount(pre_b, minlength=n_kc)
    s = (np.bincount(pre_a, weights=np.asarray(g_a, dtype=np.float64), minlength=n_kc) / np.maximum(na, 1)
         + np.bincount(pre_b, weights=np.asarray(g_b, dtype=np.float64), minlength=n_kc) / np.maximum(nb, 1))
    return s, (na > 0) & (nb > 0)


def _selftest():
    ok = [0, 0]

    def chk(name, cond, extra=""):
        ok[1] += 1
        ok[0] += bool(cond)
        print("  [%s] %s %s" % ("통과" if cond else "실패", name, extra))

    # 1) 경계에 안 닿는 경우: 같은 값을 빼면 합이 목표 — KC0: a(160,180) b(150,150) → S = 170+150 = 320 → c = 10 → a(150,170) b(140,140)
    pa = np.array([0, 0, 1]); ga = np.array([160., 180., 100.])
    pb = np.array([0, 0, 1, 1]); gb = np.array([150., 150., 200., 220.])
    na_, nb_, d0, d1 = project(pa, ga, pb, gb, 3, 300.0, 0.0, 300.0)
    chk("KC0 정답(감산 c=10)", np.allclose(na_[:2], [150, 170]) and np.allclose(nb_[:2], [140, 140]), str((na_, nb_)))
    # KC1: a(100) b(200,220) → S = 100 + 210 = 310 → c = 5 → a 95, b 195·215
    chk("KC1 정답(c=5)", np.isclose(na_[2], 95) and np.allclose(nb_[2:], [195, 215]))
    chk("정규화 전·뒤 차", np.isclose(d0, 20) and d1 < 1e-6, "%.6f %.2e" % (d0, d1))
    # 2) 경계에 닿는 경우: a(290,300) b(100,100) → S = 295 + 100 = 395, c = 47.5 이면 a(242.5,252.5) b(52.5,52.5) 합 300 — 경계 안
    na_, nb_, _, d1 = project(np.array([0, 0]), np.array([290., 300.]), np.array([0, 0]), np.array([100., 100.]), 1, 300.0, 0.0, 300.0)
    chk("경계 안 c=47.5", np.allclose(na_, [242.5, 252.5]) and np.allclose(nb_, [52.5, 52.5]), str((na_, nb_)))
    # 3) 하한에 닿는 경우: a(300,300) b(30,0) → S = 300 + 15 = 315 → c = 15 이면 b(15, −15→0): 합 = 285 + 7.5 = 292.5 < 300 → 정확한 c 는 c < 15:
    #    c ∈ [0, 30): b = (30−c, 0) → 합 = (300−c) + (30−c)/2 = 315 − 1.5c = 300 → c = 10 → a 290·290, b 20·0
    na_, nb_, _, d1 = project(np.array([0, 0]), np.array([300., 300.]), np.array([0, 0]), np.array([30., 0.]), 1, 300.0, 0.0, 300.0)
    chk("하한 자르기 포함 정확한 투영(c=10)", np.allclose(na_, [290, 290]) and np.allclose(nb_, [20, 0]) and d1 < 1e-6, str((na_, nb_, d1)))
    # 4) 이미 부호 고정점(300/0)은 그대로
    na_, nb_, d0, d1 = project(np.array([0, 0]), np.array([300., 300.]), np.array([0, 0]), np.array([0., 0.]), 1, 300.0, 0.0, 300.0)
    chk("부호 고정점 불변", np.allclose(na_, [300, 300]) and np.allclose(nb_, [0, 0]) and d0 == 0)
    # 5) 한쪽 시냅스 없는 KC 는 불변, 행 안 순서 무관
    na_, nb_, _, _ = project(np.array([0, 1]), np.array([200., 250.]), np.array([0]), np.array([200.]), 2, 300.0, 0.0, 300.0)
    chk("한쪽 없는 KC 불변", np.isclose(na_[1], 250) and np.allclose([na_[0], nb_[0]], [150, 150]), str((na_, nb_)))
    perm = np.array([2, 0, 1])
    a1, b1, _, _ = project(pa, ga, pb, gb, 3, 300.0, 0.0, 300.0)
    a2, b2, _, _ = project(pa[perm], ga[perm], pb, gb, 3, 300.0, 0.0, 300.0)
    chk("행 안 순서 무관", np.allclose(a2, a1[perm]) and np.allclose(b2, b1))
    # 6) 공통 모드만 있는 KC(두 motor 같이 +20) → 초기값으로 복귀, 차만 있는 KC(+20/−20) → 그대로
    na_, nb_, _, _ = project(np.array([0, 1]), np.array([170., 170.]), np.array([0, 1]), np.array([170., 130.]), 2, 300.0, 0.0, 300.0)
    chk("공통 모드 제거·차 보존", np.allclose([na_[0], nb_[0]], [150, 150]) and np.allclose([na_[1], nb_[1]], [170, 130]), str((na_, nb_)))
    # 7) 큰 무작위 배열: 합 정확·경계 안·차(a 평균 − b 평균) 보존(경계 안 KC)
    rs = np.random.RandomState(0)
    n = 300
    pa = np.sort(rs.randint(0, n, 30000)); pb = np.sort(rs.randint(0, n, 30000))
    ga = np.clip(150 + rs.normal(0, 40, pa.size) + 30 * (pa % 3 == 0), 0, 300); gb = np.clip(150 + rs.normal(0, 40, pb.size) + 25, 0, 300)
    na_, nb_, d0, d1 = project(pa, ga, pb, gb, n, 300.0, 0.0, 300.0)
    s, both = kc_sums(pa, na_, pb, nb_, n)
    chk("무작위: 합 300(1e-6)·경계 안", np.max(np.abs(s[both] - 300)) < 1e-6 and na_.min() >= 0 and nb_.max() <= 300, "%.2e" % np.max(np.abs(s[both] - 300)))
    # 8) (E201) 경로 평균: 경계에 안 닿으면 같은 값을 뺌 — (160,170,180) → 평균 150 이면 s = 20 → (140,150,160)
    x, d0, d1 = center_path(np.array([160., 170., 180.]), 150.0, 0.0, 300.0)
    chk("경로 평균 1차 해(s=20)", np.allclose(x, [140, 150, 160]) and np.isclose(d0, 20) and d1 < 1e-9, str((x, d0, d1)))
    # 9) 하한 자르기: (0, 0, 300, 300, 300) 평균 180 → 목표 150: s 를 빼면 0 은 그대로 → 3(300 − s)/5 = 150 → s = 50 → (0,0,250,250,250)
    x, d0, d1 = center_path(np.array([0., 0., 300., 300., 300.]), 150.0, 0.0, 300.0)
    chk("경로 평균 자르기 포함(s=50)", np.allclose(x, [0, 0, 250, 250, 250]) and d1 < 1e-9, str((x, d1)))
    # 10) 상한 쪽: (300, 300, 0, 100) 평균 175 → 목표 150 — 0 은 그대로, 나머지 (300−s)·2 + (100−s) = 600 → s = 100/3
    x, _, d1 = center_path(np.array([300., 300., 0., 100.]), 150.0, 0.0, 300.0)
    chk("경로 평균 셋 이동(s=100/3)", np.allclose(x, [300 - 100 / 3, 300 - 100 / 3, 0, 100 - 100 / 3]) and d1 < 1e-9, str(x))
    chk("경로 평균 이미 목표면 불변", np.allclose(center_path(np.array([100., 200.]), 150.0, 0.0, 300.0)[0], [100, 200]))
    for nm, fn in (("pre·g 크기 불일치 거부", lambda: project(pa, ga[:-1], pb, gb, n, 300.0, 0.0, 300.0)),
                   ("KC 범위 밖 거부", lambda: project(pa, ga, pb, gb, n - 1, 300.0, 0.0, 300.0)),
                   ("경로 목표 경계 밖 거부", lambda: center_path(np.array([1.0]), 400.0, 0.0, 300.0))):
        try:
            fn()
            chk(nm, False)
        except ValueError:
            chk(nm, True)
    print("합성 %d/%d" % tuple(ok))
    return ok[0] == ok[1]


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
