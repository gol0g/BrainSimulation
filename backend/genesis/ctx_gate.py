"""E204: 맥락 게이팅 KC 무리(L4 병목 정리 2026-10-18 4절 G1) — 무리 배정·억제 연결 마스크·무리별 발화 세기(순수 함수, 합성 시험용).

KC 를 쪽마다 맥락 X 무리·Y 무리로 나누고, 맥락 X 집단(zz_ctx2)은 Y 무리 KC 를, 맥락 Y 집단(zz_ctx)은 X 무리 KC 를 직접 억제한다.
그러면 각 무리의 KC→motor 시냅스가 그 맥락 전용 판독 경로가 된다(새 뉴런 집단 없음 — 정적 억제 시냅스만).
난수는 호스트 지역 RandomState 하나를 쪽 l → r 순서로, 쪽마다 배정 → X 집단 마스크 → Y 집단 마스크 순서로 쓴다(재현).
"""
import numpy as np


def check_params(w, p, frac):
    if not (w < 0):
        raise ValueError("ctx_gate_w 는 음수(억제)여야 한다: %r" % (w,))
    if not (0.0 < p <= 1.0):
        raise ValueError("ctx_gate_p 는 (0, 1] 이어야 한다: %r" % (p,))
    if not (0.0 < frac < 1.0):
        raise ValueError("ctx_gate_frac 는 (0, 1) 이어야 한다: %r" % (frac,))


def side_masks(rs, n_kc, n_x, n_y, frac, p):
    """쪽 하나의 배정과 마스크.
    반환 (isx, m_x, m_y): isx = KC 별 맥락 X 무리 여부(bool, n_kc),
    m_x = 맥락 X 집단(n_x 뉴런) → Y 무리 KC 억제 마스크(n_x × n_kc), m_y = 맥락 Y 집단(n_y) → X 무리 KC 억제 마스크(n_y × n_kc)."""
    isx = rs.random_sample(int(n_kc)) < float(frac)
    m_x = (rs.random_sample((int(n_x), int(n_kc))) < float(p)) & (~isx)[None, :]
    m_y = (rs.random_sample((int(n_y), int(n_kc))) < float(p)) & isx[None, :]
    return isx, m_x, m_y


def build_all(seed, n_kc, n_x, n_y, frac, p):
    """두 쪽(l → r) 배정·마스크를 한 난수열로. 반환 {"l": (isx, m_x, m_y), "r": (...)}."""
    rs = np.random.RandomState(int(seed))
    return {sd: side_masks(rs, n_kc, n_x, n_y, frac, p) for sd in ("l", "r")}


def count_by_subset(ids, isx):
    """KC 발화 id 배열 → (X 무리 발화 수, Y 무리 발화 수)."""
    ids = np.asarray(ids, dtype=np.int64).ravel()
    if ids.size == 0:
        return 0, 0
    if ids.min() < 0 or ids.max() >= isx.size:
        raise ValueError("발화 id 범위 밖")
    sx = int(np.sum(isx[ids]))
    return sx, int(ids.size) - sx


def rewire_within(pre, post, gpre, gpost, rs):
    """pre·post: 연결 배열(같은 길이), gpre: 앞 뉴런 무리(bool, 앞 집단 크기), gpost: 뒤 뉴런 무리(bool). 반환 (new_pre, new_post, 옮긴 수).
    gpre[i] == gpost[j] 인 쌍은 그대로, 아닌 쌍은 앞 뉴런을 gpost[j] 무리에서 새로 뽑는다."""
    pre = np.asarray(pre, dtype=np.int64); post = np.asarray(post, dtype=np.int64)
    if pre.shape != post.shape:
        raise ValueError("pre·post 길이 다름")
    if pre.size == 0:
        return pre.copy(), post.copy(), 0
    if pre.min() < 0 or pre.max() >= gpre.size or post.min() < 0 or post.max() >= gpost.size:
        raise ValueError("연결 번호 범위 밖")
    pool = {True: np.nonzero(gpre)[0], False: np.nonzero(~gpre)[0]}
    order = np.argsort(post, kind="stable")
    ps, qs = pre[order], post[order]
    bounds = np.flatnonzero(np.diff(qs)) + 1
    new_pre, new_post, moved = [], [], 0
    for seg_p, seg_q in zip(np.split(ps, bounds), np.split(qs, bounds)):
        j = int(seg_q[0]); gj = bool(gpost[j])
        keep = seg_p[gpre[seg_p] == gj]
        if np.unique(keep).size != keep.size:
            raise ValueError("뒤 뉴런 %d 의 원래 연결에 중복 쌍" % j)
        r = seg_p.size - keep.size
        if r:
            cand = np.setdiff1d(pool[gj], keep, assume_unique=False)
            if cand.size < r:
                raise ValueError("뒤 뉴런 %d: 같은 무리 후보 %d < 옮길 수 %d" % (j, cand.size, r))
            pick = rs.choice(cand, size=r, replace=False)
            keep = np.concatenate([keep, pick])
            moved += r
        new_pre.append(np.sort(keep)); new_post.append(np.full(keep.size, j, dtype=np.int64))
    return np.concatenate(new_pre), np.concatenate(new_post), moved


def active_inactive(counts_l, counts_r, ctx):
    """두 쪽의 (X 무리, Y 무리) 발화 수와 맥락('X'·'Y') → (활성 무리 합, 비활성 무리 합)."""
    if ctx not in ("X", "Y"):
        raise ValueError("맥락은 X 또는 Y")
    a = (counts_l[0] + counts_r[0]) if ctx == "X" else (counts_l[1] + counts_r[1])
    b = (counts_l[1] + counts_r[1]) if ctx == "X" else (counts_l[0] + counts_r[0])
    return int(a), int(b)
