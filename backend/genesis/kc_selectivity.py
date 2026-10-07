"""E138: 전체 모델 KC 선택성(발화 수 기준)·희석 지수·학습 여유 분해·이상 가중치 생성 — 순수 함수(합성 정답 검사용, GeNN 불사용).

정의(logs/E138/criteria_fixed.txt 와 같다):
  r_L, r_R = 제시당 스파이크(좌/우 제시 각 n_pres 회, 제시 창 = steps_per_pres 처리 스텝)
  b = 기준선 처리 스텝당 스파이크 × steps_per_pres (제시 창 길이로 환산)
  e_L = max(r_L − b, 0), e_R = max(r_R − b, 0), SI = (e_L − e_R)/(e_L + e_R)
  분류: 1 좌선택(e_L + e_R > eps, SI ≥ θ) / 2 우선택(SI ≤ −θ) / 3 비선택(제시 구간 스파이크 > 0, 선택 아님) / 0 무활동
정답 motor: 좌 자극(good=왼쪽) → motor_r, 우 자극 → motor_l (반사 0 교차 = 정답, K57).
"""
import numpy as np

CLS_NONE, CLS_L, CLS_R, CLS_NS = 0, 1, 2, 3


def classify(cL, cR, c0, n_pres, steps_per_pres, base_steps, theta=0.5, eps=0.02):
    cL = np.asarray(cL, dtype=np.float64); cR = np.asarray(cR, dtype=np.float64); c0 = np.asarray(c0, dtype=np.float64)
    rL = cL / n_pres; rR = cR / n_pres
    b = c0 / base_steps * steps_per_pres
    eL = np.maximum(rL - b, 0.0); eR = np.maximum(rR - b, 0.0)
    tot = eL + eR
    # 소수 9자리 반올림: 정수 계수에서 수학적으로 정확히 θ 인 경계(예: 좌 3·우 1 → 0.5)가 부동소수 오차로 0.4999… 가 되지 않게(합성 시험이 발견)
    SI = np.round(np.where(tot > 0, (eL - eR) / np.maximum(tot, 1e-12), 0.0), 9)
    # eps 비교도 같은 반올림(E138 독립 대조가 발견: 뇌 10 지속 발화 KC — 좌 292·우 308·기준선 1020/1000스텝 → 유발 합이 정확히 0.02 인데
    # 부동소수로 0.020000000000000018 > 0.02 가 되어 우선택으로 오분류. 정수 산술 대조로 확인)
    resp = np.round(tot, 9) > eps
    cls = np.zeros(cL.size, dtype=np.int64)
    cls[(cL + cR) > 0] = CLS_NS
    cls[resp & (SI >= theta)] = CLS_L
    cls[resp & (SI <= -theta)] = CLS_R
    return rL, rR, b, SI, cls


def dilution(cL, cR, cls):
    """제시 구간(좌+우) 스파이크 중 비선택 KC 몫."""
    sp = np.asarray(cL, dtype=np.float64) + np.asarray(cR, dtype=np.float64)
    tot = sp[cls != CLS_NONE].sum()
    return float(sp[cls == CLS_NS].sum() / tot) if tot > 0 else float("nan")


def per_kc_sum(pre, w, n_k):
    """시냅스 값(w)을 전시냅스 KC 별로 합한다."""
    return np.bincount(np.asarray(pre, dtype=np.int64), weights=np.asarray(w, dtype=np.float64), minlength=n_k)[:n_k]


def margin(rL, rR, dS_r, dS_l):
    """KC 별 학습 여유 기여: m = r_L·(ΔS→motor_r − ΔS→motor_l) + r_R·(ΔS→motor_l − ΔS→motor_r). 양수 = 정답(교차) 방향."""
    return rL * (dS_r - dS_l) + rR * (dS_l - dS_r)


def ideal_weights(pre, cls, pop, motor, wmax, init, mode):
    """KC→motor 한 집단(pop: 'l'|'r' KC 쪽, motor: 'l'|'r')의 시냅스 값.
    mode 'pop': 집단 교차(pop ≠ motor → wmax, 같으면 0) — E119 P2 의 rev 와 같다.
    mode 'sel': 선택 KC 만 선호 쪽 교차(좌선택 → motor_r wmax·motor_l 0, 우선택 → motor_l wmax·motor_r 0), 나머지 init."""
    pre = np.asarray(pre, dtype=np.int64)
    if mode == "pop":
        return np.full(pre.size, (wmax if pop != motor else 0.0), dtype=np.float32)
    if mode != "sel":
        raise ValueError(mode)
    c = cls[pre]
    out = np.full(pre.size, init, dtype=np.float32)
    if motor == "r":
        out[c == CLS_L] = wmax; out[c == CLS_R] = 0.0
    else:
        out[c == CLS_L] = 0.0; out[c == CLS_R] = wmax
    return out


def selonly_weights(pre, cls, w_learned, init):
    """선택 KC(좌·우선택)의 시냅스는 학습값, 나머지는 init."""
    pre = np.asarray(pre, dtype=np.int64)
    out = np.asarray(w_learned, dtype=np.float32).copy()
    out[~np.isin(cls[pre], (CLS_L, CLS_R))] = np.float32(init)
    return out


def sets_ge1(cL, cR):
    """E117 식(≥1 스파이크) 집합 수: (좌전용, 우전용, 공유, 무반응)."""
    aL = np.asarray(cL) > 0; aR = np.asarray(cR) > 0
    return int((aL & ~aR).sum()), int((aR & ~aL).sum()), int((aL & aR).sum()), int((~aL & ~aR).sum())


def overlap_stats(cnt_a, cnt_b, c0, n_pres, steps, base_steps, thr=0.5):
    """E149: 두 자극(a·b)의 KC 반응 겹침. cnt_* = 자극별 KC 발화 수 합(n_pres 제시 × steps 스텝), c0 = 기준선 발화 수 합(base_steps 스텝).
    유발 e = cnt/n_pres − steps·c0/base_steps (제시당 기준선 뺀 발화 수). 반응 = e ≥ thr.
    반환 (반응 수 a, 반응 수 b, 자카드, 유발 벡터 코사인(음수 유발은 0으로))."""
    base = steps * np.asarray(c0, dtype=np.float64) / max(base_steps, 1)
    ea = np.asarray(cnt_a, dtype=np.float64) / max(n_pres, 1) - base
    eb = np.asarray(cnt_b, dtype=np.float64) / max(n_pres, 1) - base
    ra, rb = ea >= thr, eb >= thr
    union = int((ra | rb).sum())
    jac = float((ra & rb).sum() / union) if union else float("nan")
    pa, pb = np.clip(ea, 0, None), np.clip(eb, 0, None)
    na, nb = float(np.linalg.norm(pa)), float(np.linalg.norm(pb))
    cos = float(pa @ pb / (na * nb)) if na > 0 and nb > 0 else float("nan")
    return int(ra.sum()), int(rb.sum()), jac, cos


def overlap3_stats(cnt_g, cnt_b, cnt_f, c0, n_pres, steps, base_steps, thr=0.5):
    """E152: good·bad·먹이 단독 세 자극의 KC 반응 집합(E149 정의 그대로: 유발 = 제시당 발화 − 기준선, 반응 = 유발 ≥ thr).
    겹침 O = G∩B 중 먹이 단독에도 반응하는 몫을 센다. 반환 dict:
    nG·nB·nF(반응 수), nO = |G∩B|, nOF = |G∩B∩F|, nFin = |F∩(G∪B)|, jac_gb = |G∩B|/|G∪B|(합집합 0 이면 nan)."""
    base = steps * np.asarray(c0, dtype=np.float64) / max(base_steps, 1)
    rg = np.asarray(cnt_g, dtype=np.float64) / max(n_pres, 1) - base >= thr
    rb = np.asarray(cnt_b, dtype=np.float64) / max(n_pres, 1) - base >= thr
    rf = np.asarray(cnt_f, dtype=np.float64) / max(n_pres, 1) - base >= thr
    o = rg & rb
    union = int((rg | rb).sum())
    return {"nG": int(rg.sum()), "nB": int(rb.sum()), "nF": int(rf.sum()), "nO": int(o.sum()), "nOF": int((o & rf).sum()),
            "nFin": int((rf & (rg | rb)).sum()), "jac_gb": float(o.sum() / union) if union else float("nan")}


def type_redistribute(g_good, g_bad, post_good, post_bad, counts, presented, eta, s0):
    """E153: 종류 입력 합 보존 헤브 재분배(호스트 적용, 최소 회로 K69 형). 제시된 종류(presented = 'good' | 'bad')의
    시냅스 가중치에 (1 + eta·c[post]) 를 곱하고(c = 그 제시 동안 KC 발화 수), KC 별 종류 입력 합(good + bad)을 s0(처음 값)로 되돌린다.
    발화하지 않은 KC(c=0)는 곱 1 → 합 그대로 → 변화 없음. 균형(두 종류에 같은 발화)이면 번갈아 같은 배율 → 몫 불변.
    반환 (새 g_good, 새 g_bad) — float64 로 계산(장치에는 호출자가 float32 로 싣는다)."""
    gg = np.asarray(g_good, dtype=np.float64).copy()
    gb = np.asarray(g_bad, dtype=np.float64).copy()
    c = np.asarray(counts, dtype=np.float64)
    if presented == "good":
        gg *= 1.0 + eta * c[post_good]
    elif presented == "bad":
        gb *= 1.0 + eta * c[post_bad]
    else:
        raise ValueError("presented 는 good|bad: %r" % presented)
    s0 = np.asarray(s0, dtype=np.float64)
    n = s0.size
    s = np.bincount(post_good, weights=gg, minlength=n)[:n] + np.bincount(post_bad, weights=gb, minlength=n)[:n]
    f = np.ones(n)
    m = s > 0
    f[m] = s0[m] / s[m]
    return gg * f[post_good], gb * f[post_bad]


def type_share(g_good, g_bad, post_good, post_bad, n):
    """E153: KC 별 종류 입력 중 good 몫 = Σgood / (Σgood + Σbad). 종류 입력이 없는(합 0) KC 는 nan."""
    sg = np.bincount(post_good, weights=np.asarray(g_good, dtype=np.float64), minlength=n)[:n]
    sb = np.bincount(post_bad, weights=np.asarray(g_bad, dtype=np.float64), minlength=n)[:n]
    tot = sg + sb
    out = np.full(n, np.nan)
    m = tot > 0
    out[m] = sg[m] / tot[m]
    return out
