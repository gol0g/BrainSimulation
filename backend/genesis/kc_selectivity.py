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
    cls = np.zeros(cL.size, dtype=np.int64)
    cls[(cL + cR) > 0] = CLS_NS
    cls[(tot > eps) & (SI >= theta)] = CLS_L
    cls[(tot > eps) & (SI <= -theta)] = CLS_R
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
