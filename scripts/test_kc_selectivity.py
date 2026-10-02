#!/usr/bin/env python3
"""kc_selectivity.py 합성 정답 시험(E138, 조건 1·2 — 새 측정 도구는 답을 아는 자료로 먼저).
실행: python3 scripts/test_kc_selectivity.py  (저장소 루트에서)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "genesis"))
import kc_selectivity as K

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print("%-44s %s %s" % (name, "✓" if cond else "✗", detail))


# 1) 분류: 정답을 심은 KC 8개. n_pres 100, 제시 창 3스텝, 기준선 500스텝.
#   k0 좌선택(좌 100, 우 0) / k1 우선택 / k2 비선택(좌 50 우 50) / k3 지속 발화(기준선 = 제시와 같은 비율 → 유발 0 → 비선택)
#   k4 무활동 / k5 약한 좌 편향(좌 60 우 40 → SI 0.2 → 비선택) / k6 경계(유발 0.03·0.01 → SI 0.5 → 좌선택) / k7 유발 합 ≤ eps(좌 1·우 0 → 0.01 → 비선택)
cL = np.array([100, 0, 50, 30, 0, 60, 3, 1], dtype=float)
cR = np.array([0, 100, 50, 30, 0, 40, 1, 0], dtype=float)
c0 = np.array([0, 0, 0, 50, 0, 0, 0, 0], dtype=float)   # k3: 50/500 스텝 × 3 = 0.3 = r_L = r_R
rL, rR, b, SI, cls = K.classify(cL, cR, c0, 100, 3, 500)
check("분류 정답 [1,2,3,3,0,3,1,3]", list(cls) == [1, 2, 3, 3, 0, 3, 1, 3], str(list(cls)))
check("k3 기준선 b = 0.3, SI 0", abs(b[3] - 0.3) < 1e-12 and SI[3] == 0.0, "b=%.3f SI=%.3f" % (b[3], SI[3]))
check("k5 SI = 0.2", abs(SI[5] - 0.2) < 1e-12, "%.4f" % SI[5])
check("k6 경계 SI = 0.5 → 좌선택", abs(SI[6] - 0.5) < 1e-12 and cls[6] == 1, "%.4f" % SI[6])
# θ 0.7 이면 k6 는 비선택
_, _, _, _, cls7 = K.classify(cL, cR, c0, 100, 3, 500, theta=0.7)
check("θ 0.7 에서 k6 비선택", cls7[6] == 3)

# 2) 희석 지수: 제시 구간 스파이크 중 비선택 몫 = (50+50 + 30+30 + 60+40 + 1+0) / 전체(활동)
sp = cL + cR
want = (sp[2] + sp[3] + sp[5] + sp[7]) / sp[cls != 0].sum()
check("희석 지수", abs(K.dilution(cL, cR, cls) - want) < 1e-12, "%.4f (기대 %.4f)" % (K.dilution(cL, cR, cls), want))

# 3) 여유: 좌선택 KC 가 →motor_r 로 +10, →motor_l 로 0 학습 → m = r_L·10 = 1·10 = 10. 비선택 KC 양쪽 +5 → 0.
dS_r = np.array([10, 0, 5, 5, 0, 0, 0, 0], dtype=float); dS_l = np.array([0, 10, 5, 5, 0, 0, 0, 0], dtype=float)
m = K.margin(rL, rR, dS_r, dS_l)
check("여유: 좌선택 +10, 우선택 +10, 비선택 0", abs(m[0] - 10) < 1e-12 and abs(m[1] - 10) < 1e-12 and abs(m[2]) < 1e-12 and abs(m[3]) < 1e-12, str(m[:4]))
# 반사 방향 학습(좌선택 KC 가 →motor_l 로)이면 음수
check("반사 방향 학습이면 음수", K.margin(rL, rR, np.zeros(8), np.array([10, 0, 0, 0, 0, 0, 0, 0.]))[0] == -10)

# 4) KC 별 합
pre = np.array([0, 0, 1, 2, 2, 2]); w = np.array([1, 2, 3, 4, 5, 6.])
check("per_kc_sum", list(K.per_kc_sum(pre, w, 4)) == [3, 3, 15, 0])

# 5) 이상 가중치: pop 은 집단만으로, sel 은 KC 별 선호
pre = np.array([0, 1, 2, 4, 6, 0])
check("pop kc_l→motor_r = 300", np.all(K.ideal_weights(pre, cls, "l", "r", 300, 150, "pop") == 300))
check("pop kc_l→motor_l = 0", np.all(K.ideal_weights(pre, cls, "l", "l", 300, 150, "pop") == 0))
wr = K.ideal_weights(pre, cls, "l", "r", 300, 150, "sel"); wl = K.ideal_weights(pre, cls, "l", "l", 300, 150, "sel")
check("sel →motor_r: 좌선택 300, 우선택 0, 나머지 150", list(wr) == [300, 0, 150, 150, 300, 300], str(list(wr)))
check("sel →motor_l: 좌선택 0, 우선택 300, 나머지 150", list(wl) == [0, 300, 150, 150, 0, 0], str(list(wl)))
check("sel 출력 float32", wr.dtype == np.float32)

# 6) selonly: 선택 KC 시냅스만 학습값
wlearn = np.array([11, 12, 13, 14, 15, 16.])
so = K.selonly_weights(pre, cls, wlearn, 150)
check("selonly: 선택(0,1,6,0) 학습값, 나머지 150", list(so) == [11, 12, 150, 150, 15, 16], str(list(so)))

# 7) E117 식 집합
check("≥1 스파이크 집합(좌전용, 우전용, 공유, 무반응)", K.sets_ge1(cL, cR) == (2, 1, 4, 1), str(K.sets_ge1(cL, cR)))

print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
