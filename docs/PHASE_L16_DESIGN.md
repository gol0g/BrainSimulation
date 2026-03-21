# Phase L16: Sparse Expansion Layer (Mushroom Body / Dentate Gyrus)

> 날짜: 2026-03-21
> 근거: 초파리 MB KC 패턴 + 포유류 DG sparse expansion
> 목표: 학습 가능 연결을 ~10,000 → ~30,000으로 3배 확장

---

## 문제

400ep 훈련에서 생존율 ~60% 천장. 41개 학습 시냅스 중 행동에 직접 영향 주는 ~20개, 절반 포화.
감각→BG 경로가 population 수준 SPARSE로만 연결 — 개별 뉴런 수준 패턴 분리 불가.

## 구조

```
food_eye(400×2) + good/bad_food(200×4) + IT_Food(200)
        │ SPARSE 0.10 (random, static)
        ▼
   KC_left(1500)  ←→  KC_inh_left(200)   (WTA, ~5% 활성)
   KC_right(1500) ←→  KC_inh_right(200)
        │ SPARSE 0.05 (R-STDP / Anti-Hebbian)
        ▼
   D1_left/right (Go)    ← R-STDP (eta=0.0003, w_max=3.0)
   D2_left/right (NoGo)  ← Anti-Hebb (eta=0.0002)
```

## 뉴런

| Population | 크기 | 타입 | C |
|---|---|---|---|
| kc_left | 1500 | LIF | 30.0 |
| kc_right | 1500 | LIF | 30.0 |
| kc_inhibitory_left | 200 | LIF | 1.0 |
| kc_inhibitory_right | 200 | LIF | 1.0 |
| **합계** | **+3,400** | | |

## 시냅스

### 입력 (static, 8개)
- food_eye L/R → KC L/R: 3.0, SPARSE 0.10
- good_food_eye L/R → KC L/R: 4.0, SPARSE 0.10
- bad_food_eye L/R → KC L/R: 4.0, SPARSE 0.10
- IT_Food → KC L/R: 2.0, SPARSE 0.05

### WTA (static, 4개)
- KC → KC_inh: 5.0, SPARSE 0.05
- KC_inh → KC: -8.0, SPARSE 0.08

### 출력 (learning, 4개)
- KC L → D1 L: R-STDP, init=0.5, eta=0.0003, w_max=3.0
- KC R → D1 R: R-STDP, init=0.5, eta=0.0003, w_max=3.0
- KC L → D2 L: Anti-Hebb, init=0.5, eta=0.0002, w_min=0.05
- KC R → D2 R: Anti-Hebb, init=0.5, eta=0.0002, w_min=0.05

개별 학습 연결: ~30,000 (1500×100×0.05×4)

## 검증 기준

| 지표 | 기준 | L15 baseline |
|------|------|-------------|
| Survival | >40% | 56% |
| Reward Freq | >2.5% | 2.75% |
| KC Sparsity | 3-7% | N/A |
| Pain Death | 0% | 0% |
