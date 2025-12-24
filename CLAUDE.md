# Genesis Brain - 근본 원리에서 창발하는 인공 뇌

## 핵심 목표

**뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다.**

이것은 심즈가 아니다. 수치의 나열이 아니다.
- "지루함 게이지가 차면 놀이를 한다" (X)
- "Risk가 높으면 회피 행동" (O)

모든 행동에는 **왜?**가 있어야 하고, 그 **왜?**를 계속 파고들면 **하나의 근본 원리**에 도달해야 한다.

---

## 현재 구현 상태 - True FEP v2.1

### 핵심 공식 (True FEP)

```
F = Prediction Error + Complexity
G(a) = Risk + Ambiguity
```

- **Risk** = KL[Q(o|a) || P(o)] - 진짜 KL divergence
  - Q(o|a): 행동 a 후 예측 관측 분포
  - P(o): 선호 관측 분포 (Beta distributions)
- **Ambiguity** = f(transition_std) - 전이 모델 불확실성
  - 휴리스틱 아님: 경험 → 학습 → delta_std 감소 → ambiguity 감소

### P(o) 선호 분포 (Beta distributions)

```
food_proximity:   P(o) = Beta(5, 1)  # 음식 위에 있고 싶음
danger_proximity: P(o) = Beta(1, 5)  # 위험에서 멀리
directions:       P(o) = Uniform     # 방향 선호 없음
```

### 관측 공간 (6차원)

```
observation = [food_proximity, danger_proximity, food_dx, food_dy, danger_dx, danger_dy]
```

---

## v2.1 변경사항 (2024-12-25)

### 1. 브라우저 간섭 해결 (SimulationClock)

**문제**: 브라우저가 /step을 ~20x/초로 호출 → 전이 모델 학습 오염

**해결**: SimulationClock 도입
- Fast-path: lock 전에 캐시 체크 → 중복 요청 즉시 반환
- Time-based throttling: 50ms 간격 (max 20 FPS)
- Learning downsampling: 5 tick마다 학습

### 2. 행동별 Ambiguity (FEP 정의 준수)

**이전 (휴리스틱)**: confidence = sqrt(n)/(sqrt(n)+2), STAY bonus

**현재 (FEP 정의)**: ambiguity = transition_std * 1.5

**핵심**: 경험 → 모델 학습 → delta_std 감소 → ambiguity 감소 (휴리스틱 아님)

---

## v2.0 변경사항 (2024-12-25)

1. P(o)를 확률분포로 변경: Beta distributions
2. Risk = KL divergence: 진짜 KL[Q||P]
3. STAY 패널티 제거: P(o)에 흡수
4. 전이 모델 학습: 물리 Prior + 온라인 학습

---

## 파일 구조

```
backend/
├── genesis/
│   ├── action_selection.py    # G = Risk + Ambiguity
│   ├── preference_distributions.py  # P(o) Beta 분포
│   └── agent.py
├── main_genesis.py            # FastAPI 서버
```

### API 엔드포인트

- POST /step - 한 스텝 실행
- POST /reset - 리셋
- GET /clock - 시뮬레이션 시계 상태

---

## 금지 사항

- 감정 이름을 변수로 사용 (X)
- 심즈식 욕구 게이지 (X)
- 휴리스틱으로 직접 행동 조작 (X)

---

## 다음 단계

- Complexity = KL[Q(s)||P(s)]: 믿음 업데이트 제약
- 환경 테스트 시나리오

---

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다"
