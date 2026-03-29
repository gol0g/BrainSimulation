# 개념 형성 실행 계획 (C0-C5)

> 작성일: 2026-03-29
> 근거: GPT 자문 + 최신 논문 11편 리서치
> 선행 조건: 27,910 뉴런, KC 3000, 9개 다중감각 입력, 61% 생존 (400ep 검증)

---

## Phase C0: 개념 검증 계측 프레임워크

### 목적
"개념이 생겼다"를 수치로 판정하는 테스트 + 지표. Phase 검증(Survival >40%)과 동일한 엄격함.

### 구현물

#### 1. 평가 스크립트 (`evaluate_concepts.py`)

훈련된 모델을 로드 → 특수 테스트 시나리오 실행 → 점수 산출. 훈련과 분리된 **별도 평가**.

```python
# 사용법
python evaluate_concepts.py --load-weights brain_kc3000_b8.npz --test all
# 개별 테스트
python evaluate_concepts.py --load-weights ... --test call_semantics
```

#### 2. 4가지 테스트

**Test 1: 다중감각 범주화 (C1 검증용)**
- 환경: 시각 동일 음식 3종, 구분 단서 = 소리/장소/call
- 100 trial, 각 trial에 랜덤 음식 배치
- 측정: 에이전트가 good food 방향으로 가는 비율
- 기준: >70% (random=33%)

**Test 2: 범주 일반화 (C2 검증용)**
- 학습 시 노출 안 된 변형 자극 (감각 값 ±20% 편향)
- 100 trial
- 측정: 변형 자극에도 올바른 접근/회피 비율
- 기준: >50% (random baseline 대비)

**Test 3: Call Semantics (C3 검증용)**
- 시각 단서 제거 (food_rays = 0), NPC food call만 발생
- 100 trial, call 방향 랜덤
- 측정: call 방향으로 이동한 비율
- 기준: >60% (random=50%)

**Test 4: 맥락 의존 (C4 검증용)**
- 동일 음식이 Rich Zone에서는 +25 에너지, 외부에서는 -5 에너지
- 100 trial
- 측정: 장소에 따른 접근/회피 차이
- 기준: >30% 행동 차이

#### 3. KC 패턴 분리 지표 (논문 #11: Information Theory)
- Mutual Information: KC 활성 패턴과 음식 유형 간 MI
- Sparsity: KC 활성률 (target 3-7%)
- Selectivity: KC 뉴런별 음식 유형 선호도

#### 4. 기존 지표와의 통합
```
Phase 검증:     Survival >40%, Reward >2.5%, Pain <15%
개념 검증 (C0): Test1 >70%, Test2 >50%, Test3 >60%, Test4 >30%
```

### 구현 순서
1. `evaluate_concepts.py` 스켈레톤 (ForagerBrain 로드 + 평가 루프)
2. Test 3 (Call Semantics) 먼저 — 현재 환경에서 즉시 테스트 가능
3. Test 1은 C1 환경 변경 후
4. Test 2, 4는 C2, C4 구현 후

---

## Phase C0.5: SWR Selective Replay (Science 2024 차용)

### 근거
Yang et al. (Science 2024): 보상 시점 SWR이 경험을 태깅 → 태깅된 것만 우선 replay.

### 현재 코드
- `experience_buffer`: list of (pos_x, pos_y, food_type, step, reward), max 50
- `replay_swr()`: 무작위 5회 replay
- Hebbian `learn_food_location()` 호출

### 변경
1. experience_buffer에 `tagged: bool` 필드 추가
2. 음식 섭취 시 (reward > 0): tagged = True
3. 나쁜 음식 섭취 시 (reward < 0): tagged = True, negative_tag = True
4. 배경 경험 (매 100스텝): tagged = False
5. `replay_swr()`: 80% tagged 경험 선택, 20% random
6. negative_tag 경험은 learn_food_location()에서 **anti-learning** (가중치 약화)

### 검증
- 20ep: Hippo avg_w 수렴 속도 비교 (현재 vs selective)
- Selectivity 변화
- 코드 변경: ~30행

---

## Phase C1 준비: 감각 모호성 환경 설계

### 근거
GPT 자문: "겉보기 비슷하지만 의미는 다른 자극이 필요"
논문 #3 (Nature 2024): KC multi-compartment STM/LTM

### 환경 변경 (forager_gym.py)
현재: good_food (green), bad_food (purple) — 시각적으로 구분 가능
변경: **3종 음식, 시각 동일, 다중감각 단서로만 구분**

```
Type A (good):  시각=노랑, 소리=고음,  장소=Rich Zone 1  → +25 에너지
Type B (bad):   시각=노랑, 소리=저음,  장소=Rich Zone 2  → -5 에너지
Type C (social): 시각=노랑, NPC가 먹음, 장소=변동         → +15 에너지
```

초기 단계에서는 시각 구분 유지 (green/purple) + **추가 단서 도입**부터 시작.
C1 완성 시 시각 단서를 점진적으로 제거.

### 뇌 변경
- KC multi-compartment (논문 #3): KC_gamma(STM, 1500) + KC_alpha(LTM, 1500)
- 또는 기존 KC 3000을 역할 분화 없이 유지하고 환경만 변경

### 순서
1. 환경에 "소리 단서가 다른 음식" 추가 (sound_food_type)
2. 기존 학습으로 소리 단서 구분이 되는지 테스트 (Test 3)
3. 안 되면 KC multi-compartment 구현

---

## 실행 순서 (구현 우선순위)

```
Step 1: evaluate_concepts.py 스켈레톤 + Test 3 (Call Semantics)
        → 현재 상태에서 "소리에 반응하는가" baseline 측정

Step 2: SWR Selective Replay 구현
        → experience_buffer tagged, replay 우선순위
        → 20ep 검증

Step 3: Test 3 재측정 (SWR 개선 후)
        → call semantics 점수 변화 확인

Step 4: 결과 분석 → C1 환경 설계 확정
```

---

## 성공 기준

| 단계 | 산출물 | 판정 기준 |
|------|--------|----------|
| C0 | evaluate_concepts.py | Test 3 baseline 측정 완료 |
| C0.5 | SWR selective replay | Hippo 수렴 속도 ≥1.5x |
| C0→C1 | Test 3 점수 | >60%이면 C1으로 진행 |
|  | | <40%이면 청각→BG 경로 강화 필요 |

---

## 리스크

| 리스크 | 대응 |
|--------|------|
| Test 3 baseline이 50% (random) | 청각 입력이 BG에 영향 못 미침 → A1→KC 가중치 상향 |
| SWR selective가 성능 하락 | tagged 비율 80→60%으로 조정, random 비율 증가 |
| evaluate_concepts.py가 기존 훈련과 충돌 | --no-learning 모드로 실행 (가중치 변경 없음) |
