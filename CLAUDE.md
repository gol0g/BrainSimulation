# Genesis Brain - 근본 원리에서 창발하는 인공 뇌

## 핵심 목표

**뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다.**

이것은 심즈가 아니다. 수치의 나열이 아니다.
- "지루함 게이지가 차면 놀이를 한다" (X)
- "Risk가 높으면 회피 행동" (O)

모든 행동에는 **왜?**가 있어야 하고, 그 **왜?**를 계속 파고들면 **하나의 근본 원리**에 도달해야 한다.

---

## 현재 구현 상태 - True FEP v3.3.1

### 핵심 공식 (True FEP)

**지각 (현재 스텝 믿음 업데이트)**:
```
F_t = Accuracy + Complexity_perception
    = -log P(o_t|s) + KL[Q(s_t) || P(s_t)]
```

**행동 선택 (미래 예측)**:
```
G(a) = Expected_Risk + Expected_Ambiguity + Expected_Complexity

여기서:
- Expected_Risk(a) = E[KL[Q(o|s',a) || P(o)]]
  = "이 행동 후 관측이 선호에서 얼마나 벗어날 것인가"

- Expected_Ambiguity(a) = "전이/관측이 얼마나 불확실한가"
  - 이론: E[H[P(o|s')]] + E[H[P(s'|s,a)]]
  - 현재 구현: k × transition_std (단순화)

- Expected_Complexity(a) = E[KL[Q(s_{t+1}|a) || P(s_{t+1})]]
  = "이 행동 후 믿음이 선호 상태에서 얼마나 벗어날 것인가"
```

**핵심 구분**:
- F의 Complexity: 현재 믿음 업데이트 비용 (지각)
- G의 Expected_Complexity: 미래 믿음 이탈 예측 (행동 선택)
- 둘은 다른 시점, 다른 목적 → 이중 계산 아님

### P(o) 선호 분포 (Beta distributions) - v2.5 내부 항상성

```
# INTEROCEPTION (내부 상태) - 기본 활성화 (lambda=1.0)
energy:           P(o) = Beta(3, 2)  # ~0.6 선호 (항상성 목표)
pain:             P(o) = Beta(1, 5)  # 0 선호 (통증 없음)

# EXTEROCEPTION (외부 세계) - lambda로 조절 가능
food_proximity:   P(o) = Beta(5, 1)  # 음식 위에 있고 싶음
danger_proximity: P(o) = Beta(1, 5)  # 위험에서 멀리
directions:       P(o) = Uniform     # 방향 선호 없음
```

### 관측 공간 (8차원) - v2.5

```
observation = [
    # EXTEROCEPTION (0-5)
    food_proximity, danger_proximity,
    food_dx, food_dy, danger_dx, danger_dy,
    # INTEROCEPTION (6-7) - v2.5
    energy, pain
]
```

### 구현 세부사항

**Ambiguity 현재 구현**:
```python
ambiguity = transition_std * 1.5  # 전이 불확실성만
```
- 현재 환경이 단순해서 OK
- 확장 시: `ambiguity = w1*transition_uncertainty + w2*observation_uncertainty`

**Precision 안정성**:
- EMA(Exponential Moving Average)로 오차 스무딩
- min/max clamp (0.1 ~ 5.0)
- Goal precision = preference sharpness (메타-정밀도)

---

## v3.3.1 변경사항 (2024-12-26)

### Context-weighted Transitions - 안정화

**v3.3 문제점**:
- delta_ctx 스케일이 physics보다 커서 alpha 조금만 올려도 지배
- DOWN 같은 특정 행동이 internal state 예측에서 과하게 유리
- Q(c)가 one-hot으로 굳으면 더 빠르게 붕괴
- alpha=1.0에서 {2:15} 같은 완전 붕괴 발생

**v3.3.1 개선**:
1. **delta_ctx clamp**: [-0.05, +0.05]로 스케일 제한
2. **alpha 분리**: external(0.2) / internal(0.1) 별도 설정
3. **신뢰도 기반 alpha_eff**: context entropy 높으면 alpha 감소

```python
# 신뢰도 기반 alpha 조절
entropy = -Σ Q(k) * log(Q(k))
entropy_ratio = entropy / log(K)  # 0=확실, 1=불확실
confidence_mult = 1.0 - entropy_ratio

alpha_eff = alpha_base * confidence_mult
```

**테스트 결과**:
```
v3.3 (alpha=0.5):  {1:8, 2:7}      # 집중
v3.3.1 (alpha=0.4): {1:12, 2:6, 3:5, 4:7}  # 분산 (max 40%)

신뢰도 조절 효과:
  base alpha=0.4 → effective alpha=0.09 (entropy로 감소)
```

**API**:
```bash
POST /hierarchy/alpha?alpha_ext=0.2&alpha_int=0.1&clamp=0.05&use_confidence=true
```

---

## v3.3 변경사항 (2024-12-26)

### Hierarchical Models - Context-weighted Transitions in G Calculation

**핵심 철학**: Slow layer가 배운 "세상 규칙"을 Fast layer의 행동 선택에 반영

**v3.3.1에서 안정화됨** (위 섹션 참조)

---

## v3.2 변경사항 (2024-12-26)

### Hierarchical Models - Context별 전이 모델 학습

**핵심 철학**: Context는 "관측 패턴"만이 아니라 **"세상이 어떻게 작동하는가"**를 학습

**v3.2 개선**:
- **Context별 전이 모델**: 각 context k가 "행동 → 관측 변화" 예측을 학습
- **전이 예측 정확도**: context likelihood에 전이 예측 오차 40% 반영
- **세상 모델 분화**: 지배적 context가 더 정확한 전이 예측 획득

```python
# v3.2: Context별 전이 학습
context_transition_delta[k][action] = expected_obs_delta  # 학습됨

# Context likelihood = 관측 설명력 + 전이 예측 정확도
log_lik[k] = 0.6 * obs_stats_likelihood + 0.4 * (-trans_error)
```

**테스트 결과** (2748 steps):
```
Transition errors: [0.159, 0.162, 0.103, 0.105]
                    C0     C1     C2↓    C3
Dominant: C2 (lowest error = best world model)
```

---

## v3.1 변경사항 (2024-12-25)

### Hierarchical Models - Slow Layer Context Tracking + Learning

**핵심 설계 원칙**:
1. **내부 표현**: Pure indices (`c ∈ {0..K-1}`) - 라벨 없음
2. **사후 해석**: UI/디버깅에서만 행동 패턴 기반 라벨 표시 (모델에 피드백 안 함)
3. **FEP 일관성**: Context도 Free Energy 최소화로 추론
4. **Precision 조절만**: 행동 직접 지시 안 함, 메타 파라미터만 조절

**v3.1 개선**:
- **expected[k] 학습**: 각 context가 자기 특징을 EMA로 스스로 배움
- **modulation 범위 축소**: 0.85~1.15 (숨은 손 방지)
- **context 전환 실증**: 에이전트가 다양한 상황에서 context 전환 확인

**2층 구조**:
```
Slow Layer (Context)
  - 느린 시간 스케일 (매 10 step 업데이트)
  - Context belief Q(c) 유지 (K=4)
  - 관측 통계의 "설명력"으로 추론
  - expected[k] 학습으로 자기 특징 발견
  ↓ Precision Modulation (범위 0.85~1.15)
Fast Layer (Action Selection)
  - 빠른 시간 스케일 (매 step)
  - G = Risk + Ambiguity + Complexity
  - Slow layer에서 modulation 받음
```

**expected[k] 학습 (v3.1)**:
```python
# 각 context k가 자기 특징을 스스로 배움
# Q(k)로 가중된 EMA 업데이트
effective_lr = expectation_lr * Q_context[k]
expected[k] = (1-lr) * expected[k] + lr * current_obs_stats
```

**테스트 결과** (2800+ steps):
```
Context belief: [0.006, 0.005, 0.988, 0.001]
Dominant: Context 2 (switched from Context 3!)

Learned expectations:
  C0: energy=0.49  (저에너지 상황)
  C1: danger=0.46  (위험 상황)
  C2: energy=0.96  (고에너지 - 현재 dominant)
  C3: complexity=1.22  (고복잡도 상황)
```

**Precision Modulation** (범위 축소):
```python
modulation = {
    'goal_precision_mult': 0.85~1.15,  # 숨은 손 방지
    'sensory_mod': 0.9~1.1,            # 약한 영향만
    'internal_pref_weight': 0.8~1.0,   # 내부 선호 유지
    'rollout_budget': 0.0~1.0,         # 상상 확률
}
```

**API**:
```bash
POST /hierarchy/enable?K=4&update_interval=10  # 계층적 처리 활성화
POST /hierarchy/disable                         # 비활성화
GET  /hierarchy/status                          # 현재 상태 (context_expectations 포함)
```

**구현 파일**:
- `hierarchy.py`: SlowLayerState, SlowLayerInference, HierarchicalController
- `action_selection.py`: compute_G()에 modulation 적용
- `main_genesis.py`: API 엔드포인트

---

## v2.5 변경사항 (2024-12-25)

### 내부 항상성 기반 P(o) - Interoception

**핵심 전환**: 외부 선호(food/danger proximity) → 내부 선호(energy/pain)

**테스트 결과** (100 steps × 3 runs 평균):

| λ | Phase | Avg Food | Danger | 설명 |
|---|-------|----------|--------|------|
| 0.0 | Phase 0 | 16.7 | 0 | 외부 선호만 |
| 0.3 | Phase 1 | 9.3 | 0 | 70% 외부 + 30% 내부 |
| 0.6 | Phase 1 | 17.7 | 0 | 40% 외부 + 60% 내부 |
| 1.0 | Phase 2 | **31.0** | 0 | 내부 선호만 |

**결과**: λ=1.0(내부 선호만)이 λ=0.0(외부 선호만)보다 **2배 더 많은 음식 섭취**

**왜 내부 선호가 더 robust한가**:

| 외부 선호 (λ=0) | 내부 선호 (λ=1) |
|-----------------|-----------------|
| "음식 가까이 있고 싶다" | "에너지 ~0.6 유지하고 싶다" (항상성) |
| → 목표: 관측 상태 | → 목표: 내부 안정 |
| → 음식 근처에서 만족 | → 에너지 감소 시 행동 유발 |
| → 정책이 위치 의존적 | → 정책이 더 일반적 (robust) |

**핵심 통찰**:
- 항상성 선호 P(energy) = Beta(3,2) → 모드 ~0.67 (적당한 에너지)
- 에너지는 시간에 따라 자연 감소 → 계속 음식 섭취가 유도됨
- "극대화"가 아니라 "안정화"를 추구하지만, 감소 다이나믹으로 인해 지속적 행동 필요

**기본값**: `internal_pref_weight = 1.0` (Phase 2가 기본)

**API**:
```bash
POST /preference/internal_weight?weight=0.5  # 외부/내부 혼합
GET  /preference/status                       # 현재 상태 확인
```

**관측 공간 확장**: 6차원 → 8차원
```python
obs[6] = energy  # 현재 에너지 레벨 (0.0~1.0)
obs[7] = pain    # 현재 고통 레벨 (0.0~1.0)
```

**학습에 대한 정확한 이해**:
- 환경: "food 먹으면 energy=1.0" 하드코딩 (인과 구조)
- 전이 모델: "행동 a → 관측 delta" 학습 (action-observation 상관관계)
- 에이전트가 발견하는 것: "어떤 행동이 energy를 유지시키는가"
- 발견하지 않는 것: "food가 energy를 올린다"는 인과 메커니즘

즉, 의미의 발견은 "food=좋다"가 아니라 "이 행동 패턴이 내부 안정을 유지한다"

---

## v2.4 변경사항 (2024-12-25)

### Temporal Depth - 다중 시간 스케일 상상 (n-step rollout)

**핵심 개념**:
```
G(a) = Σ_{t=1}^{H} γ^{t-1} * G_t(a)

여기서:
- H = rollout horizon (몇 스텝 앞까지 상상)
- γ = discount factor (미래 가치 할인, 0.9)
- G_t = t스텝 후의 expected free energy
```

**왜 필요한가**:
- 1-step만 보면 근시안적 결정 (당장 위험 피하지만 음식 못 찾음)
- n-step 보면 "지금은 나빠도 나중에 좋아지는" 경로 발견
- 예: 위험을 돌아가는 경로 (단기 손해, 장기 이득)

**비교 결과** (30 steps each):
```
CONFLICT 시나리오:
  OFF (1-step): food=2, dominant={risk_avoidance: 90%}
  ON  (3-step): food=6, dominant={risk_avoidance: 70%, ambiguity_reduction: 20%}
  → +4 음식, 더 탐색적

TEMPTATION 시나리오:
  OFF (1-step): food=0, dominant={risk_avoidance: 100%}
  ON  (3-step): food=7, dominant={risk_avoidance: 63%, ambiguity_reduction: 30%}
  → +7 음식, 위험 0 유지
```

**구현 파일**:
- `temporal.py`: TemporalPlanner 클래스, RolloutResult
- `action_selection.py`: enable_rollout(), select_action_with_rollout()
- `main_genesis.py`: /temporal/enable, /temporal/disable, /temporal/status

**API**:
```bash
POST /temporal/enable?horizon=3&discount=0.9  # 3-step rollout 활성화
POST /temporal/disable                         # 1-step으로 복귀
GET  /temporal/status                          # 현재 상태
```

---

## v2.3 변경사항 (2024-12-25)

### Precision Learning - 동적 주의 조절

**핵심 개념**: Precision = 1/variance = "이 정보를 얼마나 신뢰할 것인가"

```python
# 예측 오차 작으면 → precision ↑ → 더 신뢰
# 예측 오차 크면 → precision ↓ → 덜 신뢰

sensory_precision[i] += lr * (1/(error + 0.1) - sensory_precision[i])
```

**세 가지 Precision**:
- **Sensory Precision**: 각 관측 차원별 신뢰도
  - 예측이 정확한 차원 → precision ↑ → Risk 계산에서 가중치 ↑
- **Transition Precision**: 각 행동별 전이 모델 신뢰도
  - 잘 예측되는 행동 → precision ↑ → Ambiguity에 더 민감
- **Goal Precision** (= Preference Sharpness): P(o) 선호 분포의 온도/확신도
  - 높으면 → P(o) 분포가 뾰족 → 선호에 확신 → 덜 탐색적
  - 낮으면 → P(o) 분포가 평평 → 선호에 불확신 → 더 탐색적
  - **핵심**: 목표를 바꾸는 게 아니라, 선호의 확신도를 조절하는 메타-정밀도

**구현 파일**:
- `precision.py`: PrecisionLearner 클래스
- `action_selection.py`: compute_G()에서 precision 가중치 적용
- `preference_distributions.py`: compute_risk()에 precision_weights 파라미터

**관찰자 해석**:
- Sensory precision 높음 → "주의 집중"
- Goal precision 높음 → "동기 부여"
- Goal precision 낮음 → "탐색적/개방적"

---

## v2.2 변경사항 (2024-12-25)

### Complexity 구현 - 믿음 업데이트 제약

**핵심 개념**: G(a) = Risk + Ambiguity + Complexity

```
Complexity = KL[Q(s'|a) || P(s')]
           = "이 행동을 하면 믿음이 선호 상태에서 얼마나 벗어날 것인가"
```

**P(s) 상태 선호 분포**:
- 관측 선호 P(o)에서 유도
- 상태 s → 예상 관측 o → P(o)에서 P(s) 계산
- "안전하고 음식 가까운 상태"를 선호

**행동에서의 역할**:
- Risk: "나쁜 관측을 피해라" → 회피 (공포)
- Ambiguity: "불확실한 전이를 피해라" → 탐색 (호기심)
- Complexity: "선호 상태에서 벗어난 믿음을 피해라" → 인지적 관성 (습관)

**구현 파일**:
- `preference_distributions.py`: StatePreferenceDistribution 클래스
- `action_selection.py`: compute_expected_complexity_from_obs()
- `agent.py`: AgentState.complexity 필드

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
│   ├── action_selection.py         # G = Risk + Ambiguity + Complexity
│   ├── preference_distributions.py # P(o) Beta 분포 + P(s) 상태 선호
│   ├── precision.py                # Precision Learning (v2.3)
│   ├── temporal.py                 # Temporal Depth / n-step Rollout (v2.4)
│   ├── hierarchy.py                # Hierarchical Models / Stabilized Context Transitions (v3.3.1)
│   ├── scenarios.py                # 테스트 시나리오 (v2.2)
│   ├── agent.py                    # GenesisAgent
│   ├── free_energy.py              # F 계산
│   ├── generative_model.py         # 생성 모델
│   └── inference.py                # 믿음 업데이트
├── main_genesis.py                 # FastAPI 서버
frontend/
├── src/GenesisApp.jsx              # 시각화 (Risk/Ambiguity/Complexity/Precision 표시)
```

### API 엔드포인트

- POST /step - 한 스텝 실행
- POST /reset - 리셋
- GET /clock - 시뮬레이션 시계 상태
- POST /hierarchy/enable - 계층적 처리 활성화 (v3.0)
- POST /hierarchy/disable - 계층적 처리 비활성화
- GET /hierarchy/status - 현재 계층 상태
- POST /hierarchy/alpha - Context 전이 혼합 비율 설정 (v3.3)

---

## 금지 사항

- 감정 이름을 변수로 사용 (X)
- 심즈식 욕구 게이지 (X)
- 휴리스틱으로 직접 행동 조작 (X)

---

## 다음 단계

- ~~Complexity = KL[Q(s)||P(s)]: 믿음 업데이트 제약~~ ✅ 완료 (v2.2)
- ~~환경 테스트 시나리오~~ ✅ 완료 (v2.2) - scenarios.py
- ~~Precision Learning: 정밀도 동적 조정~~ ✅ 완료 (v2.3)
- ~~Temporal Depth: 다중 시간 스케일 상상~~ ✅ 완료 (v2.4) - 3-step rollout
- ~~내부 항상성 기반 P(o)~~ ✅ 완료 (v2.5) - energy/pain interoception
- ~~Hierarchical Models: 계층적 예측 및 추상화~~ ✅ 완료 (v3.3.1) - Stabilized context-weighted transitions
- Learning: 온라인 선호 학습, 전이 모델 개선

---

## 달성된 목표: 내부 항상성 기반 P(o) ✅

### v2.5에서 구현 완료

**이전 구조 (v2.4 이전)**:
```python
# 외부 대상에 대한 선호 (하드코딩)
food_proximity:   P(o) = Beta(5, 1)  # "음식 가까이"
danger_proximity: P(o) = Beta(1, 5)  # "위험에서 멀리"
```

**새 구조 (v2.5)**:
```python
# 내부 상태에 대한 항상성 선호
energy: P(o) = Beta(3, 2)  # ~0.6 선호 (적당한 에너지)
pain:   P(o) = Beta(1, 5)  # 0 선호 (통증 없음)

# 외부와의 연결은 전이 모델이 학습
# "음식 먹으면 energy ↑" → transition_model에서 발견
```

### 결과

| | v2.4 (외부 선호) | v2.5 (내부 항상성) |
|---|---|---|
| 선호 대상 | 음식, 위험 | 에너지, 고통 |
| "음식=좋다" | 하드코딩 | 전이 모델 학습 |
| 성능 | food 16.7/100 | food 31.0/100 (**2배**) |
| 의미의 원천 | 프로그래머 | 에이전트 경험 |

**핵심 통찰**: 내부 항상성 선호가 더 robust한 정책을 유도함
- 외부 선호: "음식 근처에 있고 싶다" → 관측 상태 목표 → 위치 의존적
- 내부 선호: "에너지 ~0.6 유지" → 내부 안정 목표 → 일반화 가능

---

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다"
