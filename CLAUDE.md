# Genesis Brain - 근본 원리에서 창발하는 인공 뇌

## 핵심 목표

**뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다.**

이것은 심즈가 아니다. 수치의 나열이 아니다.
- "지루함 게이지가 차면 놀이를 한다" (X)
- "Risk가 높으면 회피 행동" (O)

모든 행동에는 **왜?**가 있어야 하고, 그 **왜?**를 계속 파고들면 **하나의 근본 원리**에 도달해야 한다.

---

## 현재 구현 상태 - True FEP v4.4

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

### P(o) 선호 분포 (Beta distributions) - v3.5 학습 가능

```
# INTEROCEPTION (내부 상태) - v3.5: 경험에서 학습 가능
energy:           P(o) = Beta(α, β)  # 초기 Beta(3, 2) → 학습으로 조정
pain:             P(o) = Beta(α, β)  # 초기 Beta(1, 5) → 학습으로 조정

# mode/concentration 파라미터화:
#   α = mode * (c - 2) + 1
#   β = (1 - mode) * (c - 2) + 1
#   mode: 선호하는 값 (0~1)
#   concentration: 선호의 확신도 (높을수록 뾰족)

# EXTEROCEPTION (외부 세계) - 고정 (lambda로 조절)
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

## v4.0 변경사항 (2024-12-26)

### Long-Term Memory - "미래 F/G를 줄이는 압축 모델"

**핵심 철학**: 기억은 "행동을 지시"하지 않고 "G를 조정"

**왜 FEP에 자연스러운가**:
- 불확실성과 서프라이즈를 줄이는 데 도움이 되는 경험을 압축 보존
- 기억 = 미래 예측의 Prior → F/G 감소에 기여
- 불확실할 때 기억에 더 의존 (새 상황에서 과적합 방지)

**에피소드 구조**:
```python
Episode = {
    t: int,                    # 타임스텝
    context_id: int,           # dominant context
    obs_summary: np.ndarray,   # 관측 상태 (8차원)
    action: int,               # 선택한 행동
    delta_energy: float,       # Δenergy
    delta_pain: float,         # Δpain
    delta_uncertainty: float,  # Δglobal_uncertainty
    delta_surprise: float,     # Δprediction_error
    outcome_score: float,      # G 감소량 (내부 기준)
    store_count: int,          # 병합 횟수
    recall_count: int,         # 회상 횟수
}
```

**저장 게이트 (memory_gate 사용)**:
```python
store_prob = sigmoid((memory_gate - θ) * k)
# memory_gate = f(surprise, uncertainty)
# 놀라움/불확실성 높을 때 → 저장
```

**회상 메커니즘 (G bias)**:
```python
# 행동 직접 지시 X, G를 조정 O
G(a) = G(a) + recall_weight * memory_bias[a]

# recall_weight: uncertainty 높을수록 기억에 더 의존
recall_weight = 0.2 + 0.8 * uncertainty

# memory_bias[a]: 과거 경험에서 계산
# outcome > 0 → bias < 0 → G ↓ → 더 선택됨
# outcome < 0 → bias > 0 → G ↑ → 덜 선택됨
```

**중복 억제 (similarity-based)**:
```python
if cosine_similarity(obs1, obs2) > 0.95:
    # 병합: store_count++, EMA로 값 업데이트
```

**테스트 결과 (50 steps)**:
```
stats:
  total_episodes: 2
  total_store_attempts: 3
  total_stored: 2
  total_merged: 1       # 중복 억제 작동
  total_recalls: 3
  store_rate: 66.7%
  merge_rate: 33.3%

last_recall:
  memory_bias: [0, 0.0001, 0, 0, 0, 0]  # UP 약간 유리
  recalled_count: 1
  recall_weight: 0.587  # uncertainty 기반
```

**API**:
```bash
POST /memory/enable?store_threshold=0.5  # 활성화
POST /memory/disable                      # 비활성화
POST /memory/reset                        # 초기화
GET  /memory/status                       # 상태 조회
GET  /memory/episodes?limit=10            # 에피소드 목록
```

**구현 파일**:
- `genesis/memory.py`: LTMStore, Episode, RecallResult
- `action_selection.py`: recall_from_memory(), store_episode(), G에 memory_bias 적용
- `main_genesis.py`: API 엔드포인트, step에서 recall/store 호출

---

## v4.1 변경사항 (2024-12-26)

### Memory Consolidation (Sleep) - "기억이 뇌 구조가 된다"

**핵심 철학**: 기억이 "조언자"에서 "prior"로 변환 → 미래 F/G 감소

**왜 FEP에 자연스러운가**:
- 수면 = "졸림 게이지"가 아닌 **내부 신호** 기반 트리거
- 통합 = transition model 재학습 → ambiguity(transition_std) 감소
- 프로토타입 = 반복 패턴 압축 → 일반화된 prior

**Sleep 트리거 조건**:
```python
# 3가지 조건 모두 충족 시 sleep
should_sleep = low_surprise AND high_redundancy AND stable_context

low_surprise:    avg(prediction_error[-30:]) < 0.3  # 최근 예측 정확
high_redundancy: merge_rate[-10:] > 0.4             # 유사 경험 반복
stable_context:  most_common_context_ratio > 0.7     # context 안정
```

**Sleep 중 수행**:
```python
# 1. Episode Replay → Transition Model 재학습
for ep in top_episodes:
    action_selector.update_from_replay(ep.action, ep.delta, lr=0.1)
    transition_std *= 0.98  # 2% 감소 bias

# 2. Episode Clustering → Prototype 생성
for (context, action) group:
    prototype = {
        obs_centroid: mean(obs),
        avg_outcome: mean(outcome),
        confidence: sqrt(n)/(sqrt(n)+k)
    }

# 3. Context Model Update (optional)
slow_layer.expected[k] 업데이트
```

**테스트 결과**:
```
Before Sleep:
  transition_std: 0.2706
  episodes: 7

After Sleep:
  transition_std: 0.2499  (7.7% 감소)
  prototypes: 3
  episodes_replayed: 7
```

**완료 기준**:
- [x] sleep 후 transition_std 하락 (0.2706 → 0.2499)
- [x] 프로토타입 생성 (3개)
- [x] 생존 유지

**API**:
```bash
POST /consolidation/enable?auto_trigger=true   # 활성화
POST /consolidation/disable                     # 비활성화
POST /consolidation/trigger                     # 수동 sleep
GET  /consolidation/status                      # 상태 조회
POST /consolidation/reset                       # 초기화
```

**구현 파일**:
- `genesis/consolidation.py`: MemoryConsolidator, Prototype, SleepTriggerState
- `action_selection.py`: update_from_replay(), enable_consolidation()
- `main_genesis.py`: API 엔드포인트, step에서 트리거 체크

---

## v4.3 변경사항 (2024-12-26)

### Uncertainty/Confidence - "대시보드"가 아닌 "자기조절 신호"

**핵심 목표**: 에이전트가 불확실함을 '느끼고', 그 결과로 내부 메커니즘이 자동으로 바뀌게 만드는 것.

**4가지 불확실성 소스**:
```
1) Belief Uncertainty: H(Q(c))
   - context belief 엔트로피
   - "내가 지금 어떤 상황인지 모르겠다"

2) Action Uncertainty: H(π(a))
   - action 분포 엔트로피
   - "뭘 해야 할지 모르겠다"

3) Model Uncertainty: transition_std
   - 전이 모델 불확실성
   - "이 행동의 결과를 잘 모르겠다"

4) Surprise: prediction_error
   - 예측 오차
   - "세상이 내 예측과 다르다"
```

**불확실성 → 행동 조절 (자동)**:
```python
# A) THINK 조절
# 불확실성 ↑ → THINK bias ↓ → THINK 더 유리
think_bias = 0.2 - 0.5 * uncertainty  # u=0: +0.2, u=1: -0.3

# B) Precision 조절 (메타-정밀도)
# 불확실성 ↑ → precision ↓ → 관측을 덜 믿음
sensory_precision_mult = 1.0 - 0.3 * uncertainty
goal_precision_mult = 1.0 + 0.3 * (1 - uncertainty)

# C) 탐색/회피 조절
# 불확실성 ↑ → 탐색 보너스 ↑, risk 민감도 ↓
exploration_bonus = 0.2 * uncertainty
risk_sensitivity = 1.0 - 0.3 * uncertainty

# D) 기억 저장 게이트 (v4.0 준비)
# 불확실성/surprise ↑ → memory_gate ↑ → 기억할 가치
memory_gate = f(surprise, uncertainty)  # 0~1
```

**테스트 결과**:
```
=== 10 steps 실행 ===
Global Uncertainty: 0.467
Global Confidence:  0.533
Top Factor: action (action entropy가 가장 높음)

Modulation:
  THINK bias:        -0.033 (THINK 약간 유리)
  Sensory precision: 0.86  (관측 덜 믿음)
  Goal precision:    1.16  (목표 더 확신)
  Exploration bonus: 0.09  (탐색 보너스)
  Memory gate:       0.47  (중간 저장 가치)
```

**왜 "대시보드"가 아닌가**:
- 불확실성 값을 보여주기만 하면 관찰자용 UI일 뿐
- v4.3: 불확실성이 THINK, Precision, 탐색/회피를 **자동으로** 조절
- 즉, 에이전트가 "모르는 것을 안다" → 행동이 바뀜

**API**:
```bash
POST /uncertainty/enable?sensitivity=1.0  # 활성화 (가중치 조절 가능)
POST /uncertainty/disable                  # 비활성화
POST /uncertainty/reset                    # 상태 리셋
GET  /uncertainty/status                   # 상태 조회
GET  /uncertainty/memory_gate              # 기억 게이트 값 (v4.0 준비)
```

**구현 파일**:
- `genesis/uncertainty.py`: UncertaintyTracker, UncertaintyModulation
- `action_selection.py`: compute_G(), compute_G_think()에 modulation 적용
- `main_genesis.py`: API 엔드포인트

---

## v4.3.1 안전점검 (2024-12-26)

### v4.0 Memory 진입 전 필수 점검 3가지

**1. Risk sensitivity 하한선**:
```python
# 문제: 불확실할 때 risk_sensitivity ↓ → 위험 앞에서 "자살 버튼"?
# 해결: clamp(0.6, 1.2) + 위험 근접 시 min 상향

risk_sensitivity = max(0.6, min(1.2, 1.0 - 0.3 * uncertainty))

# 추가: danger_proximity > 0.3이면 min 상향 (0.6 → 0.9)
if danger_proximity > 0.3:
    min_sensitivity = 0.6 + 0.3 * (danger_proximity - 0.3) / 0.7
    risk_sensitivity = max(min_sensitivity, risk_sensitivity)
```

**2. THINK bias 부호 검증**:
```
Low uncertainty (u=0.35):  think_bias = +0.026 → THINK 불리 ✓
High uncertainty (u=0.52): think_bias = -0.062 → THINK 유리 ✓

G_think = G_best - improvement + cost + think_bias
→ think_bias < 0이면 G_think ↓ → THINK 선택 확률 ↑
```

**3. Uncertainty 감소 경로**:
```
THINK cooldown = 5 steps (연속 THINK 방지)
→ 물리 행동 → 전이 모델 학습 → model_uncertainty ↓
→ 예측 정확 → surprise ↓
→ 결정 개선 → action_entropy ↓

테스트: 50 steps 후 global_u 0.500 → 0.483 (-3.4%)
EMA alpha = 0.1 (느린 추적, 안정성 우선)
```

**v4.0 Memory 준비**:
- `memory_gate`가 "이 경험을 기억할 가치"를 제공
- surprise ↑ = 기억할 가치
- uncertainty ↑ = 학습/통합 필요
- context entropy ↑ = 상황 전환점

---

## v4.4 변경사항 (2024-12-27)

### Counterfactual + Regret - "후회가 학습을 바꾸는 구조"

**핵심 철학**: 후회(regret)는 "감정 변수"가 아니라, 선택한 행동이 대안 행동보다 얼마나 더 큰 G를 초래했는지에 대한 **'사후 EFE 차이'**.

**왜 FEP에 자연스러운가**:
- 정책을 직접 바꾸는 보상학습 X
- 모델/정밀도/기억 게이트 쪽으로 연결 O
- → **"후회가 '학습/추론 자원 배분'을 바꾸는 구조"**

**Counterfactual 계산**:
```python
# 매 step에서:
# 1. 선택 시점의 G_pred 저장
# 2. 관측 후 G_post 계산 (전이 모델 기반 반사실 추론)
# 3. Regret 신호 계산

G_pred = {action: G_value}  # 선택 시점 예측
G_post = {action: G_value}  # 관측 후 재평가

# 판단 오류 후회: 내가 잘못 판단했나?
regret_pred = G_pred(chosen) - min_a G_pred(a)

# 실제 결과 후회: 세상이 바뀌었나?
regret_real = G_post(chosen) - min_a G_post(a)
```

**Regret 연결 (FEP스럽게)**:
```python
# 1) Memory gate boost: regret 큰 사건 = 저장 우선순위 ↑
memory_gate_boost = 0.3 * sigmoid(regret - 0.3)  # 0~0.3

# 2) LR boost: regret spike면 모델 재학습 필요
# regret_pred 높고 regret_real도 높으면 = 판단도 틀리고 결과도 나빴음
if is_spike and regret_pred > 0.2:
    lr_boost_factor = 1.5  # 50% lr 증가
elif regret > 0.3:
    lr_boost_factor = 1.2
else:
    lr_boost_factor = 1.0

# 3) THINK benefit boost: 누적 regret 높으면 메타인지 가치 ↑
think_benefit_boost = min(0.2, cumulative_regret * 0.5)
```

**Regret State 추적**:
```python
RegretState:
  cumulative_regret: float  # EMA 누적
  regret_baseline: float    # 최근 평균
  is_spike: bool            # baseline의 2배 이상
  optimality_ratio: float   # 최적 선택 비율
```

**API**:
```bash
POST /regret/enable           # Counterfactual + Regret 활성화
POST /regret/disable          # 비활성화
POST /regret/reset            # 상태 초기화
GET  /regret/status           # 상태 조회
GET  /regret/modulation       # 조절 파라미터 반환
POST /ablation/apply?regret=true  # ablation 테스트에 포함
```

**구현 파일**:
- `genesis/regret.py`: CounterfactualEngine, CounterfactualResult, RegretState
- `action_selection.py`: enable_regret(), compute_counterfactual(), get_regret_modulation()
- `main_genesis.py`: API 엔드포인트

**핵심 통찰**:
- regret_pred ≠ regret_real: 판단 오류 vs 세상 변화 분리
- 정책 직접 수정 X → memory_gate, lr_boost, THINK에 연결 O
- "후회가 '무엇을 배울지'와 '얼마나 생각할지'를 바꾸는 구조"

### v4.4.1 보완 (2024-12-27)

**해석 가능성 개선**:

1. **Optimal 기준 명시**: `optimal_basis: "G_post"` - 사후 재평가 기준임을 명시
2. **Regret Z-score**: `regret_z = (regret - mean) / std` - 최근 분포 대비 상대적 크기
3. **Normalized Regret**: `regret / (|G_best| + 0.01)` - 스케일 불변 비교 가능
4. **Spike 원인 분류**:
   - `judgment_error`: 판단도 틀리고 결과도 나쁨 (regret_pred > 0.2 && regret > 0.3)
   - `model_mismatch`: 예측과 현실 괴리 (regret > regret_pred * 2)
   - `environment_change`: 환경 변화

**프론트엔드 패널 (GenesisApp.jsx)**:
```
┌─────────────────────────────────────────┐
│ Counterfactual + Regret    Optimal: G_post │
├─────────────────────────────────────────┤
│ Counterfactuals │ Optimal Ratio │ Regret Z │
├─────────────────────────────────────────┤
│    Real    │    Pred    │  Normalized   │
│ Action N (Optimal/Suboptimal)           │
├─────────────────────────────────────────┤
│ Memory Gate │ LR Boost │ THINK Benefit  │
│ [SPIKE: 원인]                           │
└─────────────────────────────────────────┘
```

---

## v3.5 변경사항 (2024-12-26)

### Online Preference Learning - 경험에서 선호 학습

**핵심 개념**: 내부 선호(energy, pain)의 Beta 파라미터를 경험에서 학습

**왜 필요한가**:
- v2.5에서 P(o)는 고정된 Beta 분포 (예: energy ~ Beta(3,2))
- 하지만 "적당한 에너지"가 무엇인지는 경험에서 배워야 함
- v3.5: G(a) 결과에 따라 선호 분포 자체를 업데이트

**학습 메커니즘**:
```python
# G_surprise = G_baseline - G_value
# positive = 좋은 결과, negative = 나쁜 결과

# Good outcome → mode를 현재 상태 쪽으로
# Bad outcome → mode를 현재 상태 반대쪽으로

if G_surprise > 0:
    energy_mode → current_energy
else:
    energy_mode ← away from current_energy
```

**Beta 파라미터화 (mode/concentration)**:
```
α = mode * (concentration - 2) + 1
β = (1 - mode) * (concentration - 2) + 1

mode: 분포의 최빈값 (0 < m < 1)
concentration: 분포의 뾰족함 (c > 2)
```

**안정화 기법**:
- EMA smoothing (lr = 0.01~0.05)
- Parameter clamps (mode: 0.1~0.9, concentration: 3~15)
- Learning inertia (급격한 변화 방지)
- G_baseline tracking (상대적 surprise 계산)

**테스트 결과** (50 steps):
```
Energy: mode 0.67 → 0.76 (더 높은 에너지 선호 학습)
Pain:   mode 0.05 → 0.04 (더 낮은 통증 선호 강화)
Update count: 25 (downsampled)
G_baseline: 1.92 (평균 G 추적)
```

**API**:
```bash
POST /preference/learning/enable?mode_lr=0.02&concentration_lr=0.01
POST /preference/learning/disable
POST /preference/learning/reset
GET  /preference/learning/status
```

**구현 파일**:
- `preference_distributions.py`: LearnableBetaParams, PreferenceLearner
- `action_selection.py`: enable/disable/update_preference_learning()
- `main_genesis.py`: API 엔드포인트

**핵심 통찰**:
- 선호는 고정된 것이 아니라 경험에서 형성됨
- "적당한 에너지"의 기준이 경험에 따라 조정됨
- FEP 원리: G 최소화 → 좋은 결과 → 해당 상태 선호 강화

---

## v3.5.1 변경사항 (2024-12-26)

### 에이전트 "멈춤" 문제 해결

**문제 증상**:
- 에이전트가 Y축으로만 진동 (UP/DOWN 반복)
- LEFT/RIGHT 선택 안 함 → 음식을 찾지 못함
- Risk = 0 → 행동 선택에 영향 없음

**원인 분석**:

| 문제 | 원인 |
|------|------|
| Risk=0 | `internal_pref_weight=1.0`이고 모든 행동의 energy 예측이 동일 |
| energy mode 폭주 | 0.67 → 0.90으로 학습됨 → 현재 energy(0.88)가 선호와 일치 → Risk=0 |
| 외부 선호 무시 | `internal_pref_weight=1.0`이라 food proximity 선호가 무시됨 |

**해결책**:

```python
# 1. internal_pref_weight 기본값 변경
internal_pref_weight: float = 0.5  # 이전: 1.0

# 2. energy_mode_clamp 추가 (항상성 범위 제한)
energy_mode_clamp: Tuple[float, float] = (0.5, 0.75)
```

**테스트 결과** (500 steps):
```
이전: Y축 진동만, food 0개
이후: 모든 방향 이동, food 112개 (22.4/100 steps)
      UP: 24%, DOWN: 32%, LEFT: 28%, RIGHT: 15%
```

---

## v3.8 변경사항 (2024-12-26)

### Docker Packaging - 환경 재현성

**핵심 개념**: 컨테이너로 의존성/환경 차이를 제거하여 완전한 재현성 보장.

**왜 필요한가**:
- v3.7의 "논리적 재현성" + v3.8의 "환경적 재현성" = G0 게이트 완전 통과
- 다른 PC/서버로 이식 시 동일 동작 보장
- 배포/공유 간편

**파일 구성**:
```
backend/
├── Dockerfile          # Python 3.12-slim 기반
├── requirements.txt    # 버전 고정 의존성
├── .dockerignore      # 불필요 파일 제외
docker-compose.yml      # 서비스 정의
```

**의존성 버전 고정**:
```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
numpy==2.2.1
scipy==1.15.0
```

**실행 방법**:
```bash
# 1. 이미지 빌드
docker compose build backend

# 2. 서버 실행
docker compose up backend

# 3. 재현성 테스트 (컨테이너 내부)
docker compose --profile test run test-reproducibility
```

**검증 체크리스트**:
- [x] 파일 구성 완료 (Dockerfile, requirements.txt, docker-compose.yml, .dockerignore)
- [x] docker compose build 성공
- [x] docker compose up으로 서버 기동
- [x] 컨테이너 내부 재현성 테스트 통과 (hash: d3ff2147f3de, 3회 동일)
- [x] GET /info, POST /step 정상 동작

**G0 게이트 통과** - 논리적 재현성(v3.7) + 환경적 재현성(v3.8) 완료

---

## G1 Gate: Generalization Test (2024-12-26)

### 핵심 질문: 에이전트가 일반화하는가, 암기하는가?

**왜 필요한가**:
- LTM/Consolidation이 "도움"이 되는지 "방해"가 되는지 검증
- 환경이 변해도 적응할 수 있는지 확인
- 과적합(overfitting) vs 일반화(generalization) 구분

**테스트 시나리오 (DRIFT)**:
```
Phase 1 (Training):    100 steps 학습
Phase 2 (Drift):       환경 dynamics 변경 (행동 효과 회전)
Phase 3 (Adaptation):  50 steps, 적응 속도 측정
```

**Drift 종류**:
- `rotate`: 행동 회전 (UP→RIGHT, RIGHT→DOWN, ...)
- `flip_x`: 좌우 반전 (LEFT↔RIGHT)
- `flip_y`: 상하 반전 (UP↔DOWN)
- `reverse`: 전체 반전

**통과 기준**:

| 기준 | 설명 | 통과 조건 |
|------|------|----------|
| 1. Drift 감지 | transition_std 증가 | std_increase > 5% |
| 2. 생존 | post-drift에서도 food 획득 | post_food > 0 |
| 3. 빠른 적응 | G가 pre-drift 수준으로 복귀 | adaptation_steps < 50 |

**테스트 결과 (100 steps)**:
```
Pre-drift:  food=22, avg_G=2.148, std=0.378
Post-drift: food=15, avg_G=1.971, std=0.374

PASS: total food=37
PASS: G 안정 (91.8%)
```

**사용법**:
```bash
# API 기반 테스트 (서버 필요)
python test_g1_gate.py --api --drift-after 50 --total-steps 100

# 독립 실행 테스트 (Mock 환경)
python test_g1_gate.py --drift-after 100 --total-steps 150 --drift-type rotate
```

**구현 파일**:
- `genesis/scenarios.py`: ScenarioType.DRIFT, G1GateResult
- `backend/test_g1_gate.py`: G1 Gate 테스트 스크립트

---

## v3.7 변경사항 (2024-12-26)

### Reproducibility - 시드 고정 & 재현성 테스트

**핵심 개념**: 동일 seed → 동일 결과 보장. G0 게이트(재현성) 통과를 위한 기반.

**왜 필요한가**:
- 개선/퇴행 판정: 같은 조건에서 비교해야 의미 있음
- 디버깅: 특정 버그 상황 재현
- 벤치마크: 공정한 성능 비교

**시드 관리**:
```python
# 모든 랜덤 소스 고정 (np.random + random)
set_global_seed(42)

# 재현성 테스트
result = run_reproducibility_test(
    agent, world, action_selector,
    seed=42, n_steps=100, n_runs=3
)
# → 3회 실행 모두 동일한 fingerprint
```

**SimulationFingerprint**:
```python
fingerprint = {
    'seed': 42,
    'step_count': 100,
    'total_food': 23,
    'total_deaths': 0,
    'final_energy': 0.847,
    'agent_pos': (5, 3),
    'avg_G': 2.134,
    'action_counts': {0: 12, 1: 25, 2: 28, 3: 18, 4: 17}
}
# → MD5 해시로 비교
```

**API**:
```bash
POST /seed?seed=42              # 시드 설정
GET  /seed                      # 현재 시드 상태
POST /reproducibility/test      # 재현성 테스트 실행
     ?seed=42&n_steps=100&n_runs=3
```

**테스트 결과**:
```
Seed 42:  Hash = d3ff2147f3de (3회 동일) ✓
Seed 123: Hash = 3f18e3c5c6fb (3회 동일) ✓
→ 동일 seed = 동일 결과, 다른 seed = 다른 결과
```

**구현 파일**:
- `genesis/reproducibility.py`: SeedManager, run_reproducibility_test
- `genesis/checkpoint.py`: HeadlessRunner에 seed 지원
- `main_genesis.py`: /seed, /reproducibility/test 엔드포인트

---

## v3.6 변경사항 (2024-12-26)

### Checkpoint & Headless Evaluation - 재현성과 벤치마크

**핵심 개념**: 학습된 뇌 상태를 저장/복원하고, UI 없이 N 에피소드 평가 실행

**왜 필요한가**:
- 재현성: 동일한 학습 상태에서 시작할 수 있어야 함
- 벤치마크: UI 없이 성능을 정량적으로 측정해야 함
- 대회/챌린지: 학습된 모델 제출 가능해야 함

**체크포인트 저장 대상**:
```python
checkpoint = {
    'metadata': {version, timestamp, step_count, total_food, total_deaths, description},
    'transition_model': {delta_mean, delta_std, count},  # 전이 모델
    'precision': {sensory, transition, goal, ema_error, volatility},  # 정밀도
    'hierarchy': {K, context_beliefs, expectations, transitions},  # 계층 컨트롤러
    'preference_learning': {energy_mode, pain_mode, concentrations},  # 선호 학습
    'world': {agent_pos, food_pos, danger_pos, energy, step_count}  # 월드 상태
}
```

**헤드리스 평가**:
```python
# N 에피소드 자동 실행
result = runner.run(n_episodes=100, max_steps_per_episode=500)

# 결과
{
    'n_episodes': 100,
    'total_food': 2350,
    'total_deaths': 3,
    'avg_food_per_episode': 23.5,
    'survival_rate': 0.97,
    'episodes': [...]  # 개별 에피소드 상세
}
```

**API**:
```bash
# Checkpoint
POST /checkpoint/save?filename=brain.json&description=text
POST /checkpoint/load?filename=brain.json
GET  /checkpoint/list

# Headless Evaluation
POST /evaluate?n_episodes=100&max_steps=500
POST /evaluate/save?n_episodes=100&filename=result.json
```

**World 통계 추적 (v3.6)**:
```python
world.total_food    # 총 음식 섭취 수
world.total_deaths  # 총 사망 횟수
```

**구현 파일**:
- `genesis/checkpoint.py`: BrainCheckpoint, HeadlessRunner
- `main_genesis.py`: API 엔드포인트

**테스트 결과**:
```
Checkpoint Save/Load: ✓ Step=31, Food=11 상태 저장/복원 확인
Headless Eval (5 eps): avg_food=20.6, survival=100%
```

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
│   ├── action_selection.py         # G = Risk + Ambiguity + Complexity + THINK (v3.4) + PreferenceLearning (v3.5) + Uncertainty (v4.3)
│   ├── preference_distributions.py # P(o) Beta 분포 + P(s) 상태 선호 + PreferenceLearner (v3.5)
│   ├── precision.py                # Precision Learning (v2.3)
│   ├── temporal.py                 # Temporal Depth / n-step Rollout (v2.4)
│   ├── hierarchy.py                # Hierarchical Models / Stabilized Context Transitions (v3.3.1)
│   ├── checkpoint.py               # Checkpoint & Headless Evaluation (v3.6)
│   ├── reproducibility.py          # Seed Management & Reproducibility Tests (v3.7)
│   ├── uncertainty.py              # Uncertainty/Confidence Tracking (v4.3)
│   ├── memory.py                   # Long-Term Memory (v4.0)
│   ├── consolidation.py            # Memory Consolidation / Sleep (v4.1)
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
- POST /think/enable - THINK action 활성화 (v3.4)
- POST /think/disable - THINK action 비활성화
- GET /think/status - THINK 상태 조회
- POST /preference/learning/enable - 선호 학습 활성화 (v3.5)
- POST /preference/learning/disable - 선호 학습 비활성화
- POST /preference/learning/reset - 선호 학습 리셋
- GET /preference/learning/status - 선호 학습 상태 조회
- POST /checkpoint/save - 체크포인트 저장 (v3.6)
- POST /checkpoint/load - 체크포인트 로드 (v3.6)
- GET /checkpoint/list - 체크포인트 목록 (v3.6)
- POST /evaluate - 헤드리스 N 에피소드 평가 (v3.6)
- POST /evaluate/save - 평가 결과 JSON 저장 (v3.6)
- POST /seed - 글로벌 시드 설정 (v3.7)
- GET /seed - 현재 시드 상태 (v3.7)
- POST /reproducibility/test - 재현성 테스트 실행 (v3.7)
- POST /uncertainty/enable - 불확실성 추적 활성화 (v4.3)
- POST /uncertainty/disable - 불확실성 추적 비활성화 (v4.3)
- POST /uncertainty/reset - 불확실성 상태 리셋 (v4.3)
- GET /uncertainty/status - 불확실성 상태 조회 (v4.3)
- GET /uncertainty/memory_gate - 기억 저장 게이트 값 (v4.3)
- POST /memory/enable - 장기 기억 활성화 (v4.0)
- POST /memory/disable - 장기 기억 비활성화 (v4.0)
- POST /memory/reset - 장기 기억 초기화 (v4.0)
- GET /memory/status - 기억 상태 조회 (v4.0)
- GET /memory/episodes - 저장된 에피소드 목록 (v4.0)
- POST /consolidation/enable - 수면/통합 활성화 (v4.1)
- POST /consolidation/disable - 수면/통합 비활성화 (v4.1)
- POST /consolidation/trigger - 수동 수면 트리거 (v4.1)
- GET /consolidation/status - 통합 상태 조회 (v4.1)
- POST /consolidation/reset - 통합 시스템 초기화 (v4.1)

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
- ~~Online Preference Learning: 경험에서 선호 학습~~ ✅ 완료 (v3.5) - Beta 파라미터 학습
- ~~Uncertainty/Confidence: 자기조절 신호~~ ✅ 완료 (v4.3) - THINK/Precision/탐색 자동 조절
- ~~Memory (LTM): 경험 저장/회상~~ ✅ 완료 (v4.0) - memory_gate 사용
- ~~Consolidation: Awake Replay로 모델 불확실성 감소~~ ✅ 완료 (v4.1)
- ~~G1 Gate: Generalization Test~~ ✅ 완료 - DRIFT 시나리오
- ~~Counterfactual + Regret: 반사실적 추론과 후회 기반 학습~~ ✅ 완료 (v4.4) - memory_gate, lr_boost, THINK 연결
- Server-side Drift: API로 환경 dynamics 변경 지원

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
