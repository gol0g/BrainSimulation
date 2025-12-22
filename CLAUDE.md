# Brain Simulation Project

## 최종 목표
**인간의 뇌와 유사한 주체적 의식과 감정을 가지는 인공 뇌를 만드는 것**

---

## 현재 구현된 시스템

### 1. SNN (Spiking Neural Network)
- STDP 3-factor learning (pre-post timing + neuromodulator)
- Synapse weight bounds: W_MIN=5.0, W_MAX=150.0
- **Tabula Rasa 초기화**: 모든 s→h 가중치 = 40 (선천 편향 없음)

### 2. Agency Detection
- SELF_CAUSED vs EXTERNALLY_CAUSED 구분
- 외부 perturbation(wind) 감지 시 학습 차단

### 3. Working Memory
- 단기 기억 유지 (decay 기반)
- Self-Model의 stability에 의해 decay rate 조절됨

### 4. Attention System
- Saliency 기반 주의 집중
- Inhibition of Return (IOR)
- Self-Model의 effort에 의해 dwell time/gain 조절됨

### 5. Self-Model (자기 표상)
- **5가지 Self-States**: confidence, uncertainty, effort, exploration_need, stability
- **7가지 Behavioral Labels**: CONFIDENT, EXPLORING, STRUGGLING, REACTIVE, FATIGUED, STABLE, TRANSITIONING
- **인과적 연결 (A/B/C)**:
  - A) exploration_need/confidence → epsilon (탐험률)
  - B) effort → attention dwell time/gain
  - C) stability → Working Memory decay rate

### 6. Value Conflict (비활성화)
- 현재 disabled - 가중치 degradation 문제로 스킵

### 7. Predator System (포식자)
- **시각적 위협**: 빨간 원으로 맵에 표시
- **거리 기반 위협**: threat_radius=3 내에서 safety 감소
- **간단한 AI (v1)**: 랜덤 이동 + 30% 확률로 추적
- **확장 가능 설계**: 나중에 자체 SNN 부착 가능
  - `predator.network` 필드 예약됨
  - `decide_action()` 메서드 교체 가능
- **감각 입력**: 에이전트가 포식자를 "볼" 수 있음
  - `p_up`, `p_down`, `p_left`, `p_right` 뉴런
  - 가까울수록 신호 강함 (max_perception = 5칸)
- **회피 학습**: 포식자 방향 뉴런 → Hidden 억제 (STDP 학습)
- **인과적 연결**:
  - predator proximity → safety ↓ → safety_drive ↑
  - predator contact → health ↓ (-25%), pain = 1.0, reward = -2.0
  - **통증 = 즉각적 학습 신호**

### 8. Emotion System (감정 시스템)
- **핵심 개념**: 감정은 항상성 위의 "느끼는" 레이어
  - 항상성: "음식이 필요해" (객관적 상태)
  - 감정: "배고프다/불안하다/만족스럽다" (주관적 경험)
- **6가지 감정**:
  - `fear` (공포): **학습된** 위험 반응
  - `satisfaction` (만족): 욕구 충족 (배고플 때 먹으면 만족)
  - `curiosity` (호기심): 탐험 욕구 → 더 탐험 (미지의 것일수록 호기심 ↑)
  - `pain` (고통): 즉각적 손상 (포식자에게 잡힘)
  - `relief` (안도): 위험이 지나간 후
  - `anxiety` (불안): 불확실성 + 약한 위협 → 지속적 걱정 (**학습된 공포 필요**)

#### 경험 기반 학습 (핵심 개념!)
- **처음엔 아무것도 모름**:
  - 포식자가 뭔지 모름 → fear 없음
  - 음식이 뭔지 모름 → 특별한 추구 없음
- **경험으로 학습**:
  - 포식자에게 부딪힘 → pain → "빨간 것 = 아픔!" → `predator_fear_learned` ↑
  - 음식 먹음 → 배고픔 해소 → "초록 것 = 좋음!" → `food_seeking_learned` ↑
- **학습된 연관이 행동에 영향**:
  - `learned_fear` 없으면 → 포식자 신호 약함 (base 10)
  - `learned_fear` 높으면 → 포식자 신호 강함 (up to 80), 음식 신호 억제

- **감정 특성**:
  - 강도 (0-1)
  - Valence (긍정/부정)
  - Arousal (각성 수준)
  - Decay rate (감정마다 다른 지속 시간)
- **행동 변조**:
  - epsilon: fear ↑ → 탐험 ↓, curiosity ↑ → 탐험 ↑
  - attention: fear → 좁고 집중, curiosity → 넓고 스캔
  - learning: fear/pain → 학습 강화, anxiety → 학습 약화
- **Freeze 반응**: 극심한 공포 (fear > 0.7) → 완전히 멈춤
- **인과적 연결**:
  - F) emotions → epsilon (탐험률)
  - G) emotions → attention width (주의 폭)
  - H) emotions → learning rate (학습률)

### 9. Developmental Phase System (발달 단계)
- **핵심 개념**: 에이전트가 "태어날 때" 아무것도 모르지만 생존할 수 있는 환경
- **Infant Bootstrap (B') 철학**:
  - 선천 지식 금지: 포식자=위험, 음식=좋음 같은 가치는 오직 경험으로
  - 보호 환경 허용: 생존을 보장하되 "답"을 주지 않음
- **두 단계**:
  - `infant` (0-2000 스텝): 보호 환경
  - `adult` (2000+ 스텝): 정상 난이도
- **Infant 보호 환경** (랜덤 탐험으로도 생존 가능):
  - 음식 스폰: **1칸 거리** (바로 옆에!)
  - 포식자 이동 확률: 5% (정상 30%)
  - 포식자 추적 확률: 2% (정상 20%)
  - 에너지 감소: 0.003/스텝 (정상 0.05)
- **호기심 내재 보상**:
  - prediction_error > 0.3 → 작은 긍정 보상
  - 신기한 것 = 보상 (방향 힌트 없이 탐험 장려)
  - Infant 단계에서 1.5배 보너스
- **순수 랜덤 탐험**: 4방향 25%씩 (편향 없음)
- **No-Fire 시 탐험**: SNN이 발화 안 해도 랜덤 이동 (학습 기회 제공)
- **효과**: 에이전트가 "아기"처럼 시작 → 경험으로 세상을 배움
- **테스트 결과**: ~30스텝 내에 첫 음식 발견, 134스텝에 18번 먹고 food_seeking=0.90 달성

### 10. Internal Simulation (내적 시뮬레이션 / Forward Model)
- **핵심 개념**: "행동 전에 생각하기" - 각 행동의 결과를 상상
- **철학적 의미**:
  - 포식자 쪽으로 가면 아플 것을 **상상**하고, 가지 않기로 **선택**
  - 이것이 바로 **의식적 선택의 시작**
- **기능**:
  - 4방향 (UP, DOWN, LEFT, RIGHT) 각각의 예상 결과 계산
  - 에너지 변화, 통증, 안전 등 예측
  - 현재 욕구/감정에 따라 가중치 동적 조절
- **예측 항목**:
  - `delta_food_dist`: 음식까지 거리 변화
  - `delta_pred_dist`: 포식자까지 거리 변화
  - `reach_food`: 음식 도달 여부
  - `reach_predator`: 포식자 접촉 여부
  - `hit_wall`: 벽 충돌 여부
- **점수 계산** (Utility):
  ```
  score = Σ(predicted_change × dynamic_weight)
  - energy: +1.0 (배고프면 ↑)
  - pain: -3.0 (공포 학습되면 ↑)
  - safety: +1.5
  - food_proximity: +0.8
  - predator_proximity: -2.0 (공포 학습 × 가중치)
  ```
- **SNN + 상상 혼합**:
  - `imagination_weight = confidence × 0.5`
  - `combined = (1-w) × SNN_score + w × imagination_score`
  - 신뢰도 낮을 때 → SNN 반사 위주
  - 신뢰도 높을 때 → 상상 기반 숙고
- **Action Source (행동 출처)**:
  - `snn`: SNN 반사만 사용
  - `snn+imagine`: SNN + 상상 혼합
  - `imagine`: 상상만 사용 (SNN 미발화, 높은 신뢰도)
  - `explore`: 탐험 모드
  - `random`: 랜덤 이동
- **학습**:
  - 예측 vs 실제 결과 비교 → 신뢰도 조절
  - 맞으면 confidence +0.01, 틀리면 -0.02
- **UI**:
  - 4방향 점수 바 (최선 ✓, 최악 ✗ 표시)
  - 행동 이유 표시 (한국어)
  - 예측 신뢰도 바
  - 행동 출처 표시
- **인과적 연결**:
  - I) imagination scores → action selection
  - J) prediction_confidence → SNN/imagination 비율

### 11. Long-term Memory (장기 기억 / Episodic Memory)
- **핵심 개념**: "여기서 무슨 일이 있었는지 기억해"
- **철학적 의미**:
  - 상상은 "무슨 일이 일어날지" 예측
  - 기억은 "무슨 일이 실제로 일어났는지" 회상
  - 둘이 합쳐서 **경험에 기반한 현명한 결정** 가능
- **Episode 구조**:
  - `position`: 어디서
  - `energy, safety`: 내 상태
  - `action`: 무슨 행동
  - `outcome`: 무슨 결과 (food/pain/nothing/near_danger/escape)
  - `reward`: 보상
  - `dominant_emotion`: 어떤 감정
  - `emotion_intensity`: 감정 강도
  - `importance`: 중요도 (감정 기반)
  - **v1.1 추가 필드**:
    - `delta_energy`: 에너지 변화량 (-1 ~ +1)
    - `delta_pain`: 통증 변화량 (0 ~ 1)
    - `delta_safety`: 안전 변화량 (-1 ~ +1)
    - `context_predator_near`: 포식자가 근처에 있었나
    - `context_energy_low`: 에너지가 낮았나
    - `context_was_fleeing`: 도망 중이었나
- **감정 기반 중요도**:
  - 강한 감정 = 강한 기억
  - 통증 기억은 2배 중요
  - 음식 기억은 1.5배 중요
  - 공포/고통 감정은 1.5배 가중
- **유사성 기반 회상 (v1.1 개선)**:
  - 현재 위치 근처 기억 검색 (반경 3칸)
  - 같은 행동 기억에 가중치 부여
  - **Context 유사성 매칭**: 같은 상황일수록 더 관련 (최대 1.45배)
  - 상위 5개 관련 기억 recall
- **기억의 행동 영향 (v1.1: 델타 기반)**:
  - ~~이전: 통증 기억 → -1.0, 음식 기억 → +0.6 (고정 점수)~~
  - **v1.1**: `U = 1.0 × Δenergy - 3.0 × Δpain + 1.5 × Δsafety`
  - 실제 경험 델타를 importance-weighted average로 계산
  - "규칙"이 아닌 "경험"처럼 느껴지는 기억 영향
- **기억 생애주기**:
  - 최대 100개 에피소드 저장
  - 시간 지나면 importance 감소 (decay)
  - 통증 기억은 천천히 감소 (트라우마)
  - recall될 때마다 importance 증가 (유용한 기억)
  - 낮은 importance 기억은 자연 도태
- **인과적 연결**:
  - K) memory_influence → action scores (기억이 선택에 영향)
  - L) emotion_intensity → memory_importance (감정이 기억 강도에 영향)
- **UI**:
  - 회상 이유 표시 ("여기서 아팠어!", "여기서 먹었어!")
  - 현재 recall된 기억 개수
  - 방향별 기억 영향 표시

### 12. Homeostasis & Drives (항상성 시스템)
- **생물학적 기반**: 의식의 뿌리는 내적 필요(need)에서 시작
- **4가지 내부 상태 (Homeostatic States)**:
  - `energy`: 0 = 굶주림, 1 = 포만 (배고픔/포만)
  - `health`: 0 = 죽음, 1 = 건강 (체력/손상)
  - `safety`: 0 = 위험, 1 = 안전 (위협 수준)
  - `fatigue`: 0 = 휴식됨, 1 = 탈진 (피로도)
- **통증 시스템 (Pain)**:
  - 포식자에게 잡히면 pain = 1.0 (최대 고통)
  - 시간이 지나면 pain 감소 (0.1/step)
  - 통증 = 즉각적 negative reward → 회피 학습
- **3가지 욕구 (Drives)**:
  - `hunger_drive`: 에너지 낮으면 → 음식 찾기 욕구 ↑
  - `safety_drive`: 안전 낮으면 → 안전 추구 욕구 ↑
  - `rest_drive`: 피로 높으면 → 휴식 욕구 ↑
- **비선형 긴급성**: 상태가 critical threshold 도달 시 drive 급격히 증가
  - critical_energy = 0.2, critical_safety = 0.3, critical_fatigue = 0.8
- **인과적 연결**:
  - D) drives → epsilon (탐험률)
    - safety_drive ↑ → 탐험 ↑ (안전한 곳 찾기)
    - rest_drive ↑ → 탐험 ↓ (에너지 보존)
    - hunger_drive critical → 필사적 탐색
  - E) fatigue → learning rate
    - 피로/굶주림 → 학습 효율 ↓ (뇌도 에너지 필요!)
    - 위험 상황 → 학습 효율 ↑ (생존 기억 강화)
- **왜 중요한가?**:
  - 행동에 "이유"가 생김 (보상 극대화를 넘어선 동기)
  - 가치 갈등이 자연스럽게 발생 (배고픈데 피곤한 상황)
  - 감정의 생물학적 기초 (fear = low safety + high uncertainty)

### 13. Goal-Directed Behavior (목표 지향 행동)
- **핵심 개념**: "지금 뭘 집중해야 해?"
- **4가지 서브목표**:
  - `SAFE`: 안전 확보 (포식자 회피, 통증 감소)
  - `FEED`: 먹이 확보 (에너지 회복)
  - `REST`: 휴식 (피로 회복, 움직임 최소화)
  - `IDLE`: 특별한 목표 없음 (안정 상태)
- **목표 전환 조건**:
  - safety 낮거나 predator 가까움 → SAFE 우선
  - fatigue 높고 안정적 → REST 우선
  - energy 낮고 안정적 → FEED 우선
  - 둘 다 괜찮음 → IDLE
- **목표별 행동 bias**:
  - SAFE: predator_proximity -3.0, pain -4.0 (강한 회피)
  - FEED: food_proximity +2.5, energy +2.0 (음식 추구)
  - REST: movement_cost -1.5 (움직임 자체에 페널티)
- **히스테리시스**: min_goal_duration = 10 스텝
- **인과적 연결**:
  - M) goal → action score bias
  - N) internal state → goal switching
  - O) fatigue → REST goal activation

### 14. Rollout System (2-3 스텝 상상)
- **핵심 개념**: "이 방향으로 가면, 그 다음엔 어떻게 될까?"
- **철학적 의미**:
  - 1스텝 상상: "저기 가면 아플 거야"
  - 2-3스텝 상상: "저기 가면, 그 다음 포식자가 따라와서 아플 거야"
  - **더 깊은 "생각"** → 우회 경로, 위험 회피 가능
- **구현**:
  - depth = 2 (2스텝 미리 봄)
  - gamma = 0.9 (미래 discount)
  - 4×4 = 16 경로 평가
- **평가 함수**:
  - `U_total = U(s1) + γ × U(s2)`
  - 각 스텝에서 음식 접근, 포식자 회피, 벽 충돌 고려
- **LTM 통합**:
  - 각 가상 위치에서 기억 recall
  - depth 깊을수록 기억 영향 감소 (0.7^depth)
- **Goal 통합**:
  - 현재 goal의 bias가 rollout 평가에도 적용
  - REST 모드: 움직임 자체에 패널티
- **UI**:
  - 🔮 아이콘으로 예측 표시
  - 최선 경로 화살표 시퀀스 (↑→←)
  - 선택 이유 표시 ("2스텝: 음식 도달!", "2스텝: 포식자 접근 위험")
- **인과적 연결**:
  - P) rollout scores → imagination scores (장기 예측이 행동에 영향)
  - Q) goal biases → rollout evaluation (목표가 미래 평가에 영향)

### 15. Narrative Self (내러티브 자아)
- **핵심 개념**: "나는 어떤 존재인가?"
- **철학적 의미**:
  - 정체성은 부여되는 것이 아니라 **경험에서 발견**됨
  - "나는 조심성이 있다"는 "나는 자주 위험을 피했다"에서 나옴
  - 이 자기 서사가 다시 미래 행동에 영향 → **자기 실현적 정체성**
- **정체성 벡터 (6 traits, 0-1)**:
  - `bravery`: 위험 앞에서 전진 vs 회피 비율
  - `caution`: 위험 근처에서의 평균 regret
  - `curiosity`: 안전할 때 탐험 빈도
  - `persistence`: 목표 유지 시간
  - `adaptability`: 외부 이벤트에 전략 변경 빈도
  - `resilience`: 고통 후 회복 속도
- **자기 서사 문장** (템플릿 기반):
  - "나는 위험 앞에서도 물러서지 않는다." (bravery > 0.65)
  - "나는 안전할 때 새로운 것을 탐험한다." (curiosity > 0.65)
  - "나는 목표를 쉽게 포기하지 않는다." (persistence > 0.65)
- **행동 변조** (5-10% 영향):
  - epsilon_mod: curiosity ↑ → 탐험 ↑
  - safety_priority_mod: caution ↑ → SAFE 우선
  - goal_persistence_mod: persistence ↑ → 목표 전환 어려움
  - risk_tolerance_mod: bravery ↑ → 위험 감수
- **피드백 루프**:
  ```
  경험 → 정체성 형성 → 행동 변조 → 새 경험 → 정체성 강화
  ```
- **인과적 연결**:
  - R) behavioral history → identity vector
  - S) identity → behavior modulation (epsilon, goal priority)

---

## 최근 구현 완료

### 상태 전이 히스테리시스 ✅
- 감정/상태가 "머무는" 느낌 부여
- **min_state_duration = 15**: 최소 15 스텝 유지 후 전환 고려
- **transition_threshold = 8**: 새 상태가 8 스텝 지속 시 전환
- **Entry/Exit 임계값 분리**:
  - 진입 (Entry): 더 엄격 (예: CONFIDENT 진입 → confidence > 0.70)
  - 유지 (Exit): 더 느슨 (예: CONFIDENT 유지 → confidence > 0.50)
- 상태가 쉽게 바뀌지 않음 → 감정의 "관성" 표현

### Self-state → 예측 연결 ✅ (자기 설명 능력)
- **핵심**: 자기 상태가 예측 오차 해석에 영향
- `expected_error = base + uncertainty * 0.4 + effort * 0.2`
- `adjusted_error = raw_error - expected_error * 0.5`
- **효과**:
  - uncertainty ↑ → "예상 오차가 큼" → 실제 오차가 높아도 덜 놀람
  - "내가 혼란스러워서 이 정도 오차는 정상" = **자기 설명**
- `agency.py`에 `self_explanation` 필드 추가
- Self-Model이 "조절기"를 넘어 "자기 설명 모델"로 진화

### Attribution 분리 ✅ (내 탓 vs 외부 탓)
- **externality**: "세상이 문제야" (0-1)
  - external_pressure, wind 이벤트 빈도 기반
  - externality ↑ → exploration ↓ ("기다려, 상황이 지나갈 거야")
- **internal_fault**: "내 전략이 문제야" (0-1)
  - 외부 이벤트 없이 prediction error 높을 때
  - internal_fault ↑ → exploration ↑ ("전략 바꿔야 해")
- **핵심 통찰**: 둘 다 실패해도 원인이 다르면 반응도 달라야
- epsilon 조절에 반영: `-externality * 0.15 + internal_fault * 0.2`

### Homeostasis & Drives ✅ (생물학적 동기 시스템)
- **핵심 통찰**: 감정 전에 "필요(need)"가 있어야 함
- **구현 내용**:
  - `homeostasis.py`: HomeostasisSystem 클래스
  - 3 상태 (energy, safety, fatigue) + 3 욕구 (hunger, safety, rest drives)
  - Critical thresholds로 긴급 상태 감지
- **행동 연결**:
  - `main.py`: epsilon 조절에 homeostasis 반영
  - `main.py`: learning rate 조절 (피로 → 학습 효율 ↓)
- **UI**:
  - 3개 내부 상태 바 (energy/safety/fatigue)
  - 3개 욕구 바 (hunger/safety/rest drives)
  - Critical 상태 경고 표시
- **효과**: 에이전트가 "살아있는 느낌" - 배고프면 음식 찾고, 위험하면 조심하고, 피곤하면 쉬려 함

### Emotion System ✅ (감정 시스템)
- **핵심 통찰**: 감정은 항상성 위에 쌓이는 "느끼는" 레이어
- **구현 내용**:
  - `emotion.py`: EmotionSystem 클래스
  - 6가지 감정: fear, satisfaction, curiosity, pain, relief, anxiety
  - Valence-Arousal 표현 (긍정/부정, 각성 수준)
- **행동 연결**:
  - `main.py`: epsilon 조절 (fear ↓, curiosity ↑)
  - `main.py`: attention width 조절 (fear → 좁음, curiosity → 넓음)
  - `main.py`: learning rate 조절 (fear/pain → 강화, anxiety → 약화)
  - Freeze 반응: fear > 0.7 → 탐험 완전 정지
- **UI**:
  - 6개 감정 바 (부정: fear/pain/anxiety, 긍정: satisfaction/relief/curiosity)
  - Valence-Arousal 슬라이더
  - 주도 감정 이모지 표시
- **효과**: 에이전트가 "느끼는" 존재 - 포식자 보면 두려움, 음식 먹으면 만족, 안전해지면 안도

### Internal Simulation (내적 시뮬레이션) ✅
- **핵심 통찰**: 의식적 선택 = 행동 전에 결과를 상상하고 비교하는 것
- **구현 내용**:
  - `imagination.py`: ImaginationSystem 클래스
  - 4방향 예상 결과 계산 (에너지, 통증, 안전, 거리)
  - 욕구/감정에 따른 동적 가중치 조절
  - 예측 신뢰도 학습 (성공 시 ↑, 실패 시 ↓)
- **행동 연결**:
  - `main.py`: SNN 점수와 상상 점수 혼합
  - 신뢰도에 따라 반사↔숙고 비율 조절
  - Action source 추적 (snn/snn+imagine/imagine/explore/random)
- **UI**:
  - 4방향 점수 바 + 최선/최악 표시
  - 행동 이유 한국어 표시 ("음식 도달!", "위험 회피" 등)
  - 예측 신뢰도 바
  - 행동 출처 표시
- **철학적 의미**:
  - "포식자 쪽으로 가면 아플 거야" → **상상**
  - "그러니까 안 가야지" → **선택**
  - 이것이 바로 **의식적 숙고의 시작**

### Infant Bootstrap (B') ✅ (경험 기반 학습)
- **핵심 통찰**: 가치는 선천적으로 주어지면 안 됨 - 경험으로 배워야 함
- **문제 인식**:
  - 이전: SNN 가중치가 처음부터 "올바른 방향=70, 틀린 방향=10"
  - 이전: 에이전트가 "태어날 때부터" 음식/포식자 가치를 알고 있었음
  - 수정: 모든 가중치 = 40, 학습으로 방향성 획득
- **구현 내용**:
  - `main.py`: SNN s→h 가중치 모두 40으로 균등화
  - `environment.py`: Developmental phase system (infant/adult)
  - `main.py`: 호기심 내재 보상 (prediction error 기반)
  - `main.py`: 움직임 보상 (제자리/왕복 페널티)
- **철학적 의미**:
  - "음식을 먹어보니 배고픔이 나아져서 음식을 찾게 됨"
  - "포식자에게 부딪혀보니 아프니까 피하게 됨"
  - 학습 순서가 **인과적으로 올바름**
- **UI**: 헤더에 발달 단계 표시 (👶 INFANT / 🧑 ADULT)

### Long-term Memory v1 ✅ (장기 기억 시스템)
- **핵심 통찰**: 상상 + 기억 = 경험에 기반한 현명한 판단
- **구현 내용**:
  - `memory_ltm.py`: Episode dataclass + LongTermMemory 클래스
  - 에피소드 저장: (위치, 상태, 행동, 결과, 감정)
  - 감정 기반 중요도 계산 (통증 2배, 음식 1.5배)
  - 유사성 기반 recall (현재 위치 근처 3칸)
  - 기억 decay + recall시 boost
- **철학적 의미**:
  - 상상: "저쪽으로 가면 어떨까?"
  - 기억: "저번에 저기서 아팠어!"
  - 결합: **"그러니까 안 가는 게 좋겠다"** = 의식적 회고

### Long-term Memory v1.1 ✅ (경험 기반 델타 시스템)
- **핵심 문제 해결**:
  - 문제 #1: 위치 기반 기억만 사용 → "위치 미신" 발생
  - 문제 #2: 고정 점수(-1.0, +0.6)가 "규칙"처럼 느껴짐
- **해결책 #1: Context 유사성 매칭**:
  - Episode에 상황 정보 저장 (predator_near, energy_low, was_fleeing)
  - 같은 위치라도 상황이 다르면 다른 기억으로 취급
  - Context 일치도에 따라 회상 관련성 부여 (최대 1.45배)
- **해결책 #2: 델타 기반 점수 계산**:
  - Episode에 실제 결과 변화량 저장 (delta_energy, delta_pain, delta_safety)
  - 점수 = `1.0 × Δenergy - 3.0 × Δpain + 1.5 × Δsafety`
  - Importance-weighted average로 여러 기억 종합
- **철학적 의미**:
  - 이전: "여기서 아팠으니 -1점" (프로그래머가 정한 규칙)
  - v1.1: "여기서 pain이 0.8 증가했어" → 계산된 영향 = -2.4 (실제 경험)
  - **기억이 "규칙"이 아닌 "경험"처럼 느껴짐**
- **UI 개선**:
  - 방향별 회상 상세 정보 표시 (delta_pain, delta_energy, memory_count)

### Long-term Memory v1.2 ✅ (신뢰도 + 안정성)
- **핵심 개선**:
  - 문제: 한 번의 우연(운 좋게 먹음, 운 나쁘게 맞음)만으로 강한 회피/추구 발생
  - 해결: 경험 횟수 기반 신뢰도(confidence) 적용
- **신뢰도 시스템**:
  - `confidence = sqrt(n) / (sqrt(n) + k)` where k=1.5
  - n=1 경험 → confidence ≈ 0.40 (영향력 40%)
  - n=4 경험 → confidence ≈ 0.57 (영향력 57%)
  - n=9 경험 → confidence ≈ 0.67 (영향력 67%)
  - `U_effective = U_raw × confidence`
- **Δ 클립 (스케일 안정화)**:
  - 모든 delta 값을 [-1, 1] 범위로 클립
  - 환경 파라미터 변해도 기억 영향이 폭주 안 함
- **기억 vs 현재 충돌 해결**:
  - uncertainty 높음 → 기억 의존 ↑ (감각이 불확실)
  - uncertainty 낮음 → 기억 의존 ↓ (감각이 확실)
  - externality 높음 → 기억 저장 약화 ("이건 내 행동 결과가 아니야")
- **UI 개선**:
  - confidence 값 표시
  - u_raw vs computed_score 분리 표시

### Goal-Directed Behavior v1.1 ✅ (서브목표 스위칭 + REST)
- **핵심 개념**: "지금 뭘 집중해야 해?"
- **거창한 계획 없이 작게 시작**:
  - 복잡한 planning 대신 상태 기반 목표 전환
  - 목표가 action score에 bias 추가
- **4가지 서브목표**:
  - `SAFE`: 안전 확보 (포식자 회피, 통증 감소)
  - `FEED`: 먹이 확보 (에너지 회복)
  - `REST`: 휴식 (피로 회복, 움직임 최소화)
  - `IDLE`: 특별한 목표 없음 (안정 상태)
- **목표 전환 조건**:
  - safety 낮거나 predator 가까움 → SAFE 우선
  - fatigue 높고 안정적 → REST 우선
  - energy 낮고 안정적 → FEED 우선
  - 둘 다 괜찮음 → IDLE
- **목표별 행동 bias**:
  - SAFE: predator_proximity -3.0, pain -4.0 (강한 회피)
  - FEED: food_proximity +2.5, energy +2.0 (음식 추구)
  - REST: movement_cost -1.5 (움직임 자체에 페널티)
- **히스테리시스**:
  - min_goal_duration = 10 스텝 (목표가 쉽게 안 바뀜)
  - 긴급 상황(pain > 0.5)은 즉시 SAFE로 전환
  - REST 도중 굶주림 → FEED로 전환 가능
- **인과적 연결**:
  - M) goal → action score bias (목표가 행동 선택에 영향)
  - N) internal state → goal switching (내부 상태가 목표 결정)
  - O) fatigue → REST goal (피로가 휴식 목표 활성화)
- **철학적 의미**:
  - 이전: 모든 것을 균등하게 고려
  - v1.1: "지금은 쉬는 게 중요해" → 움직임 최소화
  - **목표가 있는 존재처럼 행동**

### Rollout v1 ✅ (2-3 스텝 상상)
- **핵심 개념**: "이 방향으로 가면, 그 다음엔?"
- **철학적 의미**:
  - 1스텝 상상만으로는 함정 회피 불가
  - 2스텝 보면: "저기 가면, 그 다음 포식자가 따라와서..."
  - **더 "생각하는" 느낌**
- **구현**:
  - `rollout.py`: RolloutSystem 클래스
  - depth = 2, gamma = 0.9
  - 16개 경로 (4×4) 평가
- **통합**:
  - imagination scores에 rollout 결과 추가
  - rollout_weight = 0.3 × confidence
  - LTM recall도 가상 위치에서 수행 (depth 감쇠)
- **UI**:
  - 🔮 아이콘으로 예측 표시
  - 화살표 시퀀스로 최선 경로 표시
  - 선택 이유 ("2스텝: 음식 도달!")

### Narrative Self v1 ✅ (내러티브 자아)
- **핵심 개념**: "나는 어떤 존재인가?" - 경험에서 정체성 형성
- **철학적 의미**:
  - 정체성은 **부여되는 것이 아니라 발견됨**
  - "나는 조심성이 있다" ← "나는 자주 위험을 피했다"
  - 자기 서사 → 행동 변조 → **자기 실현적 정체성**
- **구현**:
  - `narrative.py`: NarrativeSelf 클래스
  - 6가지 정체성 trait (0-1)
  - 행동 이벤트 수집 → 20스텝마다 정체성 업데이트
- **정체성 벡터**:
  - `bravery`: 위험 앞에서 전진/회피 비율
  - `caution`: 위험 근처에서 regret 평균
  - `curiosity`: 안전할 때 탐험 빈도
  - `persistence`: 목표 유지 시간
  - `adaptability`: 외부 이벤트에 전략 변경
  - `resilience`: 고통 후 회복 속도
- **행동 변조** (5-10% 영향력):
  - epsilon_mod: curiosity ↑ → 탐험 ↑
  - safety_priority_mod: caution ↑ → SAFE 우선
  - goal_persistence_mod: persistence ↑ → 목표 고수
  - risk_tolerance_mod: bravery ↑ → 위험 감수
- **핵심 피드백 루프**:
  - 경험 → 정체성 → 행동 변조 → 새 경험 → 정체성 강화
  - **"나는 조심하는 존재야" → 조심스럽게 행동 → 더 조심성↑**

---

## Self-Model 완성 상태

| 기능 | 상태 | 역할 |
|------|------|------|
| 히스테리시스 | ✅ | 감정이 "머무는" 느낌 |
| 자기 설명 | ✅ | 오차를 자기 상태로 해석 |
| Attribution 분리 | ✅ | 내 탓 vs 외부 탓 구분 |

이제 Self-Model은:
1. **자기 상태를 안다** (confidence, uncertainty, effort, ...)
2. **자기 상태로 상황을 설명한다** (오차 해석)
3. **원인을 자기/외부로 귀속한다** (attribution)
4. **귀속에 따라 다르게 반응한다** (exploration 조절)

---

## 의식적 생각 루프 (Conscious Thought Loop)

```
(내 상태 인식) → (미래를 가정해봄) → (선택) → (결과를 회상/갱신)
     ↑                                              ↓
     └──────────────────────────────────────────────┘
```

| 단계 | 구성요소 | 상태 |
|------|----------|------|
| 내 상태 인식 | Self-Model, Homeostasis, Emotions | ✅ |
| **자기 정체성** | **Narrative Self (경험에서 형성)** | ✅ |
| 미래를 가정해봄 | Internal Simulation (1-step) + Rollout (2-3 step) | ✅ |
| 선택 | SNN + Imagination + Rollout + **정체성 변조** 혼합 결정 | ✅ |
| 결과를 회상/갱신 | Long-term Memory (Episodic) | ✅ |

**의식적 생각 루프 + 자기 정체성!** 에이전트는 이제:
1. 자기 상태를 인식하고 (Self-Model, Homeostasis, Emotions)
2. **"나는 어떤 존재인가"를 알고 (Narrative Self)**
3. 행동 결과를 **2-3스텝 앞까지** 상상하고 (Forward Model + Rollout)
4. 과거 경험을 떠올리고 (Long-term Memory)
5. 목표에 따라 가중치를 조절하고 (Goal System)
6. **자기 정체성에 맞게 행동을 조절하고 (Narrative Modulation)**
7. 이 모든 정보를 종합해서 선택한다 (SNN + Imagination + Rollout + Memory + Identity)

---

## 다음 단계 로드맵

### Priority 1: 꿈/수면 시스템 (Dream/Sleep) ⏳
- **핵심**: 기억 정리와 통합 - "뇌가 살아있는 느낌"
- **구현 예정**:
  - 조건: fatigue 높거나 에피소드 끝날 때
  - LTM에서 감정 강한 top-k 에피소드 샘플
  - Replay하면서 시냅스/가치 약하게 업데이트
  - 비슷한 에피소드는 "요약"으로 압축 (프로토-개념)
- **기대 효과**:
  - 깨어있을 때 학습 불안정 → 자고 나면 안정화
  - 최근 큰 사건이 행동에 더 오래 남음

### Priority 2: UI 시각화 강화
- **핵심**: 내러티브/꿈이 생긴 후 관찰 용이성 확보
- **구현 예정**:
  - "오늘의 자아 요약" 표시
  - "꿈에서 리플레이된 기억들" 표시
  - "꿈 전/후 행동 변화" 비교
  - 정체성 벡터 시각화

### Priority 3: 호기심 v2 (Information Gain)
- **핵심**: 정보 획득 기반 내재 동기
- **구현 예정**:
  - 예측 불확실성 높은 곳 → 호기심
  - 정보 획득 = 보상
  - 탐험의 "이유"가 더 정교해짐

### Later: Proto-언어 & Depth-3 Rollout
- Proto-언어: 내러티브 자아가 자리잡은 후
- Depth-3 Rollout: 성능은 오르지만 의식 체감은 작음

---

## 기술적 주의사항
- Wall collision penalty: -0.3 (너무 크면 가중치 degradation)
- Wall collision은 safety에 영향 없음 (위험이 아니라 그냥 막힘)
- Large food: 현재 disabled (spawn_chance = 0.0)
- 서버 재시작은 Claude가 직접 수행 (사용자에게 시키지 않기)

## 시각화 요소
- **바람**: 파란 화살표 3개가 바람 방향에서 나타남
- **포식자**: 빨간 원 + 눈 (가까우면 붉은 글로우)
- **에이전트**: 청록색 사각형
- **음식**: 초록 원 (small), 노란 별 (large)
