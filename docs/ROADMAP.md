# Genesis Brain 로드맵

> 최종 목표: 인간 뇌처럼 학습하고 개념을 형성하는 인공 뇌

---

## 현재 상태: Phase L16 + 환경 고도화 (24,510 뉴런, 800×800 맵)

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase L16: Sparse Expansion + 800맵 (2026-03-25)               ║
║  KC(1500×2) sparse expansion → 30K 학습 연결                    ║
║  food_eye 35.0 하드코딩 제거 → R-STDP 학습 접근                ║
║  800×800맵, 장애물 1개, Rich Zones 2개                          ║
║  생존율 65%, Reward 2.60%, Selectivity 0.67                     ║
╚═══════════════════════════════════════════════════════════════╝
```

### Phase 상태

| Phase | 영역 | 뉴런 | 상태 | 생존율 |
|-------|------|------|------|--------|
| 1 | 뇌간/척수 (Slither.io) | 13,800 | ✓ 졸업 | - |
| 2a | 시상하부 (항상성) | +1,400 | ✓ 완료 | - |
| 2b | 편도체 (공포) | +1,000 | ✓ 완료 | - |
| 3a-c | 해마 (공간 기억) | +600 | ✓ 완료 | - |
| 4 | 기저핵 (습관/보상) | +900 | ✓ 완료 | - |
| 5 | 전전두엽 (목표/억제) | +400 | ✓ 완료 | - |
| 6a | 소뇌 (운동 조정) | +550 | ✓ 완료 | - |
| 6b | 시상 (감각 게이팅) | +350 | ✓ 완료 | - |
| 7 | 통합 테스트 | - | ✓ 완료 | 80% |
| 8 | V1 시각 피질 | +400 | ✓ 완료 | - |
| 9 | V2/V4 고차 시각 | +600 | ✓ 완료 | - |
| 10 | IT Cortex (측두엽) | +1,000 | ✓ 완료 (M1!) | 80% |
| 11 | 청각 피질 (A1/A2) | +900 | ✓ 완료 | **60%** |
| 12 | 다중 감각 통합 (STS) | +800 | ✓ 수정 완료 | 60% |
| 13 | 두정엽 (PPC) | +1,000 | ✓ 수정 완료 | 60% |
| 14 | 전운동 피질 (PMC) | +800 | ✓ 수정 완료 | 60% |
| 15a | 사회적 뇌 (Social Brain) | +1,500 | ✓ 완료 | 60% |
| 15b | 거울 뉴런 (Mirror Neurons) | +600 | ✓ 완료 | 65% |
| 15c | Theory of Mind (ToM) | +500 | ✓ 완료 | **70%** |
| 16 | 연합 피질 (Association Cortex) | +700 | ✓ 완료 | 50% |
| 17 | 언어 회로 (Broca/Wernicke) | +1,000 | ✓ 완료 | 55% |
| 18 | 작업 기억 확장 (WM Expansion) | +620 | ✓ 완료 | 50% |
| 19 | 메타인지 (Metacognition) | +380 | ✓ 완료 | 55% |
| 20 | 자기 모델 (Self-Model) | +440 | ✓ 완료 | **50%** |
| **L1** | **R-STDP + BG Push-Pull** | **-** | **✓ 완료** | **100%** |
| **L2** | **D1/D2 MSN 분리** | **-** | **✓ 완료** | **95%** |
| **L3** | **Homeostatic R-STDP** | **-** | **✓ 완료** | **90%** |
| **L4** | **Anti-Hebbian D2** | **-** | **✓ 완료** | **100%** |
| **L5** | **지각 학습 (피질 R-STDP + 다중 음식)** | **+800** | **✓ 완료** | **60%** |
| **L6** | **예측 오차 회로 (Predictive Coding)** | **+200** | **✓ 완료** | **45%** |
| **L7** | **음식 유형별 BG 학습 (Discriminative BG)** | **-** | **✓ 완료** | **45%** |
| **L8** | **혐오 도파민 딥 (Aversive Dopamine Dip)** | **-** | **✓ 완료** | **60%** |
| **L9** | **피질→BG 하향 연결 (Cortical→BG Top-Down)** | **-** | **✓ 완료** | **70%** |
| **L10** | **TD Learning (NAc Critic → RPE 도파민)** | **+110** | **✓ 완료** | **65%** |
| **L11** | **SWR Replay (해마 기억 재생)** | **+200** | **✓ 완료** | **45%** |
| **L12** | **Global Workspace (주의 기반 경쟁)** | **+160** | **✓ 완료** | **50%** |
| **L13** | **조건 맛 혐오 (Garcia Effect)** | **0** | **✓ 완료** | **55%** |
| **L14** | **에이전시 감지 (Agency Detection)** | **+50** | **✓ 완료** | **~65%** |
| **L15** | **내러티브 자기 (Narrative Self)** | **0** | **✓ 완료** | **55%** |
| **L16** | **Sparse Expansion (KC 패턴 분리)** | **+3,400** | **✓ 완료** | **65%** |
| **Opt** | **Spike recording 배치화** | **0** | **✓ 완료** | GPU 60% |
| **Fix** | **food_eye 하드코딩→학습 교체** | **0** | **✓ 완료** | **89%** (400) |
| **E1** | **장애물 (obstacle_rays 분리)** | **+400** | **✓ 완료** | 60% |
| **E2** | **Rich Zones (영역 다양성)** | **0** | **✓ 완료** | Reward+0.46pp |
| **Map** | **맵 확장 400→800** | **0** | **✓ 완료** | **65%** (800) |

### Phase 12-14 수정 이력

**1차 수정 (2026-02-02): Motor 직접 연결 비활성화 → 35% 생존**

| Phase | 문제점 | 수정 |
|-------|--------|------|
| 12 | STS→Motor 충돌 | Motor 직접 연결 제거 (0.0) |
| 13 | PPC→Motor 노이즈 | Motor 직접 연결 제거 (0.0) |
| 14 | PMC가 회피 반응 지연 | Motor_Prep 약화 (15→2), PMd/PMv 비활성화 |

**2차 수정 (2026-02-08): 간접 경로 약화 → 60% 생존 ✓**

| 경로 | 문제점 | 수정 |
|------|--------|------|
| STS→Amygdala | 18.0 → Fear 과다 증폭 | **18→8** |
| STS→Hippocampus | 15.0 → Food_Memory 간접 간섭 | **15→8** |
| PPC→Hippocampus | 10.0 → 노이즈 | **10→5** |

**교훈: 직접 연결뿐 아니라 간접 경로(Amygdala, Hippocampus 경유)도 Motor에 간섭할 수 있음**

---

## 단기 계획 (Phase 8-10) ✓ 완료

### Phase 8: V1 시각 피질 ✓ 완료
- **구조**: V1_Food L/R (200) + V1_Danger L/R (200)
- **기능**: Lateral Inhibition, 방향 정보 보존
- **연결**: Food Eye → V1 → Motor/Hippocampus

### Phase 9: V2/V4 고차 시각 ✓ 완료
- **구조**: +600 뉴런 (V2_Edge 300 + V4_Object 300)
- **기능**: 에지 검출, 물체 분류, WTA 경쟁
- **연결**: V1 → V2 (수렴) → V4 (분류) → Hippocampus/Amygdala/Dopamine
- **Top-Down**: Hunger/Fear/Goal → V2/V4 (주의 조절)

### Phase 10: 측두엽 (하측두 피질) ✓ 완료 - M1 마일스톤!
- **구조**: +1,000 뉴런 (IT_Food/Danger/Neutral + Association + Buffer)
- **기능**: 물체 기억, 범주화, "음식/위험" 개념 표상
- **연결**: V4 → IT → Hippocampus/Amygdala/PFC/Motor
- **양방향**: IT ↔ Hippocampus (기억 저장/인출)
- **성능**: 생존율 80%, Reward Freq 3.05%

**M1 마일스톤 달성**: 10,000 뉴런, 시각 개념 형성

---

## 중기 계획 (Phase 11-15)

### Phase 11: 청각 피질 ✓ 완료 (검증 통과)
- **구조**: +900 뉴런 (Sound Input 400 + A1 300 + A2 200)
- **기능**: 소리 자극 처리, 좌우 방향 정보, 청각-공포 조건화
- **연결**: Sound → A1 → Amygdala/IT/Motor
- **Top-Down**: Fear/Hunger → A1 (주의 조절)
- **검증 결과 (2026-02-02)**: 생존율 60% ✓, Reward Freq 2.97% ✓

### Phase 12: 다중 감각 통합 ✓ 수정 완료
- **구조**: +800 뉴런 (STS_Food/Danger 400 + Congruence/Mismatch 250 + Buffer 150)
- **기능**: 시청각 통합, 일치/불일치 감지
- **연결**: IT/A1 → STS → Hippocampus/Amygdala/PFC (Motor 직접 연결 비활성화)
- **수정 (2026-02-08)**: STS→Motor 비활성화, STS→Amygdala 18→8, STS→Hippo 15→8

### Phase 13: 두정엽 ✓ 수정 완료
- **구조**: +1,000 뉴런 (PPC_Space L/R 300 + PPC_Goal Food/Safety 300 + Attention 200 + Path_Buffer 200)
- **기능**: 공간 표상, 목표 벡터 계산, 공간 주의, 경로 계획 기초
- **연결**: V1/IT/STS → PPC_Space → PPC_Goal → PMC (Motor 직접 연결 비활성화)
- **수정 (2026-02-08)**: PPC→Motor 비활성화, PPC→Hippo 10→5

### Phase 14: 전운동 피질 ✓ 수정 완료
- **구조**: +800 뉴런 (PMd L/R 200 + PMv Approach/Avoid 200 + SMA 150 + pre-SMA 100 + Motor_Prep 150)
- **기능**: 공간/물체 기반 운동 계획, 시퀀스 생성, 운동 준비
- **연결**: PPC → PMd, IT/STS → PMv, PFC → SMA, Motor_Prep → Motor (2.0)
- **수정 (2026-02-08)**: Motor_Prep 15→2, PMd/PMv→Motor 비활성화
- **통합 검증**: 생존율 60% ✓, Reward Freq 3.30% ✓, Pain Avoidance 91.1% ✓

### Phase 15a: 사회적 뇌 - 환경 + 감지 ✓ 완료
- **환경**: ForagerGym에 NPC 에이전트 추가 (forager/predator 행동)
- **감각**: Agent Eye L/R (200×2) + Agent Sound L/R (100×2)
- **구조**: +1,500 뉴런
  - STS_Social (200): 다른 에이전트 시청각 통합
  - TPJ (300): Self(100)/Other(100)/Compare(100) - 자기-타자 비교
  - ACC (200): Conflict(100)/Monitor(100) - 갈등 감지
  - Social_Valuation (200): Approach(100)/Avoid(100) - 접근/회피 동기
- **연결**: Motor 직접 연결 없음 (0.0), 간접 경로 ≤6.0
- **검증 (2026-02-08)**: 생존율 60% ✓, Reward Freq 3.33% ✓, Pain Avoidance 90.9% ✓

### Phase 15b: 거울 뉴런 & 관찰 학습 ✓ 완료
- **환경**: NPC 먹기 이벤트 감지 (5개 관찰 채널 추가)
- **구조**: +600 뉴런
  - Social_Observation (200): NPC 목표지향 움직임 감지
  - Mirror_Food (150): 자기+타인 먹기 거울 뉴런
  - Vicarious_Reward (100): 관찰 예측 오차 (대리 보상)
  - Social_Memory (150): 사회적 음식 위치 기억 (Hebbian)
- **학습**: Vicarious_Reward → Social_Memory Hebbian 학습
- **연결**: Motor 직접 연결 없음 (0.0), 간접 경로 ≤6.0
- **검증 (2026-02-08)**: 생존율 65% ✓, Reward Freq 3.34% ✓, Pain Avoidance 91.0% ✓

### Phase 15c: Theory of Mind & 협력/경쟁 ✓ 완료
- **환경**: NPC 의도/경쟁 감지 (3개 관찰 채널 추가)
- **구조**: +500 뉴런
  - ToM_Intention (100): NPC 의도 추론 ("NPC가 음식을 원한다")
  - ToM_Belief (80): NPC 신념 추적
  - ToM_Prediction (80): NPC 행동 예측 (Recurrent)
  - ToM_Surprise (60): 예측 오차 (NPC 예상 외 행동)
  - CoopCompete_Coop (80): 협력 가치 (Hebbian 학습)
  - CoopCompete_Compete (100): 경쟁 감지
- **학습**: ToM_Intention → Coop Hebbian 학습 (음식 먹기 시 트리거)
- **연결**: Motor 직접 연결 없음 (0.0), 간접 경로 ≤6.0, Coop↔Compete WTA (-8.0)
- **검증 (2026-02-08)**: 생존율 70% ✓, Reward Freq 2.85% ✓, Pain Avoidance 91.1% ✓, Hebbian avg_w 5.0→6.45

**Phase 15 완료**: 16,100 뉴런

### Phase 16: 연합 피질 (Association Cortex) ✓ 완료
- **구조**: +700 뉴런
  - Assoc_Edible (120): "먹을 수 있는 것" 초범주 (IT_Food + STS_Food + A1_Food + Social_Memory + Mirror_Food)
  - Assoc_Threatening (120): "위험한 것" 초범주 (IT_Danger + STS_Danger + A1_Danger + Fear)
  - Assoc_Animate (100): "살아있는 것" 초범주 (ToM_Intention + Social_Obs + STS_Social)
  - Assoc_Context (100): "익숙한 장소" 맥락 (Place_Cells + PPC_Space + Food_Memory)
  - Assoc_Valence (80): "좋다/나쁘다" 가치 (Dopamine + Edible + Threatening)
  - Assoc_Binding (100): 교차 연합 학습 (Hebbian DENSE)
  - Assoc_Novelty (80): 새로운 조합 탐지 (IT_Neutral + STS_Mismatch - Binding)
- **학습**: Edible→Binding, Context→Binding Hebbian DENSE (eta=0.06, w_max=18.0)
- **연결**: Motor 직접 연결 없음 (0.0), 간접 경로 ≤6.0, Edible↔Threatening WTA (-6.0)
- **검증 (2026-02-08)**: 생존율 50% ✓, Reward Freq 3.46% ✓, Pain Avoidance 91.0% ✓, Hebbian avg_w 2.0→7.12

**Phase 16 완료**: 16,800 뉴런

### Phase 17: 언어 회로 (Broca/Wernicke) ✓ 완료
- **구조**: +1,000 뉴런
  - Call Input (200 SensoryLIF): Call_Food L/R (50×2) + Call_Danger L/R (50×2) - NPC 발성 청취
  - Wernicke's Area (280 LIF): Food(80)/Danger(80)/Social(60)/Context(60) - 발성 이해
  - Broca's Area (280 LIF): Food(80)/Danger(80)/Social(60)/Sequence(60) - 발성 생산
  - Vocal Gate / PAG (80 SensoryLIF): 발성 AND 게이트 (Arousal + Broca - Fear 억제)
  - Call Mirror (80 LIF): 듣기+생산 양쪽 활성 (거울 공명)
  - Call Binding (80 LIF, Hebbian DENSE): 소리-의미 연합 학습
- **연결**: Arcuate Fasciculus (Wernicke↔Broca 양방향), ~42 시냅스
- **환경**: NPC 발성 (food call, danger call) + 에이전트 발성 출력 (vocalize_type)
- **학습**: Call Binding Hebbian DENSE (eta=0.06, w_max=18.0) - 음식/위험 call 의미 연합
- **연결 안전**: Motor 직접 연결 없음 (0.0), 간접 경로 ≤6.0, Pain Push-Pull 간섭 <5%
- **검증 (2026-02-08)**: 생존율 55% ✓, Reward Freq 3.19% ✓, Pain Avoidance 90.9% ✓

**Phase 17 완료**: 17,800 뉴런

### Phase 18: 작업 기억 확장 (WM Expansion) ✓ 완료
- **구조**: +620 뉴런
  - WM_Thalamic (100 LIF): MD thalamus analog, WM↔Thalamic 양방향 유지 루프
  - WM_Update_Gate (50 SensoryLIF): 도파민/갈등/신기 게이트 (I_input: dopamine×12 + novelty×15 + conflict×15)
  - Temporal_Recent (80 LIF): 현재 이벤트 ~1초 유지 (recurrent 7.0)
  - Temporal_Prior (40 LIF): 이전 이벤트 ~3초 유지 (recurrent 4.0)
  - Goal_Pending (80 LIF): 대기 중 다음 목표, 활성 목표가 억제 (-8.0)
  - Goal_Switch (70 LIF): 문맥 전환 감지기 (burst-only, self-inhibition -8.0)
  - WM_Context_Binding (100 LIF, Hebbian DENSE): 시간 패턴 학습 (eta=0.05, w_max=16.0)
  - WM_Inhibitory (100 LIF): 국소 억제, 이중 루프 폭주 방지
- **연결**: ~42 시냅스, 1 Hebbian DENSE (Temporal_Recent → WM_Context_Binding)
- **설계 원칙**: LSTM-style 3-gate 미사용 → 시상 릴레이 + 도파민 + TRN 게이팅으로 생물학적 구현
- **연결 안전**: Motor 직접 연결 없음 (0.0), 간접 경로 3+ 홉, Pain Push-Pull 간섭 <1%
- **검증 (2026-02-08)**: 생존율 50% ✓, Reward Freq 3.02% ✓, Pain Avoidance 91.0% ✓

**Phase 18 완료**: 18,420 뉴런

---

### Phase 19: 메타인지 (Metacognition) ✓ 완료
- **구조**: +380 뉴런
  - Meta_Confidence (80 LIF): 확신 누적기 (전방 섬엽 analog)
  - Meta_Uncertainty (80 LIF): 불확실성 누적기 (dACC error-likelihood analog)
  - Meta_Evaluate (80 SensoryLIF): 자기평가 게이트 (mPFC analog, I_input: uncert×6 - confid×5 + DA×4)
  - Meta_Arousal_Mod (70 LIF): 불확실성→각성 커플링 (Locus coeruleus/NE analog)
  - Meta_Inhibitory (70 LIF): 국소 억제 균형
- **메커니즘**: Confidence vs Uncertainty WTA 경쟁 (-5.0), Evaluate gate가 불확실성 높을 때 목표 전환/각성 증폭
- **학습**: assoc_valence → meta_confidence Hebbian DENSE (eta=0.04, w_max=14.0, avg_w 2.0→6.22)
- **연결 안전**: Motor 직접 연결 없음 (0.0), 모든 출력 ≤2.0 (very gentle modulator)
- **검증 (2026-02-11)**: 생존율 55% ✓, Reward Freq 2.96% ✓, Pain Avoidance 91.0% ✓

**Phase 19 완료**: 18,800 뉴런

### Phase 20: 자기 모델 (Self-Model) ✓ 완료
- **구조**: +440 뉴런
  - Self_Body (80 SensoryLIF): 내수용감각 통합 (섬엽/Insular Cortex analog, I_input: energy×8 + hunger×-6 + satiety×5)
  - Self_Efference (80 LIF): 운동 명령 복사 (소뇌 Efference Copy analog)
  - Self_Predict (70 SensoryLIF): 감각 결과 예측 (소뇌 순행 모델, I_input: efference_rate×6 + food_eye×5)
  - Self_Agency (70 LIF): 행위주체감 (각회/Angular Gyrus analog)
  - Self_Narrative (80 LIF): 자기 서사 (DMN/mPFC analog, 자기 참조 유지 recurrent)
  - Self_Inhibitory (60 LIF): 국소 억제 균형
- **메커니즘**: Body(내수용) + Efference(운동복사) → Predict(예측) + Agency(행위주체) → Narrative(자기 서사) 계층적 통합
- **학습**: self_body → self_narrative Hebbian DENSE (eta=0.04, w_max=14.0)
- **연결 안전**: Motor 직접 연결 없음 (0.0), 모든 출력 ≤1.5 (whisper level modulator)
- **검증 (2026-02-11)**: 생존율 50% ✓, Reward Freq 3.17% ✓, Pain Avoidance 90.8% ✓

**Phase 20 완료**: 19,240 뉴런

---

## 학습 기반 (Learning Foundation)

> Phase 1-20에서 뇌 구조 완성. 이제 실제 학습이 일어나게 만드는 단계.

### Phase L1: R-STDP + BG Push-Pull ✓ 완료
- **문제**: 95%+ 시냅스가 하드코딩, Episode 1 = Episode 20 성능 동일 (학습 없음)
- **구현**: BG 측면화 (Striatum L/R), Food_Eye→Striatum R-STDP, BG Push-Pull (교차 억제)
- **환경**: food_approach_signal 셰이핑 리워드 추가
- **결과**: 100% 생존, Food 48.4→60.8 (+25%), Correct Turn +3.8pp
- **한계**: Reward Freq 1.81% < 2.5% 목표, Go-NoGo 상쇄 (단일 Striatum이 Go+NoGo 동시 강화)

### Phase L2: D1/D2 MSN 분리 ✓ 완료
- **문제**: Go-NoGo 상쇄 → R-STDP 학습 효과 상쇄
- **해결**: D1 MSN (Go, R-STDP 학습) / D2 MSN (NoGo, Static) 분리
- **구조**: D1_L/R (100×2) + D2_L/R (100×2) = 400 뉴런 (변동 없음, 총 19,240 유지)
- **도파민**: MSN 레벨로 이동 (DA→D1 +15, DA→D2 -12)
- **D1↔D2 경쟁**: -5.0 측면 억제 (SPARSE 0.1)
- **food_weight 튜닝**: 25→30→35→38(fail)→40(fail) → **35 최적**
- **검증 (2026-02-12)**: 생존율 95% ✓, Reward Freq 2.33% (L1 대비 +28.7%), Pain Avoidance 100% ✓
- **한계**: Reward Freq 2.33% < 2.5% 목표 (food_weight=35가 최적, 38+ 시 starvation)
- **분석**: R-STDP가 ep 1-2에서 w_max=5.0 포화 → 점진적 학습 대신 빠른 수렴

### Phase L3: Homeostatic R-STDP ✓ 완료
- **문제**: R-STDP가 ep 1-2에서 w_max=5.0 포화 (trace 무한 누적 + 높은 eta)
- **해결**: (1) 적격 추적 상한 1.0, (2) eta 0.001→0.0005 감소, (3) 항상성 가중치 감쇠 0.00003/step
- **생물학적 근거**: 항상성 시냅스 스케일링 (homeostatic synaptic scaling)
- **학습 곡선**: w=1.35(ep1) → 2.23(ep5) → 3.15(ep10) → 3.91(ep15) → 4.29(ep20)
- **Food 향상**: ep 1-5 avg 48.4 → ep 11-15 avg 65.8 (+36%)
- **검증 (2026-02-12)**: 생존율 90% ✓, Reward Freq 2.13%, Pain 0% ✓
- **트레이드오프**: 집계 Reward Freq 하락 (L2: 2.33% → L3: 2.13%) - 초반 약한 가중치가 평균 하락시킴
- **의의**: 포화→점진적 학습으로 전환. 인간 뇌의 시냅스 가소성과 유사한 학습 역학 달성

### Phase L4: Anti-Hebbian D2 ✓ 완료
- **문제**: D2 MSN이 Static (학습 불가) → D1 학습만으로 Go/NoGo 비율 변화 제한적
- **해결**: D2에 Anti-Hebbian 학습 추가 (도파민 시 가중치 감소 = NoGo 약화)
- **생물학적 근거**: D2 MSN LTD (long-term depression) - 도파민이 D2 시냅스 약화
- **메커니즘**: D1 R-STDP (보상→강화) + D2 Anti-Hebbian (보상→약화) = 경쟁적 Go/NoGo 학습
- **D2 trace**: pre-synaptic (food_eye) 기반 — D1↔D2 경쟁으로 D2 post-synaptic 발화 억제되므로
- **파라미터**: eta_d2=0.0003 (D1의 60%), w_min=0.1 (완전 소멸 방지)
- **학습 곡선**: D1: 1.37→4.53 (20ep, 점진적), D2: 1.0→0.10 (ep5에서 w_min 도달)
- **검증 (2026-02-14)**: 생존율 100% ✓, Reward Freq 2.23%, Pain 0% ✓, Avg Food 67.0
- **관찰**: D2 학습이 빠름 (ep5에 w_min 도달) — 향후 eta_d2 감소 또는 w_min 상향 가능
- **체크포인트**: `checkpoints/brain_L4_20ep.npz` (15 시냅스)

### Phase L5: 지각 학습 (피질 R-STDP + 다중 음식) ✓ 완료
- **문제**: 모든 음식이 동일 (좋은/나쁜 구별 없음) → 학습 압력 부족
- **환경 변경**: 좋은 음식 (green, +25 energy) / 나쁜 음식 (purple, -5 energy), 60/40 비율
- **감각**: good/bad_food_eye L/R (200×2×2 = +800 뉴런)
- **학습**: 피질 R-STDP 8시냅스 — good→IT_Food↑, good→IT_Danger↓, bad→IT_Danger↑, bad→IT_Food↓
- **맛 혐오 (Garcia Effect)**: bad food → danger_sensor I_input (NOT lateral_amygdala Ioffset)
- **파라미터**: eta=0.0008, w_max=8.0, w_min=0.1, init_w=2.0, trace_decay=0.90
- **학습 곡선 (20ep)**: Good→IT_Food 2.0→2.30, Bad→IT_Danger 2.0→2.18, Bad→IT_Food 2.0→1.90
- **검증 (2026-02-14)**: 생존율 60% ✓, Pain Death 0% ✓, Avg Food 51.1, Selectivity ~0.60

### Phase L6: 예측 오차 회로 (Predictive Coding) ✓ 완료
- **문제**: 학습 신호가 보상 시점에만 발생 → 감각-보상 사이 매핑 부족
- **구조**: +200 뉴런 (PE_Food 100, PE_Danger 100) — 예측 오차 뉴런
- **메커니즘**: V1 (실제 감각) vs IT (예측/기대) → PE = V1 - IT, 놀라움 신호
- **학습**: PE→IT R-STDP 4시냅스 (PE_Food→IT_Food, PE_Danger→IT_Danger, L/R)
- **환경**: 음식 클러스터 리스폰 (먹은 위치 근처에 새 음식 생성)
- **파라미터**: eta=0.0008, w_max=5.0, init_w=1.0, trace_decay=0.90
- **학습 곡선 (20ep)**: PE_Food→IT_Food 1.0→1.18, PE_Danger→IT_Danger 1.0→1.22
- **검증 (2026-02-14)**: 생존율 45% ✓, Pain Death 0% ✓, Food Correct +4.6pp 학습 효과

### Phase L7: 음식 유형별 BG 학습 (Discriminative BG) ✓ 완료
- **문제**: food_eye(무차별)가 BG D1(Go) + Reflex(35.0) 구동 → 모든 음식에 동일한 접근
- **해결**: good/bad food_eye를 BG D1/D2에 직접 연결 → 도파민 차등으로 음식 유형별 Go/NoGo 학습
- **생물학적 근거**: 복측 선조체 D1/D2 MSN은 안와전두피질(OFC)에서 범주 특이적 입력 수신
- **구조**: 뉴런 변경 없음 (20,240 유지), +8 SPARSE 시냅스 (23→31 학습 시냅스)
  - good_food_eye → D1 L/R (R-STDP, 도파민 시 강화)
  - bad_food_eye → D1 L/R (R-STDP, 도파민 없어 실질 static)
  - good_food_eye → D2 L/R (Anti-Hebbian, 도파민 시 약화)
  - bad_food_eye → D2 L/R (Anti-Hebbian, 도파민 없어 실질 static)
- **파라미터**: init_w=1.0, sparsity=0.08, 기존 D1/D2 학습률 재사용
- **학습 곡선 (20ep)**: Good→D1: 1.0→2.79 ↑, Bad→D1: 1.0→2.30 ↑ (delta 0.49)
  - Good→D2: 1.0→0.10 ↓, Bad→D2: 1.0→0.16 ↓ (차등 감소)
  - Bad→D1도 학습하는 이유: 음식 근접 시 trace 중첩 (시간적 신용 할당 노이즈, 생물학적으로 현실적)
- **검증 (2026-02-15)**: 생존율 45% ✓, Pain Death 0% ✓, Reward Freq 2.50%
- **체크포인트**: `checkpoints/brain_L7_20ep.npz` (35 시냅스)

### Phase L8: 혐오 도파민 딥 (Aversive Dopamine Dip) ✓ 완료
- **문제**: L7에서 Selectivity 0.60 = 음식 비율과 동일 → 행동적 구별 없음. Bad→D1도 1.0→2.30으로 증가 (시간적 노이즈)
- **근본 원인**: 나쁜 음식 섭취 시 도파민이 "없음"(0)뿐, 적극적 벌칙 신호 없음
- **해결**: dopamine_level을 음수로 허용 (-1.0~+1.0). 나쁜 음식 → 도파민 딥 (-0.5)
- **생물학적 근거**: 혐오 자극 시 도파민 뉴런 기저선 아래로 억제 (Frank 2005, Schultz 1997, Shen et al 2008)
  - D1 MSN: LTD (Go 약화) — `Δw = +η × trace × (-0.5)` → 감소
  - D2 MSN: LTP (NoGo 강화) — `Δw = -η_d2 × trace × (-0.5)` → 이중 부정 → 증가
  - 경로: LHb → RMTg → VTA 억제
- **구조**: 뉴런 변경 없음 (20,240 유지), 시냅스 변경 없음 (31 유지)
  - Config: `dopamine_dip_enabled=True`, `dopamine_dip_magnitude=0.5`
  - `release_dopamine()`: `np.clip(-1.0, 1.0)`, 딥 시 뉴런 전류 차단 (`max(0, DA) * 80`)
  - `decay_dopamine()`: `abs()` 임계값으로 음수 딥 유지
  - `_update_rstdp_weights()`: `abs()` 임계값으로 음수 도파민도 학습 트리거
- **학습 곡선 (20ep)**: Good→D1: 1.0→2.42, Bad→D1: 1.0→1.96 (dip이 성장 0.34 억제)
  - Good→D2: 1.0→0.10 (w_min), Bad→D2: 1.0→0.38 (dip이 감소를 방지)
  - D2 차등: L7 delta 0.06 → L8 delta **0.28** (4.7x 강화)
- **검증 (2026-02-16)**: 생존율 **60%** ✓ (+15pp vs L7), Pain Death 0% ✓, Reward Freq 2.54% ✓
- **체크포인트**: `checkpoints/brain_L8_20ep.npz` (35 시냅스)

---

## 학습 시리즈 로드맵: 의식을 향한 경로

> L1-L8: 감각운동 학습 완성 ✓
> L9-L11: 인지적 학습 (개념→행동, 예측, 시간) ✓
> L12-L13: 주의 + 조건화 학습 ✓
> L14-L15: 자기 참조 학습 (에이전시, 내러티브) ✓

### 인지적 학습 단계 (L9-L11 계획)

| Phase | 이름 | 핵심 메커니즘 | 뉴런 | 선행 조건 |
|-------|------|--------------|------|-----------|
| **L9** | **피질→BG 하향 연결 ✓** | IT 학습 표상 → D1/D2 의사결정 | 0 | L5 (IT 학습) |
| L10 | TD Learning (RPE 도파민) | NAc Critic → VTA 보상 예측 오차 | ~200 | L9 (상태 표상) |
| L11 | 해마 시퀀스 + 리플레이 | 세타 압축 + SWR 리플레이 | ~370 | L10 (TD 오차) |

**전략적 근거:**
- L9: L5-L8의 피질 학습이 행동에 반영되지 않는 문제 해결. 기존 시냅스 유형 재사용, 최소 위험.
- L10: 도파민을 "보상 신호"→"보상 예측 오차"로 전환 (Schultz 1997). Campbell et al. (2025): TD 회로는 하드와이어드, 가치 표상만 학습. Alpha-blending으로 안전 도입.
- L11: "영원한 현재" 탈출. 세타 위상 전이 + STDP ≈ TD-lambda on Successor Representation (George et al. 2023). 시간적 인지의 시작.

### 주의 + 조건화 학습 (L12-L13)

| Phase | 이름 | 핵심 메커니즘 | 의식 이론 | 상태 |
|-------|------|--------------|-----------|------|
| **L12** | **주의 선택 (Attention)** | **Global Workspace 게이팅** | **Dehaene/Baars** | **✓ 완료** |
| **L13** | **조건 맛 혐오 (Garcia Effect)** | **bad_food_eye→LA Hebbian** | **Garcia & Koelling 1966** | **✓ 완료** |

### 자기 참조 학습 단계 (L14-L15) ✓ 완료

| Phase | 이름 | 핵심 메커니즘 | 의식 이론 | 상태 |
|-------|------|--------------|-----------|------|
| **L14** | **에이전시 감지** | **예측-결과 비교 → "내가 했다"** | **Frith** | **✓ 완료** |
| **L15** | **내러티브 자기** | **에이전시 게이팅 자서전 학습** | **Damasio** | **✓ 완료** |

### Phase L12: Global Workspace (주의 기반 경쟁) ✓ 완료
- **문제**: food_memory→motor 12.0이 항상 동일 강도 → 위험해도 음식 추적 (상태 의존적 행동 없음)
- **해결**: Global Workspace Theory (Dehaene & Changeux 2011) — GW_Food vs GW_Safety 경쟁, 승자가 motor에 브로드캐스트
- **구조**: +160 뉴런 (GW_Food L/R 50×2 + GW_Safety 60), C=30 LIF
- **메커니즘**:
  - food_memory→motor 직접 경로 12.0→5.0 약화
  - GW_Food→motor +4.0 (hunger 게이팅): 안전+배고픔=9.0, 위험=5.0, 배부름=6.0
  - GW_Safety→GW_Food 억제 (-12.0): fear+LA 활성 시 음식 탐색 차단
- **입력**: food_memory(6.0) + hunger(5.0) + good_food_eye(3.0) → GW_Food; fear(12.0) + LA(8.0) → GW_Safety
- **시각 강화**: 궤적 트레일 (초록/빨강/파랑), 에이전트 헤일로, 뇌 패널 GW 섹션
- **시냅스**: 12 static (학습 없음), 기존 37 학습 시냅스 유지
- **검증 (2026-02-17)**: 생존율 50% ✓ (+5pp), Pain 0% ✓, Reward 2.49%, GW Food 0.022~0.216, Safety 0.300
- **체크포인트**: `checkpoints/brain_L12_20ep.npz` (42 시냅스)

**Phase L12 완료**: 20,710 뉴런, 37 학습 시냅스, 12 new static

### Phase L13: 조건 맛 혐오 (Conditioned Taste Aversion) ✓ 완료
- **문제**: 나쁜 음식 시각 → 접근 반사(35.0) 그대로 작동, 시각적 회피 학습 없음
- **해결**: Garcia Effect (Garcia & Koelling 1966) — 나쁜 음식 시각(CS)과 내장 불쾌(US) 연합 학습
- **구조**: 뉴런 변경 없음 (20,710 유지), +2 DENSE Hebbian 시냅스 (bad_food_eye L/R → lateral_amygdala)
- **메커니즘**:
  - 나쁜 음식 섭취 → trigger_taste_aversion() → learn_taste_aversion()
  - Hebbian: Δw = η × prev_bad_food_activity (timing bug 해결: 이전 스텝 활성 사용)
  - bad_food_eye → LA → CEA → Fear → Motor Push-Pull (기존 경로 재사용)
- **eta 튜닝**: 0.003 (너무 느림, w=0.45/20ep) → **0.02** (w=1.95/20ep, 4.3x 빠름)
- **학습 곡선 (eta=0.02, 20ep)**: BadFood→LA 0.1→1.95, Selectivity 0.58→0.66 (피크 0.72)
- **검증 (2026-02-21)**: 생존율 55% ✓, Pain 0% ✓, Predator 0% ✓, Reward 2.57% ✓
- **체크포인트**: `checkpoints/brain_L13_eta02_20ep.npz` (44 시냅스)

**Phase L13 완료**: 20,710 뉴런, 39 학습 시냅스 (37 + 2 DENSE Hebbian)

### Phase L14: Agency Detection (에이전시 감지) ✓ 완료
- **목표**: 자기 행동의 감각적 결과를 예측하여 "내가 했다" vs "외부 사건" 구분
- **생물학적 근거**: Frith (2005) — 소뇌 전방 모델 (efference copy → predicted sensory)
- **구조**: Agency_PE(50 LIF) = +50 뉴런 (20,710 유지, Phase 20 Self_Predict 재사용)
- **Forward Model**: self_efference → self_predict (DENSE Hebbian, eta=0.005, w_max=10.0)
  - 학습: 음식 접근/섭취, 고통 경험, 배경(매 5스텝) 총 4개 학습 호출 위치
  - DENSE pull_from_device → view.copy() → reshape → modify → clip → flatten → view[:] = → push_to_device
- **Agency Prediction Error**: V1_Food(+8.0, 실제 감각) + Self_Predict(-6.0, 예측, 억제) → PE
  - 높은 PE = 예측과 불일치 = 낮은 에이전시
  - Agency_PE → Self_Agency (-2.0): 감쇠 신호
- **환경 변경**: motor_noise (σ=0.05) + sensor_jitter (σ=0.03)
  - **핵심 버그 수정**: pain_rays는 sensor_jitter에서 제외 (Push-Pull 60/-40 정밀도 보호)
- **eta 튜닝**: 0.04 (ep3에 포화) → **0.005** (20ep에 걸쳐 점진적 학습)
- **학습 곡선 (20ep)**: FM 1.0→1.17(ep1)→1.46(ep2)→~7.6(ep13)
- **검증**: 생존율 ~65% ✓, Pain 0% ✓, Reward Freq ~2.7% ✓
- **체크포인트**: `checkpoints/brain_L14_20ep.npz`

**Phase L14 완료**: 20,710 뉴런, 40 학습 시냅스 (39 + 1 DENSE Hebbian)

### Phase L15: Narrative Self (내러티브 자기) ✓ 완료 (2026-02-21)
- **이론**: Damasio (2010) — 자기 서사는 자기 원인 경험에서 더 강하게 형성
- **핵심 메커니즘**: Agency-Gated Autobiographical Learning
  1. **Agency gate**: agency_rate/baseline → 자기 원인(high agency) 시 body→narrative 학습 ×2.0
  2. **Salience gate**: |Δbody_rate| × scale → 신체 상태 변화가 클수록 학습 ×3.0
  3. **Agency→Narrative DENSE Hebbian**: eta=0.01, w_max=8.0, init_w=1.0
- **변경 요약**: 0 new neurons, +1 DENSE Hebbian synapse, learn_self_narrative() 수정
- **기존 body→narrative에 agency_gate×salience_gate 곱셈 적용**
- **검증**: 생존율 55% ✓, Pain 0% ✓, Predator 0% ✓, Reward 2.79% ✓, Selectivity 0.70 (best)
- **체크포인트**: `checkpoints/brain_L15_20ep.npz` (46 시냅스)

**Phase L15 완료**: 20,710 뉴런, 41 학습 시냅스 (40 + 1 DENSE Hebbian)

#### 100ep 중기 검증 결과 (2026-02-21)
- **생존율 56%** ✓ (+1pp vs 20ep), Reward 2.75% ✓, Pain Death 0%, Predator Death 1%
- **포화 시냅스**: BadFood→LA(5.0/5.0 ~ep60), Agency→Narr(8.0/8.0 ~ep55), FM(10.0/10.0 ~ep25), Body→Narr(14.0/14.0 ~ep5)
- **계속 성장**: Hippo 2.12→11.84/18.0 (65%), D1 1.08→2.04/5.0 (41%), NAc 1.09→2.11/3.0 (70%)
- **Correct Turn**: early 56.4% → late 53.2% (-3.2pp, 약간 하락)
- **Pain Escape**: 41.6% (개선 여지), Avg Dist→Pain: 107px (경계 따라감)
- **체크포인트**: `brain_L15_100ep.npz` (46 시냅스)

### Phase L16: Sparse Expansion Layer ✓ 완료 (2026-03-21)
- **문제**: L15 400ep 장기 훈련에서 생존율 ~60% 천장. 학습 가능 시냅스(41개) 부족
- **해결**: 초파리 Mushroom Body KC 패턴 — sparse random projection + WTA + R-STDP
- **구조**: KC(1500×2 LIF) + KC_inh(200×2 SensoryLIF, homeostatic PI control)
- **입력**: food_eye/good_bad_food/IT_Food → KC (SPARSE 0.10, static)
- **출력**: KC → D1/D2 (SPARSE 0.05, R-STDP/Anti-Hebbian) = ~30,000 학습 연결
- **검증**: 800맵 100ep: 65% 생존 ✓, Reward 2.60% ✓

### 하드코딩 제거: food_eye→Motor 학습화 ✓ 완료 (2026-03-22)
- **문제**: food_eye→Motor 35.0 (하드코딩)이 good+bad 무차별 접근 → 학습된 회피를 압도
- **해결**: food_eye 10.0(탐색, static) + good_food_eye 25.0(R-STDP, 학습)
- **효과**: 나쁜 음식 접근 반사 제거 → 회피가 Garcia+도파민 딥에서 창발
- **결과**: 400맵 생존율 56% → **89%** (+33pp), Selectivity 0.67-0.72

### 환경 고도화: 800×800 맵 + 장애물 + Rich Zones ✓ 완료 (2026-03-22~24)
- **맵 확장**: 400→800, 모든 거리 파라미터 비례 스케일링
- **장애물**: obstacle_rays 분리 (wall_rays와 독립), obstacle_eye(200×2) Push-Pull 8/-4, 1개
- **Rich Zones**: 2개 (radius 120px), 70% food clustering, zone_richness observation 추가
- **포식자 밸런스**: speed 2.5 (agent 4.5의 56%), chase_range 150px
- **학습 추이 시각화**: 실시간 미니 그래프 (D1/Hippo/Garcia/KC + food selectivity)
- **렌더링**: 800px 내부 → 400px 축소 표시, 패널 위치 유지
- **Spike recording 배치화**: pull 1,310회→1회, GPU 3D 90%→60%

### 환경: 모터 노이즈 + 센서 지터 추가 ✓ 완료 (2026-02-21)
- **motor_noise**: 각도 변화에 Gaussian 노이즈 (σ=0.05) → 완벽한 모터 제어 불가
- **sensor_jitter**: 감각 레이에 곱셈 노이즈 (σ=0.03) → 환경 인식 불확실성
- **pain_rays 제외**: Push-Pull(60/-40)은 정밀한 L/R 차이에 의존 → jitter 적용 시 회피 파괴
- **Config**: `motor_noise_enabled=True`, `motor_noise_std=0.05`, `sensor_jitter_enabled=True`, `sensor_jitter_std=0.03`

### 환경: 포식자 추가 ✓ 완료 (2026-02-18)
- **설계**: "이동형 pain source" — 뇌 변경 제로, 기존 pain_rays + danger_signal 재사용
- **PredatorAgent**: wander(1.5)/chase(2.0) 상태 기계, nest=safe zone
- **Config**: speed=2.0 (< agent 3.0), chase_range=120, pain_intensity=0.8
- **검증**: 70% 생존 ✓, 5% predator death, 0% pain death

---

## 환경 진화 계획

| 단계 | 환경 | 복잡도 | 대응 Phase | 상태 |
|------|------|--------|------------|------|
| 기본 | ForagerGym (2D, 400×400) | ⭐ | Phase 1-10 | ✓ |
| 다중 음식 | good/bad food (60/40) | ⭐⭐ | L5 | ✓ |
| 포식자 | 이동형 위협 (PredatorAgent) | ⭐⭐ | L12 후 | ✓ |
| 모터 노이즈 + 센서 지터 | motor_noise σ=0.05, sensor_jitter σ=0.03 | ⭐⭐⭐ | L14 | ✓ |
| **맵 확장** | **800×800, 스케일링 파라미터** | ⭐⭐⭐ | **L16** | **✓** |
| **장애물** | **obstacle_rays 분리, Push-Pull 8/-4** | ⭐⭐⭐ | **E1** | **✓** |
| **영역 다양성** | **Rich Zones 2개, 70% food clustering** | ⭐⭐⭐ | **E2** | **✓** |
| **학습 추이 시각화** | **실시간 그래프 (D1/Hippo/Garcia/KC)** | ⭐⭐ | **E4/E5** | **✓** |
| 시간 변화 | rich zone 이동, 음식 고갈 | ⭐⭐⭐⭐ | E3 | 계획 |
| 장기 | 다중 에이전트 | ⭐⭐⭐⭐ | L16+ | 계획 |
| 최종 | 언어 환경 | ⭐⭐⭐⭐⭐ | L16+ | 계획 |

---

## 현실 세계 연결 (ESP32 로봇) 로드맵

> 2륜 차동 구동 로봇. I/O 매핑은 깔끔하나 센서 드리프트/조명/통신 지터에 주의 필요.

| 단계 | 시점 | 내용 | 비고 | 상태 |
|------|------|------|------|------|
| 0 | 지금 | SensorAdapter / MotorAdapter 추상화 인터페이스 정의 | 코드 레벨만 | 계획 |
| 1 | L14 | 시뮬에 모터 노이즈(±10%) + 외란 + 센서 지터 추가 | 에이전시 회로를 non-trivial하게 | ✓ 완료 |
| 2 | L14 후 | ESP32 센서 연결 (거리/터치/전압만). **카메라는 PC 웹캠** | 통신 RTT/지터 실측 | **VL53L0X 확인 ✓** |
| 3 | L15 후 | place cell을 ego-centric(Δx,Δθ)으로 재설계 or 외부 기준(AprilTag) | IMU 드리프트 대응 | 계획 |
| 4 | 최종 | 2륜 구동 + 실제 주행 (모터/바퀴/미끄러짐) | 전체 폐루프 | 계획 |

### 현실 연결 시 필수 사항
- **카메라**: 초기엔 PC USB 웹캠 (ESP32-CAM은 조명 변화에 불안정)
- **위치**: IMU+엔코더 데드레커닝은 드리프트 → 외부 기준 or ego-centric 필수
- **에너지↔전압**: 모터 부하로 전압 출렁임 → EMA 저역통과 필터 + 부하 보정
- **통신**: 패킷에 timestamp/seq, RTT 히스토그램 실측, 지연 포함 관측 전략
- **SNN 특성**: R-STDP + weight_decay로 online 재적응 가능 → DNN식 과적합/forgetting 우려는 낮음

---

## 핵심 원칙 (불변)

```
┌─────────────────────────────────────────────────────────────────┐
│  "인공 뇌의 학습 방식과 구조는 인간 뇌와 같아야 한다"           │
└─────────────────────────────────────────────────────────────────┘
```

### 금지 사항
- ✗ 사전 학습된 모델 붙이기
- ✗ 지식 DB에 정보 저장
- ✗ 개념/단어 직접 주입
- ✗ LLM 방식 (다음 토큰 예측)
- ✗ 휴리스틱으로 행동 조작

### 허용 메커니즘
- ✓ STDP/R-STDP 시냅스 가소성
- ✓ 생물학적 회로 구조
- ✓ 경험에서 개념 창발
- ✓ 감각 → 지각 → 인지 → 행동

---

## 마일스톤 체크포인트

| 마일스톤 | 기준 | 목표 시점 |
|----------|------|-----------|
| M1 | 10K 뉴런, 시각 개념 형성 | Phase 10 |
| M2 | 50K 뉴런, 다중 감각 통합 | Phase 15 |
| M3 | 100K 뉴런, 기초 언어 | Phase 17 |
| M4 | 500K 뉴런, 추상 추론 | Phase 20 |

---

## 검증 교훈 (2026-02-02)

```
╔═══════════════════════════════════════════════════════════════╗
║  "검증 없이 진행하면 나중에 더 많은 작업이 필요하다"           ║
╚═══════════════════════════════════════════════════════════════╝
```

### 필수 검증 프로세스

1. **새 Phase 구현 완료 후**
2. **20 에피소드 검증 실행**
3. **성공 기준 확인:**
   - Survival Rate > 40%
   - Reward Freq > 2.5%
   - Pain Avoidance > 85%
4. **실패 시 수정 후 재검증** (다음 Phase 진행 금지)

### 회로 설계 원칙

- 새 회로는 기존 **생존 반사(Pain→Motor)를 방해하지 않아야** 함
- Motor 직접 연결보다 **기존 회로 조절** 우선
- 새 가중치는 Pain Push-Pull(60/-40)보다 **약하게** 설정

---

*최종 업데이트: 2026-02-21 (Phase L13 완료, 20,710 뉴런, 39 학습 시냅스, 55% 생존율, ESP32 VL53L0X 확인)*
