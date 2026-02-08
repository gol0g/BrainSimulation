# Genesis Brain 로드맵

> 최종 목표: 인간 뇌처럼 학습하고 개념을 형성하는 인공 뇌

---

## 현재 상태: Phase 16 검증 완료 (16,800 뉴런)

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 16 검증 통과 ✓ (2026-02-08)                            ║
║  생존율 50%, Reward Freq 3.46%, Pain Avoidance 91.0%         ║
║  Association Cortex 700 뉴런 추가 (초범주 개념 형성)         ║
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
| 16 | 연합 피질 (Association Cortex) | +700 | ✓ 완료 | **50%** |

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

---

## 장기 계획 (Phase 17-20)

### Phase 17: 언어 회로 (Broca/Wernicke)
- 소리 패턴 → 의미 연결
- 처음엔 단순 신호 (경고음, 음식 신호)
- 점진적 복잡화

### Phase 18: 작업 기억 확장
- 다단계 추론
- 계획의 계획
- 시간적 추상화

### Phase 19: 메타인지
- "내가 뭘 모르는지 아는 것"
- 불확실성 표상
- 탐색 vs 활용 의식적 결정

### Phase 20: ???
- 의식의 창발?
- 자기 모델?
- 열린 질문...

**Phase 20 완료 시**: ~500,000+ 뉴런

---

## 환경 진화 계획

| 단계 | 환경 | 복잡도 | 대응 Phase |
|------|------|--------|------------|
| 현재 | ForagerGym (2D) | ⭐ | Phase 1-10 |
| 다음 | 포식자 추가 | ⭐⭐ | Phase 11-12 |
| 중기 | 3D 환경 | ⭐⭐⭐ | Phase 13-14 |
| 장기 | 다중 에이전트 | ⭐⭐⭐⭐ | Phase 15-17 |
| 최종 | 언어 환경 | ⭐⭐⭐⭐⭐ | Phase 18-20 |

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

*최종 업데이트: 2026-02-08 (Phase 16 검증 통과, 생존율 50%, 16,800 뉴런)*
