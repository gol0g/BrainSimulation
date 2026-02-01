# Phase 13: 두정엽 (Parietal Cortex) 설계

> 목표: 공간 추론 및 경로 계획 - "목표까지 어떻게 갈 것인가"

---

## 1. 생물학적 배경

### 두정엽 (Parietal Cortex)
- 공간 인지와 주의의 핵심 영역
- "Where" pathway의 종착점
- 시각-운동 변환 (visuomotor transformation)
- 경로 계획과 도달 운동 (reaching) 조절

### 주요 영역
```
후두정 피질 (Posterior Parietal Cortex, PPC):
├── Area 7a: 시각-공간 통합
├── LIP (Lateral Intraparietal): 시선 방향, 공간 주의
├── MIP (Medial Intraparietal): 도달 운동 계획
└── AIP (Anterior Intraparietal): 물체 파지 (grasping)
```

### 핵심 기능
1. **공간 좌표 변환**: 망막 중심 → 머리 중심 → 몸 중심 좌표
2. **공간 주의**: "어디에 주의를 기울일 것인가"
3. **경로 계획**: 현재 위치 → 목표 위치로의 벡터 계산
4. **운동 의도**: 행동 이전의 운동 계획 표상

### 핵심 연결
```
시각 경로:    V1 → V2 → MT/V5 → PPC
체감각 경로:  S1 → S2 → PPC
               ↓
          PPC (통합)
               ↓
         전운동 피질 → M1 → 운동
               ↓
         해마 (공간 기억)
```

---

## 2. 구현 범위

### 단순화된 Parietal Cortex
```
PPC_Space: 공간 위치 표상
├── STS/IT → PPC_Space (물체 위치)
├── Place Cells → PPC_Space (자기 위치)
└── PPC_Space → PFC (목표 기반 행동)

PPC_Goal_Vector: 목표 방향 벡터
├── Goal (PFC) + Place Cells → PPC_Goal_Vector
└── PPC_Goal_Vector → Motor (목표 방향 이동)

PPC_Attention: 공간 주의
├── Salience (음식/위험) → PPC_Attention
├── PPC_Attention → V1/V2 (top-down 주의)
└── PPC_Attention → STS (다감각 주의)

PPC_Path: 경로 계획 (단순화)
├── PPC_Space + PPC_Goal_Vector → PPC_Path
└── PPC_Path → Motor (순차적 행동)
```

---

## 3. 뉴런 구조

| 영역 | 뉴런 수 | 역할 |
|------|---------|------|
| PPC_Space_Left | 150 | 왼쪽 공간 표상 |
| PPC_Space_Right | 150 | 오른쪽 공간 표상 |
| PPC_Goal_Food | 150 | 음식 방향 벡터 |
| PPC_Goal_Safety | 150 | 안전 방향 벡터 |
| PPC_Attention | 200 | 공간 주의 조절 |
| PPC_Path_Buffer | 200 | 경로 계획 버퍼 |
| **총계** | **1,000** | |

**Phase 13 완료 시 총 뉴런: 11,700 + 1,000 = 12,700**

---

## 4. 시냅스 연결

### 4.1 감각 → PPC (공간 입력)

#### 4.1.1 시각 → PPC_Space
```
V1_Food_Left → PPC_Space_Left (시각적 음식 위치)
V1_Food_Right → PPC_Space_Right
V1_Danger_Left → PPC_Space_Left (시각적 위험 위치)
V1_Danger_Right → PPC_Space_Right
IT_Food → PPC_Space (물체 인식 기반 위치)
IT_Danger → PPC_Space
STS_Food → PPC_Space (다감각 위치)
STS_Danger → PPC_Space
```
- 가중치: 15.0

#### 4.1.2 해마 → PPC_Space (자기 위치)
```
Place_Cells → PPC_Space_Left/Right (현재 위치 맵핑)
Food_Memory → PPC_Space (기억된 음식 위치)
```
- 가중치: 12.0

### 4.2 PFC → PPC (목표 설정)

```
Goal_Food → PPC_Goal_Food (음식 목표 활성화)
Goal_Safety → PPC_Goal_Safety (안전 목표 활성화)
Working_Memory → PPC_Path_Buffer (목표 유지)
```
- 가중치: 18.0

### 4.3 PPC 내부 연결

#### 4.3.1 공간-목표 통합
```
PPC_Space_Left + Goal_Food → PPC_Goal_Food (왼쪽에 음식 목표)
PPC_Space_Right + Goal_Food → PPC_Goal_Food
PPC_Space + Goal_Safety → PPC_Goal_Safety (위험 반대 방향)
```
- 가중치: 15.0

#### 4.3.2 경로 계획
```
PPC_Goal_Food/Safety → PPC_Path_Buffer (경로 버퍼 활성화)
PPC_Path_Buffer → PPC_Path_Buffer (자기 유지)
```
- 가중치: 10.0

#### 4.3.3 WTA 경쟁
```
PPC_Space_Left ↔ PPC_Space_Right (좌우 경쟁)
PPC_Goal_Food ↔ PPC_Goal_Safety (목표 경쟁)
```
- 가중치: -8.0

#### 4.3.4 주의 조절
```
PPC_Goal_Food → PPC_Attention (목표 기반 주의)
PPC_Goal_Safety → PPC_Attention
Amygdala_Fear → PPC_Attention (공포 시 주의 강화)
```
- 가중치: 12.0

### 4.4 PPC → 출력 연결

#### 4.4.1 PPC → Motor (공간 유도 행동)
```
PPC_Goal_Food + PPC_Space_Left → Motor_Left (왼쪽 음식으로 이동)
PPC_Goal_Food + PPC_Space_Right → Motor_Right
PPC_Goal_Safety → Motor (위험 반대 방향)
PPC_Path_Buffer → Motor (경로 실행)
```
- 가중치: 15.0

#### 4.4.2 PPC → V1/STS (Top-Down 주의)
```
PPC_Attention → V1_Food (음식 시각 강화)
PPC_Attention → V1_Danger (위험 시각 강화)
PPC_Attention → STS (다감각 주의 조절)
```
- 가중치: 8.0

#### 4.4.3 PPC → Hippocampus (공간 기억)
```
PPC_Space → Place_Cells (공간 표상 업데이트)
PPC_Goal_Food → Food_Memory (목표 위치 기억)
```
- 가중치: 10.0

### 4.5 Top-Down 조절

```
Hunger → PPC_Goal_Food (배고플 때 음식 목표 강화)
Fear → PPC_Goal_Safety (공포 시 안전 목표 강화)
Dopamine → PPC_Attention (보상 예측 시 주의 강화)
```
- 가중치: 10.0

---

## 5. 학습 메커니즘

### 5.1 공간-보상 연합 학습
```
특정 방향 이동 + 보상 → PPC_Goal_Food 강화
  → "이 방향 = 음식" 학습

특정 방향 이동 + 고통 → PPC_Goal_Safety 강화
  → "이 방향 = 위험, 반대로" 학습
```

### 5.2 경로 학습
```
연속 행동 + 최종 보상 → 경로 버퍼 강화
  → eligibility trace로 이전 행동까지 credit 전파
```

---

## 6. 시각화

```
┌─────────────────────────────────────────────────────────────────┐
│                 Phase 13: Parietal Cortex                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  V1/IT/STS ──────► PPC_Space_L/R ──────┬──► Motor             │
│       │                  │              │                       │
│       │                  │              └──► Hippocampus       │
│       │                  ▼                                      │
│       │           PPC_Attention ────────► V1/STS (Top-Down)    │
│       │                  ▲                                      │
│       │                  │                                      │
│  Place Cells ────► PPC_Goal_Food/Safety ──► Motor              │
│       │                  │                                      │
│       │                  ▼                                      │
│       │           PPC_Path_Buffer ──────► Motor (Sequential)   │
│       │                  ▲                                      │
│       │                  │                                      │
│  PFC ─┴──────► Goal_Food/Safety ────────► PPC_Goal            │
│                                                                 │
│         Top-Down: Hunger/Fear/Dopamine → PPC                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 성공 기준

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| PPC_Goal_Food 활성화 | 배고픔 + 음식 방향에서 > 0.4 | 스파이크율 |
| PPC_Attention 활성화 | 자극 존재 시 > 0.3 | 스파이크율 |
| 목표 지향 개선 | 음식 방향 정확도 증가 | 모터 출력 분석 |
| 성능 유지 | 생존율 > 70% | 기존 대비 |

---

## 8. 구현 순서

1. **뉴런 생성**: PPC_Space_L/R, PPC_Goal_Food/Safety, PPC_Attention, PPC_Path_Buffer
2. **감각 → PPC 연결**: V1/IT/STS → PPC_Space
3. **PFC → PPC 연결**: Goal → PPC_Goal
4. **PPC 내부 연결**: 공간-목표 통합, WTA, 주의 조절
5. **PPC → 출력 연결**: Motor, V1/STS (Top-Down), Hippocampus
6. **Top-Down 연결**: Hunger/Fear/Dopamine → PPC
7. **스파이크 추적 및 시각화**
8. **테스트**: 5 에피소드

---

## 9. 예상 문제점

| 문제 | 해결책 |
|------|--------|
| PPC 과활성화 | WTA 경쟁 강화, 억제 가중치 조정 |
| 기존 경로와 충돌 | PPC 출력 가중치 점진적 조정 |
| 공간 표상 불안정 | Place Cells → PPC 가중치 조정 |
| Top-Down 과조절 | 주의 가중치 적절히 제한 |

---

## 10. 두정엽의 인지적 의의

```
╔═══════════════════════════════════════════════════════════════╗
║  두정엽의 역할: "어디로 갈 것인가"                            ║
╠═══════════════════════════════════════════════════════════════╣
║  1. 공간 통합: 시각 + 청각 + 체감각 → 통합 공간 표상         ║
║  2. 목표 벡터: 현재 위치 → 목표 위치 방향 계산               ║
║  3. 주의 조절: 중요한 위치에 선택적 주의 배분                ║
║  4. 경로 계획: 연속적 행동 시퀀스 생성의 기초                ║
║  5. 운동 의도: 행동 전 운동 계획의 표상                      ║
╚═══════════════════════════════════════════════════════════════╝
```

---

*작성: 2026-02-01*
