# Phase 14: 전운동 피질 (Premotor Cortex) 설계

> 목표: 운동 계획 및 시퀀스 생성 - "다음에 무엇을 할 것인가"

---

## 1. 생물학적 배경

### 전운동 피질 (Premotor Cortex, PMC)
- 1차 운동 피질(M1) 앞에 위치
- 운동 계획과 준비의 핵심 영역
- 감각 정보 → 운동 명령 변환
- 운동 시퀀스의 시간적 조직화

### 주요 영역
```
전운동 피질 구성:
├── PMd (Dorsal Premotor): 공간 기반 운동 계획
├── PMv (Ventral Premotor): 물체 기반 운동 계획, 파지
├── SMA (Supplementary Motor Area): 내부 생성 시퀀스
└── pre-SMA: 운동 의도, 시퀀스 학습
```

### 핵심 기능
1. **운동 준비**: 실제 움직임 전 운동 계획 표상
2. **감각-운동 변환**: 시각/청각 → 적절한 운동 명령
3. **시퀀스 생성**: 연속적 행동의 시간적 조직화
4. **운동 선택**: 여러 가능한 행동 중 선택

### 핵심 연결
```
입력:
  PPC (공간 정보) → PMd
  IT/STS (물체 정보) → PMv
  PFC (목표/계획) → SMA/pre-SMA
  Basal Ganglia → PMC (행동 선택)

출력:
  PMC → M1 (1차 운동 피질) → 척수 → 근육
  PMC → Cerebellum (운동 조정)
  PMC → PFC (피드백)
```

---

## 2. 구현 범위

### 단순화된 Premotor Cortex
```
PMd (Dorsal Premotor): 공간 기반 운동 계획
├── PPC_Goal → PMd (목표 방향 입력)
├── PMd → Motor (방향 운동 출력)
└── PMd 내부 시퀀스 유지

PMv (Ventral Premotor): 물체 기반 운동 계획
├── IT/STS → PMv (물체 인식 입력)
├── PMv → Motor (접근/회피 출력)
└── Hunger/Fear → PMv (동기 조절)

SMA (Supplementary Motor Area): 시퀀스 생성
├── PFC Goal → SMA (목표 기반 시퀀스)
├── SMA → PMd/PMv (시퀀스 실행)
└── SMA 내부 재귀 (시퀀스 유지)

Motor_Preparation: 운동 준비 버퍼
├── PMd/PMv/SMA → Motor_Prep (운동 계획 통합)
├── Motor_Prep → Motor (실행)
└── BG Direct → Motor_Prep (Go 신호)
```

---

## 3. 뉴런 구조

| 영역 | 뉴런 수 | 역할 |
|------|---------|------|
| PMd_Left | 100 | 왼쪽 방향 운동 계획 |
| PMd_Right | 100 | 오른쪽 방향 운동 계획 |
| PMv_Approach | 100 | 접근 운동 계획 |
| PMv_Avoid | 100 | 회피 운동 계획 |
| SMA_Sequence | 150 | 시퀀스 생성/유지 |
| pre_SMA | 100 | 운동 의도/선택 |
| Motor_Preparation | 150 | 운동 준비 버퍼 |
| **총계** | **800** | |

**Phase 14 완료 시 총 뉴런: 12,700 + 800 = 13,500**

---

## 4. 시냅스 연결

### 4.1 PPC → PMd (공간 기반 운동 계획)

```
PPC_Goal_Food + PPC_Space_Left → PMd_Left (왼쪽 음식 방향 계획)
PPC_Goal_Food + PPC_Space_Right → PMd_Right (오른쪽 음식 방향 계획)
PPC_Goal_Safety + PPC_Space_Left → PMd_Right (왼쪽 위험 → 오른쪽 회피)
PPC_Goal_Safety + PPC_Space_Right → PMd_Left (오른쪽 위험 → 왼쪽 회피)
```
- 가중치: 18.0

### 4.2 IT/STS → PMv (물체 기반 운동 계획)

```
IT_Food_Category → PMv_Approach (음식 인식 → 접근 계획)
IT_Danger_Category → PMv_Avoid (위험 인식 → 회피 계획)
STS_Food → PMv_Approach (다감각 음식 → 접근)
STS_Danger → PMv_Avoid (다감각 위험 → 회피)
```
- 가중치: 15.0

### 4.3 PFC → SMA (목표 기반 시퀀스)

```
Goal_Food → SMA_Sequence (음식 목표 → 탐색 시퀀스)
Goal_Safety → SMA_Sequence (안전 목표 → 회피 시퀀스)
Working_Memory → pre_SMA (작업 기억 → 운동 의도)
Inhibitory_Control → pre_SMA (억제 → 운동 중단)
```
- 가중치: 15.0

### 4.4 PMC 내부 연결

#### 4.4.1 시퀀스 유지 (재귀 연결)
```
SMA_Sequence → SMA_Sequence (자기 유지, 시퀀스 지속)
pre_SMA → SMA_Sequence (의도 → 시퀀스 시작)
```
- 가중치: 8.0 (재귀), 12.0 (의도→시퀀스)

#### 4.4.2 PMd/PMv 통합
```
PMd_Left + PMv_Approach → Motor_Preparation (왼쪽 접근 계획)
PMd_Right + PMv_Approach → Motor_Preparation (오른쪽 접근 계획)
PMd_Left + PMv_Avoid → Motor_Preparation (오른쪽 회피 계획)
PMd_Right + PMv_Avoid → Motor_Preparation (왼쪽 회피 계획)
SMA_Sequence → Motor_Preparation (시퀀스 실행)
```
- 가중치: 12.0

#### 4.4.3 WTA 경쟁
```
PMd_Left ↔ PMd_Right (좌우 경쟁)
PMv_Approach ↔ PMv_Avoid (접근/회피 경쟁)
```
- 가중치: -10.0

### 4.5 PMC → Motor (운동 출력)

```
Motor_Preparation → Motor_Left/Right (계획 → 실행)
PMd_Left → Motor_Left (직접 경로)
PMd_Right → Motor_Right
PMv_Approach → Motor (양측 활성화, 전진)
PMv_Avoid → Motor (반대측 활성화, 회피)
```
- 가중치: 15.0

### 4.6 PMC → Cerebellum (운동 조정)

```
Motor_Preparation → Granule_Cells (efference copy)
PMd/PMv → Deep_Nuclei (운동 계획 피드백)
```
- 가중치: 10.0

### 4.7 Basal Ganglia → PMC (행동 선택/억제)

```
Direct_Pathway → Motor_Preparation (Go 신호)
Indirect_Pathway → Motor_Preparation (NoGo 신호)
Dopamine → SMA_Sequence (보상 → 시퀀스 강화)
```
- 가중치: 12.0 (Direct), -8.0 (Indirect), 10.0 (DA)

### 4.8 Top-Down 조절

```
Hunger → PMv_Approach (배고플 때 접근 계획 강화)
Fear → PMv_Avoid (공포 시 회피 계획 강화)
Arousal → Motor_Preparation (각성 → 운동 준비 강화)
```
- 가중치: 10.0

---

## 5. 학습 메커니즘

### 5.1 운동 계획 학습
```
운동 계획 + 성공(보상) → 해당 PMC-Motor 연결 강화
  → "이 상황에서 이 계획이 좋았다" 학습

운동 계획 + 실패(고통) → 해당 PMC-Motor 연결 약화
  → "이 상황에서 이 계획은 피해야" 학습
```

### 5.2 시퀀스 학습
```
행동 A → 행동 B → 보상:
  → SMA 내에서 A→B 시퀀스 강화
  → 다음에 A 상황에서 B를 준비
```

---

## 6. 시각화

```
┌─────────────────────────────────────────────────────────────────┐
│                 Phase 14: Premotor Cortex                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PPC_Goal/Space ────► PMd_L/R ──────┐                          │
│                           │         │                           │
│                           ▼         │                           │
│  IT/STS ────────────► PMv_App/Avoid ├──► Motor_Preparation     │
│                           │         │         │                 │
│                           ▼         │         ▼                 │
│  PFC Goal/WM ───────► SMA_Sequence ─┘    Motor_L/R             │
│                           │                   ▲                 │
│                           ▼                   │                 │
│                       pre_SMA ────────────────┘                 │
│                           ▲                                     │
│                           │                                     │
│  Basal Ganglia ──────► Go/NoGo → Motor_Preparation             │
│                                                                 │
│         Top-Down: Hunger/Fear/Arousal → PMv/Motor_Prep         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 성공 기준

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| PMd 활성화 | 목표 방향에서 > 0.3 | 스파이크율 |
| PMv_Approach | 음식 근처에서 > 0.3 | 스파이크율 |
| PMv_Avoid | 위험 근처에서 > 0.3 | 스파이크율 |
| Motor_Prep | 행동 전 > 0.2 | 스파이크율 |
| 성능 유지 | Reward Freq > 2.5% | 기존 대비 |

---

## 8. 구현 순서

1. **뉴런 생성**: PMd_L/R, PMv_Approach/Avoid, SMA_Sequence, pre_SMA, Motor_Preparation
2. **PPC → PMd 연결**: 공간 기반 운동 계획
3. **IT/STS → PMv 연결**: 물체 기반 운동 계획
4. **PFC → SMA 연결**: 목표 기반 시퀀스
5. **PMC 내부 연결**: 재귀, 통합, WTA
6. **PMC → Motor 연결**: 운동 출력
7. **PMC → Cerebellum 연결**: 운동 조정
8. **BG → PMC 연결**: 행동 선택
9. **Top-Down 연결**: Hunger/Fear/Arousal
10. **스파이크 추적 및 시각화**
11. **테스트**: 5 에피소드

---

## 9. 예상 문제점

| 문제 | 해결책 |
|------|--------|
| PMC 과활성화 | WTA 경쟁 강화, 억제 가중치 조정 |
| 기존 Motor 경로와 충돌 | PMC 출력 가중치 점진적 조정 |
| 시퀀스 불안정 | SMA 재귀 가중치 조정 |
| 운동 준비 지연 | Motor_Preparation 임계값 조정 |

---

## 10. 전운동 피질의 인지적 의의

```
╔═══════════════════════════════════════════════════════════════╗
║  전운동 피질의 역할: "어떻게 움직일 것인가"                    ║
╠═══════════════════════════════════════════════════════════════╣
║  1. 운동 계획: 실제 움직임 전 행동 준비                       ║
║  2. 감각-운동 변환: 지각 → 적절한 운동 명령                   ║
║  3. 시퀀스 생성: 연속적 행동의 시간적 조직화                  ║
║  4. 운동 선택: 여러 가능한 행동 중 최적 선택                  ║
║  5. 운동 억제: 부적절한 행동 억제 (with BG)                   ║
╚═══════════════════════════════════════════════════════════════╝
```

---

*작성: 2026-02-01*
