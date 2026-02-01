# Phase 10: 하측두 피질 (Inferior Temporal Cortex) 설계

> 목표: 물체 기억 및 범주화 - "이것은 음식이다" 개념의 뉴런 표상

---

## 1. 생물학적 배경

### 하측두 피질 (IT Cortex)
- 시각 처리의 최상위 단계 ("what" pathway의 끝)
- V4에서 입력을 받아 물체의 정체성(identity) 표상
- 학습을 통해 범주별 뉴런 군집 형성
- 해마와 양방향 연결 (기억 저장/인출)

### 핵심 특성
- **불변 표상**: 위치, 크기, 조명에 무관한 물체 인식
- **범주 선택성**: 특정 범주(얼굴, 물체, 장소)에 선택적 반응
- **경험 의존적**: 학습을 통해 새로운 범주 형성

---

## 2. 구현 범위

### 단순화된 IT Cortex
```
IT_Food_Category: "음식" 범주 뉴런
├── V4_Food_Object → IT_Food (음식 물체 입력)
├── Hippocampus → IT_Food (기억에서 인출)
└── IT_Food → Motor (음식 쪽으로 이동)

IT_Danger_Category: "위험" 범주 뉴런
├── V4_Danger_Object → IT_Danger (위험 물체 입력)
├── Amygdala → IT_Danger (공포 기억)
└── IT_Danger → Motor (회피 행동)

IT_Association: 연합 학습 영역
├── IT_Food + IT_Danger → 복합 표상
└── 새로운 범주 학습 가능
```

---

## 3. 뉴런 구조

| 영역 | 뉴런 수 | 역할 |
|------|---------|------|
| IT_Food_Category | 200 | "음식" 범주 표상 |
| IT_Danger_Category | 200 | "위험" 범주 표상 |
| IT_Neutral_Category | 150 | 중립/미분류 물체 |
| IT_Association | 200 | 범주 간 연합 |
| IT_Memory_Buffer | 250 | 단기 물체 기억 |
| **총계** | **1,000** | |

**Phase 10 완료 시 총 뉴런: 9,000 + 1,000 = 10,000** (M1 마일스톤!)

---

## 4. 시냅스 연결

### 4.1 V4 → IT (순방향)
```
V4_Food_Object → IT_Food_Category (강한 분류)
V4_Danger_Object → IT_Danger_Category
V4_Novel_Object → IT_Neutral_Category
```
- 가중치: 25.0 (강한 범주화)

### 4.2 IT ↔ Hippocampus (양방향)
```
IT_Food → Hippocampus Place Cells (음식 범주 기억 저장)
Place Cells → IT_Food (위치 기반 음식 기억 인출)
Food Memory → IT_Food (음식 기억 활성화)
```
- 순방향: 15.0, 역방향: 12.0

### 4.3 IT ↔ Amygdala (양방향)
```
IT_Danger → Amygdala LA (위험 범주 → 공포)
Fear Response → IT_Danger (공포 → 위험 인식 강화)
```
- 가중치: 18.0

### 4.4 IT → Motor (행동 출력)
```
IT_Food_Category → Motor (ipsi) - 음식 쪽으로 이동
IT_Danger_Category → Motor (contra) - 위험 회피
```
- 가중치: 12.0

### 4.5 IT → PFC (목표 설정)
```
IT_Food → Goal_Food (음식 인지 → 음식 목표)
IT_Danger → Goal_Safety (위험 인지 → 안전 목표)
```
- 가중치: 15.0

### 4.6 IT 내부 연결
```
IT_Food ↔ IT_Danger (WTA 경쟁)
IT_Food/Danger → IT_Association (연합 학습)
IT_Memory_Buffer → IT_Categories (버퍼에서 인출)
```

### 4.7 Top-Down 조절
```
Hunger → IT_Food (배고플 때 음식 범주 민감도 증가)
Fear → IT_Danger (공포 시 위험 범주 민감도 증가)
Working Memory → IT_Memory_Buffer (작업 기억 유지)
```

---

## 5. 학습 메커니즘

### 5.1 범주 형성 (Hebbian)
```
음식 먹을 때:
  V4_Food 활성 + Reward → IT_Food 강화
  → "이 패턴 = 음식" 학습

고통 받을 때:
  V4_Danger 활성 + Pain → IT_Danger 강화
  → "이 패턴 = 위험" 학습
```

### 5.2 연합 학습
```
IT_Food + IT_Danger 동시 활성:
  → IT_Association에 "위험한 음식" 표상 형성
  → 복합 범주 학습
```

### 5.3 기억 통합 (IT ↔ Hippocampus)
```
낮: IT → Hippocampus (새 경험 저장)
밤(휴식): Hippocampus → IT (기억 공고화) [미구현]
```

---

## 6. 시각화

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 10: IT Cortex                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    V4_Food ──────► IT_Food_Category ──┬──► Motor (ipsi)    │
│         │              │    ▲         │                     │
│         │              │    │         └──► Goal_Food        │
│         │              ▼    │                               │
│         │         IT_Association ◄──► IT_Memory_Buffer     │
│         │              ▲    │              ▲                │
│         │              │    │              │                │
│    V4_Danger ────► IT_Danger_Category ──┬──► Motor (contra)│
│                        │    ▲           │                   │
│                        │    │           └──► Goal_Safety    │
│                        ▼    │                               │
│                   Hippocampus ◄──────────────────────────  │
│                        ▲                                    │
│                        │                                    │
│              Top-Down: Hunger/Fear/WM                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 성공 기준

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| IT_Food 활성화 | 음식 근처에서 > 0.4 | 스파이크율 |
| IT_Danger 활성화 | Pain Zone 근처에서 > 0.4 | 스파이크율 |
| 범주 분리 | IT_Food vs IT_Danger 상관 < 0.3 | 활성화 패턴 |
| 성능 유지 | 생존율 > 50% | 기존 대비 |
| 총 뉴런 | 10,000 | M1 마일스톤 |

---

## 8. 구현 순서

1. **뉴런 생성**: IT_Food/Danger/Neutral_Category, IT_Association, IT_Memory_Buffer
2. **순방향 연결**: V4 → IT
3. **양방향 연결**: IT ↔ Hippocampus, IT ↔ Amygdala
4. **출력 연결**: IT → Motor, IT → PFC
5. **내부 연결**: WTA, Association
6. **Top-Down 연결**: Hunger/Fear/WM → IT
7. **스파이크 추적 및 시각화**
8. **테스트**: 10 에피소드

---

## 9. 예상 문제점

| 문제 | 해결책 |
|------|--------|
| IT 과활성화 | WTA 경쟁 강화, 억제 가중치 조정 |
| V4와 중복 | V4는 즉각적 분류, IT는 기억 기반 분류로 역할 분리 |
| 학습 불안정 | Hebbian 학습률 조정, 가중치 클리핑 |
| 기억 간섭 | Memory Buffer로 단기/장기 분리 |

---

## 10. M1 마일스톤 의의

```
╔═══════════════════════════════════════════════════════════════╗
║  M1: 10,000 뉴런 달성                                         ║
╠═══════════════════════════════════════════════════════════════╣
║  • 시각 경로 완성: Eye → V1 → V2 → V4 → IT                   ║
║  • 개념 형성 시작: "음식", "위험" 범주의 뉴런 표상           ║
║  • 기억-지각 통합: IT ↔ Hippocampus 양방향 연결             ║
║  • 목표 지향 행동: IT → PFC → Motor                          ║
╚═══════════════════════════════════════════════════════════════╝
```

---

*작성: 2025-02-01*
