# Phase 9: V2/V4 고차 시각 피질 설계

> 목표: 패턴 인식 및 물체 분류 능력

---

## 1. 생물학적 배경

### V2 (이차 시각 피질)
- V1의 출력을 받아 더 복잡한 특징 추출
- 방향성 에지, 질감, 윤곽 처리
- 크기 불변성 시작

### V4 (사차 시각 피질)
- 색상, 형태, 물체 분류
- 주의(attention) 조절을 받음
- 측두엽(IT)으로 정보 전달

---

## 2. 구현 범위

### 단순화된 V2
```
V2_Edge: 에지/윤곽 검출
├── V1_Food → V2_Edge_Food (음식 윤곽)
└── V1_Danger → V2_Edge_Danger (위험 윤곽)
```

### 단순화된 V4
```
V4_Object: 물체 분류
├── V2_Edge_Food → V4_Food_Object (음식 물체)
└── V2_Edge_Danger → V4_Danger_Object (위험 물체)
```

---

## 3. 뉴런 구조

| 영역 | 뉴런 수 | 역할 |
|------|---------|------|
| V2_Edge_Food | 150 | 음식 관련 에지/윤곽 |
| V2_Edge_Danger | 150 | 위험 관련 에지/윤곽 |
| V4_Food_Object | 100 | "이것은 음식이다" 표상 |
| V4_Danger_Object | 100 | "이것은 위험이다" 표상 |
| V4_Novel_Object | 100 | 새로운/미분류 물체 |
| **총계** | **600** | |

**Phase 9 완료 시 총 뉴런: 8,400 + 600 = 9,000**

---

## 4. 시냅스 연결

### 4.1 V1 → V2 (수렴)
```
V1_Food_Left/Right → V2_Edge_Food (수렴, 방향 통합)
V1_Danger_Left/Right → V2_Edge_Danger
```
- 좌우 정보를 통합 (크기 불변성)
- 가중치: 15.0

### 4.2 V2 → V4 (분류)
```
V2_Edge_Food → V4_Food_Object (음식으로 분류)
V2_Edge_Danger → V4_Danger_Object (위험으로 분류)
```
- 가중치: 20.0 (강한 분류)

### 4.3 V4 → 상위 영역
```
V4_Food_Object → Hippocampus (음식 기억 강화)
V4_Food_Object → Hunger Drive (음식 인지 시 배고픔 활성화)
V4_Danger_Object → Amygdala (위험 인지 시 공포 활성화)
V4_Novel_Object → Dopamine (새로운 물체 → 호기심)
```

### 4.4 Top-Down 조절
```
Hunger → V4_Food_Object (배고플 때 음식 탐지 증가)
Fear → V4_Danger_Object (공포 시 위험 탐지 증가)
Goal_Food → V2_Edge_Food (목표에 따른 주의 조절)
```

---

## 5. 학습 메커니즘

### 5.1 Hebbian 학습 (V2 → V4)
- 음식 먹을 때: V2_Edge_Food → V4_Food_Object 강화
- 고통 받을 때: V2_Edge_Danger → V4_Danger_Object 강화

### 5.2 Novelty Detection
```
V1 활성화 + V4_Object 비활성화 → V4_Novel_Object 활성화
→ Dopamine 방출 → 탐색 행동
```
- 본 적 없는 패턴 = 새로운 물체

---

## 6. 시각화

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 9: V2/V4                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    V1_Food_L/R ──┬──► V2_Edge_Food ──► V4_Food_Object      │
│                  │         │                  │             │
│                  │         ▼                  ▼             │
│                  │    (수렴/통합)      → Hippocampus        │
│                  │                     → Hunger             │
│                  │                                          │
│    V1_Danger_L/R ┴──► V2_Edge_Danger ──► V4_Danger_Object  │
│                            │                  │             │
│                            ▼                  ▼             │
│                       (수렴/통합)      → Amygdala           │
│                                                             │
│              Top-Down: Hunger/Fear/Goal → V2/V4            │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 성공 기준

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| V4_Food 활성화 | 음식 근처에서 > 0.5 | 스파이크율 |
| V4_Danger 활성화 | Pain Zone 근처에서 > 0.5 | 스파이크율 |
| 학습 효과 | V2→V4 가중치 증가 | 50 에피소드 후 |
| 성능 유지 | 생존율 > 70% | 기존 대비 하락 없음 |

---

## 8. 구현 순서

1. **뉴런 생성**: V2_Edge_Food/Danger, V4_Food/Danger/Novel_Object
2. **순방향 연결**: V1 → V2 → V4
3. **Top-Down 연결**: Hunger/Fear/Goal → V2/V4
4. **V4 → 상위 연결**: Hippocampus, Amygdala, Dopamine
5. **학습 규칙**: Hebbian (음식/고통 시)
6. **스파이크 추적 및 시각화**
7. **테스트**: 20 에피소드

---

## 9. 예상 문제점

| 문제 | 해결책 |
|------|--------|
| V4 과활성화 | WTA 경쟁 추가 (Food vs Danger vs Novel) |
| 학습 불안정 | 가중치 클리핑, 학습률 조정 |
| Top-Down 과잉 | 조절 가중치 약화 |

---

*작성: 2025-02-01*
