# Phase 12: 다중 감각 통합 (Multimodal Integration) 설계

> 목표: 시각 + 청각 통합, "보이는 것과 들리는 것" 일치 학습

---

## 1. 생물학적 배경

### 상측두고랑 (Superior Temporal Sulcus, STS)
- 시각과 청각 정보가 수렴하는 핵심 영역
- 생물학적 움직임 인식 (biological motion)
- 사회적 신호 처리 (얼굴 + 목소리)
- 시청각 일치/불일치 감지

### 다중 감각 통합 원칙
1. **공간적 일치**: 같은 방향에서 오는 시각/청각 → 통합 강화
2. **시간적 일치**: 동시에 발생하는 자극 → 연합 학습
3. **의미적 일치**: 음식 보임 + 음식 소리 → "음식" 개념 강화

### 핵심 연결
```
시각 경로: V1 → V2 → V4 → IT
청각 경로: Sound → A1 → A2
         ↘         ↙
          STS (통합)
              ↓
     Hippocampus / Amygdala / Motor
```

---

## 2. 뉴런 구조

| 영역 | 뉴런 수 | 역할 |
|------|---------|------|
| STS_Food | 200 | 음식 관련 시청각 통합 |
| STS_Danger | 200 | 위험 관련 시청각 통합 |
| STS_Congruence | 150 | 시청각 일치 감지 (보상 신호) |
| STS_Mismatch | 100 | 시청각 불일치 감지 (주의 신호) |
| Multimodal_Buffer | 150 | 다중 감각 작업 기억 |
| **총계** | **800** | |

**Phase 12 완료 시 총 뉴런: 10,900 + 800 = 11,700**

---

## 3. 시냅스 연결

### 3.1 시각 → STS (IT Cortex에서)
```
IT_Food_Category → STS_Food (시각적 음식)
IT_Danger_Category → STS_Danger (시각적 위험)
```
- 가중치: 20.0

### 3.2 청각 → STS (A1/A2에서)
```
A1_Food → STS_Food (청각적 음식)
A1_Danger → STS_Danger (청각적 위험)
A2_Association → STS_Food/Danger (연합 청각)
```
- 가중치: 20.0

### 3.3 STS 내부 연결

#### 3.3.1 일치 감지 (Congruence Detection)
```
STS_Food (visual + auditory 동시 활성) → STS_Congruence
STS_Danger (visual + auditory 동시 활성) → STS_Congruence
```
- 조건: 시각 AND 청각 모두 활성화될 때만
- 구현: 두 입력의 곱셈적 상호작용 (multiplicative gating)
- 가중치: 15.0

#### 3.3.2 불일치 감지 (Mismatch Detection)
```
(시각 음식 + 청각 위험) → STS_Mismatch
(시각 위험 + 청각 음식) → STS_Mismatch
```
- 교차 입력 감지
- 가중치: 12.0

#### 3.3.3 WTA 경쟁
```
STS_Food ↔ STS_Danger (상호 억제)
STS_Congruence ↔ STS_Mismatch (상호 억제)
```
- 가중치: -8.0

### 3.4 STS → 출력 연결

#### 3.4.1 STS → Hippocampus
```
STS_Food → Hippocampus Food Memory (다중 감각 기억)
STS_Danger → Hippocampus (위험 장소 기억)
STS_Congruence → Place Cells (일치 시 기억 강화)
```
- 가중치: 15.0

#### 3.4.2 STS → Amygdala
```
STS_Danger → Amygdala LA (다중 감각 공포)
STS_Mismatch → Amygdala LA (불일치 = 경계)
```
- 가중치: 18.0

#### 3.4.3 STS → Motor
```
STS_Food → Motor (ipsi) (통합된 음식 방향)
STS_Danger → Motor (contra) (통합된 위험 회피)
```
- 가중치: 12.0

#### 3.4.4 STS → PFC
```
STS_Congruence → Working Memory (확실한 정보)
STS_Mismatch → Goal_Safety (불확실 = 안전 우선)
```
- 가중치: 10.0

### 3.5 Top-Down 조절
```
Hunger → STS_Food (배고플 때 음식 통합 민감도 증가)
Fear → STS_Danger (공포 시 위험 통합 민감도 증가)
Working Memory → STS_Congruence (목표 집중 시 일치 감지 강화)
```
- 가중치: 8.0

---

## 4. 학습 메커니즘

### 4.1 Hebbian 연합 학습
```
시각 음식 + 청각 음식 + 보상 → STS_Food 강화
  → "보고 + 듣고 + 먹음" = 연합 형성

시각 위험 + 청각 위험 + 고통 → STS_Danger 강화
  → "보고 + 듣고 + 아픔" = 연합 형성
```

### 4.2 불일치 학습
```
불일치 감지 → 도파민 방출 (novelty)
  → 탐색 행동 증가
  → 새로운 연합 형성 기회
```

---

## 5. 시각화

```
┌─────────────────────────────────────────────────────────────────┐
│                 Phase 12: Multimodal Integration                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IT_Food ─────────┐                                             │
│                   ├──► STS_Food ──────┬──► Hippocampus         │
│  A1_Food ─────────┘        │          │                         │
│                            │          └──► Motor (ipsi)         │
│                            ▼                                    │
│                    STS_Congruence ────► Working Memory          │
│                            ▲                                    │
│                            │                                    │
│  IT_Danger ───────┐        │                                    │
│                   ├──► STS_Danger ────┬──► Amygdala LA         │
│  A1_Danger ───────┘        │          │                         │
│                            │          └──► Motor (contra)       │
│                            ▼                                    │
│                    STS_Mismatch ──────► Goal_Safety             │
│                                                                 │
│        Cross-modal: IT_Food + A1_Danger → STS_Mismatch         │
│                                                                 │
│              Top-Down: Hunger/Fear/WM → STS                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 성공 기준

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| STS_Food 활성화 | 시각+청각 동시 > 0.5 | 스파이크율 |
| STS_Congruence | 일치 자극 시 > 0.4 | 스파이크율 |
| 성능 유지 | 생존율 > 70% | 기존 대비 |
| 통합 이점 | 시각만 vs 시청각 반응 차이 | 모터 출력 비교 |

---

## 7. 구현 순서

1. **뉴런 생성**: STS_Food/Danger, STS_Congruence/Mismatch, Multimodal_Buffer
2. **시각 → STS 연결**: IT → STS
3. **청각 → STS 연결**: A1/A2 → STS
4. **STS 내부 연결**: 일치/불일치 감지, WTA
5. **STS → 출력 연결**: Hippocampus, Amygdala, Motor, PFC
6. **Top-Down 연결**: Hunger/Fear/WM → STS
7. **스파이크 추적 및 시각화**
8. **테스트**: 5 에피소드

---

## 8. 예상 문제점

| 문제 | 해결책 |
|------|--------|
| STS 과활성화 | WTA 경쟁 강화, 억제 가중치 조정 |
| 단일 감각 우세 | 시청각 가중치 균형 조정 |
| 불일치 과민감 | Mismatch 임계값 조정 |
| 기존 경로와 충돌 | STS 출력 가중치 점진적 조정 |

---

## 9. 다중 감각 통합의 의의

```
╔═══════════════════════════════════════════════════════════════╗
║  다중 감각 통합의 인지적 이점                                  ║
╠═══════════════════════════════════════════════════════════════╣
║  1. 신뢰성 향상: 두 감각이 일치 → 확신 증가                  ║
║  2. 노이즈 감소: 한 감각이 불확실해도 다른 감각으로 보완     ║
║  3. 경계 강화: 불일치 감지 → 주의/탐색 증가                  ║
║  4. 개념 형성: "음식" = (시각적 특징 + 청각적 특징)          ║
╚═══════════════════════════════════════════════════════════════╝
```

---

*작성: 2026-02-01*
