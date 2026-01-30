# Phase 6a 설계 문서: Cerebellum (소뇌)

> **Genesis Brain Project - Phase 6a: Motor Coordination**
>
> 날짜: 2025-01-31
> 상태: **구현 중**

---

## 1. 목표

소뇌(Cerebellum)를 구현하여 **운동 조정(Motor Coordination)**을 추가한다.

### 1.1 생물학적 역할

소뇌는 뇌의 "자동 조종 장치"로서:
- **운동 조정(Motor Coordination)**: 부드럽고 정확한 움직임
- **타이밍 학습(Timing Learning)**: 정확한 동작 타이밍
- **오류 기반 학습(Error-based Learning)**: 운동 오류로부터 학습
- **예측(Prediction)**: 행동의 감각적 결과 예측

### 1.2 Phase 6a 목표

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 6a 목표 (Cerebellum)                                   ║
╠═══════════════════════════════════════════════════════════════╣
║  1. 운동 출력 평활화 (Smoothing)                              ║
║     - 급격한 방향 전환 감소                                   ║
║     - 부드러운 궤적                                           ║
║                                                               ║
║  2. 오류 기반 학습 (Error Learning)                           ║
║     - 벽 충돌 시 운동 패턴 수정                               ║
║     - Pain Zone 진입 시 회피 패턴 강화                        ║
║                                                               ║
║  3. 예측적 운동 조절 (Predictive Control)                     ║
║     - 현재 속도/방향 기반 미래 위치 예측                      ║
║     - 충돌 전 사전 회피                                       ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 2. 아키텍처

### 2.1 소뇌 구조 (생물학적)

```
┌─────────────────────────────────────────────────────────────┐
│                    Cerebellum (소뇌)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  입력 (Mossy Fibers)          입력 (Climbing Fibers)        │
│  - 감각 정보                  - 오류 신호                   │
│  - 운동 명령 복사             - 예상치 못한 결과            │
│         │                            │                      │
│         ▼                            │                      │
│  ┌─────────────┐                     │                      │
│  │  Granule    │ ← 가장 많은 뉴런    │                      │
│  │  Cells      │   희소 표현         │                      │
│  └──────┬──────┘                     │                      │
│         │ Parallel Fibers            │                      │
│         ▼                            ▼                      │
│  ┌─────────────────────────────────────┐                    │
│  │         Purkinje Cells              │                    │
│  │    (주요 출력, 억제성)               │                    │
│  │    ← 평행섬유 + 등반섬유 학습        │                    │
│  └──────────────┬──────────────────────┘                    │
│                 │ (억제)                                    │
│                 ▼                                           │
│  ┌─────────────────────────────────────┐                    │
│  │       Deep Cerebellar Nuclei        │                    │
│  │         (심부 소뇌핵)                │                    │
│  │       → 운동 출력으로 전달          │                    │
│  └─────────────────────────────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 단순화된 Phase 6a 구조

```
┌─────────────────────────────────────────────────────────────┐
│                 Phase 6a 구현 (단순화)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Granule Cells (과립세포)                │    │
│  │                  (300 뉴런)                          │    │
│  │   - Motor 명령 + Sensory 입력 통합                   │    │
│  │   - 희소 표현 (Sparse Representation)                │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │ Parallel Fibers                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Purkinje Cells (푸르키네 세포)          │    │
│  │                  (100 뉴런)                          │    │
│  │   - 운동 출력 조절                                   │    │
│  │   - Climbing Fiber 오류 신호로 학습                  │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │ (억제)                         │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Deep Nuclei (심부핵)                    │    │
│  │                  (100 뉴런)                          │    │
│  │   - 운동 출력 최종 조절                              │    │
│  │   - Motor 뉴런으로 전달                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Error Signal (오류 신호)                │    │
│  │                  (50 뉴런)                           │    │
│  │   - Pain/Wall 충돌 시 활성화                         │    │
│  │   - Climbing Fiber 역할                              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  총 뉴런: 550개                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 연결 구조

```
입력:
  Motor L/R ──────────────────┐
  Sensory (Food, Wall, Pain) ─┼──► Granule Cells
  Current Velocity ───────────┘

내부:
  Granule Cells ───────────────► Purkinje Cells (Parallel Fibers)
  Error Signal ────────────────► Purkinje Cells (Climbing Fibers)
  Purkinje Cells ──────────────► Deep Nuclei (억제)

출력:
  Deep Nuclei ─────────────────► Motor L/R (조절된 출력)
```

---

## 3. 메커니즘

### 3.1 운동 평활화 (Motor Smoothing)

**문제**: 현재 모터 출력이 급격하게 변함 → 불안정한 움직임

**해결**: Deep Nuclei가 이전 출력과 현재 명령을 통합

```python
# 평활화 메커니즘
Motor_Command (raw) → Granule → Purkinje → Deep Nuclei
                                              ↓
                                    Motor (smoothed)

# Deep Nuclei = Low-pass filter 역할
# 급격한 변화를 완충
```

### 3.2 오류 기반 학습 (Error-based Learning)

**Climbing Fiber 역할**:
- 벽 충돌/Pain Zone 진입 시 Error Signal 활성화
- Purkinje Cell의 시냅스 가중치 조절 (LTD)

```python
def on_error_event(self, error_type):
    """
    오류 발생 시 Climbing Fiber 활성화

    error_type: 'wall_collision', 'pain_entry', 'food_miss'
    """
    # Error Signal 뉴런 활성화
    self.error_signal.vars["I_input"].view[:] = 80.0

    # Purkinje Cells 학습 트리거
    # → 현재 활성화된 Granule-Purkinje 시냅스 약화 (LTD)
```

### 3.3 예측적 조절 (Predictive Control)

**Forward Model**:
- 현재 운동 명령 → 예상 결과 예측
- 예상과 실제 불일치 시 조절

```python
# Granule Cells = 현재 상태 + 운동 명령 조합
# Purkinje Cells = 예상 결과 인코딩
# 불일치 시 Deep Nuclei 출력 수정
```

---

## 4. 구현 계획

### 4.1 뉴런 집단

| 집단 | 뉴런 수 | 역할 |
|------|---------|------|
| Granule Cells | 300 | 입력 통합, 희소 표현 |
| Purkinje Cells | 100 | 운동 조절, 학습 |
| Deep Nuclei | 100 | 최종 출력 |
| Error Signal | 50 | 오류 감지 (Climbing Fiber) |
| **합계** | **550** | |

### 4.2 시냅스 연결

```python
# 입력 → Granule
Motor_L/R → Granule (efference copy)
Sensory → Granule (current state)

# Granule → Purkinje (Parallel Fibers, 학습 가능)
Granule → Purkinje (dense, plastic)

# Error → Purkinje (Climbing Fibers)
Error_Signal → Purkinje (strong, triggers LTD)

# Purkinje → Deep Nuclei (억제)
Purkinje → Deep_Nuclei (inhibitory)

# Deep Nuclei → Motor (조절)
Deep_Nuclei → Motor_L/R (excitatory, modulatory)
```

### 4.3 시각화 업데이트

```
┌─────────────────────────────┐
│  CEREBELLUM                 │
│  ▓▓▓░░░░░░░ Granule   30%   │
│  ▓▓░░░░░░░░ Purkinje  20%   │
│  ▓▓▓▓░░░░░░ DeepNuc   40%   │
│  ░░░░░░░░░░ Error      0%   │
└─────────────────────────────┘
```

---

## 5. 예상 효과

### 5.1 행동 변화

| 상황 | Phase 5 (PFC만) | Phase 6a (소뇌 추가) |
|------|-----------------|----------------------|
| 방향 전환 | 급격함 | 부드러움 |
| 벽 근처 | 늦은 반응 | 예측적 회피 |
| 반복 충돌 | 같은 실수 | 점진적 개선 |

### 5.2 성능 기대

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 6a 성능 기대치                                         ║
╠═══════════════════════════════════════════════════════════════╣
║  생존율:       60% → 65-70% (예측적 회피로 개선)              ║
║  Pain Avoid:   94% → 96%+ (오류 학습으로 개선)                ║
║  움직임:       더 부드럽고 효율적                             ║
╚═══════════════════════════════════════════════════════════════╝
```

---

## 6. 뉴런 구성 요약

```
Phase 6a 추가 뉴런: 550개

전체 뉴런 수: 7,100 (Phase 5) + 550 (Phase 6a) = 7,650 뉴런
```

---

*Phase 6a 설계 문서*
*2025-01-31*
