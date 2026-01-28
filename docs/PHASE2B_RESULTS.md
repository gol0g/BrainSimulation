# Phase 2b 최종 결과 보고서

> **Genesis Brain Project - Phase 2b: 편도체 (Amygdala)**
>
> 날짜: 2025-01-28
> 상태: **검증 완료 - 목표 달성!**

---

## 1. 최종 테스트 결과

### 1.1 최종 통계 (20 에피소드)

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 2b 최종 결과 (20 Episodes) - 목표 달성!                ║
╠═══════════════════════════════════════════════════════════════╣
║  생존율:        50.0% ✓ (목표: >40%)                          ║
║  평균 생존:     2614.8 steps                                  ║
║  평균 음식:     26.4개                                        ║
║  Reward Freq:   1.01% ✗ (목표: >5%)                           ║
║  Homeostasis:   28.3%                                         ║
║  Pain Avoidance:95.3% ✓ (목표: <15% pain time)                ║
║  사망 원인:     starve 50%, timeout 50% (pain 0%!)            ║
╚═══════════════════════════════════════════════════════════════╝
```

### 1.2 초기 vs 최종 비교

| 지표 | 초기 (3ep) | 최종 (20ep) | 개선 |
|------|-----------|-------------|------|
| 생존율 | 0% | **50%** | +50% |
| Pain Death | 33% | **0%** | -33% |
| Pain Time | 10.4% | **4.7%** | -5.7% |
| Avg Steps | 1456 | **2615** | +80% |
| Avg Food | 13.7 | **26.4** | +93% |
| Pain Avoidance | 89.6% | **95.3%** | +5.7% |

### 1.3 회로 검증 결과

| 회로 | 설계 | 결과 | 상태 |
|------|------|------|------|
| Pain Eye L/R | Pain Zone 방향 감지 | 작동 확인 | ✓ |
| Danger Sensor | Pain Zone 거리 감지 | 작동 확인 | ✓ |
| Pain → LA | 무조건 반사 (US) | 작동 확인 | ✓ |
| Danger → LA | 조건 자극 (CS) | 작동 확인 (고정) | ✓ |
| LA → CEA → Fear | 내부 연결 | 작동 확인 | ✓ |
| **Pain → Motor** | **방향성 Push-Pull** | **작동 확인!** | ✓ |
| Hunger ↔ Fear 경쟁 | 동기 경쟁 | **작동 확인!** | ✓ |

---

## 2. 핵심 튜닝

### 2.1 튜닝 내역

**문제 1: MOTOR DEAD**
```
원인: Satiety → Motor 억제가 너무 강함
해결: satiety_to_motor_weight: -8.0 → -4.0
결과: Motor 활성화 개선
```

**문제 2: Pain Zone 탈출 불가 (핵심!)**
```
원인: Fear → Motor가 양쪽 동시 활성화 (무방향)
      → Pain Zone에서 회전하지 못하고 갇힘

해결: Pain Eye L/R → Motor Push-Pull (방향성)
      Pain_L → Motor_R (Push +60)
      Pain_L → Motor_L (Pull -40)
      Pain_R → Motor_L (Push +60)
      Pain_R → Motor_R (Pull -40)

결과: Pain Time 10.4% → 4.7%, Pain Death 33% → 0%!
```

### 2.2 최종 가중치

```python
# Phase 2b 최종 설정
satiety_to_motor_weight: -4.0     # (초기 -8.0)
fear_push_weight: 60.0            # Pain→Motor Push
fear_pull_weight: -40.0           # Pain→Motor Pull

# 기타 (변경 없음)
pain_to_la_weight: 50.0
danger_to_la_weight: 25.0
la_to_cea_weight: 30.0
cea_to_fear_weight: 25.0
hunger_to_fear_weight: -15.0
fear_to_hunger_weight: -10.0
```

---

## 3. 아키텍처

### 3.1 뉴런 구성

```
Total: 5,200 neurons

Sensory:
  Food Eye L/R:        400 × 2 = 800
  Wall Eye L/R:        200 × 2 = 400
  Pain Eye L/R:        200 × 2 = 400   ← Phase 2b
  Danger Sensor:       200              ← Phase 2b

Hypothalamus:
  Low Energy Sensor:   200
  High Energy Sensor:  200
  Hunger Drive:        500
  Satiety Drive:       500

Amygdala:                               ← Phase 2b
  Lateral Amygdala:    500
  Central Amygdala:    300
  Fear Response:       200

Motor:
  Motor L/R:           500 × 2 = 1,000
```

### 3.2 시냅스 구조

```
Phase 2b 핵심 시냅스:

1. 무조건 반사 (US)
   Pain_L/R → LA:      +50.0

2. 조건 자극 (CS)
   Danger → LA:        +25.0

3. Amygdala 내부
   LA → CEA:           +30.0
   CEA → Fear:         +25.0

4. 방향성 회피 반사 (핵심!)
   Pain_L → Motor_R:   +60.0 (Push)
   Pain_L → Motor_L:   -40.0 (Pull)
   Pain_R → Motor_L:   +60.0 (Push)
   Pain_R → Motor_R:   -40.0 (Pull)

5. Hunger-Fear 경쟁
   Hunger → CEA:       -15.0 (배고프면 공포 감소)
   CEA → Hunger:       -10.0 (공포 시 식욕 감소)
```

---

## 4. 교훈

### 4.1 핵심 발견

**방향성이 중요하다**
```
✗ Fear → Motor (양쪽 동시) → 회전 불가 → 갇힘
✓ Pain_L/R → Motor Push-Pull → 방향성 회피 → 탈출!

생물학적 근거:
- 실제 동물도 고통 방향에서 반대로 회피
- 무방향 "공포"만으로는 불충분
```

**MOTOR DEAD의 원인**
```
Satiety + Fear 동시 활성화 → Motor 억제 누적
→ Satiety 억제 완화 (-8 → -4)로 해결
```

### 4.2 실험 프로세스 효과

| 단계 | 에피소드 | 발견 |
|------|----------|------|
| 초기 테스트 | 3 | 회로 작동 확인, MOTOR DEAD 발견 |
| Satiety 튜닝 | 5 | 생존율 20% (첫 timeout) |
| 방향성 수정 | 10 | Pain Death 0%, 생존율 30% |
| 최종 검증 | 20 | **생존율 50% 달성!** |

---

## 5. 다음 단계 (Phase 3)

### 5.1 남은 과제

```
Reward Freq: 1.01% (목표 5% 미달)
- 원인: 음식/완주 보상 빈도 낮음
- 해결: R-STDP 학습 또는 보상 설계 개선 (Phase 3)
```

### 5.2 Phase 3 후보

1. **R-STDP 학습**: Danger → LA 시냅스 학습
2. **해마 (Hippocampus)**: 공간 기억
3. **기저핵 (Basal Ganglia)**: 습관/절차 학습

---

## 6. 실행 명령어

```bash
# Phase 2b 테스트 (WSL)
wsl -d Ubuntu-24.04 -- bash -c "
export CUDA_PATH=/usr/local/cuda-12.6;
export PATH=\$CUDA_PATH/bin:/usr/local/bin:/usr/bin:/bin;
export LD_LIBRARY_PATH=\$CUDA_PATH/lib64;
source ~/pygenn_wsl/bin/activate;
cd ~/pygenn_test;
python /mnt/c/.../backend/genesis/forager_brain.py \
    --episodes 20 --log-level normal --render none
"

# 시각화 모드
python forager_brain.py --episodes 5 --render pygame
```

---

## 7. 결론

### Phase 2b 상태: 검증 완료!

```
✓ 달성:
  - 생존율 50% (목표 >40%)
  - Pain Avoidance 95.3% (목표 <15% pain time)
  - Pain Death 0%
  - Hunger-Fear 경쟁 관찰
  - 방향성 Pain 회피 회로

✗ 미달:
  - Reward Freq 1.01% (목표 >5%)

핵심 교훈:
  - 방향성(directional) Push-Pull이 핵심
  - 무방향 Fear로는 탈출 불가
  - MOTOR DEAD는 Satiety 억제 완화로 해결
```

---

*Phase 2b 최종 결과 보고서*
*2025-01-28*
