# Phase 3 결과 보고서

> **Genesis Brain Project - Phase 3: Hippocampus (Place Cells + Hebbian Learning)**
>
> 날짜: 2025-01-29
> 상태: **Phase 3b 완료 + Learning Checkpoint 실험 완료**

---

## 1. 최종 결과

### 1.1 Phase 3b 통계 (20 에피소드)

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 3b 최종 결과 (20 Episodes) - Hebbian 학습              ║
╠═══════════════════════════════════════════════════════════════╣
║  생존율:        60.0% ✓ (목표: >40%)                          ║
║  평균 생존:     2703.8 steps                                  ║
║  평균 음식:     27.6개                                        ║
║  Reward Freq:   1.02% ✗ (목표: >5%)                           ║
║  Pain Avoidance:94.9% ✓ (목표: <15% pain time)                ║
║  사망 원인:     starve 40%, timeout 60%                       ║
║  학습 이벤트:   552회 (평균 27.6/ep)                          ║
╚═══════════════════════════════════════════════════════════════╝
```

### 1.2 Phase 비교

| 지표 | Phase 2b | Phase 3a | Phase 3b | 변화 (2b→3b) |
|------|----------|----------|----------|--------------|
| 생존율 | 50% | 50% | **60%** | +10% |
| Starve | 50% | 50% | **40%** | -10% |
| Avg Steps | 2614.8 | 2289.6 | **2703.8** | +3.4% |
| Avg Food | 26.4 | 21.8 | **27.6** | +4.5% |
| Pain Avoid | 95.3% | 94.9% | **94.9%** | -0.4% |

**결론**: Hebbian 학습이 Starve 비율을 10% 감소시킴

---

## 2. Learning Checkpoint 실험

### 2.1 실험 목적

에피소드 간 학습 보존이 누적 학습 효과를 가져오는지 검증

### 2.2 실험 결과 (20 에피소드)

| 조건 | 생존율 | Starve | Avg Food | Final Avg Weight |
|------|--------|--------|----------|------------------|
| **persist OFF** | **60%** | **40%** | **27.6** | 5.12 (에피소드 내) |
| **persist ON** | 30% | 70% | 20.9 | 6.47 (누적) |

### 2.3 분석

```
╔═══════════════════════════════════════════════════════════════╗
║  Learning Checkpoint 결과: 성능 저하!                         ║
╠═══════════════════════════════════════════════════════════════╣
║  원인:                                                        ║
║  1. 음식이 매 에피소드 랜덤 위치에 스폰                       ║
║  2. 학습된 "음식 위치"가 다음 에피소드에서는 무의미           ║
║  3. 누적된 가중치가 노이즈로 작용 → 간섭 증가                 ║
║                                                               ║
║  결론:                                                        ║
║  - 공간 기억 누적은 고정 환경에서만 유효                      ║
║  - 랜덤 환경에서는 에피소드별 학습이 더 효과적                ║
║  - --persist-learning 플래그는 비권장                         ║
╚═══════════════════════════════════════════════════════════════╝
```

### 2.4 persist-learning 사용 시기

```
✓ 권장: 고정 음식 위치 (학습 환경)
✓ 권장: 특정 영역에 음식 밀집 (패턴 존재)
✗ 비권장: 완전 랜덤 음식 스폰 (현재 환경)
```

---

## 3. Phase 3b 구현

### 3.1 Hebbian 학습 메커니즘

```python
def learn_food_location(self):
    """
    음식 발견 시 Hebbian 학습

    Δw = η * pre_activity
    - η = 0.1 (학습률)
    - pre_activity = Place Cell 활성화 (가우시안)
    """
    # 1. 현재 가중치 가져오기
    weights = self.place_to_food_memory.vars["g"].view

    # 2. 활성화된 Place Cells 기반 학습
    for i in range(n_place_cells):
        if active_cells[i] > 0.1:
            delta_w = eta * active_cells[i]
            weights[i, :] += delta_w
            weights[i, :] = clip(weights[i, :], 0, w_max)

    # 3. 가중치 다시 저장
    self.place_to_food_memory.vars["g"].push_to_device()
```

### 3.2 Learning Checkpoint 메서드

```python
def save_hippocampus_weights(self, filepath=None):
    """학습된 가중치 저장"""
    np.save(filepath, weights)

def load_hippocampus_weights(self, filepath=None):
    """저장된 가중치 복원"""
    weights = np.load(filepath)
    self.place_to_food_memory.vars["g"].view[:] = weights
    self.place_to_food_memory.vars["g"].push_to_device()
```

### 3.3 시냅스 구성

```
변경 사항 (Phase 3a → 3b):

Place Cells → Food Memory:
  - SPARSE (고정) → DENSE (학습 가능)
  - 초기 가중치: 5.0
  - 학습률 (η): 0.1
  - 최대 가중치: 30.0
```

---

## 4. 아키텍처

```
┌────────────────────────────────────────────────────┐
│              Phase 3b 아키텍처                     │
├────────────────────────────────────────────────────┤
│                                                    │
│  Position (x, y)                                   │
│        │                                           │
│        ▼                                           │
│  ┌───────────┐                                     │
│  │  Place    │  ← 가우시안 수용장                  │
│  │  Cells    │    (20x20 = 400)                    │
│  └─────┬─────┘                                     │
│        │ Hebbian STDP (η=0.1)                      │
│        │ ← 음식 발견 시 강화!                      │
│        ▼                                           │
│  ┌───────────┐     Hunger                          │
│  │   Food    │◄────────────┐                       │
│  │  Memory   │  w=10.0     │                       │
│  └─────┬─────┘             │                       │
│        │ w=5.0             │                       │
│        ▼                   │                       │
│  ┌───────────┐      ┌──────┴─────┐                 │
│  │  Motor    │◄─────│  Phase 1-2 │                 │
│  │  L / R    │      │  회로들    │                 │
│  └───────────┘      └────────────┘                 │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 5. 교훈

### 5.1 핵심 발견

**1. Hebbian 학습은 에피소드 내에서 효과적**
```
- 음식 발견 시 Place Cell 연결 강화
- 같은 위치 재방문 시 Food Memory 활성화 증가
- Starve 비율 10% 감소
```

**2. 에피소드 간 학습 보존은 환경에 의존**
```
- 랜덤 환경: 누적 학습 = 노이즈 누적 (성능 저하)
- 고정 환경: 누적 학습 = 패턴 학습 (성능 향상 예상)
```

**3. Dense 연결이 학습에 필수**
```
- SPARSE: 가중치 접근 어려움
- DENSE: pull/push로 가중치 수정 가능
```

### 5.2 한계점

```
1. 공간 기억의 한계
   - 랜덤 환경에서 위치 기억은 무의미
   - 방향성 기억 또는 패턴 기억이 더 유용할 수 있음

2. Food Memory → Motor 연결이 방향성 없음
   - 현재 좌/우 구분 없이 Motor 활성화
   - 개선: Place Cell 위치 기반 방향 편향
```

---

## 6. 다음 단계

### 6.1 옵션

| 옵션 | 설명 | 효과 예상 |
|------|------|----------|
| ~~에피소드 간 학습 보존~~ | ~~가중치 저장/복원~~ | ~~✗ 랜덤 환경에서 무효~~ |
| **방향성 Food Memory** | 위치별 Motor 편향 | 목표 지향 이동 |
| **기저핵 추가** | 습관/절차 학습 | 행동 자동화 |

### 6.2 권장 다음 단계

```
Option A: 방향성 Food Memory (Phase 3c)
- Place Cell 좌측 활성화 → Motor_L 편향
- Place Cell 우측 활성화 → Motor_R 편향
- 음식이 있던 방향으로 이동 성향

Option B: Phase 4 (기저핵)
- 반복 행동의 자동화
- "습관" 형성 메커니즘
```

---

## 7. 실행 명령어

```bash
# Phase 3b 테스트 (WSL) - 권장
wsl -d Ubuntu-24.04 -- bash -c "
export CUDA_PATH=/usr/local/cuda-12.6;
export PATH=\$CUDA_PATH/bin:/usr/local/bin:/usr/bin:/bin;
export LD_LIBRARY_PATH=\$CUDA_PATH/lib64;
source ~/pygenn_wsl/bin/activate;
cd ~/pygenn_test;
python /mnt/c/.../backend/genesis/forager_brain.py \
    --episodes 20 --log-level normal --render none
"

# Learning Checkpoint 테스트 (비권장 - 랜덤 환경)
python forager_brain.py --episodes 20 --persist-learning

# 시각화 모드
python forager_brain.py --episodes 3 --render pygame
```

---

## 8. 결론

### Phase 3 상태: 완료!

```
✓ Phase 3a 완료:
  - Place Cells (400 neurons, 20x20 격자)
  - Food Memory (200 neurons)
  - 위치 기반 활성화

✓ Phase 3b 완료:
  - Hebbian 학습 구현
  - 음식 발견 시 연결 강화
  - Starve 비율 10% 감소 확인

✓ Learning Checkpoint 실험 완료:
  - 기능 구현 완료 (--persist-learning)
  - 랜덤 환경에서는 비효과적 확인
  - 고정 환경 실험 보류

성과:
  - 생존율: 50% → 60% (+10%)
  - Starve: 50% → 40% (-10%)
  - 학습 메커니즘 검증 완료

한계:
  - Reward Freq 여전히 낮음 (1.02%)
  - 공간 기억은 랜덤 환경에서 제한적
```

---

*Phase 3 결과 보고서*
*2025-01-29*
