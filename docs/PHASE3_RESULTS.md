# Phase 3a 결과 보고서

> **Genesis Brain Project - Phase 3a: Place Cells 기본 구조**
>
> 날짜: 2025-01-28
> 상태: **Phase 3a 완료 - Phase 3b (학습) 필요**

---

## 1. 테스트 결과

### 1.1 Phase 3a 통계 (10 에피소드)

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 3a 결과 (10 Episodes) - Place Cells 기본 구조          ║
╠═══════════════════════════════════════════════════════════════╣
║  생존율:        50.0% ✓ (목표: >40%)                          ║
║  평균 생존:     2289.6 steps                                  ║
║  평균 음식:     21.8개                                        ║
║  Reward Freq:   0.95% ✗ (목표: >5%)                           ║
║  Pain Avoidance:94.9% ✓ (목표: <15% pain time)                ║
║  사망 원인:     starve 50%, timeout 50%                       ║
╚═══════════════════════════════════════════════════════════════╝
```

### 1.2 가중치 튜닝 결과

| 설정 | 생존율 | Starve | 비고 |
|------|--------|--------|------|
| 강한 가중치 (15, 20) | 30% | 70% | 노이즈로 음식 탐색 방해 |
| 약한 가중치 (5, 10) | **50%** | 50% | Phase 2b와 동등 |

**결론**: 학습 없는 Hippocampus는 효과 없음 (neutral)

### 1.3 Phase 비교

| 지표 | Phase 2b | Phase 3a | 변화 |
|------|----------|----------|------|
| 생존율 | 50% | 50% | 0% |
| Avg Steps | 2614.8 | 2289.6 | -12.4% |
| Avg Food | 26.4 | 21.8 | -17.4% |
| Pain Avoidance | 95.3% | 94.9% | -0.4% |

---

## 2. 구현 내용

### 2.1 뉴런 추가

```
Phase 3a 신규 (600 neurons):
  Place Cells:     400   # 20x20 격자
  Food Memory:     200   # 음식 위치 기억

Total: 5,200 + 600 = 5,800 neurons
```

### 2.2 시냅스 추가

```python
# Place Cells → Food Memory (고정, 학습 예정)
place_to_food_memory_weight: 5.0

# Food Memory → Motor (약한 편향)
food_memory_to_motor_weight: 5.0

# Hunger → Food Memory (배고플 때 활성화)
hunger_to_food_memory_weight: 10.0
```

### 2.3 Place Cell 활성화

```python
def _compute_place_cell_input(pos_x, pos_y):
    """가우시안 수용장 활성화"""
    for i, (cx, cy) in enumerate(place_cell_centers):
        dist_sq = (pos_x - cx)**2 + (pos_y - cy)**2
        activation = exp(-dist_sq / (2 * sigma**2))
        currents[i] = activation * 50.0
```

---

## 3. 교훈

### 3.1 핵심 발견

**학습 없는 Hippocampus는 노이즈**
```
문제:
- Place Cells가 위치에 따라 활성화됨 ✓
- 하지만 Food Memory는 무작위 활성화
- Motor에 무작위 신호 추가 → 음식 탐색 방해

해결:
- 가중치 감소로 간섭 최소화 (15→5, 20→10)
- Phase 3b에서 Hebbian 학습 필요
```

### 3.2 다음 단계: Phase 3b

```
Phase 3b: Hebbian STDP 학습

목표:
- 음식 발견 시 Place Cells → Food Memory 연결 강화
- "이 위치에 음식이 있었다" 기억 형성
- 배고플 때 기억된 위치로 이동

구현 필요:
1. Hebbian STDP 시냅스 모델 추가
2. 음식 발견 시 학습 신호 (neuromodulation)
3. 기억 → 방향 변환 로직
```

---

## 4. 아키텍처

```
┌────────────────────────────────────────────────────┐
│              Phase 3a 아키텍처                     │
├────────────────────────────────────────────────────┤
│                                                    │
│  Position (x, y)                                   │
│        │                                           │
│        ▼                                           │
│  ┌───────────┐                                     │
│  │  Place    │  ← 가우시안 수용장                  │
│  │  Cells    │    (20x20 = 400)                    │
│  └─────┬─────┘                                     │
│        │ w=5.0 (고정)                              │
│        ▼                                           │
│  ┌───────────┐     Hunger                          │
│  │   Food    │◄────────────┐                       │
│  │  Memory   │  w=10.0     │                       │
│  └─────┬─────┘             │                       │
│        │ w=5.0 (약함)      │                       │
│        ▼                   │                       │
│  ┌───────────┐      ┌──────┴─────┐                 │
│  │  Motor    │◄─────│  Phase 1-2 │                 │
│  │  L / R    │      │  회로들    │                 │
│  └───────────┘      └────────────┘                 │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 5. 실행 명령어

```bash
# Phase 3a 테스트 (WSL)
wsl -d Ubuntu-24.04 -- bash -c "
export CUDA_PATH=/usr/local/cuda-12.6;
export PATH=\$CUDA_PATH/bin:/usr/local/bin:/usr/bin:/bin;
export LD_LIBRARY_PATH=\$CUDA_PATH/lib64;
source ~/pygenn_wsl/bin/activate;
cd ~/pygenn_test;
python /mnt/c/.../backend/genesis/forager_brain.py \
    --episodes 20 --log-level normal --render none
"
```

---

## 6. 결론

### Phase 3a 상태: 기본 구조 완료

```
✓ 완료:
  - Place Cells (400 neurons, 20x20 격자)
  - Food Memory (200 neurons)
  - Hunger → Food Memory 조절
  - 위치 기반 Place Cell 활성화
  - Phase 2b 성능 유지 (50% 생존율)

✗ 미완료:
  - Hebbian STDP 학습 (Phase 3b)
  - 음식 위치 기억 형성
  - 기억 기반 탐색

다음 단계:
  - Phase 3b: Hebbian 학습 구현
  - 또는: Phase 3 스킵, 다른 방향 탐색
```

---

*Phase 3a 결과 보고서*
*2025-01-28*
