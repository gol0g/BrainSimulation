# Phase 3 설계: 해마 (Hippocampus)

> **Genesis Brain Project - Phase 3: 공간 기억**
>
> 날짜: 2025-01-28
> 상태: 설계 중

---

## 1. 목표

### 1.1 현재 문제

```
Phase 2b 결과:
- 생존율: 50% (목표 달성)
- 사망 원인: starve 50%, timeout 50%
- 문제: 음식을 효율적으로 찾지 못함
```

### 1.2 Phase 3 목표

```
1. 공간 기억: 음식이 있었던 위치 기억
2. 목표 지향 탐색: 기억된 음식 위치로 이동
3. 생존율 향상: starve 비율 감소
```

### 1.3 성공 기준

| 지표 | 현재 | 목표 |
|------|------|------|
| 생존율 | 50% | >60% |
| Starve Death | 50% | <30% |
| Avg Food | 26.4 | >35 |
| Reward Freq | 1.01% | >3% |

---

## 2. 생물학적 배경

### 2.1 해마의 역할

```
해마 (Hippocampus):
- 위치: 측두엽 내측
- 기능: 공간 기억, 일화 기억
- 주요 세포: Place Cells, Grid Cells

Place Cell (장소 세포):
- 특정 위치에서만 발화
- "인지 지도" 형성
- O'Keefe & Nadel (1978) - 노벨상 2014
```

### 2.2 회로 구조

```
실제 해마 회로:
EC (내후각피질) → DG (치상회) → CA3 → CA1 → EC

단순화 모델 (Phase 3):
Position Input → Place Cells → Food Memory → Motor Bias
```

### 2.3 학습 메커니즘

```
Hebbian Learning: "함께 발화하는 뉴런은 함께 연결된다"

Place Cell + Food 감지 → 연결 강화
→ 나중에 Place Cell만 활성화되어도 "음식 기대" 활성화
→ 해당 방향으로 이동 편향
```

---

## 3. 아키텍처 설계

### 3.1 신규 뉴런

```
Hippocampus (Phase 3 신규):
  Position Encoder:    200   # x, y 위치 인코딩
  Place Cells:         400   # 공간 표상 (격자 배치)
  Food Memory:         200   # 음식 위치 기억
  Goal Signal:         100   # 목표 방향 신호

Total 추가: 900 neurons
Phase 3 총합: 5,200 + 900 = 6,100 neurons
```

### 3.2 Place Cell 구현

```python
# Place Cell 배치: 20x20 격자 (400개)
# 각 Place Cell은 맵의 특정 영역에서 활성화

place_cell_centers = []
for i in range(20):
    for j in range(20):
        x = (i + 0.5) * (map_width / 20)
        y = (j + 0.5) * (map_height / 20)
        place_cell_centers.append((x, y))

# 활성화 함수: 가우시안 (거리 기반)
def place_cell_activation(agent_pos, cell_center, sigma=30):
    dist = distance(agent_pos, cell_center)
    return exp(-dist^2 / (2 * sigma^2))
```

### 3.3 시냅스 연결

```
Phase 3 시냅스:

1. Position → Place Cells (고정)
   - 위치 입력을 Place Cell 활성화로 변환
   - 가우시안 수용장 (Receptive Field)

2. Place Cells → Food Memory (학습!)
   - Hebbian STDP
   - 음식 발견 시 활성화된 Place Cell 강화
   - 이것이 "음식이 여기 있었다" 기억

3. Food Memory → Motor Bias (고정)
   - 기억된 음식 위치 방향으로 약한 편향
   - Hunger와 결합하여 배고플 때만 작동

4. Hunger → Food Memory (조절)
   - 배고플 때 Food Memory 활성화 증폭
   - 배부를 때는 기억 탐색 억제
```

### 3.4 회로 다이어그램

```
                    Agent Position (x, y)
                           │
                           ▼
                   ┌───────────────┐
                   │ Position      │
                   │ Encoder       │
                   └───────┬───────┘
                           │ 가우시안 수용장
                           ▼
                   ┌───────────────┐
                   │ Place Cells   │◄────── Current Position
                   │ (400 neurons) │        → 특정 셀 활성화
                   └───────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
      ┌─────────┐    ┌─────────┐    ┌─────────┐
      │ Food    │    │ Pain    │    │ (미래)  │
      │ Memory  │    │ Memory  │    │ 기타    │
      └────┬────┘    └────┬────┘    └─────────┘
           │              │
           │   Hebbian    │   (Phase 3b)
           │   Learning   │
           │              │
           ▼              ▼
      ┌─────────┐    ┌─────────┐
      │ Goal    │    │ Avoid   │
      │ Signal  │    │ Signal  │
      └────┬────┘    └────┬────┘
           │              │
           └──────┬───────┘
                  │
                  ▼
            ┌───────────┐
            │  Motor    │
            │  L / R    │
            └───────────┘
```

---

## 4. 구현 계획

### 4.1 Phase 3a: Place Cells (기본)

```
목표: Place Cell 활성화 검증

구현:
1. Position Encoder: x, y → 전류
2. Place Cells: 20x20 격자, 가우시안 수용장
3. 시각화: Place Cell 활성화 맵

검증:
- 에이전트 이동 시 Place Cell 패턴 변화 확인
- 같은 위치에서 같은 셀 활성화 확인
```

### 4.2 Phase 3b: Food Memory

```
목표: 음식 위치 기억 학습

구현:
1. Place Cells → Food Memory: Hebbian STDP
2. 음식 발견 시 학습 신호
3. Food Memory → Goal Signal

검증:
- 음식 발견 후 해당 Place Cell 강화 확인
- Food Memory 활성화 패턴 확인
```

### 4.3 Phase 3c: Goal-Directed Navigation

```
목표: 기억된 위치로 이동

구현:
1. Goal Signal → Motor Bias
2. Hunger × Food Memory → 목표 활성화
3. Motor Bias: 목표 방향으로 약한 회전

검증:
- 배고플 때 기억된 음식 위치로 이동 경향
- Starve 비율 감소
```

---

## 5. 기술 상세

### 5.1 Place Cell 활성화

```python
# Gym에서 제공할 정보
observation["position"] = (agent_x, agent_y)  # 절대 위치

# Brain에서 Place Cell 활성화 계산
def compute_place_cell_input(position):
    """위치를 Place Cell 입력 전류로 변환"""
    x, y = position
    currents = np.zeros(400)  # 20x20 Place Cells

    for i, (cx, cy) in enumerate(place_cell_centers):
        dist_sq = (x - cx)**2 + (y - cy)**2
        sigma = 30.0  # 수용장 크기
        currents[i] = np.exp(-dist_sq / (2 * sigma**2)) * max_current

    return currents
```

### 5.2 Hebbian STDP

```python
# PyGeNN Hebbian STDP 모델
hebbian_stdp = create_weight_update_model(
    "HebbianSTDP",
    params=["eta", "wMax", "wMin"],
    vars=[("g", "scalar")],

    # Pre-synaptic spike
    pre_spike_code="""
        const scalar dt = t - sT_post;
        if (dt > 0) {
            g += eta * exp(-dt / 20.0);  // LTP
            g = min(g, wMax);
        }
    """,

    # Post-synaptic spike
    post_spike_code="""
        const scalar dt = t - sT_pre;
        if (dt > 0) {
            g -= 0.5 * eta * exp(-dt / 20.0);  // LTD (약함)
            g = max(g, wMin);
        }
    """
)
```

### 5.3 Food Memory 학습

```python
# 음식 발견 시 학습 신호
def on_food_eaten():
    # 현재 활성화된 Place Cells와 Food Memory 연결 강화
    # Hebbian: 동시 활성화 시 연결 강화
    learning_signal = 1.0  # 음식 발견 = 강한 학습

# 시간 경과 시 약한 망각
def on_step():
    # 음식 없이 Place Cell 활성화 = 약한 LTD
    # 잘못된 기억 점진적 삭제
```

---

## 6. 파라미터

### 6.1 Place Cell 파라미터

```python
# Place Cell 설정
n_place_cells: int = 400         # 20x20 격자
place_cell_sigma: float = 30.0   # 수용장 크기 (픽셀)
place_cell_max_current: float = 50.0

# 격자 크기 = map_size / 20 = 500 / 20 = 25 픽셀
# sigma = 30이면 약간 겹침 (smooth coverage)
```

### 6.2 학습 파라미터

```python
# Hebbian STDP
hebbian_eta: float = 0.1         # 학습률
hebbian_w_max: float = 30.0      # 최대 가중치
hebbian_w_min: float = 0.0       # 최소 가중치 (억제 없음)
hebbian_tau: float = 20.0        # 시간 상수 (ms)
```

### 6.3 Goal Signal 파라미터

```python
# Food Memory → Goal Signal
food_memory_to_goal: float = 20.0  # 흥분

# Goal Signal → Motor (약한 편향)
goal_to_motor: float = 10.0        # 음식 동측과 비슷하게
```

---

## 7. 예상 문제

### 7.1 Place Cell 안정성

```
문제: Place Cell이 너무 넓거나 좁으면 공간 해상도 문제
해결: sigma 튜닝 (20-40 범위 테스트)
```

### 7.2 기억 간섭

```
문제: 여러 음식 위치가 겹치면 혼란
해결:
1. 최근 기억 우선 (recency)
2. 강한 기억 우선 (strength)
3. Hunger가 높을 때만 기억 활성화
```

### 7.3 학습 속도

```
문제: 너무 빠른 학습 = 불안정, 너무 느린 학습 = 효과 없음
해결: eta 튜닝 (0.05-0.2 범위 테스트)
```

---

## 8. 검증 계획

### 8.1 Phase 3a 검증 (Place Cells)

```bash
# 5 에피소드, Place Cell 활성화 로그
python forager_brain.py --episodes 5 --log-level debug --render pygame

# 확인 사항:
# - 위치 변화 → Place Cell 패턴 변화
# - 같은 위치 재방문 → 같은 패턴
```

### 8.2 Phase 3b 검증 (Food Memory)

```bash
# 20 에피소드, 학습 효과 확인
python forager_brain.py --episodes 20 --log-level normal

# 확인 사항:
# - 음식 발견 후 시냅스 강화
# - 기억 패턴 형성
```

### 8.3 Phase 3c 검증 (Navigation)

```bash
# 50 에피소드, 성능 향상 확인
python forager_brain.py --episodes 50 --log-level normal

# 목표:
# - Starve Death < 30%
# - Avg Food > 35
```

---

## 9. 구현 순서

```
1. Gym 수정: position 정보 추가 (observation)
2. Brain 수정: Place Cell 뉴런 추가
3. Place Cell 활성화 테스트
4. Hebbian STDP 시냅스 추가
5. Food Memory 학습 테스트
6. Goal Signal → Motor 연결
7. 통합 테스트 및 튜닝
```

---

## 10. 결론

### Phase 3 요약

```
목표: 공간 기억으로 음식 탐색 효율화

핵심 구성요소:
1. Place Cells: 공간 표상
2. Food Memory: 음식 위치 기억 (Hebbian 학습)
3. Goal Signal: 목표 지향 이동

예상 효과:
- Starve Death 50% → <30%
- 생존율 50% → >60%

생물학적 근거:
- Place Cell (O'Keefe, 노벨상 2014)
- Hebbian Learning ("함께 발화하면 함께 연결")
```

---

*Phase 3 설계 문서*
*2025-01-28*
