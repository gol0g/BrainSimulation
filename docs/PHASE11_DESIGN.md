# Phase 11: 청각 피질 (Auditory Cortex) 설계

> 목표: 소리 자극 처리 및 청각-공포/기대 조건화

---

## 1. 생물학적 배경

### 청각 경로
```
소리 → 달팽이관 → 청각신경 → 뇌간 → 시상(MGN) → A1 → 연합 청각 피질
```

### 주요 영역
- **A1 (Primary Auditory Cortex)**: 기본 소리 특성 (주파수, 강도)
- **Belt/Parabelt**: 복잡한 소리 패턴 인식
- **청각-편도체 경로**: 위험 소리 → 즉각적 공포 반응

### 핵심 기능
- 주파수 토노토픽 조직 (저음 → 고음 배열)
- 시간적 패턴 인식
- 청각-시각 통합 (다감각 처리 기초)

---

## 2. 환경 확장: 소리 자극

### ForagerGym 소리 시스템
```python
# 새로운 감각 입력
observation["sound_danger_left"]   # 왼쪽 위험 소리 (0~1)
observation["sound_danger_right"]  # 오른쪽 위험 소리 (0~1)
observation["sound_food_left"]     # 왼쪽 음식 소리 (0~1)
observation["sound_food_right"]    # 오른쪽 음식 소리 (0~1)
```

### 소리 생성 규칙
```
위험 소리 (Pain Zone):
  - Pain Zone 근처에서 "경고음" 발생
  - 거리에 반비례하는 강도
  - 좌우 귀에 다른 강도 (방향 정보)

음식 소리:
  - 음식 클러스터 근처에서 "먹이 소리" 발생
  - 여러 음식이 모여있을 때 더 강함
  - 좌우 방향 정보 포함
```

---

## 3. 뉴런 구조

| 영역 | 뉴런 수 | 역할 |
|------|---------|------|
| Sound_Danger_Left | 100 | 왼쪽 위험 소리 입력 |
| Sound_Danger_Right | 100 | 오른쪽 위험 소리 입력 |
| Sound_Food_Left | 100 | 왼쪽 음식 소리 입력 |
| Sound_Food_Right | 100 | 오른쪽 음식 소리 입력 |
| A1_Danger | 150 | 위험 소리 처리 (A1) |
| A1_Food | 150 | 음식 소리 처리 (A1) |
| A2_Association | 200 | 청각 연합 영역 |
| **총계** | **900** | |

**Phase 11 완료 시 총 뉴런: 10,000 + 900 = 10,900**

---

## 4. 시냅스 연결

### 4.1 Sound Input → A1 (순방향)
```
Sound_Danger_L/R → A1_Danger (좌우 수렴)
Sound_Food_L/R → A1_Food (좌우 수렴)
```
- 가중치: 20.0

### 4.2 A1 → Amygdala (청각-공포 경로)
```
A1_Danger → Amygdala LA (위험 소리 → 공포)
```
- 가중치: 22.0 (빠른 공포 반응)
- 생물학적 근거: 청각-편도체 직접 경로

### 4.3 A1 → IT Cortex (청각-시각 통합)
```
A1_Danger → IT_Danger_Category (청각 위험 → 위험 범주)
A1_Food → IT_Food_Category (청각 음식 → 음식 범주)
```
- 가중치: 15.0

### 4.4 A1 → Motor (청각 유도 행동)
```
A1_Danger_from_L → Motor_R (왼쪽 위험 소리 → 오른쪽 회피)
A1_Food_from_L → Motor_L (왼쪽 음식 소리 → 왼쪽 접근)
```
- 가중치: 12.0

### 4.5 A2 Association (다감각 통합 준비)
```
A1_Danger + A1_Food → A2_Association
IT_Food/Danger → A2_Association (시각-청각 연합)
```
- 가중치: 10.0

### 4.6 Top-Down 조절
```
Fear → A1_Danger (공포 시 위험 소리 민감도 증가)
Hunger → A1_Food (배고플 때 음식 소리 민감도 증가)
```
- 가중치: 8.0

---

## 5. 학습 메커니즘

### 5.1 청각-공포 조건화
```
소리(CS) + 고통(US) → 소리만으로 공포 반응
A1_Danger → LA 시냅스 강화
```
- Hebbian 학습: 동시 활성화 시 가중치 증가

### 5.2 청각-음식 연합
```
음식 소리 + 음식 섭취 → 소리만으로 접근 행동
A1_Food → IT_Food 시냅스 강화
```

---

## 6. 시각화

```
┌─────────────────────────────────────────────────────────────┐
│                Phase 11: Auditory Cortex                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sound_Danger_L/R ──► A1_Danger ──┬──► Amygdala LA         │
│                           │       │                         │
│                           │       └──► IT_Danger            │
│                           │                                 │
│                           ▼                                 │
│                     A2_Association ◄─── IT_Food/Danger     │
│                           ▲                                 │
│                           │                                 │
│  Sound_Food_L/R ────► A1_Food ────┬──► IT_Food             │
│                           │       │                         │
│                           │       └──► Hunger Drive        │
│                           │                                 │
│                           ▼                                 │
│                      Motor L/R                              │
│                                                             │
│              Top-Down: Fear/Hunger → A1                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 환경 수정 (ForagerGym)

### 7.1 Config 추가
```python
# Sound settings
sound_enabled: bool = True
danger_sound_range: float = 80.0    # Pain Zone 소리 범위
food_sound_range: float = 60.0      # 음식 소리 범위
sound_decay: float = 2.0            # 거리에 따른 감쇠
```

### 7.2 Observation 확장
```python
def _get_observation(self):
    ...
    # Sound inputs
    sound_danger_l, sound_danger_r = self._compute_danger_sound()
    sound_food_l, sound_food_r = self._compute_food_sound()

    obs["sound_danger_left"] = sound_danger_l
    obs["sound_danger_right"] = sound_danger_r
    obs["sound_food_left"] = sound_food_l
    obs["sound_food_right"] = sound_food_r
```

### 7.3 소리 계산
```python
def _compute_danger_sound(self):
    """Pain Zone에서 발생하는 위험 소리 계산"""
    # Pain Zone 중심까지 거리
    dx = self.pain_zone_center[0] - self.agent_pos[0]
    dy = self.pain_zone_center[1] - self.agent_pos[1]
    dist = np.sqrt(dx**2 + dy**2)

    if dist > self.config.danger_sound_range:
        return 0.0, 0.0

    # 거리에 따른 강도
    intensity = 1.0 - (dist / self.config.danger_sound_range) ** self.config.sound_decay

    # 좌우 분리 (에이전트 방향 기준)
    angle_to_danger = np.arctan2(dy, dx) - self.agent_angle
    left = max(0, intensity * (1 + np.sin(angle_to_danger)))
    right = max(0, intensity * (1 - np.sin(angle_to_danger)))

    return left, right
```

---

## 8. 성공 기준

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| A1_Danger 활성화 | Pain Zone 근처에서 > 0.3 | 스파이크율 |
| 청각-공포 반응 | 소리 → Fear 증가 | LA 활성화 |
| 청각 유도 회피 | 위험 소리 → 반대 방향 | 모터 출력 |
| 성능 유지 | 생존율 > 70% | 기존 대비 |

---

## 9. 구현 순서

1. **ForagerGym 수정**: 소리 감각 입력 추가
2. **뉴런 생성**: Sound Input, A1, A2
3. **순방향 연결**: Sound → A1 → Amygdala/IT/Motor
4. **Top-Down 연결**: Fear/Hunger → A1
5. **스파이크 추적 및 시각화**
6. **테스트**: 10 에피소드

---

## 10. 예상 문제점

| 문제 | 해결책 |
|------|--------|
| 소리가 너무 강함 | 감쇠 계수 조정, 입력 정규화 |
| 시각-청각 간섭 | A2에서 WTA 경쟁 추가 |
| 학습 불안정 | Hebbian 학습률 낮게 설정 |
| 방향 정보 부정확 | 좌우 lateral inhibition 강화 |

---

*작성: 2025-02-01*
