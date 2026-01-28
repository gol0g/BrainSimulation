# Phase 2b: 편도체 - 공포의 탄생

> **Genesis Brain Project - Phase 2b: The Amygdala**
>
> 목표: 고통 경험을 통해 공포를 학습하는 뇌 구현
> 상태: **설계 중 (In Design)**

---

## 1. 프로젝트 개요

### 1.1 Phase 2b의 위치

```
┌─────────────────────────────────────────────────────────────────┐
│  Genesis Brain 로드맵                                           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: 반사하는 뇌 (Reactive Brain) - ✓ 완료               │
│  └── Slither.io: 뇌간/척수 수준 반사 회로                      │
│                                                                 │
│  Phase 2: 느끼는 뇌 (Affective Brain) - 현재                   │
│  ├── 2a: 시상하부 - 항상성 ✓ 완료                             │
│  ├── 2b: 편도체 - 공포 조건화 ← ★ 이 문서                     │
│  └── 2c: 해마 - 공간 기억                                      │
│                                                                 │
│  Phase 3: 기억하는 뇌 (Mnemonic Brain) - 미래                  │
│  Phase 4: 계획하는 뇌 (Executive Brain) - 미래                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Phase 2b 목표

| 목표 | 설명 | 검증 방법 |
|------|------|----------|
| 고통 감각 구현 | Pain Zone에서 고통 신호 발생 | Pain Sensor 스파이크 측정 |
| 공포 조건화 | 중립 자극 + 고통 → 학습된 공포 | Pain Zone 접근 빈도 감소 |
| Fear vs Hunger 경쟁 | 배고파도 위험 회피, 극도로 배고프면 위험 감수 | 행동 패턴 관찰 |
| 회피 학습 (R-STDP) | 고통 경험 후 해당 영역 회피 학습 | 시냅스 가중치 변화 |

### 1.3 생물학적 근거

```
편도체 (Amygdala)의 역할:
├── 공포 조건화 (Fear Conditioning)
│   └── Pavlov: 중립자극(소리) + 무조건자극(전기충격) → 조건반응(공포)
├── 위협 탐지 및 평가
├── 정서적 기억 형성
└── Fight-or-Flight 반응 조절

Phase 2b에서 구현하는 것:
├── 1. Pain Receptor: 고통 자극 감지
├── 2. Amygdala: 공포 학습 및 표현
└── 3. Fear → Motor 회피 반응

핵심 메커니즘: Pavlovian Fear Conditioning
├── US (Unconditioned Stimulus): 고통 (Pain Zone)
├── CS (Conditioned Stimulus): 위치/색상/거리 (시각 신호)
├── UR (Unconditioned Response): 회피 반사
└── CR (Conditioned Response): 학습된 회피 (예측적)
```

### 1.4 Phase 2a에서 물려받는 것

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2a 유산 (완전 보존)                                      │
├─────────────────────────────────────────────────────────────────┤
│  환경:                                                          │
│  ├── ForagerGym: 400x400 맵, Nest, Food                        │
│  └── Energy 시스템: 감소/증가, 항상성                          │
│                                                                 │
│  뇌:                                                            │
│  ├── Food Eye L/R: 음식 감지                                   │
│  ├── Wall Eye L/R: 벽 감지                                     │
│  ├── Motor L/R + WTA: 운동 출력                                │
│  ├── Energy Sensor: 내부 감각                                  │
│  ├── Hunger/Satiety Drive: 동기 시스템                         │
│  └── 모든 Phase 1 반사 회로                                    │
│                                                                 │
│  검증된 행동:                                                   │
│  ├── 음식 추적 (Food Tracking)                                 │
│  ├── 벽 회피 (Wall Avoidance)                                  │
│  ├── 배고픔 → 활동 증가                                        │
│  └── 포만감 → 활동 감소                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 환경 설계: ForagerGym v2 (위험 추가)

### 2.1 환경 변경 사항

```
┌─────────────────────────────────────────────────────────────────┐
│                    ForagerGym v2 (Phase 2b)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  기존 (Phase 2a):                                               │
│  ├── 400 x 400 맵                                              │
│  ├── Nest (중앙, 안전)                                         │
│  ├── Field (음식 존재)                                         │
│  └── Energy 시스템                                              │
│                                                                 │
│  신규 (Phase 2b):                                               │
│  ├── Pain Zone (빨간색 영역)                                   │
│  │   ├── 맵 가장자리 또는 특정 영역                            │
│  │   ├── 진입 시 Pain 신호 + Energy 감소                       │
│  │   └── 시각적으로 구분 (빨간색)                              │
│  │                                                              │
│  ├── Danger Cue (경고 신호)                                    │
│  │   ├── Pain Zone 근처에서 거리 비례 신호                     │
│  │   └── CS (조건 자극) 역할 - 학습 대상                       │
│  │                                                              │
│  └── 음식 배치 변경                                             │
│      └── 일부 음식이 Pain Zone 근처에 배치 (유혹)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

맵 레이아웃 (예시):
┌───────────────────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  ░ = Pain Zone (가장자리)
│ ░                                           ░ │
│ ░     ●                          ●          ░ │  ● = Food
│ ░           ┌─────────────┐                 ░ │
│ ░           │             │                 ░ │  중앙 = Nest (안전)
│ ░     ●     │    NEST     │     ●           ░ │
│ ░           │   (safe)    │                 ░ │
│ ░           └─────────────┘                 ░ │
│ ░                              ●            ░ │
│ ░     ●                                     ░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└───────────────────────────────────────────────┘
```

### 2.2 Pain Zone 상세

```python
# Pain Zone 설정
pain_zone_config = {
    "type": "border",           # 가장자리 띠 형태
    "thickness": 30,            # 30 픽셀 두께
    "pain_intensity": 1.0,      # 고통 강도 (0~1)
    "energy_damage": 0.5,       # 매 스텝 Energy 감소
    "color": (200, 50, 50),     # 빨간색 (시각적 구분)
}

# Danger Cue: Pain Zone까지 거리
def get_danger_signal(agent_pos, pain_zone):
    """
    Pain Zone까지 거리를 위험 신호로 변환

    Returns:
        danger: 0.0 (안전) ~ 1.0 (Pain Zone 내부)
    """
    distance = distance_to_pain_zone(agent_pos)
    if distance <= 0:  # Pain Zone 내부
        return 1.0
    elif distance < 50:  # 접근 중
        return 1.0 - (distance / 50)
    else:
        return 0.0
```

### 2.3 관찰 공간 확장

```python
# Phase 2a 관찰 (보존)
food_rays: np.ndarray      # [n_rays] 음식까지 거리
wall_rays: np.ndarray      # [n_rays] 벽까지 거리
energy: float              # 에너지 (0~1)
in_nest: bool              # 둥지 내 여부

# Phase 2b 신규 관찰
pain_rays: np.ndarray      # [n_rays] Pain Zone까지 거리 ← NEW!
pain_signal: float         # 현재 고통 강도 (0~1) ← NEW!

# pain_rays 설명:
# - 각 방향별 Pain Zone까지 거리
# - Food/Wall rays와 동일한 ray-cast 방식
# - Danger Cue (CS)로 사용됨
```

### 2.4 보상 구조 확장

```python
# Phase 2a 보상 (보존)
reward_food = +1.0           # 음식 섭취
reward_starve = -10.0        # 굶어 죽음
reward_homeostasis = +0.01   # Energy 40-70 유지

# Phase 2b 신규 보상
reward_pain = -0.5           # Pain Zone에서 매 스텝 ← NEW!
reward_escape = +0.1         # Pain Zone 탈출 시 ← NEW!

# 중요: Pain Zone 내 음식 섭취 시
# reward = reward_food + reward_pain = +1.0 - 0.5 = +0.5
# → 위험 감수하고 음식 먹으면 보상 감소 (학습 신호)
```

### 2.5 종료 조건 확장

```python
done = (
    energy <= 0 or           # 굶어 죽음 (Phase 2a)
    steps >= max_steps or    # 시간 초과 (Phase 2a)
    pain_damage >= 100 or    # 고통으로 죽음 ← NEW!
)

# pain_damage: Pain Zone에서 누적된 데미지
# 너무 오래 머물면 사망 (즉사는 아님, 회피 기회 제공)
```

---

## 3. 뇌 아키텍처: 편도체 회로

### 3.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2b BRAIN ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    SENSORY LAYER (확장)                          │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │  │
│  │  │Food Eye│ │Wall Eye│ │Pain Eye│ │Danger  │ │Energy  │         │  │
│  │  │  L/R   │ │  L/R   │ │  L/R   │ │ Sensor │ │ Sensor │         │  │
│  │  └────┬───┘ └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘         │  │
│  │       │          │         │          │          │               │  │
│  └───────┼──────────┼─────────┼──────────┼──────────┼───────────────┘  │
│          │          │         │          │          │                   │
│          ▼          ▼         ▼          ▼          ▼                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    LIMBIC LAYER (Phase 2a+2b)                    │  │
│  │                                                                   │  │
│  │  ┌─────────────────────┐       ┌─────────────────────┐          │  │
│  │  │    HYPOTHALAMUS     │       │      AMYGDALA       │          │  │
│  │  │    (Phase 2a)       │       │     (Phase 2b)      │          │  │
│  │  │  ┌───────┐ ┌─────┐  │       │  ┌────────────────┐ │          │  │
│  │  │  │Hunger │ │Sati-│  │       │  │  Fear Circuit  │ │          │  │
│  │  │  │ Drive │←│ety  │  │       │  │                │ │          │  │
│  │  │  └───┬───┘ └──┬──┘  │       │  │ ┌────┐ ┌────┐ │ │          │  │
│  │  │      │        │     │       │  │ │LA  │→│CEA │ │ │          │  │
│  │  │      │        │     │       │  │ │    │ │    │ │ │          │  │
│  │  │      │        │     │       │  │ └──┬─┘ └──┬─┘ │ │          │  │
│  │  │      │        │     │       │  │    │      │   │ │          │  │
│  │  │      │        │     │  경쟁 │  │    │      │   │ │          │  │
│  │  │      │        │     │◄─────►│  │    ▼      ▼   │ │          │  │
│  │  │      │        │     │       │  │  ┌──────────┐ │ │          │  │
│  │  │      │        │     │       │  │  │Fear Resp.│ │ │          │  │
│  │  │      │        │     │       │  │  └────┬─────┘ │ │          │  │
│  │  └──────┼────────┼─────┘       │  └───────┼───────┘ │          │  │
│  │         │        │             │          │         │          │  │
│  └─────────┼────────┼─────────────┼──────────┼─────────┼──────────┘  │
│            │        │             │          │         │              │
│            ▼        ▼             ▼          ▼         ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      MOTOR LAYER (통합)                          │  │
│  │                                                                   │  │
│  │        ┌─────────────────────────────────────────────┐           │  │
│  │        │              Motor L / R (WTA)               │           │  │
│  │        │                                              │           │  │
│  │        │  입력:                                       │           │  │
│  │        │  - Food 동측 배선 (+)                        │           │  │
│  │        │  - Wall Push-Pull (±)                        │           │  │
│  │        │  - Hunger 조절 (+Food)                       │           │  │
│  │        │  - Satiety 억제 (-)                          │           │  │
│  │        │  - Fear 회피 (+Push, -Pull) ← NEW!          │           │  │
│  │        │                                              │           │  │
│  │        └─────────────────────────────────────────────┘           │  │
│  │                                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 편도체 회로 상세

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AMYGDALA CIRCUIT (Phase 2b)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  입력 (Inputs):                                                         │
│  ┌────────────┐     ┌────────────┐                                     │
│  │ Pain Eye   │     │  Danger    │                                     │
│  │   (US)     │     │  Sensor    │                                     │
│  │ 무조건자극 │     │   (CS)     │                                     │
│  └─────┬──────┘     └─────┬──────┘                                     │
│        │                  │                                             │
│        │ 고정 연결        │ 학습 연결 (R-STDP)                         │
│        │ (Static)         │ (Plastic)                                  │
│        │                  │                                             │
│        ▼                  ▼                                             │
│  ┌────────────────────────────────────────┐                            │
│  │         LA (Lateral Amygdala)          │                            │
│  │              (500 neurons)             │                            │
│  │                                        │                            │
│  │  역할: 공포 학습의 핵심 영역           │                            │
│  │  - Pain(US) → LA: 무조건 활성화       │                            │
│  │  - Danger(CS) → LA: 학습됨 (R-STDP)   │                            │
│  │                                        │                            │
│  │  Hebbian: "같이 발화하면 같이 연결"   │                            │
│  │  Pain + Danger 동시 발화 → 연결 강화  │                            │
│  └──────────────────┬─────────────────────┘                            │
│                     │                                                   │
│                     │ 고정 연결                                         │
│                     ▼                                                   │
│  ┌────────────────────────────────────────┐                            │
│  │       CEA (Central Amygdala)           │                            │
│  │              (300 neurons)             │                            │
│  │                                        │                            │
│  │  역할: 공포 반응 출력                  │                            │
│  │  - LA 활성화 → CEA 활성화             │                            │
│  │  - CEA → Fear Response                │                            │
│  └──────────────────┬─────────────────────┘                            │
│                     │                                                   │
│                     │ 고정 연결                                         │
│                     ▼                                                   │
│  ┌────────────────────────────────────────┐                            │
│  │         Fear Response Circuit          │                            │
│  │              (200 neurons)             │                            │
│  │                                        │                            │
│  │  역할: 운동 회로에 회피 명령           │                            │
│  │  - Fear → Motor 교차 배선 (회피)      │                            │
│  │  - 좌측 위험 → 우측 회전              │                            │
│  └────────────────────────────────────────┘                            │
│                                                                         │
│  학습 메커니즘 (R-STDP):                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  1. Danger 신호 (CS) 발생                                       │   │
│  │  2. Pain 신호 (US) 발생 (수백 ms 후)                            │   │
│  │  3. LA에서 CS-US 동시 활성화                                    │   │
│  │  4. Pain = 부정적 보상 신호                                      │   │
│  │  5. Danger→LA 시냅스 강화 (다음에 Danger만으로 LA 활성화)       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Hunger vs Fear 경쟁

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HUNGER vs FEAR COMPETITION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  생물학적 근거:                                                         │
│  - 배고픈 동물은 위험을 감수하고 먹이를 찾음                           │
│  - 배부른 동물은 작은 위험에도 회피 반응                                │
│  - "Risk-sensitive foraging theory"                                    │
│                                                                         │
│  구현 메커니즘:                                                         │
│                                                                         │
│  ┌─────────────┐                    ┌─────────────┐                    │
│  │   HUNGER    │◄──── 상호 억제 ────►│    FEAR     │                    │
│  │   Drive     │      (WTA)         │   (CEA)     │                    │
│  └──────┬──────┘                    └──────┬──────┘                    │
│         │                                  │                            │
│         │ 조절                             │ 조절                       │
│         ▼                                  ▼                            │
│  ┌─────────────┐                    ┌─────────────┐                    │
│  │ Food Seeking│                    │Fear Response│                    │
│  │   (Motor)   │                    │   (Motor)   │                    │
│  └─────────────┘                    └─────────────┘                    │
│                                                                         │
│  시나리오:                                                              │
│                                                                         │
│  [높은 Energy, 낮은 위험]                                               │
│    Hunger=0.2, Fear=0.3                                                │
│    → Fear 우세 → 회피 (안전 우선)                                      │
│                                                                         │
│  [낮은 Energy, 낮은 위험]                                               │
│    Hunger=0.8, Fear=0.3                                                │
│    → Hunger 우세 → 음식 추구 (위험 감수)                               │
│                                                                         │
│  [낮은 Energy, 높은 위험]                                               │
│    Hunger=0.8, Fear=0.9                                                │
│    → Fear 우세 (생존 본능) → 회피                                      │
│    → 하지만 Hunger가 Fear를 부분 억제                                  │
│    → 결과: 조심스러운 접근 (완전 회피가 아님)                          │
│                                                                         │
│  시냅스 구현:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Hunger → CEA:  -15.0 (억제)  # 배고프면 공포 감소              │   │
│  │  CEA → Hunger:  -10.0 (억제)  # 공포 시 식욕 감소               │   │
│  │                                                                  │   │
│  │  비대칭 설계: 공포가 더 강함 (생존 우선)                        │   │
│  │  - Fear 억제 가중치 > Hunger 억제 가중치                        │   │
│  │  - 극도의 공포는 극도의 배고픔도 이김                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 시냅스 연결 상세

```python
# === PHASE 2b 신규 시냅스 ===

# 1. PAIN SENSING (무조건 반사)
# Pain Eye → LA (고정, 강함)
syn_pain_to_la:
    type: StaticPulse
    weight: +50.0  # 강한 흥분 (무조건 자극)
    connectivity: Dense (1:1 or 확률적)

# 2. DANGER SENSING (학습 대상)
# Danger Sensor → LA (가소성, R-STDP)
syn_danger_to_la:
    type: RSTDP  # 보상 조절 STDP
    initial_weight: 5.0  # 약한 초기값
    w_max: 40.0
    w_min: 0.0
    tau_eligibility: 2000.0  # 2초 (CS-US 간격)
    eta: 0.02
    # 보상 신호: pain = -1.0 (부정적 경험 = 학습 신호)

# 3. LA → CEA (고정)
# 내부 연결
syn_la_to_cea:
    type: StaticPulse
    weight: +30.0

# 4. CEA → Fear Response (고정)
syn_cea_to_fear_response:
    type: StaticPulse
    weight: +25.0

# 5. FEAR → MOTOR (회피 반사)
# 교차 배선 (Push-Pull, Phase 1 스타일)
syn_fear_to_motor_push:
    type: StaticPulse
    weight: +60.0  # Fear_L → Motor_R (반대편 활성화)

syn_fear_to_motor_pull:
    type: StaticPulse
    weight: -40.0  # Fear_L → Motor_L (같은편 억제)

# 6. HUNGER ↔ FEAR 경쟁
syn_hunger_to_fear:
    type: StaticPulse
    weight: -15.0  # Hunger → CEA 억제

syn_fear_to_hunger:
    type: StaticPulse
    weight: -10.0  # CEA → Hunger 억제
```

### 3.5 뉴런 구성 (Phase 2b)

```python
# === 기존 (Phase 2a 그대로 보존) ===
n_food_eye = 800 (L: 400, R: 400)
n_wall_eye = 400 (L: 200, R: 200)
n_motor = 1000 (L: 500, R: 500)
n_energy_sensor = 200
n_hunger_drive = 500
n_satiety_drive = 500

# === 신규 (Phase 2b) ===
n_pain_eye = 400 (L: 200, R: 200)       # Pain Zone 방향 감지
n_danger_sensor = 200                    # Pain Zone 거리 (근접도)
n_lateral_amygdala = 500                 # LA: 공포 학습
n_central_amygdala = 300                 # CEA: 공포 출력
n_fear_response = 200                    # 회피 반응

# === 총 뉴런 수 ===
# Phase 2a: 3,400
# Phase 2b 추가: 1,600
# 총: ~5,000 뉴런 (여전히 경량)
```

---

## 4. 구현 계획

### 4.1 단계별 구현

```
Step 1: 환경 확장 - Pain Zone 추가 (0.5일)
├── forager_gym.py 수정
├── Pain Zone 구현 (가장자리 띠)
├── pain_rays 레이캐스트 추가
├── pain_signal 계산
├── 보상 구조 확장
└── 테스트: 랜덤 에이전트로 Pain Zone 진입/탈출 확인

Step 2: Pain Sensing 회로 (0.5일)
├── Pain Eye L/R 뉴런 그룹 추가
├── Danger Sensor 뉴런 그룹 추가
├── pain_rays → Pain Eye 입력 인코딩
├── pain_signal → Danger Sensor 입력 인코딩
└── 테스트: Pain Zone 진입 시 스파이크 확인

Step 3: Amygdala 기본 회로 (1일)
├── LA (Lateral Amygdala) 뉴런 그룹
├── CEA (Central Amygdala) 뉴런 그룹
├── Fear Response 뉴런 그룹
├── Pain → LA 고정 시냅스
├── LA → CEA → Fear Response 고정 시냅스
└── 테스트: Pain 신호 → Fear 활성화 확인

Step 4: Fear → Motor 회피 반응 (0.5일)
├── Fear → Motor Push-Pull 시냅스
├── Phase 1 회피 회로와 통합
└── 테스트: Fear 활성화 시 회피 행동 확인

Step 5: Hunger ↔ Fear 경쟁 (0.5일)
├── Hunger → CEA 억제 시냅스
├── CEA → Hunger 억제 시냅스
└── 테스트: Energy 수준에 따른 Fear 반응 차이 확인

Step 6: 공포 조건화 학습 (R-STDP) (1일)
├── Danger → LA R-STDP 시냅스
├── Pain을 부정적 보상 신호로 사용
├── 학습 파라미터 튜닝
└── 테스트: 학습 전/후 회피 행동 변화 확인

Step 7: 통합 검증 (1일)
├── Phase 2 검증 (20 에피소드)
├── Phase 3 검증 (100 에피소드)
├── 지표 수집 및 분석
└── 문서화 (PHASE2B_RESULTS.md)
```

### 4.2 파일 구조

```
backend/genesis/
├── forager_gym.py              # 수정: Pain Zone 추가
├── forager_brain.py            # 수정: Amygdala 추가
├── slither_pygenn_biological.py # Phase 1 (참조용)
└── checkpoints/
    ├── forager_hypothalamus/   # Phase 2a 체크포인트
    └── forager_amygdala/       # NEW: Phase 2b 체크포인트

docs/
├── PHASE2A_DESIGN.md           # Phase 2a 설계
├── PHASE2A_RESULTS.md          # Phase 2a 결과
├── PHASE2B_DESIGN.md           # 이 문서
└── PHASE2B_RESULTS.md          # Phase 2b 완료 후 작성
```

### 4.3 디버깅 로그 및 시각화 (MANDATORY)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  ⚠️  Phase 2a와 동일: 모든 개발 단계에서 디버그 로그와 시각화 필수!       ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

#### 4.3.1 실시간 콘솔 로그 (매 스텝)

```python
# Phase 2b 확장 로그
print(f"[Step {step:4d}] "
      f"Energy={energy:5.1f} | "
      f"Hunger={hunger:.2f} Satiety={satiety:.2f} | "
      f"Pain={pain:.2f} Fear={fear:.2f} | "  # NEW
      f"Motor L={motor_l:.2f} R={motor_r:.2f} | "
      f"Zone={'PAIN!' if in_pain else 'safe'}")  # NEW

# 예시 출력:
# [Step  142] Energy= 35.2 | Hunger=0.78 Satiety=0.12 | Pain=0.00 Fear=0.15 | Motor L=0.65 R=0.31 | Zone=safe
# [Step  200] Energy= 28.5 | Hunger=0.85 Satiety=0.05 | Pain=0.90 Fear=0.72 | Motor L=0.25 R=0.85 | Zone=PAIN!
```

#### 4.3.2 에피소드 요약 로그

```python
print(f"\n{'='*60}")
print(f"Episode {ep} Summary")
print(f"{'='*60}")
print(f"  Steps:           {steps}")
print(f"  Final Energy:    {energy:.1f}")
print(f"  Death Cause:     {death_cause}")
print(f"  Food Eaten:      {total_food}")
print(f"  Pain Zone Visits:{pain_visits}")           # NEW
print(f"  Pain Zone Time:  {pain_time} steps")       # NEW
print(f"  Fear Avg:        {avg_fear:.3f}")          # NEW
print(f"  Hunger vs Fear:  H={avg_hunger:.2f} F={avg_fear:.2f}")  # NEW
print(f"{'='*60}\n")
```

#### 4.3.3 학습 상태 로그 (R-STDP)

```python
# 학습 시냅스 가중치 변화 추적
if step % 100 == 0:
    danger_la_weights = syn_danger_to_la.vars["g"].view
    print(f"  [R-STDP] Danger→LA weights: "
          f"mean={np.mean(danger_la_weights):.3f}, "
          f"max={np.max(danger_la_weights):.3f}, "
          f"min={np.min(danger_la_weights):.3f}")
```

#### 4.3.4 시각화 (Pygame 확장)

```python
# 환경 시각화 확장
render_elements = {
    # Phase 2a 기존
    "map":        "400x400, Nest 표시",
    "agent":      "원형, 방향 화살표",
    "foods":      "노란색 점",
    "energy_bar": "Energy 게이지",
    "hunger_bar": "Hunger 활성도",
    "satiety_bar": "Satiety 활성도",

    # Phase 2b 신규
    "pain_zone":  "빨간색 영역 (가장자리)",  # NEW
    "fear_bar":   "Fear 활성도 (빨간색)",    # NEW
    "danger_indicator": "Pain Zone 접근 경고",  # NEW
    "agent_color_by_fear": "공포 시 에이전트 색상 변경",  # NEW
}
```

#### 4.3.5 이상 상황 감지

```python
def check_anomalies_2b(state):
    """Phase 2b 이상 상황 감지"""
    warnings = []

    # Phase 2a 경고 (그대로 유지)
    if state["hunger_rate"] < 0.1 and state["energy"] < 30:
        warnings.append("⚠️ LOW ENERGY but HUNGER NOT ACTIVE!")

    # Phase 2b 신규 경고
    if state["pain"] > 0.5 and state["fear"] < 0.3:
        warnings.append("⚠️ HIGH PAIN but FEAR NOT ACTIVE!")

    if state["in_pain_zone"] and state["fear"] < 0.5:
        warnings.append("⚠️ IN PAIN ZONE but LOW FEAR - circuit issue?")

    if state["fear"] > 0.7 and state["hunger"] > 0.7:
        warnings.append("⚠️ BOTH FEAR AND HUNGER HIGH - competition test!")

    for w in warnings:
        print(f"\n{'!'*60}\n{w}\n{'!'*60}\n")
```

---

## 5. 성공 기준

### 5.1 Phase 2 검증 (20 에피소드)

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| 생존율 | > 40% | 굶거나 Pain으로 죽지 않은 비율 |
| Fear 활성화 | 관찰됨 | Pain Zone 진입 시 Fear > 0.5 |
| 회피 반응 | 관찰됨 | Pain Zone에서 탈출 시도 (motor delta 변화) |
| Pain Zone 체류 | < 15% | 전체 시간 중 Pain Zone 내 시간 비율 |

### 5.2 Phase 3 검증 (100 에피소드)

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| 평균 생존 시간 | > 800 steps | mean(episode_length) |
| Pain Zone 회피 학습 | 개선됨 | 후반 50 ep의 Pain Zone 방문 횟수 < 전반 50 ep |
| Hunger-Fear 경쟁 | 관찰됨 | 낮은 Energy에서 Pain Zone 근처 음식 접근 시도 |
| R-STDP 가중치 변화 | > 20% | Danger→LA 시냅스 평균 가중치 변화율 |

### 5.3 조기 중단 기준

| 조건 | 조치 |
|------|------|
| 20 에피소드 후 생존율 < 10% | 중단, Pain damage 조정 |
| Fear 활성화 없음 | 중단, Pain → LA 시냅스 검토 |
| 회피 반응 없음 | 중단, Fear → Motor 시냅스 검토 |
| Hunger-Fear 경쟁 없음 | 중단, 억제 시냅스 가중치 조정 |

---

## 6. 위험 분석 및 대응

### 6.1 예상 위험

| 위험 | 확률 | 영향 | 대응 |
|------|------|------|------|
| Pain이 너무 강함 | 중 | 에이전트가 못 움직임 | pain_intensity 조정 |
| Fear가 Hunger를 완전 억제 | 중 | 음식 못 먹고 굶어 죽음 | 억제 가중치 비대칭 조정 |
| R-STDP 학습 안 됨 | 높 | 공포 조건화 실패 | eligibility tau 연장, 보상 강도 조정 |
| Pain Zone이 너무 좁음 | 낮 | 학습 기회 부족 | Pain Zone 크기 확대 |

### 6.2 R-STDP 학습 조건 검증

```
Phase 2b R-STDP 조건 체크:

| 조건 | 요구사항 | Phase 2b 상황 |
|------|----------|---------------|
| 보상 빈도 | > 1% | Pain Zone 진입 빈도 ~5-10% (예상) ✓ |
| 시간 지연 | < τ | Danger 신호 → Pain: < 2초 ✓ |
| 인과관계 | 명확 | Danger(거리) → Pain(진입) 명확 ✓ |

핵심: Pain을 "부정적 보상"으로 사용
- pain_signal > 0.5 → reward = -1.0
- Danger → LA 시냅스가 강화됨
- 다음에 Danger만으로 Fear 활성화
```

### 6.3 대응 전략

```
문제 발생 시 조정 순서:
1. 환경 파라미터 (Pain Zone 크기, 강도)
2. 시냅스 가중치 (Fear → Motor, Hunger ↔ Fear)
3. 학습 파라미터 (R-STDP τ, η)
4. 뉴런 수 (LA, CEA 크기)
5. 회로 구조 (마지막 수단)
```

---

## 7. 코드 스켈레톤

### 7.1 ForagerGym v2 확장

```python
# forager_gym.py 수정 부분

@dataclass
class ForagerConfig:
    # ... (Phase 2a 설정 그대로)

    # Phase 2b 신규
    pain_zone_thickness: float = 30.0    # Pain Zone 두께
    pain_intensity: float = 1.0          # 고통 강도
    pain_damage: float = 0.5             # Energy 감소량
    pain_max_damage: float = 100.0       # 누적 데미지 한계


class ForagerGym:
    def __init__(self, ...):
        # ... (Phase 2a 초기화)
        self.pain_damage_accumulated = 0.0  # NEW

    def reset(self):
        # ... (Phase 2a 리셋)
        self.pain_damage_accumulated = 0.0  # NEW

    def step(self, action):
        # ... (Phase 2a 이동, 에너지 감소)

        # Phase 2b: Pain Zone 처리
        in_pain = self._in_pain_zone()
        pain_signal = 0.0

        if in_pain:
            pain_signal = self.config.pain_intensity
            self.energy -= self.config.pain_damage
            self.pain_damage_accumulated += self.config.pain_damage
            reward += self.config.reward_pain

        # ... (Phase 2a 음식 섭취, 항상성 보상)

        # 종료 조건 확장
        if self.pain_damage_accumulated >= self.config.pain_max_damage:
            done = True
            death_cause = "pain"

        # 관찰에 Pain 정보 추가
        info["in_pain"] = in_pain
        info["pain_signal"] = pain_signal
        info["pain_damage"] = self.pain_damage_accumulated

        return observation, reward, done, info

    def _in_pain_zone(self) -> bool:
        """Pain Zone (가장자리) 내부인지 확인"""
        t = self.config.pain_zone_thickness
        return (self.agent_x < t or
                self.agent_x > self.config.width - t or
                self.agent_y < t or
                self.agent_y > self.config.height - t)

    def _get_observation(self) -> dict:
        obs = super()._get_observation()  # Phase 2a 관찰

        # Phase 2b 신규
        obs["pain_rays"] = self._cast_pain_rays()
        obs["pain_signal"] = self._get_pain_signal()

        return obs

    def _cast_pain_rays(self) -> np.ndarray:
        """Pain Zone 방향 레이캐스트"""
        rays = np.zeros(self.config.n_rays)
        # 각 방향별로 Pain Zone까지 거리 계산
        for i in range(self.config.n_rays):
            angle = self.agent_angle + (i / self.config.n_rays - 0.5) * np.pi
            dist = self._ray_to_pain_zone(angle)
            rays[i] = 1.0 - min(dist / self.config.view_range, 1.0)
        return rays

    def _get_pain_signal(self) -> float:
        """Pain Zone까지 거리 기반 위험 신호"""
        dist = self._distance_to_pain_zone()
        if dist <= 0:
            return 1.0
        elif dist < 50:
            return 1.0 - (dist / 50)
        else:
            return 0.0
```

### 7.2 ForagerBrain 확장 (Amygdala)

```python
# forager_brain.py 수정 부분

class ForagerBrain:
    def __init__(self, ...):
        # ... (Phase 2a 초기화)

        # Phase 2b 신규
        self._build_amygdala()
        self._build_fear_motor()
        self._build_hunger_fear_competition()

    def _build_amygdala(self):
        """편도체 회로 구축"""

        # 1. Pain Eye (L/R)
        self.pain_eye_left = self.model.add_neuron_population(
            "pain_eye_left", 200, sensory_lif_model, ...
        )
        self.pain_eye_right = self.model.add_neuron_population(
            "pain_eye_right", 200, sensory_lif_model, ...
        )

        # 2. Danger Sensor
        self.danger_sensor = self.model.add_neuron_population(
            "danger_sensor", 200, sensory_lif_model, ...
        )

        # 3. Lateral Amygdala (LA)
        self.lateral_amygdala = self.model.add_neuron_population(
            "lateral_amygdala", 500, "LIF", lif_params, lif_init
        )

        # 4. Central Amygdala (CEA)
        self.central_amygdala = self.model.add_neuron_population(
            "central_amygdala", 300, "LIF", lif_params, lif_init
        )

        # 5. Fear Response
        self.fear_response = self.model.add_neuron_population(
            "fear_response", 200, "LIF", lif_params, lif_init
        )

        # === 시냅스 ===

        # Pain → LA (고정, 무조건 반사)
        self.syn_pain_la = self._create_static_synapse(
            "pain_la",
            [self.pain_eye_left, self.pain_eye_right],
            self.lateral_amygdala,
            weight=50.0
        )

        # Danger → LA (R-STDP, 학습!)
        self.syn_danger_la = self._create_rstdp_synapse(
            "danger_la",
            self.danger_sensor,
            self.lateral_amygdala,
            initial_weight=5.0,
            w_max=40.0,
            tau_eligibility=2000.0,
            eta=0.02
        )

        # LA → CEA (고정)
        self.syn_la_cea = self._create_static_synapse(
            "la_cea", self.lateral_amygdala, self.central_amygdala,
            weight=30.0
        )

        # CEA → Fear Response (고정)
        self.syn_cea_fear = self._create_static_synapse(
            "cea_fear", self.central_amygdala, self.fear_response,
            weight=25.0
        )

    def _build_fear_motor(self):
        """Fear → Motor 회피 반사"""

        # Fear L → Motor R (Push, 반대편 활성화)
        self.syn_fear_motor_push_l = self._create_static_synapse(
            "fear_motor_push_l", self.fear_response, self.motor_right,
            weight=60.0
        )

        # Fear R → Motor L (Push)
        self.syn_fear_motor_push_r = self._create_static_synapse(
            "fear_motor_push_r", self.fear_response, self.motor_left,
            weight=60.0
        )

        # Fear L → Motor L (Pull, 같은편 억제)
        self.syn_fear_motor_pull_l = self._create_static_synapse(
            "fear_motor_pull_l", self.fear_response, self.motor_left,
            weight=-40.0
        )

        # Fear R → Motor R (Pull)
        self.syn_fear_motor_pull_r = self._create_static_synapse(
            "fear_motor_pull_r", self.fear_response, self.motor_right,
            weight=-40.0
        )

    def _build_hunger_fear_competition(self):
        """Hunger ↔ Fear 경쟁 회로"""

        # Hunger → CEA 억제 (배고프면 공포 감소)
        self.syn_hunger_cea = self._create_static_synapse(
            "hunger_cea", self.hunger_drive, self.central_amygdala,
            weight=-15.0
        )

        # CEA → Hunger 억제 (공포 시 식욕 감소)
        self.syn_cea_hunger = self._create_static_synapse(
            "cea_hunger", self.central_amygdala, self.hunger_drive,
            weight=-10.0
        )

    def process(self, observation: dict) -> Tuple[float, dict]:
        # ... (Phase 2a 외부/내부 감각 입력)

        # Phase 2b: Pain 감각 입력
        self._set_pain_input(observation.get("pain_rays", np.zeros(16)))
        self._set_danger_input(observation.get("pain_signal", 0.0))

        # ... (시뮬레이션, 모터 출력)

        # 디버그 정보 확장
        debug["fear"] = self._get_spike_rate(self.fear_response)
        debug["la_activity"] = self._get_spike_rate(self.lateral_amygdala)
        debug["cea_activity"] = self._get_spike_rate(self.central_amygdala)

        return angle_delta, debug

    def apply_reward(self, reward: float, pain_signal: float):
        """보상 신호 적용 (R-STDP 학습)"""

        # Pain을 부정적 보상으로 사용
        if pain_signal > 0.5:
            reward_signal = -1.0  # 강한 부정적 신호
        else:
            reward_signal = reward  # 일반 보상

        # R-STDP 시냅스에 보상 전달
        self.syn_danger_la.vars["reward"].view[:] = reward_signal
        self.syn_danger_la.vars["reward"].push_to_device()
```

---

## 8. 새 세션을 위한 체크리스트

### 8.1 구현 시작 전 확인

```
□ PHASE2B_DESIGN.md (이 문서) 전체 읽기
□ PHASE2A_RESULTS.md 읽기 (Phase 2a 결과)
□ PHASE2A_DESIGN.md 참조 (설계 패턴)
□ forager_gym.py, forager_brain.py 현재 상태 확인
□ CLAUDE.md의 실험 프로세스 확인
```

### 8.2 구현 순서

```
1. forager_gym.py에 Pain Zone 추가
2. Pain Zone 환경 테스트 (랜덤 에이전트)
3. forager_brain.py에 Pain Eye, Danger Sensor 추가
4. Amygdala 기본 회로 (LA, CEA, Fear Response)
5. Fear → Motor 회피 연결
6. Hunger ↔ Fear 경쟁 회로
7. R-STDP 학습 시냅스 (Danger → LA)
8. 통합 테스트
9. Phase 2 검증 (20 에피소드)
10. Phase 3 검증 (100 에피소드)
```

### 8.3 WSL 실행 명령어

```bash
# Phase 2b 테스트
wsl -d Ubuntu-24.04 -e bash -c "
export CUDA_PATH=/usr/local/cuda-12.6
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
python /mnt/c/.../BrainSimulation/backend/genesis/forager_brain.py \
    --episodes 20 --render pygame --log-level debug
"
```

### 8.4 핵심 원칙 (반드시 준수)

```
1. Phase 2a 보존: 기존 회로 수정 금지, 확장만
2. 한 번에 하나씩: 먼저 Pain Zone, 그다음 Amygdala, 마지막에 R-STDP
3. 측정 필수: Pain, Fear, LA, CEA 스파이크율 로깅
4. 경쟁 테스트: Hunger vs Fear 시나리오 명시적 테스트
5. 점진적 검증: 20 ep → 100 ep
6. 빠른 실패: 20 ep 후 Fear 활성화 없으면 즉시 중단
```

---

## 9. 예상 결과

### 9.1 공포 조건화 시나리오

```
Episode 시작 (학습 전):
  에이전트가 Pain Zone에 무심코 진입
  Pain 신호 발생 → LA 활성화 (US 반응)
  Danger 신호도 동시 발생 (CS)
  R-STDP: Danger → LA 시냅스 강화

Episode 중반 (학습 중):
  에이전트가 Pain Zone에 접근
  Danger 신호 발생 (아직 Pain 없음)
  LA가 약간 활성화 (학습된 반응)
  Fear Response 약간 증가
  회피 반응 시작 (완전하지 않음)

Episode 후반 (학습 완료):
  에이전트가 Pain Zone에 접근
  Danger 신호만으로 LA 강하게 활성화!
  Fear Response 강함 → 즉시 회피
  Pain Zone 진입 없이 방향 전환

결과: Pain 경험 없이도 위험 예측하여 회피
```

### 9.2 Hunger vs Fear 시나리오

```
시나리오 1: 높은 Energy, 낮은 위험
  Energy=70%, Pain Zone 근처 음식 발견
  Hunger=0.2 (약함), Fear=0.4 (중간)
  → Fear 우세 → 음식 포기, 안전한 곳으로 이동

시나리오 2: 낮은 Energy, 낮은 위험
  Energy=25%, Pain Zone 근처 음식 발견
  Hunger=0.8 (강함), Fear=0.4 (중간)
  → Hunger가 Fear를 부분 억제
  → Fear=0.2로 감소
  → 음식 방향으로 조심스럽게 접근

시나리오 3: 낮은 Energy, 높은 위험
  Energy=20%, Pain Zone 내부 음식 발견
  Hunger=0.9, Fear=0.8
  → 치열한 경쟁
  → Hunger가 Fear를 억제하지만 완전히는 못함
  → 결과: Pain Zone 가장자리에서 망설임
  → Energy가 더 떨어지면 위험 감수하고 진입
```

---

## 10. 결론

### Phase 2b의 의의

```
Phase 2a: "내부 상태(배고픔)에 따라 행동이 달라지는 뇌"
Phase 2b: "경험(고통)을 통해 공포를 학습하고 위험을 예측하는 뇌"

핵심 진화:
├── 무조건 반응 (US-UR) → 조건 반응 (CS-CR)
├── 현재 자극 반응 → 미래 위험 예측
├── 단일 동기 → 동기 간 경쟁 (Hunger vs Fear)
└── 반사적 회피 → 학습된 회피 (예방적)
```

### 다음 단계 연결

```
Phase 2b 완료 조건:
├── 공포 조건화 성공 (Danger만으로 Fear 활성화)
├── Hunger-Fear 경쟁 관찰
├── Pain Zone 회피 학습
└── 생존율 > 40%

→ Phase 2c (해마):
   "어디서 음식을 찾았는지, 어디가 위험한지 기억하는 뇌"
   공간 기억 추가
```

---

*Genesis Brain Project - Phase 2b Design Document*
*2025-01-28*
