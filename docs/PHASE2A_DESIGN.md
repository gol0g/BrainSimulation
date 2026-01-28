# Phase 2a: 시상하부 - 항상성의 탄생

> **Genesis Brain Project - Phase 2a: The Hypothalamus**
>
> 목표: 내부 상태(배고픔)에 의해 행동이 조절되는 뇌 구현
> 상태: **설계 완료 (Ready for Implementation)**

---

## 1. 프로젝트 개요

### 1.1 Phase 2a의 위치

```
┌─────────────────────────────────────────────────────────────────┐
│  Genesis Brain 로드맵                                           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: 반사하는 뇌 (Reactive Brain) - ✓ 완료               │
│  └── Slither.io: 뇌간/척수 수준 반사 회로                      │
│                                                                 │
│  Phase 2: 느끼는 뇌 (Affective Brain) - 현재                   │
│  ├── 2a: 시상하부 - 항상성 ← ★ 이 문서                        │
│  ├── 2b: 편도체 - 공포 조건화                                  │
│  └── 2c: 해마 - 공간 기억                                      │
│                                                                 │
│  Phase 3: 기억하는 뇌 (Mnemonic Brain) - 미래                  │
│  Phase 4: 계획하는 뇌 (Executive Brain) - 미래                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Phase 2a 목표

| 목표 | 설명 | 검증 방법 |
|------|------|----------|
| 내부 상태 구현 | Energy 변수로 배고픔 표현 | Energy 값 로깅 |
| 항상성 행동 | Energy 낮으면 먹이 찾기, 높으면 안전 행동 | 행동 패턴 관찰 |
| 동기 조절 | 배고픔이 Fear를 억제 | Fear-Hunger 상호작용 측정 |
| 학습 통합 | 내부 상태가 학습 신호에 영향 | R-STDP 활성화 빈도 |

### 1.3 생물학적 근거

```
시상하부 (Hypothalamus)의 역할:
├── 배고픔/포만감 조절 (Lateral/Ventromedial Hypothalamus)
├── 체온 조절
├── 수분 균형
└── 스트레스 반응 (HPA 축)

Phase 2a에서 구현하는 것:
└── 배고픔-포만감 조절만 (최소 기능)

이유: "하나씩 검증" 원칙 (Slither.io 교훈)
```

---

## 2. 환경 설계: ForagerGym

### 2.1 환경 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                      ForagerGym v1                               │
├─────────────────────────────────────────────────────────────────┤
│  크기: 400 x 400 (단순한 2D 공간)                               │
│  구역:                                                          │
│    - Nest (둥지): 중앙 100x100 영역, 안전 + Energy 소모 감소    │
│    - Field (들판): 나머지 영역, 음식 존재                       │
│                                                                 │
│  에이전트: 단순한 원형 (복잡한 뱀 구조 제거)                    │
│    - 위치: (x, y)                                               │
│    - 방향: angle                                                │
│    - 속도: 고정 (복잡성 제거)                                   │
│                                                                 │
│  내부 상태:                                                      │
│    - Energy: 0~100 (50에서 시작)                                │
│    - 감소: 매 스텝 -0.1 (Field), -0.05 (Nest)                   │
│    - 증가: 음식 섭취 시 +10~20                                  │
│    - 사망: Energy ≤ 0                                           │
│                                                                 │
│  음식:                                                          │
│    - Field에만 생성                                             │
│    - 고정 위치 (처음에는 랜덤 위치 변경 없음)                   │
│    - 섭취 시 즉시 재생성 (다른 위치)                            │
│                                                                 │
│  적/위험: 없음 (Phase 2a에서는 항상성만 검증)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 관찰 공간 (Observation Space)

```python
# 외부 감각 (Exteroception) - Phase 1에서 재사용
food_rays: np.ndarray  # [n_rays] 음식까지 거리 (0~1)
wall_rays: np.ndarray  # [n_rays] 벽까지 거리 (0~1)

# 내부 감각 (Interoception) - Phase 2a 신규!
energy: float          # 현재 에너지 (0~1 정규화)
in_nest: bool          # 둥지 안에 있는지 (0 or 1)

# 총 관찰: food_rays + wall_rays + [energy, in_nest]
```

### 2.3 행동 공간 (Action Space)

```python
# Phase 1과 동일 (단순화)
angle_delta: float  # -1 ~ +1 (회전)
# boost 제거 (복잡성 감소)
```

### 2.4 보상 구조

```python
# 항상성 보상 - 즉각적이고 빈번함! (R-STDP 조건 충족)
reward_food = +1.0      # 음식 섭취 (즉시)
reward_starve = -10.0   # 굶어 죽음 (에피소드 종료)

# 항상성 유지 보상 (Shaping Reward)
# Energy가 적정 범위(40-70)에 있으면 작은 보상
reward_homeostasis = +0.01 if 40 <= energy <= 70 else 0.0

# 학습 신호 검증용 로깅
log_reward_frequency = (n_food_eaten / n_steps) * 100  # 목표: > 5%
```

### 2.5 종료 조건

```python
done = (
    energy <= 0 or           # 굶어 죽음
    steps >= max_steps or    # 시간 초과 (기본 3000)
    # Phase 2b에서 추가: 적에게 죽음
)
```

---

## 3. 뇌 아키텍처

### 3.1 Phase 1 회로 재사용

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 회로 (보존) - 이미 검증됨                              │
├─────────────────────────────────────────────────────────────────┤
│  Push-Pull 반사: 벽 회피                                        │
│    Wall_L → Motor_R (+80), Wall_L → Motor_L (-60)              │
│                                                                 │
│  WTA 회로: 좌우 모터 경쟁                                       │
│    Motor_L ↔ Motor_R (-3)                                       │
│                                                                 │
│  음식 동측 배선: 음식 방향으로 이동                             │
│    Food_L → Motor_L (+20)                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Phase 1 회로 (수정/확장)                                       │
├─────────────────────────────────────────────────────────────────┤
│  Fear 회로 → 적 없으므로 비활성화 (Phase 2b에서 복원)           │
│  Hunt 회로 → 비활성화                                           │
│  Disinhibition → 비활성화                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase 2a 신규 회로

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2a 신규: 시상하부 회로 (Hypothalamus)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Energy Sensor│  ← 내부 감각 (Interoceptive)                 │
│  │   (200 N)    │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │ Hunger Drive │ ←───│ Satiety Drive│  ← 상호 억제            │
│  │   (500 N)    │     │   (500 N)    │                         │
│  └──────┬───────┘     └──────┬───────┘                         │
│         │                    │                                  │
│         │ 배고플 때          │ 배부를 때                        │
│         ▼                    ▼                                  │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │ Food Seeking │     │ Rest/Safety  │                         │
│  │   Boost      │     │   Behavior   │                         │
│  └──────────────┘     └──────────────┘                         │
│                                                                 │
│  핵심 메커니즘:                                                 │
│  1. Energy 낮음 → Hunger 활성화 → Food_Eye 가중치 증폭         │
│  2. Energy 높음 → Satiety 활성화 → Nest 방향 선호              │
│  3. Hunger ↔ Satiety 상호 억제 (WTA)                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 전체 아키텍처

```
                    ┌─────────────────┐
                    │  Energy Sensor  │ ← 내부 감각 (Phase 2a 신규)
                    │   (Intero)      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       ┌─────────────┐              ┌─────────────┐
       │   Hunger    │◄────────────►│  Satiety    │  ← WTA (상호 억제)
       │   Drive     │              │   Drive     │
       └──────┬──────┘              └──────┬──────┘
              │                            │
              │ 조절 (Modulation)          │ 조절
              ▼                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    PHASE 1 CIRCUITS (보존)                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐           │
│  │ Food Eye │      │ Wall Eye │      │(Fear Eye)│ ← 비활성  │
│  │   L/R    │      │   L/R    │      │   L/R    │           │
│  └────┬─────┘      └────┬─────┘      └──────────┘           │
│       │                 │                                    │
│       │ Hunger가        │                                    │
│       │ 증폭 (×1.5)     │                                    │
│       ▼                 ▼                                    │
│  ┌──────────────────────────────────────┐                   │
│  │           Motor L / R                 │                   │
│  │              (WTA)                    │                   │
│  └──────────────────────────────────────┘                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.4 시냅스 연결 상세

```python
# === ENERGY SENSOR ===
# Energy 값을 스파이크 패턴으로 인코딩
# High Energy → High Spike Rate (Satiety)
# Low Energy → Low Spike Rate (Hunger triggers)

# === HUNGER DRIVE 활성화 ===
# Energy가 낮을 때 (< 40) Hunger Drive 활성화
# 구현: Energy Sensor → Hunger (억제 연결)
#       Energy 높으면 Hunger 억제, 낮으면 억제 해제 (Disinhibition)

syn_energy_to_hunger:
    type: StaticPulse (고정)
    weight: -30.0  # 억제
    # Energy 높음 → Hunger 억제
    # Energy 낮음 → 억제 약함 → Hunger 활성화

# === SATIETY DRIVE 활성화 ===
# Energy가 높을 때 (> 60) Satiety Drive 활성화
syn_energy_to_satiety:
    type: StaticPulse (고정)
    weight: +20.0  # 흥분
    # Energy 높음 → Satiety 활성화

# === HUNGER ↔ SATIETY 상호 억제 (WTA) ===
syn_hunger_satiety_mutual:
    type: StaticPulse (고정)
    weight: -15.0
    # 승자 독식: 한 쪽만 활성화

# === HUNGER → FOOD SEEKING 증폭 ===
# 배고프면 음식 신호에 더 민감해짐
syn_hunger_to_food_eye:
    type: StaticPulse (고정)
    weight: +15.0  # Food Eye 활성화 증폭
    # Hunger가 높으면 Food_Eye → Motor 신호 강화

# === SATIETY → REST BEHAVIOR ===
# 배부르면 활동 감소, 둥지 선호
syn_satiety_to_motor:
    type: StaticPulse (고정)
    weight: -10.0  # 모터 출력 억제 (전반적 활동 감소)
```

### 3.5 뉴런 구성 (Phase 2a)

```python
# 기존 (Phase 1에서 재사용)
n_food_eye = 800 (L: 400, R: 400)  # 음식 감지
n_wall_eye = 400 (L: 200, R: 200)  # 벽 감지 (body_eye 재사용)
n_motor = 1000 (L: 500, R: 500)    # 운동 출력

# 신규 (Phase 2a)
n_energy_sensor = 200              # 에너지 상태 인코딩
n_hunger_drive = 500               # 배고픔 동기
n_satiety_drive = 500              # 포만감 동기

# 비활성화 (Phase 2a에서는 불필요)
n_enemy_eye = 0     # 적 없음
n_fear_circuit = 0  # 공포 없음
n_attack_circuit = 0 # 공격 없음

# 총: ~3,400 뉴런 (매우 경량)
```

---

## 4. 구현 계획

### 4.1 단계별 구현

```
Step 1: ForagerGym 환경 구현 (1일)
├── forager_gym.py 새 파일 생성
├── 기본 환경: 400x400, 음식, 에이전트
├── Energy 시스템 구현
├── Nest/Field 구역 구현
└── 테스트: 랜덤 에이전트로 환경 동작 확인

Step 2: 내부 감각 입력 구현 (0.5일)
├── Energy Sensor 뉴런 그룹 추가
├── Energy → Spike Rate 인코딩
└── 테스트: Energy 변화에 따른 스파이크 패턴 확인

Step 3: Hunger/Satiety 드라이브 구현 (1일)
├── Hunger Drive 뉴런 그룹
├── Satiety Drive 뉴런 그룹
├── Energy → Hunger/Satiety 시냅스
├── Hunger ↔ Satiety 상호 억제
└── 테스트: Energy에 따른 드라이브 활성화 확인

Step 4: 조절 회로 연결 (1일)
├── Hunger → Food Eye 증폭 시냅스
├── Satiety → Motor 억제 시냅스
├── Phase 1 회로와 통합
└── 테스트: 배고플 때/배부를 때 행동 차이 확인

Step 5: 학습 및 검증 (2일)
├── 20 에피소드 Phase 2 검증
├── 100 에피소드 Phase 3 검증
├── 지표 수집 및 분석
└── 문서화
```

### 4.2 파일 구조

```
backend/genesis/
├── forager_gym.py              # NEW: Phase 2a 환경
├── forager_brain.py            # NEW: Phase 2a 뇌 (시상하부 포함)
├── slither_pygenn_biological.py # Phase 1 (참조용, 수정 안 함)
├── slither_gym.py              # Phase 1 (참조용, 수정 안 함)
└── checkpoints/
    ├── slither_pygenn_bio/     # Phase 1 체크포인트
    └── forager_hypothalamus/   # NEW: Phase 2a 체크포인트

docs/
├── SLITHER_GRADUATION.md       # Phase 1 보고서
├── PHASE2A_DESIGN.md           # 이 문서
└── PHASE2A_RESULTS.md          # Phase 2a 완료 후 결과 보고서
```

### 4.3 디버깅 로그 및 시각화 (MANDATORY)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  ⚠️  모든 개발 단계에서 디버그 로그와 시각화는 필수!                      ║
║      "측정하지 않으면 개선할 수 없다" - Phase 1 핵심 교훈                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

#### 4.3.1 실시간 콘솔 로그 (매 스텝)

```python
# 필수 로그 항목 - 매 스텝마다 출력
print(f"[Step {step:4d}] "
      f"Energy={energy:5.1f} | "
      f"Hunger={hunger_rate:.2f} Satiety={satiety_rate:.2f} | "
      f"Motor L={motor_l:.2f} R={motor_r:.2f} | "
      f"Food={'ATE!' if food_eaten else '---'}")

# 예시 출력:
# [Step  142] Energy= 35.2 | Hunger=0.78 Satiety=0.12 | Motor L=0.65 R=0.31 | Food=---
# [Step  143] Energy= 35.1 | Hunger=0.82 Satiety=0.08 | Motor L=0.71 R=0.28 | Food=---
# [Step  144] Energy= 50.1 | Hunger=0.45 Satiety=0.41 | Motor L=0.52 R=0.48 | Food=ATE!
```

#### 4.3.2 에피소드 요약 로그

```python
# 에피소드 종료 시 출력
print(f"\n{'='*60}")
print(f"Episode {ep} Summary")
print(f"{'='*60}")
print(f"  Steps:        {steps}")
print(f"  Final Energy: {energy:.1f}")
print(f"  Food Eaten:   {total_food}")
print(f"  Death Cause:  {death_cause}")
print(f"  Reward Freq:  {(total_food/steps)*100:.2f}%")
print(f"  Energy Range: {min_energy:.1f} ~ {max_energy:.1f}")
print(f"  Homeostasis:  {homeostasis_time/steps*100:.1f}% (30-70 range)")
print(f"{'='*60}\n")
```

#### 4.3.3 뉴런 활성화 로그 (디버그 모드)

```python
# --debug 플래그 시 추가 출력
if debug_mode:
    print(f"  [Neurons]")
    print(f"    Energy Sensor: {energy_spikes:3d} spikes")
    print(f"    Hunger Drive:  {hunger_spikes:3d} spikes (rate={hunger_rate:.3f})")
    print(f"    Satiety Drive: {satiety_spikes:3d} spikes (rate={satiety_rate:.3f})")
    print(f"    Food Eye L/R:  {food_l_spikes:3d}/{food_r_spikes:3d}")
    print(f"    Motor L/R:     {motor_l_spikes:3d}/{motor_r_spikes:3d}")
```

#### 4.3.4 시각화 (Pygame 렌더링)

```python
# 환경 시각화 필수 요소
render_elements = {
    # 기본 환경
    "map":        "400x400 영역, Nest는 밝은 색",
    "agent":      "원형, 방향 표시 화살표",
    "foods":      "노란색 점",

    # 상태 바 (화면 상단/하단)
    "energy_bar": "Energy 게이지 (0-100), 색상 변화",
    "hunger_bar": "Hunger 활성도 (빨간색)",
    "satiety_bar": "Satiety 활성도 (파란색)",

    # 뉴런 활성화 시각화 (화면 우측)
    "neuron_panel": {
        "energy_sensor":  "200개 뉴런 격자",
        "hunger_drive":   "500개 뉴런 격자",
        "satiety_drive":  "500개 뉴런 격자",
        "motor_left":     "500개 뉴런 격자",
        "motor_right":    "500개 뉴런 격자",
    },

    # 정보 텍스트
    "info_text": [
        "Episode: N",
        "Step: N",
        "Energy: N.N",
        "Food Eaten: N",
        "Hunger/Satiety: N.NN / N.NN",
    ],
}
```

#### 4.3.5 그래프 시각화 (에피소드 후)

```python
# 에피소드 종료 후 matplotlib 그래프 저장
graphs_to_save = {
    "energy_over_time.png": {
        "x": "steps",
        "y": "energy",
        "annotations": ["food eaten markers", "homeostasis zone (30-70)"],
    },
    "drives_over_time.png": {
        "x": "steps",
        "y": ["hunger_rate", "satiety_rate"],
        "annotations": ["energy overlay (secondary y-axis)"],
    },
    "motor_output.png": {
        "x": "steps",
        "y": ["motor_left", "motor_right", "turn_delta"],
    },
    "trajectory.png": {
        "description": "에이전트 이동 경로, 색상=에너지 레벨",
        "annotations": ["nest boundary", "food locations"],
    },
}

# 저장 위치
save_dir = "checkpoints/forager_hypothalamus/logs/ep_{episode}/"
```

#### 4.3.6 로그 레벨 설정

```python
# 명령줄 인자로 로그 레벨 제어
parser.add_argument("--log-level", choices=["minimal", "normal", "debug", "verbose"],
                   default="normal", help="로그 출력 레벨")

# minimal: 에피소드 요약만
# normal:  에피소드 요약 + 주요 이벤트 (음식 섭취, 위험 상황)
# debug:   + 매 스텝 상태 (10스텝마다)
# verbose: + 매 스텝 뉴런 활성화
```

#### 4.3.7 체크포인트 로깅

```python
# 체크포인트 저장 시 메타데이터 포함
checkpoint = {
    "weights": {...},
    "config": {...},
    "metadata": {
        "episode": ep,
        "best_survival": best_steps,
        "avg_energy": avg_energy,
        "total_food": total_food_eaten,
        "timestamp": datetime.now().isoformat(),
    },
    "history": {
        "energy_history": [...],      # 최근 100 에피소드
        "survival_history": [...],
        "food_history": [...],
    },
}
```

#### 4.3.8 문제 진단 로그

```python
# 이상 상황 자동 감지 및 경고
def check_anomalies(state):
    warnings = []

    if state["hunger_rate"] < 0.1 and state["energy"] < 30:
        warnings.append("⚠️ LOW ENERGY but HUNGER NOT ACTIVE!")

    if state["satiety_rate"] > 0.5 and state["energy"] < 50:
        warnings.append("⚠️ LOW ENERGY but SATIETY ACTIVE!")

    if state["motor_left"] < 0.1 and state["motor_right"] < 0.1:
        warnings.append("⚠️ MOTOR DEAD - no movement!")

    if abs(state["motor_left"] - state["motor_right"]) < 0.05:
        warnings.append("⚠️ MOTOR BALANCED - spinning in place?")

    for w in warnings:
        print(f"\n{'!'*60}\n{w}\n{'!'*60}\n")
```

---

## 5. 성공 기준

### 5.1 Phase 2 검증 (20 에피소드)

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| 생존율 | > 50% | 굶어 죽지 않은 에피소드 비율 |
| 보상 빈도 | > 5% | (음식 섭취 횟수 / 총 스텝) × 100 |
| Energy 변화 | 관찰됨 | Energy 그래프에서 상승/하강 존재 |
| Hunger 활성화 | 관찰됨 | Energy < 40 시 Hunger 스파이크 증가 |

### 5.2 Phase 3 검증 (100 에피소드)

| 지표 | 기준 | 측정 방법 |
|------|------|----------|
| 평균 생존 시간 | > 1000 steps | mean(episode_length) |
| Energy 유지율 | > 60% | 시간 중 Energy 30-70 범위 비율 |
| 항상성 행동 | 관찰됨 | 배고플 때 활동 증가, 배부를 때 감소 |
| 학습 지표 | 변화 있음 | R-STDP 가중치 변화 추적 |

### 5.3 조기 중단 기준

| 조건 | 조치 |
|------|------|
| 20 에피소드 후 생존율 < 10% | 중단, Energy 감소율 조정 |
| Hunger 활성화 없음 | 중단, Energy → Hunger 시냅스 검토 |
| 보상 빈도 < 1% | 중단, 음식 밀도 또는 감지 범위 증가 |

---

## 6. 위험 분석 및 대응

### 6.1 예상 위험

| 위험 | 확률 | 영향 | 대응 |
|------|------|------|------|
| Energy 감소가 너무 빠름 | 중 | 학습 전 사망 | 감소율 조정 (0.1 → 0.05) |
| Hunger가 Food 신호를 압도 | 중 | 방향 감각 상실 | Hunger 가중치 조정 |
| WTA가 Hunger/Satiety 모두 억제 | 낮 | 동기 상실 | 억제 가중치 조정 |
| 환경이 너무 단순 | 낮 | 의미 있는 학습 없음 | Phase 2b로 확장 |

### 6.2 대응 전략

```
문제 발생 시 조정 순서:
1. 환경 파라미터 (Energy 감소율, 음식 밀도)
2. 시냅스 가중치 (Hunger → Food Eye)
3. 뉴런 수 (Hunger/Satiety 크기)
4. 회로 구조 (마지막 수단)
```

---

## 7. 코드 스켈레톤

### 7.1 ForagerGym (환경)

```python
# forager_gym.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ForagerConfig:
    """Phase 2a 환경 설정"""
    # 맵
    width: int = 400
    height: int = 400
    nest_size: int = 100  # 중앙 둥지 크기

    # 에이전트
    agent_speed: float = 2.0
    agent_radius: float = 10.0

    # 음식
    n_food: int = 20
    food_radius: float = 8.0
    food_value: float = 15.0  # Energy 회복량

    # 에너지 (항상성)
    energy_start: float = 50.0
    energy_max: float = 100.0
    energy_decay_field: float = 0.1   # Field에서 감소율
    energy_decay_nest: float = 0.05   # Nest에서 감소율 (절반)

    # 감각
    n_rays: int = 16  # 음식/벽 감지 레이 수
    view_range: float = 150.0  # 시야 거리

    # 보상
    reward_food: float = 1.0
    reward_starve: float = -10.0
    reward_homeostasis: float = 0.01  # Energy 40-70 유지 시

    # 시뮬레이션
    max_steps: int = 3000


class ForagerGym:
    """Phase 2a: 항상성 검증 환경"""

    def __init__(self, config: Optional[ForagerConfig] = None, render_mode: str = "none"):
        self.config = config or ForagerConfig()
        self.render_mode = render_mode

        # 상태
        self.agent_x: float = 0
        self.agent_y: float = 0
        self.agent_angle: float = 0
        self.energy: float = 0
        self.foods: list = []
        self.steps: int = 0

        # 통계
        self.total_food_eaten: int = 0

        self.reset()

    def reset(self) -> dict:
        """환경 초기화"""
        # 에이전트를 중앙(둥지)에 배치
        self.agent_x = self.config.width / 2
        self.agent_y = self.config.height / 2
        self.agent_angle = np.random.uniform(0, 2 * np.pi)
        self.energy = self.config.energy_start

        # 음식 생성 (Field에만)
        self.foods = []
        self._spawn_foods(self.config.n_food)

        self.steps = 0
        self.total_food_eaten = 0

        return self._get_observation()

    def step(self, action: Tuple[float]) -> Tuple[dict, float, bool, dict]:
        """
        한 스텝 실행

        Args:
            action: (angle_delta,) - 회전량

        Returns:
            observation, reward, done, info
        """
        angle_delta = np.clip(action[0], -1, 1) * 0.2  # 최대 회전각

        # 1. 이동
        self.agent_angle += angle_delta
        self.agent_x += np.cos(self.agent_angle) * self.config.agent_speed
        self.agent_y += np.sin(self.agent_angle) * self.config.agent_speed

        # 벽 충돌 처리
        self.agent_x = np.clip(self.agent_x, 0, self.config.width)
        self.agent_y = np.clip(self.agent_y, 0, self.config.height)

        # 2. 에너지 감소
        if self._in_nest():
            self.energy -= self.config.energy_decay_nest
        else:
            self.energy -= self.config.energy_decay_field

        # 3. 음식 섭취
        reward = 0.0
        food_eaten = self._check_food_collision()
        if food_eaten:
            self.energy = min(self.config.energy_max,
                            self.energy + self.config.food_value)
            reward += self.config.reward_food
            self.total_food_eaten += 1

        # 4. 항상성 보상
        if 40 <= self.energy <= 70:
            reward += self.config.reward_homeostasis

        # 5. 종료 조건
        self.steps += 1
        done = False
        death_cause = None

        if self.energy <= 0:
            done = True
            death_cause = "starve"
            reward += self.config.reward_starve
        elif self.steps >= self.config.max_steps:
            done = True
            death_cause = "timeout"

        info = {
            "energy": self.energy,
            "in_nest": self._in_nest(),
            "food_eaten": food_eaten,
            "total_food": self.total_food_eaten,
            "death_cause": death_cause,
        }

        return self._get_observation(), reward, done, info

    def _in_nest(self) -> bool:
        """둥지 안에 있는지 확인"""
        cx, cy = self.config.width / 2, self.config.height / 2
        half = self.config.nest_size / 2
        return (cx - half <= self.agent_x <= cx + half and
                cy - half <= self.agent_y <= cy + half)

    def _get_observation(self) -> dict:
        """관찰 반환"""
        food_rays = self._cast_food_rays()
        wall_rays = self._cast_wall_rays()

        return {
            "food_rays": food_rays,
            "wall_rays": wall_rays,
            "energy": self.energy / self.config.energy_max,  # 정규화
            "in_nest": float(self._in_nest()),
        }

    def _cast_food_rays(self) -> np.ndarray:
        """음식 방향 레이캐스트"""
        # 구현: 각 방향별로 가장 가까운 음식까지 거리
        rays = np.zeros(self.config.n_rays)
        # ... (상세 구현)
        return rays

    def _cast_wall_rays(self) -> np.ndarray:
        """벽 방향 레이캐스트"""
        rays = np.zeros(self.config.n_rays)
        # ... (상세 구현)
        return rays

    def _check_food_collision(self) -> bool:
        """음식 충돌 확인"""
        for i, food in enumerate(self.foods):
            dist = np.sqrt((self.agent_x - food[0])**2 +
                          (self.agent_y - food[1])**2)
            if dist < self.config.agent_radius + self.config.food_radius:
                self.foods.pop(i)
                self._spawn_foods(1)  # 새 음식 생성
                return True
        return False

    def _spawn_foods(self, n: int):
        """Field에 음식 생성"""
        for _ in range(n):
            # Nest 외부에 생성
            while True:
                x = np.random.uniform(0, self.config.width)
                y = np.random.uniform(0, self.config.height)
                # Nest 내부면 다시
                cx, cy = self.config.width / 2, self.config.height / 2
                half = self.config.nest_size / 2
                if not (cx - half <= x <= cx + half and
                       cy - half <= y <= cy + half):
                    break
            self.foods.append((x, y))
```

### 7.2 ForagerBrain (뇌)

```python
# forager_brain.py (스켈레톤)

"""
Phase 2a: Forager Brain with Hypothalamus

Phase 1 회로 재사용:
- Push-Pull 벽 회피
- 음식 동측 배선
- WTA 모터 경쟁

Phase 2a 신규:
- Energy Sensor (내부 감각)
- Hunger Drive
- Satiety Drive
- Hunger → Food Eye 조절
"""

from pygenn import GeNNModel, init_sparse_connectivity, init_weight_update
# ... (Phase 1 코드에서 import)


class ForagerBrain:
    """시상하부를 포함한 생물학적 뇌"""

    def __init__(self, config=None):
        self.config = config or ForagerConfig()

        # GeNN 모델
        self.model = GeNNModel("float", "forager_brain")
        self.model.dt = 1.0

        # === Phase 1 재사용 ===
        self._build_sensory_layer()   # Food Eye, Wall Eye
        self._build_motor_layer()     # Motor L/R, WTA
        self._build_reflex_circuits() # Push-Pull, 음식 동측

        # === Phase 2a 신규 ===
        self._build_hypothalamus()    # Energy Sensor, Hunger, Satiety
        self._build_modulation()      # Hunger → Food Eye 조절

        # 빌드
        self.model.build()
        self.model.load()

    def _build_hypothalamus(self):
        """시상하부 회로 구축"""
        # 1. Energy Sensor
        self.energy_sensor = self.model.add_neuron_population(
            "energy_sensor", 200, sensory_lif_model,
            sensory_params, sensory_init
        )

        # 2. Hunger Drive
        self.hunger_drive = self.model.add_neuron_population(
            "hunger_drive", 500, "LIF", lif_params, lif_init
        )

        # 3. Satiety Drive
        self.satiety_drive = self.model.add_neuron_population(
            "satiety_drive", 500, "LIF", lif_params, lif_init
        )

        # 4. Energy → Hunger (억제: High Energy = Low Hunger)
        self.syn_energy_hunger = self._create_static_synapse(
            "energy_hunger", self.energy_sensor, self.hunger_drive,
            weight=-30.0
        )

        # 5. Energy → Satiety (흥분: High Energy = High Satiety)
        self.syn_energy_satiety = self._create_static_synapse(
            "energy_satiety", self.energy_sensor, self.satiety_drive,
            weight=+20.0
        )

        # 6. Hunger ↔ Satiety WTA
        self.syn_hunger_satiety = self._create_static_synapse(
            "hunger_satiety", self.hunger_drive, self.satiety_drive,
            weight=-15.0
        )
        self.syn_satiety_hunger = self._create_static_synapse(
            "satiety_hunger", self.satiety_drive, self.hunger_drive,
            weight=-15.0
        )

    def _build_modulation(self):
        """조절 회로: Hunger가 Food Eye를 증폭"""
        # Hunger → Food Eye (증폭)
        # 배고프면 음식 신호에 더 민감
        self.syn_hunger_food_eye = self._create_static_synapse(
            "hunger_food_eye", self.hunger_drive, self.food_eye_left,
            weight=+15.0
        )
        # Right도 동일

        # Satiety → Motor (억제)
        # 배부르면 전반적 활동 감소
        self.syn_satiety_motor = self._create_static_synapse(
            "satiety_motor", self.satiety_drive, self.motor_left,
            weight=-10.0
        )
        # Right도 동일

    def process(self, observation: dict) -> Tuple[float, dict]:
        """
        관찰을 받아 행동 출력

        Args:
            observation: {food_rays, wall_rays, energy, in_nest}

        Returns:
            angle_delta, debug_info
        """
        # 1. 외부 감각 입력
        self._set_food_input(observation["food_rays"])
        self._set_wall_input(observation["wall_rays"])

        # 2. 내부 감각 입력 (Phase 2a 신규!)
        self._set_energy_input(observation["energy"])

        # 3. 시뮬레이션
        for _ in range(10):  # 10ms
            self.model.step_time()

        # 4. 모터 출력 디코딩
        left_rate = self._get_spike_rate(self.motor_left)
        right_rate = self._get_spike_rate(self.motor_right)

        angle_delta = (right_rate - left_rate) * 0.3

        # 5. 디버그 정보
        debug = {
            "motor_left": left_rate,
            "motor_right": right_rate,
            "hunger": self._get_spike_rate(self.hunger_drive),
            "satiety": self._get_spike_rate(self.satiety_drive),
        }

        return angle_delta, debug

    def _set_energy_input(self, energy: float):
        """에너지 상태를 Energy Sensor에 입력"""
        # 높은 Energy → 높은 스파이크율
        current = energy * 50.0  # 스케일링
        self.energy_sensor.vars["I_input"].view[:] = current
        self.energy_sensor.vars["I_input"].push_to_device()
```

---

## 8. 새 세션을 위한 체크리스트

### 8.1 구현 시작 전 확인

```
□ 이 문서(PHASE2A_DESIGN.md) 전체 읽기
□ SLITHER_GRADUATION.md 읽기 (Phase 1 교훈)
□ CLAUDE.md의 실험 프로세스 확인
□ slither_pygenn_biological.py 참조 (재사용 코드)
```

### 8.2 구현 순서

```
1. forager_gym.py 생성 (환경)
2. 환경 테스트 (랜덤 에이전트)
3. forager_brain.py 생성 (뇌)
4. Phase 1 회로 이식
5. 시상하부 회로 추가
6. 통합 테스트
7. Phase 2 검증 (20 에피소드)
8. Phase 3 검증 (100 에피소드)
```

### 8.3 WSL 실행 명령어

```bash
# Phase 2a 환경 테스트
wsl -d Ubuntu-24.04 -e bash -c "
export CUDA_PATH=/usr/local/cuda-12.6
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
python /mnt/c/.../BrainSimulation/backend/genesis/forager_brain.py \
    --episodes 20 --render pygame
"
```

### 8.4 핵심 원칙 (반드시 준수)

```
1. 한 번에 하나씩: 시상하부만 구현 (편도체, 해마는 Phase 2b, 2c)
2. Phase 1 보존: 기존 회로 수정 금지, 확장만
3. 측정 필수: Energy, Hunger, Satiety 스파이크율 로깅
4. 점진적 검증: 20 ep → 100 ep → 500 ep
5. 빠른 실패: 20 ep 후 지표 0이면 즉시 중단, 재설계
```

---

## 9. 예상 결과

### 9.1 성공 시나리오

```
Episode 시작:
  Energy = 50 (중간)
  Hunger = 약함, Satiety = 약함
  행동: 랜덤 탐색

Energy 감소 (30 이하):
  Hunger = 강함, Satiety = 억제됨
  Food Eye 증폭 → 음식 방향으로 적극 이동
  행동: 음식 찾기

음식 섭취 후 (Energy 70):
  Hunger = 억제됨, Satiety = 강함
  Motor 억제 → 활동량 감소
  행동: 둥지 근처에서 휴식 (또는 느린 이동)

결과: Energy가 30-70 범위를 유지하며 생존
```

### 9.2 학습 검증

```
R-STDP 학습 조건 (Phase 2a):
  보상 빈도: 음식 섭취 빈도 > 5% (목표 달성 가능)
  시간 지연: 음식 감지 → 섭취 < 3초 (τ=1초 내)
  인과관계: Hunger → Food 탐색 → 섭취 (명확)

기대: Food Eye → Motor 시냅스 강화 (학습 발생)
```

---

## 10. 결론

### Phase 2a의 의의

```
Phase 1 (Slither.io): "세상에 반응하는 뇌"
Phase 2a (Forager): "내부 상태에 따라 반응이 달라지는 뇌"

핵심 진화:
├── 동일 자극 (음식) → 다른 반응 (배고픔에 따라)
├── 내부 상태가 행동을 "조절"
└── 단순 반사 → 동기 유발 행동
```

### 다음 단계 연결

```
Phase 2a 완료 조건:
├── 항상성 행동 관찰됨
├── Hunger/Satiety 드라이브 작동
└── 생존율 > 50%

→ Phase 2b (편도체):
   "빨간 바닥 = 고통" 공포 조건화 추가
   환경에 위험 구역 추가
```

---

*Genesis Brain Project - Phase 2a Design Document*
*2025-01-28*
