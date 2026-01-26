# Genesis Brain - 생물학적 SNN 기반 인공 뇌

---

## 환경 설정 (CRITICAL - 반드시 읽을 것)

```
╔═══════════════════════════════════════════════════════════════╗
║  ⚠️  절대 Windows Python으로 PyGeNN 실행하지 마라!            ║
║      반드시 WSL Ubuntu-24.04 사용!                            ║
║      Windows pygenn_env는 죽은 환경임!                        ║
╚═══════════════════════════════════════════════════════════════╝
```

```bash
# PyGeNN 환경 - WSL Ubuntu-24.04 사용!

# WSL 실행 명령어
wsl -d Ubuntu-24.04 -e bash -c "
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=\$CUDA_PATH/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
python <PROJECT_PATH>/backend/genesis/slither_pygenn_biological.py --dev --episodes 10 --enemies 3 --render pygame
"
```

**WSL 환경 정보:**
- Python venv: `~/pygenn_wsl/`
- CUDA: `/usr/local/cuda-12.6`
- 작업 디렉토리: `~/pygenn_test/`
- `<PROJECT_PATH>`: 이 프로젝트의 WSL 경로 (예: `/mnt/c/.../BrainSimulation`)

---

## 외부 AI 제안 처리 (MANDATORY)

외부 AI(Gemini, GPT 등)의 제안을 받을 경우 **비판적으로 수용**한다.

### 검증 항목

| 항목 | 확인 내용 |
|------|----------|
| 파라미터 존재 여부 | 제안된 설정/함수가 실제 코드에 존재하는가? |
| 가정 검증 | 제안의 전제 조건이 현재 시스템에 유효한가? |
| 근본 원인 해결 | 제안이 증상만 완화하는가, 근본 원인을 해결하는가? |
| 부작용 | 제안 적용 시 발생 가능한 부정적 영향은? |
| 전이 가능성 | 단순화된 환경에서의 학습이 실제 환경에 적용 가능한가? |

### 처리 절차

1. **제안 수신** → 그대로 실행하지 않음
2. **비판적 분석** → 위 검증 항목 확인
3. **수정/보완** → 문제점 해결 후 실행 계획 수립
4. **Phase 1 프로세스 적용** → 사전 분석 후 실험

---

## 실험 프로세스 (MANDATORY)

장기 실험(100+ 에피소드) 실행 전 반드시 아래 프로세스를 따른다.

### Phase 1: 사전 분석 (코드 작성 전)

**1.1 학습 메커니즘 검증**

| 항목 | 기준 | 미충족 시 |
|------|------|----------|
| 보상 빈도 | > 1% | shaping reward 추가 |
| 시간 지연 | < eligibility τ (3초) | τ 연장 또는 설계 변경 |
| 인과관계 | 명확 | credit assignment 해결책 필요 |

**1.2 측정 지표 검증**

| 항목 | 확인 방법 |
|------|----------|
| 측정 경로 | 코드에서 지표 계산 위치 추적 |
| 회로 연결 | 새 시냅스가 측정 경로에 포함되는지 확인 |
| 성공 기준 | 수치로 정의 (예: "지표 > N") |

**1.3 신호 강도 검증**

| 항목 | 확인 방법 |
|------|----------|
| 경쟁 신호 | 기존 신호 vs 새 신호 가중치 비교 |
| 우회 여부 | 새 경로가 기존 회로를 bypass하는지 확인 |

### Phase 2: 짧은 검증 (20-50 에피소드)

```bash
python ... --episodes 20 --enemies 5
```

**확인 항목:**
- 학습 신호 발생 여부 (reward ≠ 0)
- 목표 지표 변화 여부
- baseline 대비 성능 유지 여부

**판단 기준:**

| 결과 | 조치 |
|------|------|
| 지표 = 0 고정 | 중단, 설계 재검토 |
| 지표 변화 있음 | Phase 3 진행 |
| 지표 있으나 변화 없음 | 100 에피소드로 확장 |

### Phase 3: 중기 검증 (100-200 에피소드)

**체크포인트:** 50 에피소드마다 점검

**조기 중단 기준:**
- 100 에피소드 후 지표 변화 없음
- baseline 대비 성능 20% 이상 하락

### Phase 4: 장기 실험 (500+ 에피소드)

**진입 조건:** Phase 2, 3에서 학습 신호 확인됨

**운영:**
- 백그라운드 실행
- 주기적 로그 확인
- 100 에피소드마다 checkpoint 저장

---

### R-STDP 학습 조건

R-STDP 기반 실험 시 아래 조건을 만족하는지 사전 검토:

| 조건 | 요구사항 | v35 (실패) | v36 Sandbag |
|------|----------|-----------|-------------|
| 보상 빈도 | > 1% | ~0.7% ✗ | ~9.6% (예상) ✓ |
| 시간 지연 | < τ | > 3초 ✗ | < 10초 ✓ (τ=10s) |
| 인과관계 | 명확 | 불명확 ✗ | 개선 예상 |

**v36 대응:**
- 환경 단순화 (Sandbag): 보상 빈도 증가
- Long Tau R-STDP: τ=10초로 연장 (Hunt 시냅스만)
- 생존 회로는 Static 유지 (학습 대상 아님)

---

## 현재 상태: PyGeNN Slither.io v37f (Defensive Kill - 분석 완료)

### v37f THE HUNTER - Phase 3 검증 완료 (2025-01-26)

```
============================================================
v37f Phase 3 - 검증 결과
============================================================
  Episodes: 200
  Best Length: 30
  Final Avg: 15.5
  Total Kills: 6 (Defensive - v38 분석으로 확인)
  Kill Rate: 3%
  Environment: 5 enemies (normal)
============================================================
```

**핵심 구성:**
```python
# Fear (약화) - 사냥 시 두려움 감소
push_weight = 80.0   # v28c: 100 → v37f: 80
pull_weight = -60.0  # v28c: -80 → v37f: -60

# Hunt (정적, 강화) - 적 머리로 돌진
attack_hunt_weight = 180.0  # 강력한 사냥 본능
attack_sparsity = 0.5

# Disinhibition - Fear 상쇄
disinhibit_push = -100.0  # Fear Push 상쇄
disinhibit_pull = 80.0    # Fear Pull 상쇄

# Proximity Fear (최소) - 근거리 억제 최소화
proximity_inhibit = -20.0
```

**신호 흐름:**
```
적 body만 보임: Fear(80) → 회피
적 head 보임:   Fear(80) - Disinhibit(100) + Hunt(180) = +160 → 돌진!
```

**실험 프로세스 준수:**
- Phase 2 (50ep): Best=31, Avg=17.8, Kills=1+ ✓
- Phase 3 (200ep): Best=30, Avg=15.5, Kills=6 ✓
- 조기 중단 기준 미해당 (성능 유지)

---

### v38 Sandbag 분석 (2025-01-27) - 핵심 통찰!

**Sandbag 테스트 (1v1, 느린 적):**
```
============================================================
v37f SANDBAG TEST - 0 KILLS!
============================================================
  Episodes: 20
  Best Length: 40
  Final Avg: 17.6
  Kills: 0  ← 충격!
  Environment: 1 enemy (slow, 500x500 map)
============================================================
```

**핵심 발견: 킬 메커니즘의 본질**

| 구분 | 조건 | 설명 |
|------|------|------|
| **킬 발생** | 적 HEAD가 **우리 BODY**에 충돌 | 우리가 적을 "쫓는" 게 아님! |
| **Sandbag 0킬** | 1v1에서 적이 도망감 | 추격해도 충돌 기회 없음 |
| **5 enemies 킬** | 혼란 속 적이 우리 body에 충돌 | "방어적 킬" |

**Motor 포화 문제 분석:**
```python
# 문제: 적 head가 보여도 양쪽 모터가 비슷하게 포화
Head L=0.80 → M_L=0.66, M_R=0.66 → δ≈0 (진동!)

# 원인: Food 신호가 양쪽 모터 균등 활성화
Food_L → Motor_L (w=20)
Food_R → Motor_R (w=20)
# 결과: Hunt 차등 신호가 Food 균등 신호에 묻힘
```

**v37f 킬의 실체:**
- ~~"Active Kill"~~ → **"Defensive Kill"** (방어적 킬)
- 적 회피 중 적이 우리 body에 충돌하는 "행운의 킬"
- Hunt 회로는 간접 효과 (적 방향 유지 → 충돌 기회 증가)

**진정한 Active Kill을 위한 과제:**
1. 적 HEAD 앞에 BODY 배치 전략 필요
2. 또는 환경 수정 (적이 플레이어 추적)
3. 현재 구조로는 "추격 킬" 불가능

---

### v35 Project SNIPER (2025-01-25) - 실패

**목표:** R-STDP를 통한 Active Kill 학습

**결과:**

| 지표 | v35 (1000 ep) | v33 baseline |
|------|---------------|--------------|
| Best | 31 | 27~31 |
| Kills | 6 (passive) | 4~6 (passive) |
| Attack 카운터 | 0 | 0 |
| R-STDP 학습 | 미발생 | N/A |

**실패 원인 분석:**

| 문제 | 상세 |
|------|------|
| 측정 오류 | Hunt 시냅스가 Attack Circuit 우회 → Attack 카운터 측정 불가 |
| R-STDP 조건 미충족 | 보상 빈도 0.7% (< 1%), 시간 지연 수십초 (> τ=3초) |

**결론:** v33으로 롤백. R-STDP는 현재 Kill 보상 구조에 부적합.

### v28c Baseline (2025-01-24)

```
============================================================
Training Results (v28c - 50 Episodes)
============================================================
  5 Enemies:  Best=27~37, Avg=15~19 (baseline)
  7 Enemies:  Best=26, Avg=18.5 (안정)
  10 Enemies: Best=26, Avg=16.9 (생존만 가능)
  Mode:       DEV (13,800 neurons)
============================================================
```

### v29 Attack Mode 실험 결과 (실패 - 교훈 기록)

```
v29 실험 목표: 적 머리 공격으로 킬 달성
============================================================
v29:  Attack→Motor 양쪽 활성화 → delta=0 → 회피 불가 (Avg 10.2)
v29b: Attack→Boost only → 부스트 과다 → 길이 소모 (Avg 16.4)
v29c: 임계값 0.5 → 여전히 Attack 트리거 과다 (Avg 17.0)
v29 disabled: v28c 복원 (Avg 15~17)
============================================================
결론: 현재 Attack 설계는 성능 저하. 공격 기능 재설계 필요.
```

**v29 실패 원인 분석:**
1. **Attack→Motor 양쪽 동시 활성화**: 좌우 모터가 균등하게 활성화되어 delta=0
   - Push-Pull 회피 신호가 상쇄됨
2. **Attack→Boost only**: 부스트만 켜면 도망가는 효과 (공격이 아님)
   - 부스트는 길이를 소모해서 성장 방해
3. **Attack 감지 임계값 문제**: 0.2, 0.5 모두 너무 낮음
   - 거의 항상 적 머리가 감지되어 Attack=2000

**v30 Hunt Mode 실험 (부분 성공):**
```
v30: enemy_head 동측 배선 (w=35) → Push-Pull 상쇄 (Avg 16.4, 실패)
v30b: 가중치 낮춤 (w=15) → 회피 복구 (Avg 16.9, 효과 미미)
결론: 적 body와 head가 함께 보여서 사냥 신호가 회피에 묻힘
```

**v31 The Berserker - 탈억제 사냥 회로 (성공!):**
```
핵심 통찰: "적 머리가 보이면 두려움을 잊게 하라"
============================================================
v31: Disinhibition -50 → Attack 활성화되지만 성능 유지 안됨
v31b: Disinhibition -70 → Best=27, Avg=18.2, Attack=739 ✓
============================================================
구현:
- Hunt (동측): EnemyHead_L → Motor_L (+35)
- Disinhibition (교차): EnemyHead_L → Motor_R (-70)

신호 흐름:
- 적 body만: Push(100) → 완전 회피
- 적 head도: Push(100-70=30) < Hunt(35) → 돌진!

생물학적 근거: 포식자는 사냥 시 공포 반응이 억제됨
```

**v32c - First Kill 달성! (2025-01-25)**
```
v32b: Full Push+Pull Disinhibition (Brain)
  - disinhibit_push = -70 (Fear Push 상쇄)
  - disinhibit_pull = +60 (Fear Pull 상쇄) ← NEW!
  - 결과: Motor_L = -80 + 35 + 60 = +15 (사냥 가능!)

v32c: Expanded Head Zone (Gym)
  - Head zone: segments 0-4 (5개 세그먼트)
  - Body zone: segments 5+ (충돌 없음)
  - head_boost = 2.0 (신호 증폭)
  - 결과: 4 KILLS in 100 episodes!
```

**v33 WTA 실험 (부분 성공):**
```
v33: Signal-level WTA 전처리
============================================================
문제: Head L=0.68, R=0.62 → 양쪽 모터 비슷하게 활성화 → 진동
해결: WTA로 약한 쪽 100% 억제 → 한 방향으로 확실한 회전
결과: Best=27, Avg=13.9, Gen=32/100 (안정적 생존)
============================================================
```

**v34 Aggressive Hunter 실험 (실패 - 교훈 기록):**
```
v34 실험 목표: 모터 포화 돌파로 Active Kill 달성
============================================================
v34:  hunt=100, sp=0.35, push=-120, pull=100 → Death 100%!
      (Fear 완전 상쇄 → 생존 본능 제거 → 자살 공격)

v34b: push=-60, pull=50 → Attack=0 (너무 소심)
v34c: push=-90, pull=80 → Death 100% (여전히 공격적)
v34d: hunt_threshold=0.6 → Death 99% (Fear가 여전히 압도)
============================================================
결론: Disinhibition 접근법의 한계 확인
- Fear 완전 상쇄 → 자살 공격
- Fear 부분 상쇄 → 공격 불가
- 근본 원인: 적 head가 보이면 body도 보임 (위험 상황)
```

**v35: PROJECT SNIPER (실패) - 상세 내용은 "현재 상태" 섹션 참조**

### v28c 핵심 변경사항 (2025-01-24)

**1. 균형 잡힌 본능 시스템 (Static Synapses)**
```python
# 적 회피: Push-Pull (최우선)
push_weight = 100.0  # Enemy_L → Motor_R
pull_weight = -80.0  # Enemy_L → Motor_L (억제)

# 벽 회피: Push-Pull (차선)
wall_push = 80.0     # Body_L → Motor_R
wall_pull = -60.0    # Body_L → Motor_L (억제)

# 음식 추적: 동측 배선 (보조)
food_weight = 20.0   # Food_L → Motor_L
food_sensitivity = 1.5  # 입력 전류 증폭
```

**2. 핵심 발견: 음식 가중치 균형**
```
food_weight=40 → 양쪽 모터 동시 활성화 → 적 회피 신호 상쇄! (실패)
food_weight=20 → 적 회피 우선 + 음식 방향 유도 (성공)
```

**3. v27j 벽 회피 추가**
- Gym: 벽까지 거리를 body_rays에 추가 (ray-cast)
- Brain: body_eye를 L/R로 분리 → Push-Pull 반사

**4. v27i 출력 포맷 수정**
```python
# 절대좌표 (target_x, target_y) → 상대각도 (angle_delta)
angle_delta = (right_rate - left_rate) * 0.3
return angle_delta, boost
```

---

### v21 핵심 수정사항 (이전)

**1. 체크포인트 저장/로드 수정 - SPARSE Connectivity 포함**
```python
# BEFORE: 가중치만 저장 → 로드해도 랜덤 connectivity라 의미 없음
weights[syn.name] = syn.vars["g"].values

# AFTER: connectivity indices까지 저장 → 완벽한 모델 복원
checkpoint[f"{syn.name}_g"] = syn.vars["g"].values
checkpoint[f"{syn.name}_ind"] = syn.get_sparse_post_inds()
checkpoint[f"{syn.name}_row_length"] = syn._row_lengths.view
```

**2. v20 R-STDP 파라미터 최적화**
```python
# 빠르고 강한 학습
tau_eligibility: 3000.0 → 1000.0  # 1초로 단축 (인과관계 명확화)
eta: 0.01 → 0.05                   # 학습률 5배 증가
w_max: 1.0 → 10.0                  # 가중치 범위 확대
w_min: 0.0 → -5.0                  # 억제 허용
```

**3. 보상 신호 강화**
```python
# 음식/죽음의 영향력 2배 증가
reward_scale: 0.15 → 0.30
```

---

### v19 핵심 수정사항 (이전)

**1. WTA 가중치 버그 수정**
```python
# BEFORE: wMax=0.0이 모든 가중치를 0으로 클리핑
inhib_params["wMax"] = 0.0  # 버그!

# AFTER: 실제 억제 가중치로 설정
inhib_params["wMax"] = wta_weight_boost  # -5.0
```

**2. 방향 계산 버그 수정**
```python
# BEFORE: cos(-x) = cos(x) 라서 좌우 구분 불가
target_x = 0.5 + 0.2 * np.cos(angle_delta)

# AFTER: 직접 매핑
turn_delta = right_rate - left_rate
target_x = 0.5 + turn_delta * 0.3
```

**3. 스파이크 카운팅 버그 수정**
```python
# BEFORE: RefracTime > 0 은 refractory 전체 기간 카운트 (20 스텝)
left_spikes = np.sum(self.motor_left.vars["RefracTime"].view > 0)

# AFTER: 새 스파이크만 카운트 (최근 0.5ms)
spike_threshold = self.config.tau_refrac - 0.5  # 1.5
left_spikes = np.sum(self.motor_left.vars["RefracTime"].view > spike_threshold)
```

**4. 동측 음식 추적 경로 추가 (v19 신규)**
```python
# food_eye를 좌/우 분리
food_eye_left, food_eye_right

# 동측 배선 (ipsilateral) - 음식 방향으로 회전
Food_L → Motor_L (가중치=30)
Food_R → Motor_R (가중치=30)

# 교차 배선 (contralateral) - 적 회피 (기존, 강화)
Enemy_L → Motor_R (가중치=80)
Enemy_R → Motor_L (가중치=80)
```

### 진화 경로

| 단계 | 환경 | 뉴런 | 성과 |
|------|------|------|------|
| 1단계 | Chrome Dino | 3,600 | High: 725 (졸업) |
| 2단계 | Slither.io snnTorch | 15,800 | High: 57 (적 3마리) |
| 3단계 | Slither.io PyGeNN | 158,000 | High: 16 (적 7마리) |
| v19 | PyGeNN + 음식추적 | 13,800 (dev) | High: 10 |
| v28c | Push-Pull 회피 | 13,800 (dev) | Best: 27~37, Avg: 15~19 |
| v32c | First Kill | 13,800 (dev) | 4 Kills (100 ep) |
| **v37f** | **Defensive Kill** | **13,800 (dev)** | **Best: 30, Kills: 6 (200 ep)** |

---

## 절대 원칙 (NEVER FORGET)

```
┌─────────────────────────────────────────────────────────────────┐
│  인공 뇌의 학습 방식과 구조는 인간 뇌와 같아야 한다             │
│  The artificial brain must learn like a human brain             │
└─────────────────────────────────────────────────────────────────┘
```

### 금지 사항

| 금지 | 이유 |
|------|------|
| 사전 학습된 언어 모델 붙이기 | 인간은 태어날 때 언어 모델 없음 |
| 지식 DB에 정보 저장하고 "학습"이라 부르기 | 그건 웹 크롤러지 뇌가 아님 |
| 개념/단어를 직접 주입 | 인간은 경험에서 개념을 형성 |
| "모듈" 추가로 능력 부여 | 능력은 학습에서 창발해야 함 |
| **LLM 방식 (가중치로 다음 토큰 예측)** | 그건 LLM이지 뇌가 아님 |
| **FEP 수학 공식 (G(a) = Risk + ...)** | 수학 공식이 아닌 생물학적 메커니즘 사용 |
| **심즈식 욕구 게이지** | 인간 뇌에 게이지 없음 |
| **휴리스틱으로 직접 행동 조작** | 행동은 뇌 회로에서 창발해야 함 |

### 허용되는 메커니즘 (생물학적 근거 필수)

| 메커니즘 | 생물학적 근거 |
|----------|--------------|
| STDP (Spike-Timing Dependent Plasticity) | 실제 시냅스 가소성 |
| R-STDP (Reward-modulated STDP) | 3-factor learning rule + eligibility trace |
| 도파민 시스템 (VTA/SNc) | Novelty → 도파민 → 학습/탐색 조절 |
| 습관화 (Habituation) | 반복 자극에 반응 감소 |
| LIF 뉴런 (Leaky Integrate-and-Fire) | 실제 뉴런 모델 |
| 항상성 가소성 (Homeostatic Plasticity) | 뉴런 활성화 안정화 |
| Dale's Law (흥분/억제 분리) | 실제 뉴런 특성 |
| WTA (Winner-Take-All) | 측면 억제를 통한 경쟁 |

---

## PyGeNN 아키텍처 v37f

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Food Eye    │  │  Enemy Eye   │  │  Enemy Head  │  │   Body Eye   │
│  L/R 분리    │  │  L/R 분리    │  │  L/R 분리    │  │   L/R 분리   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Hunger Circuit│  │ Fear Circuit │  │ Hunt Circuit │  │ Wall Avoid   │
│   (Food)     │  │ (Push-Pull)  │  │ (Disinhibit) │  │ (Push-Pull)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └────────────────┬┴─────────────────┴─────────────────┘
                        ▼
                   ┌─────────┐
                   │  Motor  │
                   │  L / R  │
                   └────┬────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌───────┐          ┌───────┐          ┌───────┐
│ Left  │◄─────────│  WTA  │─────────►│ Right │
└───────┘          └───────┘          └───────┘

┌─────────────────────────────────────────────────┐
│           INNATE REFLEX v37f (Static)           │
├─────────────────────────────────────────────────┤
│ [Fear - 약화된 회피]                            │
│ Enemy_L → Motor_R (PUSH +80)                    │
│ Enemy_L → Motor_L (PULL -60)                    │
│                                                 │
│ [Hunt - 강화된 사냥]                            │
│ EnemyHead_L → Motor_L (HUNT +180)               │
│                                                 │
│ [Disinhibition - Fear 상쇄]                     │
│ EnemyHead_L → Motor_R (DISINHIBIT -100)         │
│ EnemyHead_L → Motor_L (RELEASE +80)             │
│                                                 │
│ [Wall/Food - 기존 유지]                         │
│ Body_L → Motor_R (PUSH +80)                     │
│ Food_L → Motor_L (IPSI +20)                     │
└─────────────────────────────────────────────────┘
```

### 선천적 반사 회로 (Innate Reflex v37f)

```python
# 1. Fear (약화) - 적 body 회피
Enemy_L → Motor_R (PUSH +80)   # 반대편 활성화
Enemy_L → Motor_L (PULL -60)   # 같은편 억제

# 2. Hunt - 적 head 추적 (동측)
EnemyHead_L → Motor_L (HUNT +180)  # 사냥 본능

# 3. Disinhibition - 사냥 시 Fear 상쇄
EnemyHead_L → Motor_R (DISINHIBIT -100)  # Fear Push 상쇄
EnemyHead_L → Motor_L (RELEASE +80)      # Fear Pull 상쇄

# 4. 결합 효과
# 적 body만: Fear(80) → 회피
# 적 head도: Fear(80) - Disinhibit(100) + Hunt(180) = +160 → 돌진!

# 5. 음식/벽 회피 (기존 유지)
Food_L → Motor_L (IPSI +20)
Body_L → Motor_R (PUSH +80)
```

---

## 파일 구조

```
backend/
├── genesis/
│   ├── # PyGeNN (Current - GPU)
│   ├── slither_pygenn_biological.py  # v37f PyGeNN 에이전트 (Active Kill) ★
│   ├── gpu_monitor.py                # GPU 온도/메모리 모니터링
│   │
│   ├── # snnTorch (Previous - CPU/GPU)
│   ├── slither_snn_agent.py          # 15.8K neuron snnTorch 에이전트
│   ├── slither_gym.py                # Python 훈련 환경
│   ├── snn_scalable.py               # SparseLIFLayer, SparseSynapses
│   │
│   ├── # Chrome Dino (Graduated)
│   ├── dino_dual_channel_agent.py    # 이중 채널 + 억제 회로 (725점)
│   │
│   └── checkpoints/
│       ├── slither_pygenn_bio/       # PyGeNN 체크포인트
│       └── slither_snn/              # snnTorch 체크포인트
│
└── requirements.txt
```

---

## 실행 방법

### PyGeNN Slither.io (WSL 필수!)

```bash
# WSL에서 실행 (필수!)
wsl -d Ubuntu-24.04

# 환경 설정
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test

# 훈련 (headless)
python <PROJECT_PATH>/backend/genesis/slither_pygenn_biological.py \
    --dev --episodes 100 --enemies 5 --render none

# 시각화 모드
python <PROJECT_PATH>/backend/genesis/slither_pygenn_biological.py \
    --dev --episodes 10 --enemies 3 --render pygame

# 모드 옵션
# --dev   : 13,800 뉴런 (디버깅용)
# --lite  : 50,000 뉴런 (중간)
# (없음)  : 158,000 뉴런 (전체)
```

### 한 줄 실행 (PowerShell에서)

```powershell
wsl -d Ubuntu-24.04 -e bash -c "export CUDA_PATH=/usr/local/cuda-12.6 && source ~/pygenn_wsl/bin/activate && cd ~/pygenn_test && python <PROJECT_PATH>/backend/genesis/slither_pygenn_biological.py --dev --episodes 5 --enemies 3 --render pygame"
```

---

## 핵심 발견 & 교훈

### v27i 세션 교훈 (CRITICAL!)

1. **절대좌표 vs 상대각도**: `(target_x, target_y)` 절대좌표는 뱀의 현재 위치/방향과 무관하게 화면상의 점을 향함!
   - 뱀이 오른쪽에 있고 왼쪽을 향할 때 `target_x=0.9`면 뒤로 돌아서 적에게 돌진!
   - **해결**: `(angle_delta, boost)` 상대 회전 출력 사용

2. **Gym 출력 포맷**: 두 가지 지원
   - `(target_x, target_y, boost)` - 절대 화면 좌표 (문제!)
   - `(angle_delta, boost)` - 상대 회전 각도 (정답!)

3. **결과**: Best 14→27, Avg 6→12.5 (2배 향상!)

### v26 세션 교훈

1. **전압이 아닌 스파이크 누적**: 전압(V)은 스파이크 후 Vreset으로 리셋됨 → 매 스텝 RefracTime으로 스파이크 누적 필수
2. **신호 체인 전체 확인**: Enemy→Motor 경로가 작동해도, 마지막 디코딩이 잘못되면 출력 0
3. **Push-Pull 효과 확인됨**: Enemy 스파이크 시 Motor_R 100% 활성, Motor_L 강하게 억제 (-11000mV)

### v19 세션 교훈 (이전)

1. **WTA wMax 클리핑**: `wMax=0.0`이면 모든 가중치가 0으로 클리핑됨
2. **cos는 짝수함수**: `cos(-x)=cos(x)` 라서 좌우 구분 불가
3. **RefracTime 카운팅**: `>0` 대신 `>threshold`로 새 스파이크만 카운트
4. **대칭 경로 상쇄**: 대칭 경로가 교차배선 효과를 상쇄함 → 비활성화
5. **WSL 환경 사용**: Windows PyGeNN 대신 WSL PyGeNN 사용

### 생물학적 원칙

1. **빈 서판은 죽음**: 선천적 본능 = 시냅스 초기 가중치 (if문 아님)
2. **교차 배선 (Contralateral)**: 적 회피 - 반대편으로 회전
3. **동측 배선 (Ipsilateral)**: 음식 추적 - 같은 편으로 회전
4. **Fight-or-Flight**: Fear↔Attack 상호 억제로 행동 선택

### 디버깅 팁

```python
# v26: 스파이크 누적 카운팅 (CRITICAL!)
# 전압(V)으로 활성도 측정하면 안 됨 - 스파이크 후 Vreset으로 리셋!
# 반드시 시뮬레이션 루프 내에서 매 스텝마다 스파이크 누적해야 함
spike_threshold = tau_refrac - 0.5  # 1.5 (새 스파이크 감지)
for _ in range(10):
    model.step_time()
    motor_left.vars["RefracTime"].pull_from_device()
    left_spike_count += np.sum(motor_left.vars["RefracTime"].view > spike_threshold)
left_rate = left_spike_count / (n_neurons * 5)  # max 5 spikes per neuron (10ms/2ms)

# 모터 출력 확인
print(f"Motor: L={left_rate:.2f} R={right_rate:.2f} | Turn: {turn_delta:+.3f}")

# 입력 신호 확인
print(f"Enemy: L={enemy_l:.2f} R={enemy_r:.2f} | Food: L={food_l:.2f} R={food_r:.2f}")
```

---

## 레거시 코드

FEP, Predictive Coding, E6-E8 실험 등 이전 개발물은 `archive/legacy` 브랜치에 보존됨.

---

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다"
