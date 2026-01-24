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

## 현재 상태: PyGeNN Slither.io v28c (Stable Baseline)

### 최신 결과 (2025-01-24)

```
============================================================
Training Results (v28c - 50 Episodes)
============================================================
  5 Enemies:  Best=37, Avg=18.9 (baseline)
  7 Enemies:  Best=26, Avg=18.5 (안정)
  10 Enemies: Best=26, Avg=16.9 (생존만 가능)
  Mode:       DEV (13,800 neurons)
============================================================
BREAKTHROUGH: Avg 5.5 → 18.9 (3.4배 향상!)
============================================================
```

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
| **v19** | **PyGeNN + 음식추적** | **13,800 (dev)** | **High: 10** |

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

## PyGeNN 아키텍처 v19

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Food Eye    │  │  Enemy Eye   │  │   Body Eye   │
│  L/R 분리    │  │  L/R 분리    │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Hunger Circuit│  │ Fear Circuit │  │Attack Circuit│
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │    Fear --| Hunger (억제)         │
       │    Fear <-> Attack (상호 억제)    │
       └────────────┬──────────────────────┘
                    ▼
         ┌────────────────────┐
         │   Integration 1    │
         └─────────┬──────────┘
                   ▼
         ┌────────────────────┐
         │   Integration 2    │
         └─────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌───────┐     ┌───────┐     ┌───────┐
│ Left  │◄────│  WTA  │────►│ Right │
└───────┘     └───────┘     └───────┘
    ▲              │              ▲
    │              ▼              │
    │         ┌───────┐           │
    │         │ Boost │           │
    │         └───────┘           │
    │                             │
┌───┴─────────────────────────────┴───┐
│       INNATE REFLEX (v27i)          │
├─────────────────────────────────────┤
│ Enemy_L → Motor_R (PUSH +100)       │
│ Enemy_L → Motor_L (PULL -80)        │
│ Enemy_R → Motor_L (PUSH +100)       │
│ Enemy_R → Motor_R (PULL -80)        │
│ Food_L  → Motor_L (동측, w=15)      │
│ Food_R  → Motor_R (동측, w=15)      │
└─────────────────────────────────────┘
```

### 선천적 반사 회로 (Innate Reflex v27i)

```python
# 적 회피: Push-Pull 이중 제어 - 완벽한 회피
# 왼쪽 적 → 오른쪽 활성화 + 왼쪽 억제 = 오른쪽 회전
Enemy_L → Motor_R (PUSH +100)  # 반대편 활성화
Enemy_L → Motor_L (PULL -80)   # 같은편 억제
Enemy_R → Motor_L (PUSH +100)
Enemy_R → Motor_R (PULL -80)

# 음식 추적: 동측 배선 (ipsilateral)
# 왼쪽 음식 → 왼쪽 회전 (음식 방향으로)
Food_L → Motor_L (weight=15)
Food_R → Motor_R (weight=15)

# 적 회피가 음식보다 우선 (100 vs 15)
```

---

## 파일 구조

```
backend/
├── genesis/
│   ├── # PyGeNN (Current - GPU)
│   ├── slither_pygenn_biological.py  # v27i PyGeNN 에이전트 ★
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
