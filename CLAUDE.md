# Genesis Brain - 생물학적 SNN 기반 인공 뇌

---

## 환경 설정 (CRITICAL - 반드시 읽을 것)

```bash
# PyGeNN 환경 (GPU SNN 실행용) - 시스템 Python이 아님!
C:\Users\JungHyun\Desktop\brain\pygenn_env\Scripts\python.exe

# 실행 예시 (PowerShell)
& 'C:\Users\JungHyun\Desktop\brain\pygenn_env\Scripts\python.exe' `
    'C:\Users\JungHyun\Desktop\brain\BrainSimulation\backend\genesis\slither_pygenn_biological.py' `
    --episodes 300 --enemies 7 --render none
```

**주의**: `conda activate` 아님! 독립 venv 환경.

---

## 현재 상태: PyGeNN Slither.io (158K neurons)

### 최신 결과 (2025-01-19)

```
============================================================
Training Results (300 Episodes, 7 Enemies)
============================================================
  Best Length: 16
  Final Avg:   7.1
  Time:        1406.6s (4.69s/ep)

  GPU Status:
    Util: 65.6% avg (compute-bound, not memory-bound)
    Temp: 55.2°C avg, 58°C max (안정)
    Memory: 2.2GB / 8GB (27% 사용)
============================================================
```

### 핵심 수정사항 (이번 세션)

**1. Attack Circuit 버그 수정** - `addToPost(g)` 누락
```python
# slither_pygenn_biological.py:57-60
# BEFORE: 시냅스 전류가 전달 안됨 (Attack triggers = 0)
pre_spike_syn_code="""
    stdp_trace -= aMinus;  # ← addToPost(g) 없음!
""",

# AFTER: 전류 전달 추가 (Attack triggers = 0~517)
pre_spike_syn_code="""
    addToPost(g);          # ← 핵심 수정!
    stdp_trace -= aMinus;
""",
```

**2. Enemy→Attack 시냅스 강화**
```python
# sparsity 4배: 0.5% → 2%
# weight 2.5배: 1.0 → 2.5
self.syn_enemy_attack = create_synapse(
    "enemy_attack", self.enemy_eye, self.attack,
    sparsity=self.config.sparsity * 4,  # 2% 연결
    w_init=2.5)  # 공격이 공포보다 강하게
```

### 진화 경로

| 단계 | 환경 | 뉴런 | 성과 |
|------|------|------|------|
| 1단계 | Chrome Dino | 3,600 | High: 725 (졸업) |
| 2단계 | Slither.io snnTorch | 15,800 | High: 57 (적 3마리) |
| **3단계** | **Slither.io PyGeNN** | **158,000** | **High: 16 (적 7마리)** |

### 다음 단계

1. **추가 훈련**: 500+ 에피소드로 R-STDP 학습 안정화
2. **Attack 회로 튜닝**: Fear↔Attack 균형 조절
3. **시각화 분석**: `--render pygame`으로 행동 패턴 관찰

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

## PyGeNN 아키텍처 (158K neurons)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Food Eye   │  │  Enemy Eye   │  │   Body Eye   │
│    (8K)      │  │    (8K)      │  │    (4K)      │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Hunger Circuit│  │ Fear Circuit │  │Attack Circuit│
│    (10K)     │  │    (10K)     │  │    (5K)      │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       │    Fear --| Hunger (억제)         │
       │    Fear <-> Attack (상호 억제)    │
       └────────────┬──────────────────────┘
                    ▼
         ┌────────────────────┐
         │   Integration 1    │
         │       (50K)        │
         └─────────┬──────────┘
                   ▼
         ┌────────────────────┐
         │   Integration 2    │
         │       (50K)        │
         └─────────┬──────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌───────┐  ┌───────┐  ┌───────┐
   │ Left  │  │ Right │  │ Boost │
   │ (5K)  │  │ (5K)  │  │ (3K)  │
   └───────┘  └───────┘  └───────┘
        ↑          ↑          ↑
        └── WTA 측면억제 ──────┘
```

### R-STDP (Reward-Modulated STDP)

```python
# 2-Trace System:
# 1. stdp_trace (τ=20ms): spike timing 감지 (LTP/LTD 결정)
# 2. eligibility (τ=3000ms): 3초 전 행동도 기억

# 동작 원리:
Pre-spike  → stdp_trace -= aMinus (LTD 준비)
Post-spike → stdp_trace += aPlus  (LTP)
매 스텝   → eligibility += stdp_trace (누적)
           → 둘 다 감쇠
보상 시   → g += η * dopamine * eligibility (가중치 업데이트)
```

### Fight-or-Flight 회로

```python
# Fear ↔ Attack 상호 억제 (편도체 모델)
# Fear가 강하면 → Attack 억제 → 도망
# Attack이 강하면 → Fear 억제 → 공격

self.syn_fear_attack = create_synapse(
    w_init=-0.3)   # Fear → Attack 억제
self.syn_attack_fear = create_synapse(
    w_init=-0.7)   # Attack → Fear 억제 (공격 편향)
```

---

## 파일 구조

```
backend/
├── genesis/
│   ├── # PyGeNN (Current - GPU)
│   ├── slither_pygenn_biological.py  # 158K neuron PyGeNN 에이전트 ★
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

### PyGeNN Slither.io (현재 진행중)

```powershell
# 훈련 (headless)
& 'C:\Users\JungHyun\Desktop\brain\pygenn_env\Scripts\python.exe' `
    'C:\Users\JungHyun\Desktop\brain\BrainSimulation\backend\genesis\slither_pygenn_biological.py' `
    --episodes 300 --enemies 7 --render none

# 시각화 모드
& 'C:\Users\JungHyun\Desktop\brain\pygenn_env\Scripts\python.exe' `
    'C:\Users\JungHyun\Desktop\brain\BrainSimulation\backend\genesis\slither_pygenn_biological.py' `
    --episodes 10 --enemies 7 --render pygame

# 경량 모드 (50K neurons)
... --mode lite

# 개발 모드 (15K neurons)
... --mode dev
```

### snnTorch Slither.io (이전 버전)

```bash
cd backend/genesis
python slither_snn_agent.py --enemies 3 --render pygame
```

---

## 핵심 발견 & 교훈

### 이번 세션 교훈

1. **addToPost(g) 필수**: GeNN에서 시냅스 전류 전달은 자동이 아님
2. **GPU 메모리 vs 연산**: SNN은 compute-bound (메모리 27%인데 GPU 65%)
3. **환경 정보 기록**: pygenn_env 경로 등 환경 설정 반드시 문서화

### 생물학적 원칙

1. **빈 서판은 죽음**: 선천적 본능 = 시냅스 초기 가중치 (if문 아님)
2. **Fight-or-Flight**: Fear↔Attack 상호 억제로 행동 선택
3. **3-Factor Learning**: pre + post + reward (도파민)

### 디버깅 팁

```python
# Fear/Attack 스파이크 확인
print(f"Fear: {fear_spikes:.3f}({self.fear.vars['nSpk'].view[0]})")
print(f"Attack: {attack_spikes:.3f}({self.attack.vars['nSpk'].view[0]})")

# Attack triggers = 0이면 → addToPost(g) 확인!
```

---

## 레거시 코드

FEP, Predictive Coding, E6-E8 실험 등 이전 개발물은 `archive/legacy` 브랜치에 보존됨.

---

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다"
