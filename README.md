# Genesis Brain

**생물학적 SNN 기반 인공 뇌 시뮬레이션**

순수 생물학적 메커니즘(LIF 뉴런, R-STDP, 도파민)만으로 학습하는 Spiking Neural Network.

## 핵심 철학

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다."

- **LLM이 아니다**: 다음 토큰 예측 (X)
- **FEP 공식이 아니다**: G(a) = Risk + Ambiguity (X)
- **생물학적 메커니즘**: LIF 뉴런 + R-STDP + 도파민 (O)

## 현재 상태: Slither.io PyGeNN Agent (158K neurons)

### 진화 경로

| 단계 | 환경 | 뉴런 | 성과 |
|------|------|------|------|
| 1단계 | Chrome Dino | 3,600 | High: 725 (졸업) |
| 2단계 | Slither.io snnTorch | 15,800 | High: 57 (적 3마리) |
| **3단계** | **Slither.io PyGeNN** | **158,000** | **High: 16 (적 7마리)** |

### PyGeNN 아키텍처

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
         │   Integration      │
         │      (100K)        │
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

## 빠른 시작

### 요구사항

```bash
# PyGeNN (GPU 가속)
pip install pygenn

# snnTorch (CPU/GPU)
pip install torch snntorch numpy

# 시각화
pip install pygame
```

### Slither.io 실행

```bash
cd backend/genesis

# 훈련 (headless)
python slither_pygenn_biological.py --episodes 300 --enemies 7 --render none

# 시각화 모드
python slither_pygenn_biological.py --episodes 10 --enemies 7 --render pygame
```

### Chrome Dino (졸업)

```bash
cd backend/genesis
python dino_dual_channel_agent.py --eval
```

## 핵심 메커니즘

### 1. R-STDP (Reward-Modulated STDP)

```python
# 2-Trace System:
# 1. stdp_trace (τ=20ms): spike timing 감지
# 2. eligibility (τ=3000ms): 3초 전 행동도 기억

Pre-spike  → stdp_trace -= aMinus
Post-spike → stdp_trace += aPlus
매 스텝   → eligibility += stdp_trace
보상 시   → g += η * dopamine * eligibility
```

### 2. Fight-or-Flight 회로

```python
# Fear ↔ Attack 상호 억제 (편도체 모델)
Fear 강함 → Attack 억제 → 도망
Attack 강함 → Fear 억제 → 공격
```

### 3. 선천적 본능 (Innate Reflex)

```python
# 시냅스 초기 가중치로 구현 (if문 아님)
Enemy LEFT → RIGHT motor (교차 배선)
초기 가중치 3x 부스트 (innate_boost = 3.0)
```

## 프로젝트 구조

```
backend/genesis/
├── # PyGeNN (Current - GPU)
├── slither_pygenn_biological.py  # 158K neuron 에이전트 ★
├── gpu_monitor.py                # GPU 모니터링
│
├── # snnTorch (Previous)
├── slither_snn_agent.py          # 15.8K neuron 에이전트
├── slither_gym.py                # 훈련 환경
├── snn_scalable.py               # SparseLIFLayer
│
├── # Chrome Dino (Graduated)
├── dino_dual_channel_agent.py    # 이중 채널 (725점)
│
└── checkpoints/                  # 저장된 모델
```

## 핵심 발견

1. **addToPost(g) 필수**: GeNN에서 시냅스 전류 전달은 자동이 아님
2. **GPU 메모리 vs 연산**: SNN은 compute-bound (메모리 27%인데 GPU 65%)
3. **빈 서판은 죽음**: 선천적 본능 = 시냅스 초기 가중치
4. **채널 분리**: 감각 채널 분리로 안정적 학습

## 레거시 코드

FEP, Predictive Coding, E6-E8 실험: `archive/legacy` 브랜치

## 참고

- PyGeNN: https://genn-team.github.io/
- snnTorch: https://snntorch.readthedocs.io/
- Bi & Poo (1998). Synaptic Modifications in Cultured Hippocampal Neurons
