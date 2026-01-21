# Genesis Brain

**생물학적 SNN 기반 인공 뇌 시뮬레이션**

순수 생물학적 메커니즘(LIF 뉴런, R-STDP, 도파민)만으로 학습하는 Spiking Neural Network.

## 핵심 철학

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다."

- **LLM이 아니다**: 다음 토큰 예측 (X)
- **FEP 공식이 아니다**: G(a) = Risk + Ambiguity (X)
- **생물학적 메커니즘**: LIF 뉴런 + R-STDP + 도파민 (O)

## 현재 상태: Slither.io PyGeNN v21

### 진화 경로

| 단계 | 환경 | 뉴런 | 성과 |
|------|------|------|------|
| 1단계 | Chrome Dino | 3,600 | High: 725 (졸업) |
| 2단계 | Slither.io snnTorch | 15,800 | High: 57 (적 3마리) |
| 3단계 | Slither.io PyGeNN | 158,000 | High: 16 (적 7마리) |
| v19 | PyGeNN + 음식추적 | 13,800 (dev) | High: 10 |
| **v21** | **PyGeNN + 체크포인트 수정** | **46,500 (lite)** | **High: 17** |

### v19 아키텍처

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
         │   Integration      │
         └─────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌───────┐     ┌───────┐     ┌───────┐
│ Left  │◄────│  WTA  │────►│ Right │
└───────┘     └───────┘     └───────┘
    ▲                             ▲
    │                             │
┌───┴─────────────────────────────┴───┐
│         INNATE REFLEX (v19)         │
├─────────────────────────────────────┤
│ Enemy_L → Motor_R (교차, w=80)      │
│ Enemy_R → Motor_L (교차, w=80)      │
│ Food_L  → Motor_L (동측, w=30)      │
│ Food_R  → Motor_R (동측, w=30)      │
└─────────────────────────────────────┘
```

## 빠른 시작

### 요구사항

- **WSL2** (Ubuntu 24.04)
- **CUDA 12.x** (WSL 내 설치)
- **Python 3.12+**

### WSL 환경 설정

```bash
# 1. WSL 진입
wsl -d Ubuntu-24.04

# 2. Python venv 생성 (최초 1회)
python3 -m venv ~/pygenn_wsl
source ~/pygenn_wsl/bin/activate

# 3. 패키지 설치
pip install pygenn numpy pygame

# 4. 작업 디렉토리 생성
mkdir -p ~/pygenn_test
```

### Slither.io 실행

```bash
# WSL 진입
wsl -d Ubuntu-24.04

# 환경 설정
export CUDA_PATH=/usr/local/cuda-12.6
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test

# 훈련 (headless)
python /mnt/c/<YOUR_PATH>/BrainSimulation/backend/genesis/slither_pygenn_biological.py \
    --dev --episodes 100 --enemies 5 --render none

# 시각화 모드
python /mnt/c/<YOUR_PATH>/BrainSimulation/backend/genesis/slither_pygenn_biological.py \
    --dev --episodes 10 --enemies 3 --render pygame
```

### 모드 옵션

| 옵션 | 뉴런 수 | 용도 |
|------|---------|------|
| `--dev` | 13,800 | 디버깅, 빠른 테스트 |
| `--lite` | 50,000 | 중간 규모 |
| (없음) | 158,000 | 전체 모델 |

## 핵심 메커니즘

### 1. 선천적 반사 회로 (v19)

```python
# 적 회피: 교차 배선 (contralateral)
# 왼쪽 적 감지 → 오른쪽으로 회전 (도망)
Enemy_L → Motor_R (weight=80)
Enemy_R → Motor_L (weight=80)

# 음식 추적: 동측 배선 (ipsilateral)
# 왼쪽 음식 감지 → 왼쪽으로 회전 (추적)
Food_L → Motor_L (weight=30)
Food_R → Motor_R (weight=30)
```

### 2. R-STDP (Reward-Modulated STDP)

```python
# 2-Trace System:
# 1. stdp_trace (τ=20ms): spike timing 감지
# 2. eligibility (τ=3000ms): 3초 전 행동도 기억

Pre-spike  → stdp_trace -= aMinus
Post-spike → stdp_trace += aPlus
매 스텝   → eligibility += stdp_trace
보상 시   → g += η * dopamine * eligibility
```

### 3. Fight-or-Flight 회로

```python
# Fear ↔ Attack 상호 억제 (편도체 모델)
Fear 강함 → Attack 억제 → 도망
Attack 강함 → Fear 억제 → 공격
```

## 프로젝트 구조

```
backend/genesis/
├── slither_pygenn_biological.py  # v19 PyGeNN 에이전트 ★
├── gpu_monitor.py                # GPU 모니터링
├── slither_snn_agent.py          # snnTorch 에이전트 (이전)
├── slither_gym.py                # 훈련 환경
├── dino_dual_channel_agent.py    # Chrome Dino (졸업)
└── checkpoints/                  # 저장된 모델
```

## 핵심 발견

1. **교차 배선 (Contralateral)**: 적 회피 - 반대편으로 회전
2. **동측 배선 (Ipsilateral)**: 음식 추적 - 같은 편으로 회전
3. **빈 서판은 죽음**: 선천적 본능 = 시냅스 초기 가중치
4. **WTA wMax 주의**: `wMax=0`이면 모든 가중치가 0으로 클리핑됨

## 참고

- PyGeNN: https://genn-team.github.io/
- snnTorch: https://snntorch.readthedocs.io/
- Bi & Poo (1998). Synaptic Modifications in Cultured Hippocampal Neurons
