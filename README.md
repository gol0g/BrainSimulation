# Genesis Brain

**생물학적 SNN(Spiking Neural Network) 기반 인공 뇌 프로젝트**

PyGeNN + CUDA GPU 가속 스파이킹 뉴럴 네트워크로 구현한 인공 뇌.
LLM이나 딥러닝이 아닌 실제 뇌의 생물학적 메커니즘(STDP, Push-Pull, WTA, Hebbian Learning)으로
에이전트가 환경에서 생존하고 학습한다.

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다."

## 현재 상태

```
Phase 16 완료 | 16,800 뉴런 | 50% 생존율
```

| 지표 | 결과 |
|------|------|
| Survival Rate | 50% |
| Reward Freq | 3.46% |
| Pain Avoidance | 91.0% |

## 뇌 구조 (16 Phase)

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 16: Association Cortex (연합 피질)                     │
│  Phase 15: Social Brain (사회적 뇌 + Mirror + ToM)           │
│  Phase 12-14: Multimodal STS + PPC + PMC (다감각 통합)       │
│  Phase 11: Auditory Cortex (청각 피질)                       │
│  Phase 8-10: V1 → V2/V4 → IT Cortex (시각 경로)             │
│  Phase 6-7: Cerebellum + Thalamus (소뇌 + 시상)             │
│  Phase 5: Prefrontal Cortex (전전두엽)                       │
│  Phase 4: Basal Ganglia + Dopamine (기저핵)                  │
│  Phase 3: Hippocampus (해마 - 공간 기억)                     │
│  Phase 2: Hypothalamus + Amygdala (시상하부 + 편도체)        │
│  Phase 1: Brainstem Reflexes (뇌간 반사)                     │
└─────────────────────────────────────────────────────────────┘
```

> 전체 로드맵: [docs/ROADMAP.md](docs/ROADMAP.md)

## 핵심 원칙

- **생물학적 메커니즘만 사용**: LIF 뉴런, STDP, Hebbian, Push-Pull, WTA, Dale's Law
- **선천적 본능 = 시냅스 초기 가중치**: if문이 아닌 신경 회로로 행동 생성
- **학습은 경험에서 창발**: 사전 학습, 지식 주입, 휴리스틱 금지
- **LLM/딥러닝 금지**: 뇌의 작동 원리로만 구현

## 환경

2D Forager 환경에서 에이전트가 음식을 찾아 먹고, 위험을 회피하며 생존한다.

- **음식**: 시각/청각 단서로 탐지, 에너지 회복
- **Pain Zone**: 접촉 시 에너지 감소, Push-Pull 반사로 회피
- **NPC**: 다른 에이전트와의 사회적 상호작용

## 빠른 시작

### 요구사항

- **WSL2** (Ubuntu 24.04) - Windows Python에서 PyGeNN 실행 불가
- **CUDA 12.x** (WSL 내 설치)
- **Python 3.12+**

### 실행

```bash
# WSL에서 실행
wsl -d Ubuntu-24.04 -- bash -c "
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE
python /mnt/c/<YOUR_PATH>/backend/genesis/forager_brain.py --episodes 20 --render none
"

# 시각화 모드
python forager_brain.py --episodes 3 --render pygame
```

## 프로젝트 구조

```
backend/genesis/
├── forager_brain.py   # 메인 뇌 (16,800 뉴런, Phase 1-16)
├── forager_gym.py     # Forager 환경 (NPC + Pain Zone)
└── checkpoints/       # 학습 체크포인트

docs/
├── ROADMAP.md              # 전체 Phase 로드맵 (Phase 1-20)
├── SLITHER_GRADUATION.md   # Phase 1 (Slither.io) 졸업 보고서
├── PHASE2A_DESIGN.md       # 시상하부 설계
├── PHASE2B_DESIGN.md       # 편도체 설계
└── PHASE3_DESIGN.md        # 해마 설계
```

## 진화 경로

| 단계 | 환경 | 뉴런 | 핵심 성과 |
|------|------|------|----------|
| Phase 1 | Slither.io | 13,800 | Push-Pull 반사, 사냥 본능 (졸업) |
| Phase 2-3 | Forager | 5,800 | 항상성, 공포, 공간 기억 |
| Phase 4-7 | Forager | 8,000 | 도파민, 전전두엽, 소뇌, 시상 |
| Phase 8-10 | Forager | 10,000 | 시각 경로 (V1→IT), 물체 범주화 |
| Phase 11-14 | Forager | 13,500 | 청각, 다감각 통합, 운동 계획 |
| Phase 15-16 | Forager | **16,800** | 사회적 뇌, 연합 피질 |

## 기술 스택

- **SNN Framework**: [PyGeNN](https://genn-team.github.io/) (GPU-accelerated)
- **GPU**: CUDA 12.6
- **뉴런 모델**: LIF (Leaky Integrate-and-Fire)
- **학습 규칙**: Hebbian, STDP, R-STDP
- **환경**: Pygame 기반 2D Forager
- **OS**: WSL Ubuntu-24.04
