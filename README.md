# Genesis Brain

**생물학적 SNN(Spiking Neural Network) 기반 인공 뇌 프로젝트**

PyGeNN + CUDA GPU 가속 스파이킹 뉴럴 네트워크로 구현한 인공 뇌.
LLM이나 딥러닝이 아닌 실제 뇌의 생물학적 메커니즘(STDP, Push-Pull, WTA, Hebbian Learning)으로
에이전트가 환경에서 생존하고 학습한다.

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다."

## 현재 상태

```
27,910 뉴런 | KC 3000 다중감각 통합 | 800×800 맵
9개 감각 모달리티 → KC sparse expansion → BG 의사결정
400ep: avg 61%, peak 74% 생존율
```

| 지표 | 결과 |
|------|------|
| Survival Rate | 61% (peak 74%) |
| Reward Freq | 2.62% |
| Food Selectivity | 0.67 |
| Pain Death | 0% |
| Predator Death | 21% |

## 뇌 구조

```
┌──────────────────────────────────────────────────────────────┐
│  KC Sparse Expansion (3000×2) — 다중감각 패턴 분리            │
│  9 inputs: 시각/피질/연합/공간/사회/언어 → D1/D2 R-STDP      │
├──────────────────────────────────────────────────────────────┤
│  Phase 20: Self-Model (자기 모델)                             │
│  Phase 19: Metacognition (메타인지)                           │
│  Phase 18: Working Memory Expansion (작업 기억)               │
│  Phase 17: Language Circuit (언어 — Broca/Wernicke)           │
│  Phase 16: Association Cortex (연합 피질)                     │
│  Phase 15: Social Brain (사회적 뇌 + Mirror + ToM)            │
│  Phase 12-14: Multimodal STS + PPC + PMC (다감각 통합)        │
│  Phase 11: Auditory Cortex (청각 피질)                        │
│  Phase 8-10: V1 → V2/V4 → IT Cortex (시각 경로)              │
│  Phase 6-7: Cerebellum + Thalamus (소뇌 + 시상)              │
│  Phase 5: Prefrontal Cortex (전전두엽)                        │
│  Phase 4: Basal Ganglia + Dopamine (기저핵)                   │
│  Phase 3: Hippocampus (해마 — 공간 기억)                      │
│  Phase 2: Hypothalamus + Amygdala (시상하부 + 편도체)         │
│  Phase 1: Brainstem Reflexes (뇌간 반사)                      │
└──────────────────────────────────────────────────────────────┘
```

> 전체 로드맵: [docs/ROADMAP.md](docs/ROADMAP.md)

## 학습 시스템 (L1-L18)

| Phase | 핵심 | 메커니즘 |
|-------|------|---------|
| L1-L4 | BG 기반 학습 | R-STDP, Anti-Hebbian, D1/D2 |
| L5-L6 | 피질 학습 | Perceptual R-STDP, Prediction Error |
| L7-L8 | 음식 차등 학습 | Discriminative BG, Dopamine Dip |
| L9-L10 | 인지 학습 | IT→BG Top-Down, TD Learning (RPE) |
| L11-L13 | 기억/주의/조건화 | SWR Replay, Global Workspace, Garcia |
| L14-L15 | 자기 참조 학습 | Agency Detection, Narrative Self |
| L16 | Sparse Expansion | KC 패턴 분리 (초파리 MB 영감) |
| L17-L18 | 다중감각→BG | Social/Auditory/Assoc/PPC → KC |

## 환경 (800×800 ForagerGym)

- **800×800 맵** — 음식 45개, Rich Zones 2개, Temporal shift
- **장애물** — obstacle_rays 분리, 별도 Push-Pull
- **포식자** — 이동형 pain source (speed 2.5)
- **NPC** — 사회적 상호작용, 음식/위험 call
- **학습 추이 시각화** — 실시간 시냅스 가중치 그래프

## 핵심 원칙

- **생물학적 메커니즘만 사용**: LIF 뉴런, STDP, Hebbian, Push-Pull, WTA, Dale's Law
- **학습은 경험에서 창발**: 사전 학습, 지식 주입, 휴리스틱 금지
- **하드코딩 금지**: 행동은 뇌 회로에서 창발해야 함
- **LLM/딥러닝 금지**: 뇌의 작동 원리로만 구현

## 빠른 시작

### 요구사항

- **WSL2** (Ubuntu) - Windows Python에서 PyGeNN 실행 불가
- **CUDA 12.x** (WSL 내 설치)
- **Python 3.9+** + PyGeNN 5.4.0

### 실행

```bash
# WSL에서 실행
wsl -d Ubuntu-24.04 -- bash -c "
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE
python /mnt/c/<YOUR_PATH>/backend/genesis/forager_brain.py --episodes 20 --render none
"
```

## 프로젝트 구조

```
backend/genesis/
├── forager_brain.py   # 메인 뇌 (27,910 뉴런, Phase 1-20 + L1-L18)
├── forager_gym.py     # Forager 환경 (800×800, 장애물, Rich Zones)
└── checkpoints/       # 학습 체크포인트

docs/
├── ROADMAP.md              # 전체 로드맵
├── PHASE_L16_DESIGN.md     # KC Sparse Expansion 설계
├── research/               # 초파리 리서치, 기술 차용 분석
└── PHASE*.md               # 각 Phase 설계 문서

gpu_check.ps1              # GPU 3D 사용률 측정 (PDH 카운터)
gpu_monitor.ps1            # GPU 모니터링 + 자동 kill
```

## 기술 스택

- **SNN Framework**: [PyGeNN](https://genn-team.github.io/) 5.4.0 (GPU-accelerated)
- **GPU**: CUDA 12.3
- **뉴런 모델**: LIF + SensoryLIF (27,910 neurons)
- **학습 규칙**: R-STDP, Anti-Hebbian, Hebbian, Homeostatic
- **환경**: Pygame 기반 2D Forager (800×800)
- **OS**: WSL Ubuntu (D: 드라이브)
