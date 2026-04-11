# Next Direction After M3: M4 계획

> Date: 2026-04-11

## M3까지 달성한 것

- 28,035 LIF neurons, 20 brain regions
- C0-C5 개념 형성, C3 proto-language
- Predictive plasticity + heterosynaptic budget (포화 해소)
- ACh uncertainty gate + surprise-modulated replay
- **Revaluation SWR**: place transition graph + reverse value backup
- 동적 환경 70% 생존, detour test PASS (+14.7pp)

## 다음 방향 후보 (리서치 기반)

### Option A: Spiking World Model (Active Dendrites)

**근거**: PNAS 2025 (Sun et al.) — Spiking-WM
- Multicompartment neurons (여러 dendrite에서 비선형 정보 통합)
- 환경 dynamics 예측 → model-based planning
- GRU 수준 성능 (SNN으로!)
- GPT 전략 자문 #4에서 추천한 "active dendrites" 일치

**장점**: 가장 과학적으로 의미 있는 다음 단계. 현재 LIF 점뉴런의 한계를 넘어감.
**단점**: 구현 복잡도 높음. PyGeNN에서 multicompartment neuron 지원 필요.

### Option B: Continual Learning (과제 전환)

**근거**: Nature Comms 2025 — CH-HNN (corticohippocampal hybrid)
- 기존 지식으로 새 개념 학습 촉진
- Task-incremental + class-incremental 학습
- Catastrophic forgetting 방지

**장점**: 현재 환경에서 바로 테스트 가능 (음식 종류 추가, 규칙 변경)
**단점**: M3에서 이미 일부 달성 (revaluation SWR이 contingency change 처리)

### Option C: Dreaming / Model-Based Simulation

**근거**: Scientific Reports 2024 — dreaming in SNN
- 오프라인에서 "꿈"으로 새로운 경험 생성
- Model-based simulated environment에서 학습
- SWR replay의 자연스러운 확장

**장점**: M3의 reverse replay를 "forward imagination"으로 확장
**단점**: world model이 필요 → Option A와 겹침

### Option D: 환경 복잡도 대폭 증가

**근거**: GPT 전략 자문 — "70% environment investment"
- 다중 에이전트 협력/경쟁 심화
- 도구 사용 (음식 접근에 장애물 제거 필요)
- 시간 압력 (계절 변화, 음식 고갈)
- 사회적 신호 기반 의사결정

**장점**: 뇌 변경 최소, 기존 회로 스트레스 테스트
**단점**: 새로운 인지 능력이 아닌 기존 능력의 스트레스 테스트

## 추천 순서

1. **Option D (환경 복잡도)** — 가장 빠르게 실행 가능, 기존 회로의 한계 발견
2. **Option B (Continual Learning)** — 새 음식 종류/규칙 도입, forgetting 측정
3. **Option A (Active Dendrites)** — 가장 의미 있지만 가장 어려움, 장기 목표

## 핵심 논문

- Sun et al. (PNAS 2025): Spiking world model with multicompartment neurons, PMID 41385543
- Nature Comms 2025: Hybrid corticohippocampal continual learning
- Scientific Reports 2024: Dreaming in SNN for model-based RL
