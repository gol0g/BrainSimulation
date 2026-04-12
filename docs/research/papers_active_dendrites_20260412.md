# Paper Search: Active Dendrites for Context-Dependent SNN

> Sources: PubMed (17 hits), arXiv (5 hits), OpenAlex (5 hits)
> Date: 2026-04-12

## Top Results

### 1. The Calcitron (PLOS CompBio 2025)
- **Authors**: Moldwin T, Azran LS, Segev I
- **DOI**: 10.1371/journal.pcbi.1012754
- **PMID**: 39879254
- **Abstract**: Perceptron-like model with 4 calcium sources (local, heterosynaptic, postsynaptic, supervisor). Implements Hebbian, anti-Hebbian, frequency-dependent plasticity, one-shot learning in ONE model.
- **Relevance**: ⭐⭐⭐ — dendritic computation을 LIF에 추가하는 가장 실용적 방법. 4 calcium source → branch-specific learning. 우리 context-dependent 실패 해결에 직접 적용 가능.
- **Potential use**: PFC/Hippocampal prediction 뉴런에 calcitron-style dendritic branches 추가. 하나의 branch에 위치 정보, 다른 branch에 음식 정보 → conjunction detection.

### 2. Spiking World Model with Multicompartment Neurons (PNAS 2025)
- **Authors**: Sun Y, Zhao F, Lyu M, Zeng Y
- **PMID**: 41385543
- **Abstract**: Multicompartment neurons for model-based RL. GRU-level performance in SNN.
- **Relevance**: ⭐⭐⭐ — 장기 목표 (world model). 현재 context 문제 해결 후.

### 3. MTSpark: Multi-Task SNN with Context Signals (arXiv 2024)
- **Authors**: Devkota A, Putra RVW
- **arXiv**: 2412.04847
- **Abstract**: Task-specific context signals for multi-task SNN. Addresses catastrophic forgetting.
- **Relevance**: ⭐⭐ — context signal 메커니즘 참고. 우리 zone-dependent rule과 관련.

### 4. Minicolumn Episodic Memory with Dendrites (IEEE TNNLS 2024)
- **Authors**: Zhang Y, Chen Y, Zhang J
- **DOI**: 10.1109/TNNLS.2022.3213688
- **PMID**: 36279337
- **Abstract**: Spiking neurons with dendrites + delays for episodic memory in minicolumn architecture.
- **Relevance**: ⭐⭐ — dendritic delays 메커니즘 참고.

## Implementation Strategy

**최소 변경: "Context Branch" 추가**
- 기존 LIF 유지 (SensoryLIF의 I_input 활용)
- Context signal = place_cells 활성도를 zone indicator로 변환
- Context → D1/D2 학습에 gating: "이 위치에서의 food value" 학습
- 새 뉴런 불필요 — 기존 place_cells × food_eye conjunction을 KC에서 표상

**핵심 질문: PyGeNN에서 multicompartment가 가능한가?**
- GeNN은 custom neuron model 지원 → 2-compartment LIF 구현 가능
- 또는 process()에서 소프트웨어적으로 dendritic computation 모사
