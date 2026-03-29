# 최신 논문 리서치 (2024-2026) — Genesis Brain 차용 분석

> 조사일: 2026-03-29

---

## 즉시 적용 가능

### 1. SWR Selective Replay (Science 2024)
- **논문**: "Selection of experience for memory by hippocampal sharp wave ripples"
- **저자**: Yang, Sun, Huszar, Hainmueller, Kiselev, Buzsaki
- **핵심**: Waking SWR이 보상 시점 경험을 태깅 → sleep SWR이 태깅된 것만 선택적 replay
- **차용**: experience_buffer에 tag 필드 추가, replay 시 80% tagged 우선
- **적용 단계**: C0 (즉시)

### 2. Drosophila Larva Prediction Error (iScience 2024)
- **논문**: "Prediction error drives associative learning in spiking model of Drosophila larva"
- **저자**: Jurgensen, Sakagiannis, Schleyer, Gerber, Nawrot
- **핵심**: MBON→DAN feedback으로 brain-internal RPE. Synaptic homeostasis + PE 결합
- **차용**: KC→D1/D2→Dopamine feedback loop, PE 비례 weight_decay 조절
- **적용 단계**: C0-C1

---

## C1 (감각 모호성) 관련

### 3. KC Multi-Compartment STM/LTM (Nature 2024)
- **논문**: "Dopamine-mediated interactions between short- and long-term memory dynamics"
- **저자**: Hige, Aso et al.
- **핵심**: KC를 gamma(STM, tau=1s) / alpha(LTM, tau=10s) 분리. 같은 자극의 단기/장기 가치 별도 표상
- **차용**: KC 3000을 2구획 분리, 각각 다른 STDP time constant
- **적용 단계**: C1

### 4. Multisensory Integration (Patterns 2025)
- **논문**: "Emulating sensation by bridging neuromorphic computing and multisensory integration"
- **핵심**: Crossmodal imagination (소리→시각 표상 활성화), Temporal binding via gamma oscillation
- **차용**: Wernicke_Food→IT_Food Hebbian + KC_inh gamma oscillation
- **적용 단계**: C1

---

## C2 (범주 창발) 관련

### 5. CATS Net — Concept Bottleneck (Nature CompSci 2026)
- **논문**: "A neural network for modeling human concept formation"
- **저자**: Liangxuan Guo et al. (CAS/PKU)
- **핵심**: Concept-abstraction bottleneck이 저차원 개념 표상 추출 → task module gating
- **차용**: KC→Concept Layer(16-32 뉴런)→D1/D2 gating. 현재 KC→D1 직접 연결에 bottleneck 삽입
- **적용 단계**: C2

### 6. Hybrid Corticohippocampal (Nature Comms 2025)
- **논문**: "Hybrid neural networks for continual learning"
- **저자**: Shi, Liu, Li et al. (Tsinghua)
- **핵심**: KC=specific, Assoc=generalized. Feedforward+feedback 루프로 dual representation
- **차용**: KC→Assoc_Binding→KC 루프, Assoc_Novelty로 "새 범주 vs 기존 범주" gate
- **적용 단계**: C2

---

## Neuromodulation 관련

### 7. Unified Neuromodulator Library (PLOS CompBio 2025)
- **논문**: "A unified model library maps how neuromodulation reshapes excitability"
- **핵심**: 5 neuromodulators × 7 neuron types. Switching vs Scaling 효과 분류
- **차용**: NE→exploration(KC Vthresh↓), 5-HT→patience(D2 Vthresh↑), ACh→attention(GW gate)
- **적용 단계**: C1-C5

### 8. 3-Factor Learning Survey (Patterns 2025)
- **논문**: "Three-Factor Learning in SNN: Overview of Methods and Trends"
- **저자**: Mazurek et al.
- **핵심**: Heterogeneous third factor, kernel-based trace, cell-type별 다른 modulator
- **차용**: trace_decay를 kernel mixture로, 학습 회로별 독립 neuromodulator
- **적용 단계**: 전체

---

## Continual Learning 관련

### 9. CLP-SNN on Loihi 2 (arXiv 2025)
- **논문**: "Real-time Continual Learning on Intel Loihi 2"
- **핵심**: Neurogenesis(새 뉴런 동적 할당) + Metaplasticity(성숙 가중치 보호)
- **차용**: KC dormant 뉴런 깨우기, weight_age 기반 eta 감소
- **적용 단계**: C2-C3

### 10. Meta-Learning R-STDP (Neurocomputing 2024)
- **논문**: "Meta-learning in SNN with reward-modulated STDP"
- **핵심**: Experience→memory population 인코딩, 에피소드별 meta-eta 자동 조절
- **차용**: Meta_Confidence→R-STDP eta modulation, experience_buffer를 시냅스로 전환
- **적용 단계**: C4

---

## Pattern Separation 측정

### 11. Information-Theoretic Pattern Separation (2024)
- **논문**: "Robust measures of pattern separation based on information theory"
- **핵심**: MI, conditional entropy로 KC 패턴 분리 정량화
- **차용**: C0 개념 검증 계측의 수학적 프레임워크
- **적용 단계**: C0

---

## 우선순위

| 순위 | 논문 | 단계 | 구현 난이도 |
|------|------|------|-----------|
| 1 | SWR Selective Replay | C0 | **낮음** |
| 2 | Drosophila Larva PE | C0-C1 | 중간 |
| 3 | KC Multi-Compartment | C1 | 중간 |
| 4 | Neuromodulator Library | C1-C5 | 높음 |
| 5 | CATS Net Concept Bottleneck | C2 | 중간 |
