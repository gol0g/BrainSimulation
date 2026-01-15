# Genesis Brain - 생물학적 SNN 기반 인공 뇌

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
| DA-STDP (Dopamine-modulated STDP) | 3-factor learning rule |
| 도파민 시스템 (VTA/SNc) | Novelty → 도파민 → 학습/탐색 조절 |
| 습관화 (Habituation) | 반복 자극에 반응 감소 |
| LIF 뉴런 (Leaky Integrate-and-Fire) | 실제 뉴런 모델 |
| 항상성 가소성 (Homeostatic Plasticity) | 뉴런 활성화 안정화 |
| Dale's Law (흥분/억제 분리) | 실제 뉴런 특성 |
| 억제 시냅스 (Inhibitory Synapses) | 뉴런 간 억제 연결 |

### 자기 점검 질문

새로운 기능을 추가하기 전에 반드시 확인:

1. "인간 아기도 이렇게 배우는가?"
2. "이건 학습인가, 아니면 그냥 데이터 저장인가?"
3. "뇌가 진짜 '이해'하는가, 아니면 패턴만 저장하는가?"
4. "이 능력이 경험에서 창발하는가, 아니면 내가 주입하는가?"

---

## 현재 상태: Slither.io (15,800 neurons)

### 진화 경로

| 단계 | 환경 | 뉴런 | 성과 |
|------|------|------|------|
| 1단계 | Chrome Dino | 3,600 | High: 725 (졸업) |
| 2단계 | Slither.io Phase 1 | 15,800 | High: 64 (청소부) |
| **현재** | **Slither.io Phase 2** | **15,800** | **High: 57 (적 3마리)** |

### Phase 2 결과 (적 추가 + 진화된 본능)

```
Evolved + Curriculum Training:
  Best: 57
  Total Kills: 107
  Innate avoidance reflex enabled
```

**핵심 구현사항**:

1. **진화된 본능 (Innate Reflex as Synaptic Weights)**
   - 적 회피 반사를 **시냅스 가중치**로 구현 (if문 아님)
   - 교차 배선: Enemy LEFT → RIGHT motor (적 반대로 회전)
   - 초기 가중치 3배 부스트 (innate_boost = 3.0)
   - DA-STDP로 여전히 학습 가능 (경험으로 조절됨)

   ```python
   # 진화된 본능 = 강한 초기 시냅스 가중치
   self.syn_enemy_motor_left.weights *= innate_boost  # 3x stronger
   self.syn_enemy_motor_right.weights *= innate_boost
   ```

2. **Slither.io 규칙 구현**
   - 적 머리가 내 몸에 부딪히면 → 적 사망 + 먹이화
   - 적 AI: 400px 내 플레이어 추적

3. **GPU 최적화** (RTX 3070 8GB)
   - 벡터화된 인코딩: Python for-loop → torch.repeat_interleave
   - Lazy sparse rebuild: 학습 시 매번 재구성 → 필요할 때만
   - 캐시된 transpose: float32 변환 캐싱
   - **성능: 36 → 44.6 steps/sec (+24%)**

### 철학적 원칙: 진화된 본능 vs 하드코딩

```
┌─────────────────────────────────────────────────────────────────┐
│  "빈 서판(Blank Slate)은 죽음이다"                              │
│  - 갓 태어난 동물도 생존 본능이 있음                            │
│  - 본능 = 진화가 시냅스 가중치에 새겨놓은 것                    │
│  - if문으로 행동 조작 ≠ 본능 (그건 로봇)                        │
│  - 시냅스 가중치 초기화 = 본능 (여전히 학습 가능)               │
└─────────────────────────────────────────────────────────────────┘
```

### Slither.io 아키텍처 (153K neurons)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Food Eye   │  │  Enemy Eye   │  │   Body Eye   │
│    (8K)      │  │    (8K)      │  │    (4K)      │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐
│Hunger Circuit│  │ Fear Circuit │
│    (10K)     │  │    (10K)     │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
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
```

**핵심 메커니즘**:
- **3채널 시각**: Food Eye (attract), Enemy Eye (avoid), Body Eye (self)
- **공포 회로**: 적 감지 시 Hunger 억제 (Fear --| Hunger)
- **직접 반사**: Fear → Boost (위기 탈출)
- **Ray-cast 센서**: 32방향 레이로 주변 감지

---

## Chrome Dino 졸업 (3,600 neurons)

### 핵심 아키텍처: snnTorch 기반 Scalable SNN

```python
# SparseLIFLayer: Leaky Integrate-and-Fire 뉴런 (snnTorch)
class SparseLIFLayer:
    beta: float = 0.9          # 막전위 감쇠율
    threshold: float = 1.0     # 스파이크 임계값

# SparseSynapses: 1% 희소 연결
class SparseSynapses:
    sparsity: float = 0.01     # 연결 비율
    weights: Tensor            # 희소 가중치 행렬
    eligibility: Tensor        # STDP eligibility trace
```

### Chrome Dino 에이전트 결과

| 에이전트 | High Score | Avg Score | 특징 |
|---------|------------|-----------|------|
| 단일 채널 (JS API) | 644 | 423.8 | 기본 점프 타이밍 학습 |
| **이중 채널 (Dual-Channel)** | **725** | **367.9** | 새(Bird) 회피 성공 |

### Dual-Channel Vision 아키텍처

```
┌─────────────┐                    ┌─────────────┐
│  Ground Eye │                    │   Sky Eye   │
│  (Cacti)    │                    │  (Birds)    │
│   500 LIF   │                    │   500 LIF   │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│ Ground Hidden│                   │  Sky Hidden  │
│   1000 LIF   │                   │   1000 LIF   │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐    INHIBIT        ┌──────────────┐
│  Jump Motor  │◄──────────────────│  Duck Motor  │
│    300 LIF   │   (strength=0.8)  │    300 LIF   │
└──────────────┘                   └──────────────┘

Total: 3,600 neurons
```

**핵심 메커니즘**:
- **채널 분리**: Ground Eye는 cacti만, Sky Eye는 birds만 처리
- **억제 시냅스**: Sky Eye가 새 감지 시 Jump Motor 억제 → 점프 방지
- **격리된 경로**: 채널 간 간섭(cross-talk) 방지로 안정적 학습

### DA-STDP 학습

```python
# 3-factor learning rule
def _learn(self):
    # 1. Eligibility trace 업데이트 (pre-post spike timing)
    self.syn.update_eligibility(pre_spikes, post_spikes, tau=500.0)

    # 2. 도파민 조절 시냅스 가소성
    self.syn.apply_dopamine(
        dopamine=self.dopamine,
        a_plus=0.01,   # LTP 강도
        a_minus=0.012  # LTD 강도
    )
```

---

## 파일 구조

```
backend/
├── genesis/
│   ├── # Core SNN
│   ├── snn_scalable.py           # SparseLIFLayer, SparseSynapses (snnTorch)
│   ├── snn_brain.py              # BiologicalBrain (original)
│   ├── snn_brain_biological.py   # STDP 기반 생물학적 뇌
│   │
│   ├── # Slither.io (Current)
│   ├── slither_gym.py            # Python 훈련 환경
│   ├── slither_snn_agent.py      # 153K neuron 에이전트
│   │
│   ├── # Chrome Dino (Graduated)
│   ├── dino_dual_channel_agent.py # 이중 채널 + 억제 회로 (725점)
│   ├── dino_snn_js_agent.py      # JS API + SNN (644점)
│   │
│   └── checkpoints/              # 저장된 모델들
│
├── Dockerfile
└── requirements.txt
```

---

## 실행 방법

### Slither.io (현재 진행중)

```bash
cd backend/genesis

# Phase 1: 청소부 (먹이만, 적 없음)
python slither_snn_agent.py --render pygame

# Phase 2: 겁쟁이 (적 추가)
python slither_snn_agent.py --enemies 3 --render pygame

# 이어서 훈련
python slither_snn_agent.py --resume
```

### Chrome Dino (졸업)

```bash
cd backend/genesis
python dino_dual_channel_agent.py --eval  # 저장된 모델 평가
```

---

## 핵심 발견

1. **JS API vs 픽셀**: 게임 상태 직접 접근이 스크린샷보다 ~20x 빠름
2. **점프 타이밍**: gap=100px에서 점프 시 최적 (장애물 통과 지점에서 최고점)
3. **이중 채널 효과**: 새(Bird) 회피 문제 해결로 "600점 벽" 돌파
4. **억제 회로**: 생물학적 억제 시냅스로 행동 충돌 해결
5. **채널 격리**: 분리된 Hidden layer가 cross-talk 방지의 핵심

---

## 레거시 코드

FEP, Predictive Coding, E6-E8 실험 등 이전 개발물은 `archive/legacy` 브랜치에 보존됨.

---

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다"
