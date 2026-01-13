# Genesis Brain

**생물학적 SNN 기반 인공 뇌 시뮬레이션**

순수 생물학적 메커니즘(LIF 뉴런, STDP, 도파민)만으로 학습하는 Spiking Neural Network. Chrome Dino 게임에서 실시간 반응 학습을 검증.

## 핵심 철학

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다."

- **LLM이 아니다**: 다음 토큰 예측 (X)
- **FEP 공식이 아니다**: G(a) = Risk + Ambiguity (X)
- **생물학적 메커니즘**: LIF 뉴런 + DA-STDP (O)

## 현재 상태: Chrome Dino Agent

### 결과

| 에이전트 | High Score | Avg Score | 특징 |
|---------|------------|-----------|------|
| 단일 채널 | 644 | 423.8 | 기본 점프 타이밍 |
| **이중 채널** | **725** | **367.9** | 새(Bird) 회피 성공 |

### Dual-Channel Vision 아키텍처

```
┌─────────────┐                    ┌─────────────┐
│  Ground Eye │                    │   Sky Eye   │
│  (Cacti)    │                    │  (Birds)    │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                   ┌──────────────┐
│ Ground Hidden│                   │  Sky Hidden  │
└──────┬───────┘                   └──────┬───────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐    INHIBIT        ┌──────────────┐
│  Jump Motor  │◄──────────────────│  Duck Motor  │
└──────────────┘                   └──────────────┘

Total: 3,600 LIF neurons
```

## 빠른 시작

### Chrome Dino 에이전트 실행

```bash
cd backend/genesis

# 이중 채널 (권장)
python dino_dual_channel_agent.py

# 단일 채널
python dino_snn_js_agent.py
```

### 요구사항

```bash
pip install torch snntorch playwright numpy
playwright install chromium
```

## 핵심 메커니즘

### 1. LIF 뉴런 (snnTorch)

```python
# Leaky Integrate-and-Fire
v = v * beta + input  # 막전위 누적
if v > threshold:
    spike = 1
    v = 0  # 리셋
```

### 2. DA-STDP (3-factor learning)

```python
# Eligibility trace + Dopamine modulation
eligibility = f(pre_spike, post_spike, tau)
weight += dopamine * eligibility * (a_plus if LTP else -a_minus)
```

### 3. 억제 시냅스

```python
# Sky Eye --| Jump Motor
jump_input -= sky_activity * inhibit_strength
```

## 프로젝트 구조

```
backend/genesis/
├── # Core SNN
├── snn_scalable.py           # SparseLIFLayer, SparseSynapses
├── snn_brain.py              # BiologicalBrain
├── snn_brain_biological.py   # STDP 기반 뇌
│
├── # Chrome Dino Agents
├── dino_dual_channel_agent.py # 이중 채널 + 억제 회로
├── dino_snn_js_agent.py       # JS API + SNN
├── dino_snn_agent.py          # 픽셀 기반
│
├── # Other Agents
├── snn_browser_agent.py       # 브라우저 탐색
└── snn_desktop_agent.py       # 데스크톱
```

## 핵심 발견

1. **JS API vs 픽셀**: 게임 상태 직접 접근이 ~20x 빠름
2. **점프 타이밍**: gap=100px에서 점프 시 최적
3. **채널 분리**: Ground/Sky 분리로 행동 충돌 해결
4. **억제 회로**: 새 감지 시 점프 억제로 "600점 벽" 돌파

## 레거시 코드

FEP, Predictive Coding, E6-E8 실험 등 이전 개발물:
- Branch: `archive/legacy`

## 라이선스

Private

## 참고

- snnTorch: https://snntorch.readthedocs.io/
- Bi & Poo (1998). Synaptic Modifications in Cultured Hippocampal Neurons
