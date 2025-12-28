# Genesis Brain - 근본 원리에서 창발하는 인공 뇌

## 핵심 철학

**뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다.**

- 심즈식 "욕구 게이지" 금지 (X)
- FEP 기반 "Risk가 높으면 회피" (O)
- 모든 행동의 **왜?**는 하나의 근본 원리(Free Energy 최소화)로 귀결

---

## 현재 버전: v4.6.2

### 핵심 공식

**지각 (F)**: `F = -log P(o|s) + KL[Q(s) || P(s)]`

**행동 선택 (G)**:
```
G(a) = Risk + Ambiguity + Complexity
     = E[KL[Q(o|s',a) || P(o)]]      # 선호에서 벗어날 것인가
     + transition_std × 1.5           # 결과가 불확실한가
     + E[KL[Q(s'|a) || P(s')]]        # 믿음이 선호에서 벗어날 것인가
```

### 관측 공간 (8차원)
```
[food_prox, danger_prox, food_dx, food_dy, danger_dx, danger_dy, energy, pain]
```

### P(o) 선호 분포
```
energy: Beta(3,2) → mode ~0.67 (항상성)
pain:   Beta(1,5) → mode ~0.0  (통증 회피)
```

---

## 버전 히스토리 (핵심만)

| 버전 | 핵심 기능 |
|------|----------|
| v2.0 | Beta 분포 P(o), KL divergence Risk |
| v2.1 | SimulationClock, 행동별 Ambiguity |
| v2.2 | Complexity = KL[Q(s')\|\|P(s')] |
| v2.3 | Precision Learning (sensory/transition/goal) |
| v2.4 | Temporal Depth (n-step rollout) |
| v2.5 | **Interoception** - 내부 항상성 기반 P(o) |
| v3.1 | Hierarchical (Slow/Fast Layer) |
| v3.2 | Context별 전이 모델 학습 |
| v3.3.1 | Context-weighted Transitions 안정화 |
| v3.4 | THINK Action (메타인지) |
| v3.5 | Online Preference Learning |
| v3.6 | Checkpoint & Headless Evaluation |
| v3.7 | Reproducibility (시드 고정) |
| v3.8 | Docker Packaging |
| v4.0 | **LTM** - 기억이 G를 조정 (행동 직접 지시 X) |
| v4.1 | Memory Consolidation (Sleep) |
| v4.3 | Uncertainty → 자기조절 신호 |
| v4.4 | **Counterfactual + Regret** |
| v4.5 | Server-side Drift (7가지 타입) |
| v4.6 | Drift Adaptation Report, Ablation Matrix |
| v4.6.1 | Drift Suppression (transition error 기반) |
| v4.6.2 | Regret + Suppression 결합 |

---

## 주요 메커니즘

### Memory (v4.0)
- 기억은 **G를 조정**하지, 행동을 직접 지시하지 않음
- `memory_gate = f(surprise, uncertainty)` → 저장 우선순위
- `G(a) += recall_weight × memory_bias[a]`

### Regret (v4.4)
- `regret = G_chosen - G_optimal` (사후 평가)
- 정책 직접 수정 X → memory_gate, lr_boost, THINK에 연결

### Drift Suppression (v4.6.1-2)
- transition error spike 감지 → recall weight 억제
- "Wrong Confidence" 문제 해결: pre-drift 기억이 post-drift에서 해가 되는 것 방지
- v4.6.2: regret spike를 억제 보조 신호로 활용

### Uncertainty → Modulation (v4.3)
```python
think_bias = 0.2 - 0.5 * uncertainty      # 불확실 → THINK 유리
sensory_precision *= 1.0 - 0.3 * uncertainty
exploration_bonus = 0.2 * uncertainty
```

---

## 파일 구조

```
backend/
├── genesis/
│   ├── action_selection.py   # G 계산, 행동 선택
│   ├── preference_distributions.py  # P(o) Beta 분포
│   ├── hierarchy.py          # Slow/Fast Layer
│   ├── memory.py             # LTM
│   ├── consolidation.py      # Sleep
│   ├── uncertainty.py        # Uncertainty 추적
│   ├── regret.py             # Counterfactual + Regret
│   ├── precision.py          # Precision Learning
│   ├── temporal.py           # n-step Rollout
│   ├── scenarios.py          # Drift 시나리오
│   ├── checkpoint.py         # 저장/복원
│   └── reproducibility.py    # 시드 관리
├── main_genesis.py           # FastAPI 서버
frontend/
├── src/GenesisApp.jsx        # 시각화
```

---

## 주요 API

```bash
# 기본
POST /step              # 한 스텝 실행
POST /reset             # 리셋

# Memory
POST /memory/enable
POST /memory/drift_suppression/enable?use_regret=true

# Regret
POST /regret/enable

# Drift
POST /drift/enable?drift_type=rotate
GET  /scenario/drift_report

# Hierarchy
POST /hierarchy/enable?K=4

# Checkpoint
POST /checkpoint/save?filename=brain.json
POST /checkpoint/load?filename=brain.json

# Evaluation
POST /evaluate?n_episodes=100
```

---

## 금지 사항

- 감정 이름을 변수로 사용 (X)
- 심즈식 욕구 게이지 (X)
- 휴리스틱으로 직접 행동 조작 (X)

---

## 핵심 통찰

1. **내부 항상성 > 외부 목표**: λ=1.0(내부)이 λ=0.0(외부)보다 2배 성능
2. **기억은 G 조정**: 행동 직접 지시가 아니라 Expected Free Energy 수정
3. **후회는 학습 자원 배분**: 정책 직접 수정 X, memory_gate/lr_boost에 연결
4. **Drift 적응**: transition error spike → recall 억제 → 잘못된 기억 의존 방지

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다"
