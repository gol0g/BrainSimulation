# Genesis Brain

**Free Energy Principle 기반 인공 뇌 시뮬레이션**

근본 원리에서 모든 행동이 창발하는 인공 뇌. 감정 변수나 규칙 없이, 단일 원리(자유 에너지 최소화)만으로 회피, 탐색, 습관, 학습, 후회가 나타납니다.

## 핵심 철학

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다."

- **심즈가 아니다**: "지루함 게이지가 차면 놀이를 한다" (X)
- **FEP 기반**: "Risk가 높으면 회피 행동" (O)
- **단일 원리**: 모든 행동의 "왜?"를 파고들면 G(a) 최소화에 도달

## 현재 버전: v4.4.1

### 구현된 기능

| 버전 | 기능 | 설명 |
|------|------|------|
| v2.0 | True FEP | G = Risk + Ambiguity + Complexity |
| v2.3 | Precision Learning | 동적 주의 조절 |
| v2.4 | Temporal Depth | 다중 시간 스케일 상상 (n-step rollout) |
| v2.5 | Interoception | 내부 항상성 기반 P(o) |
| v3.0 | Hierarchical Models | 2층 구조 (Slow/Fast layer) |
| v3.4 | THINK Action | 메타인지 (생각하는 행동) |
| v3.5 | Preference Learning | 경험에서 선호 학습 |
| v3.6 | Checkpoint | 상태 저장/복원, 헤드리스 평가 |
| v3.7 | Reproducibility | 시드 고정, 재현성 테스트 |
| v3.8 | Docker | 환경 재현성 |
| v4.0 | Long-Term Memory | 장기 기억 저장/회상 |
| v4.1 | Consolidation | 수면/통합 (메모리 → 모델) |
| v4.3 | Uncertainty | 불확실성 추적 및 자기조절 |
| v4.4 | **Counterfactual + Regret** | 반사실적 추론, 후회 기반 학습 |
| v4.4.1 | Regret 해석성 | z-score, normalized, spike 원인 |

### 핵심 공식

```
G(a) = Risk + Ambiguity + Complexity

Risk       = KL[Q(o|a) || P(o)]     → "선호 위반" → 회피 (공포처럼 보임)
Ambiguity  = E[H[P(o|s')]]          → "불확실성" → 탐색 (호기심처럼 보임)
Complexity = KL[Q(s'|a) || P(s')]   → "믿음 이탈" → 습관 (관성처럼 보임)
```

## 빠른 시작

### 1. 백엔드 실행

```bash
cd backend
pip install -r requirements.txt
python main_genesis.py
# http://localhost:8002 에서 API 서버 실행
```

### 2. 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173 에서 UI 실행
```

### 3. Docker (선택)

```bash
docker compose up backend
```

## API 엔드포인트

### 기본
- `POST /step` - 한 스텝 실행
- `POST /reset` - 리셋
- `GET /info` - 시스템 정보

### 기능 토글
- `POST /temporal/enable` - 다중 시간 스케일
- `POST /hierarchy/enable` - 계층적 컨텍스트
- `POST /think/enable` - THINK 행동
- `POST /memory/enable` - 장기 기억
- `POST /consolidation/enable` - 수면/통합
- `POST /uncertainty/enable` - 불확실성 추적
- `POST /regret/enable` - Counterfactual + Regret

### 테스트
- `POST /reproducibility/test` - 재현성 테스트
- `POST /scenario/g1_gate` - G1 Gate (일반화 테스트)
- `POST /ablation/apply` - Ablation 테스트

## 프로젝트 구조

```
backend/
├── genesis/
│   ├── action_selection.py    # G = Risk + Ambiguity + Complexity
│   ├── preference_distributions.py  # P(o) Beta 분포
│   ├── precision.py           # Precision Learning
│   ├── temporal.py            # n-step Rollout
│   ├── hierarchy.py           # Slow/Fast Layer
│   ├── memory.py              # Long-Term Memory
│   ├── consolidation.py       # Sleep/Consolidation
│   ├── uncertainty.py         # Uncertainty Tracking
│   ├── regret.py              # Counterfactual + Regret
│   ├── scenarios.py           # G1 Gate, Drift 시나리오
│   ├── ablation.py            # Ablation Framework
│   ├── checkpoint.py          # 상태 저장/복원
│   └── reproducibility.py     # 재현성 관리
├── main_genesis.py            # FastAPI 서버
└── requirements.txt

frontend/
├── src/
│   ├── GenesisApp.jsx         # 메인 UI
│   └── main.jsx
└── package.json
```

## v4.4 Counterfactual + Regret

### 핵심 원칙

후회(regret)는 "감정 변수"가 아니라, **선택한 행동이 대안보다 얼마나 더 큰 G를 초래했는지에 대한 사후 EFE 차이**.

```python
# 매 step에서 counterfactual 계산
regret_pred = G_pred(chosen) - min_a G_pred(a)  # 판단 오류
regret_real = G_post(chosen) - min_a G_post(a)  # 실제 후회
```

### FEP스러운 연결

정책을 직접 바꾸지 않고, 메타 메커니즘으로 연결:

| 연결 대상 | 방식 | 효과 |
|-----------|------|------|
| Memory Gate | regret 클수록 저장 가치 ↑ | 중요 경험 보존 |
| LR Boost | spike 시 학습률 50% 증가 | 빠른 모델 수정 |
| THINK Benefit | 누적 regret → 메타인지 가치 ↑ | 더 많은 숙고 |

### v4.4.1 해석 가능성

- **Optimal 기준**: G_post (사후 재평가)
- **Regret Z**: 최근 분포 대비 z-score
- **Normalized**: G_best 대비 비율 (스케일 불변)
- **Spike 원인**: 판단 오류 / 모델 불일치 / 환경 변화

## 금지 사항

- 감정 이름을 변수로 사용 (X)
- 심즈식 욕구 게이지 (X)
- 휴리스틱으로 직접 행동 조작 (X)

## 라이선스

Private - 비공개 프로젝트

## 참고 문헌

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active inference: the free energy principle in mind, brain, and behavior.
