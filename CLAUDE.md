# Genesis Brain - 생물학적 SNN 기반 인공 뇌

> **장기 로드맵**: [docs/ROADMAP.md](docs/ROADMAP.md) - Phase 8~20 계획

---

## 환경 설정 (CRITICAL - 반드시 읽을 것)

```
╔═══════════════════════════════════════════════════════════════╗
║  ⚠️  절대 Windows Python으로 PyGeNN 실행하지 마라!            ║
║      반드시 WSL Ubuntu-24.04 사용!                            ║
║      Windows pygenn_env는 죽은 환경임!                        ║
╚═══════════════════════════════════════════════════════════════╝
```

```bash
# PyGeNN 환경 - WSL Ubuntu-24.04 사용!

# WSL 실행 명령어 (Forager Brain)
wsl -d Ubuntu-24.04 -- bash -c "
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE
python /mnt/c/<PROJECT_PATH>/backend/genesis/forager_brain.py --episodes 20 --render none
"
```

**WSL 환경 정보:**
- Python venv: `~/pygenn_wsl/`
- CUDA: `/usr/local/cuda-12.6`
- 작업 디렉토리: `~/pygenn_test/`
- `<PROJECT_PATH>`: 이 프로젝트의 WSL 경로 (예: `/mnt/c/.../BrainSimulation`)

---

## Phase 검증 필수 (CRITICAL - 반드시 준수)

```
╔═══════════════════════════════════════════════════════════════╗
║  ⚠️  새 Phase 구현 후 반드시 20 에피소드 검증 실행!           ║
║      검증 없이 다음 Phase 진행 절대 금지!                     ║
║      성능 저하 시 즉시 수정 후 재검증!                        ║
╚═══════════════════════════════════════════════════════════════╝
```

### Phase 검증 기준 (MANDATORY)

| 지표 | 기준 | 실패 시 |
|------|------|---------|
| **Survival Rate** | > 40% | Phase 진행 금지, 수정 필요 |
| **Reward Freq** | > 2.5% | 회로 연결 검토 |
| **Pain Avoidance** | > 85% | Pain 회피 반사 간섭 확인 |

### Phase 검증 명령어

```bash
# 새 Phase 구현 후 반드시 실행
wsl -d Ubuntu-24.04 -- bash -c "
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE
python /mnt/c/.../forager_brain.py --episodes 20 --render none
"
```

### 검증 프로세스

1. **새 Phase 구현 완료**
2. **20 에피소드 검증 실행**
3. **결과 확인:**
   - Survival Rate > 40% → 다음 Phase 진행
   - Survival Rate ≤ 40% → 수정 후 재검증
4. **문서 업데이트 (검증 결과 포함)**

### Phase 12-14 실패 사례 (2026-02-02) - 교훈

```
╔═══════════════════════════════════════════════════════════════╗
║  CASE STUDY: 검증 없이 진행한 결과                            ║
╠═══════════════════════════════════════════════════════════════╣
║  Phase 11 baseline:  60% 생존 ✓                               ║
║  Phase 12 추가:      30% 생존 ✗ (-50%, 검증 없이 진행)        ║
║  Phase 13 추가:       5% 생존 ✗ (-83%, 검증 없이 진행)        ║
║  Phase 14 추가:       0% 생존 ✗ (-100%, 검증 없이 진행)       ║
╠═══════════════════════════════════════════════════════════════╣
║  근본 원인: 새 회로가 기존 Pain 회피 반사를 방해               ║
║  교훈: 각 Phase마다 검증했으면 Phase 12에서 문제 발견 가능    ║
╚═══════════════════════════════════════════════════════════════╝
```

**실패 원인 분석 & 수정 (2026-02-08 해결):**
- Phase 12 (STS→Motor): 다감각 출력이 Pain Push-Pull과 충돌 → Motor 직접 연결 비활성화
- Phase 13 (PPC→Motor): 공간 정보가 모터에 노이즈로 작용 → Motor 직접 연결 비활성화
- Phase 14 (PMC→Motor): 운동 계획이 즉각적 회피 반응을 지연 → Motor_Prep 15→2 약화
- **간접 경로 간섭 발견**: STS→Amygdala(18→8), STS→Hippo(15→8), PPC→Hippo(10→5)
- **결과**: 0% → 35% (직접 경로 수정) → **60%** (간접 경로 수정) ✓

**수정 원칙:**
- 새 회로는 기존 **생존 반사를 방해하지 않아야** 함
- Motor 직접 연결보다 **기존 회로를 통한 조절** 선호
- 가중치는 기존 Pain Push-Pull(60/-40)보다 **약하게** 설정
- **간접 경로(Amygdala/Hippocampus 경유)도 Motor 간섭 가능** → 반드시 확인

---

## 외부 AI 제안 처리 (MANDATORY)

외부 AI(Gemini, GPT 등)의 제안을 받을 경우 **비판적으로 수용**한다.

### 검증 항목

| 항목 | 확인 내용 |
|------|----------|
| 파라미터 존재 여부 | 제안된 설정/함수가 실제 코드에 존재하는가? |
| 가정 검증 | 제안의 전제 조건이 현재 시스템에 유효한가? |
| 근본 원인 해결 | 제안이 증상만 완화하는가, 근본 원인을 해결하는가? |
| 부작용 | 제안 적용 시 발생 가능한 부정적 영향은? |
| 전이 가능성 | 단순화된 환경에서의 학습이 실제 환경에 적용 가능한가? |

### 처리 절차

1. **제안 수신** → 그대로 실행하지 않음
2. **비판적 분석** → 위 검증 항목 확인
3. **수정/보완** → 문제점 해결 후 실행 계획 수립
4. **Phase 1 프로세스 적용** → 사전 분석 후 실험

---

## 실험 프로세스 (MANDATORY)

장기 실험(100+ 에피소드) 실행 전 반드시 아래 프로세스를 따른다.

### Phase 1: 사전 분석 (코드 작성 전)

**1.1 학습 메커니즘 검증**

| 항목 | 기준 | 미충족 시 |
|------|------|----------|
| 보상 빈도 | > 1% | shaping reward 추가 |
| 시간 지연 | < eligibility τ (3초) | τ 연장 또는 설계 변경 |
| 인과관계 | 명확 | credit assignment 해결책 필요 |

**1.2 측정 지표 검증**

| 항목 | 확인 방법 |
|------|----------|
| 측정 경로 | 코드에서 지표 계산 위치 추적 |
| 회로 연결 | 새 시냅스가 측정 경로에 포함되는지 확인 |
| 성공 기준 | 수치로 정의 (예: "지표 > N") |

**1.3 신호 강도 검증**

| 항목 | 확인 방법 |
|------|----------|
| 경쟁 신호 | 기존 신호 vs 새 신호 가중치 비교 |
| 우회 여부 | 새 경로가 기존 회로를 bypass하는지 확인 |

### Phase 2: 짧은 검증 (20-50 에피소드)

```bash
python ... --episodes 20 --enemies 5
```

**확인 항목:**
- 학습 신호 발생 여부 (reward ≠ 0)
- 목표 지표 변화 여부
- baseline 대비 성능 유지 여부

**판단 기준:**

| 결과 | 조치 |
|------|------|
| 지표 = 0 고정 | 중단, 설계 재검토 |
| 지표 변화 있음 | Phase 3 진행 |
| 지표 있으나 변화 없음 | 100 에피소드로 확장 |

### Phase 3: 중기 검증 (100-200 에피소드)

**체크포인트:** 50 에피소드마다 점검

**조기 중단 기준:**
- 100 에피소드 후 지표 변화 없음
- baseline 대비 성능 20% 이상 하락

### Phase 4: 장기 실험 (500+ 에피소드)

**진입 조건:** Phase 2, 3에서 학습 신호 확인됨

**운영:**
- 백그라운드 실행
- 주기적 로그 확인
- 100 에피소드마다 checkpoint 저장

---

### 디버깅 및 시각화 (MANDATORY)

```
╔═══════════════════════════════════════════════════════════════╗
║  ⚠️  "측정하지 않으면 개선할 수 없다"                          ║
║      모든 실험에서 철저한 로깅과 시각화 필수!                  ║
╚═══════════════════════════════════════════════════════════════╝
```

**필수 로그 항목:**
1. **매 스텝:** 내부 상태, 뉴런 활성화율, 모터 출력
2. **매 에피소드:** 생존 시간, 보상 빈도, 사망 원인, 주요 지표
3. **이상 감지:** 자동 경고 (뉴런 비활성화, 신호 불균형 등)

**필수 시각화:**
1. **Pygame 렌더링:** 환경, 에이전트, 상태 바, 뉴런 패널
2. **그래프 저장:** Energy/Drives 시계열, 궤적, 모터 출력
3. **체크포인트 메타데이터:** 학습 이력 포함
4. **뇌 활성화 패널 (NEW):** 실시간 뇌 영역별 활성화 시각화

**뇌 활성화 패널 (Brain Activity Panel):**
```
┌─────────────────────────────┐
│  Brain Activity             │
├─────────────────────────────┤
│  HYPOTHALAMUS               │
│  ▓▓▓▓▓░░░░░ Hunger    45%   │
│  ▓▓░░░░░░░░ Satiety   20%   │
├─────────────────────────────┤
│  AMYGDALA                   │
│  ▓▓▓░░░░░░░ LA        30%   │
│  ▓▓░░░░░░░░ CEA       20%   │
│  ▓░░░░░░░░░ Fear      10%   │
├─────────────────────────────┤
│  HIPPOCAMPUS                │
│  ▓▓▓▓▓▓▓░░░ Place     70%   │
│  ▓▓▓░░░░░░░ FoodMem   30%   │
├─────────────────────────────┤
│  BASAL GANGLIA              │
│  ▓▓▓▓░░░░░░ Striatum  40%   │
│  ▓▓▓░░░░░░░ Direct    30%   │
│  ▓▓░░░░░░░░ Indirect  20%   │
│  ▓▓▓▓▓▓▓▓░░ Dopamine  80%   │
├─────────────────────────────┤
│  PREFRONTAL                 │
│  ▓▓▓▓░░░░░░ WorkMem   40%   │
│  ▓▓▓▓▓▓░░░░ GoalFood  60%   │
│  ▓▓░░░░░░░░ GoalSafe  20%   │
│  ▓░░░░░░░░░ Inhibit   10%   │
├─────────────────────────────┤
│  MOTOR OUTPUT               │
│  ▓▓▓░░░░░░░ Motor L   30%   │
│  ▓▓▓▓▓▓░░░░ Motor R   60%   │
│  Turn: >> RIGHT             │
└─────────────────────────────┘
```

> **상세 스펙:** [docs/PHASE2A_DESIGN.md - 섹션 4.3](docs/PHASE2A_DESIGN.md)

---

### R-STDP 학습 조건

R-STDP 기반 실험 시 아래 조건을 만족하는지 사전 검토:

| 조건 | 요구사항 | v35 (실패) | v36 Sandbag |
|------|----------|-----------|-------------|
| 보상 빈도 | > 1% | ~0.7% ✗ | ~9.6% (예상) ✓ |
| 시간 지연 | < τ | > 3초 ✗ | < 10초 ✓ (τ=10s) |
| 인과관계 | 명확 | 불명확 ✗ | 개선 예상 |

**v36 대응:**
- 환경 단순화 (Sandbag): 보상 빈도 증가
- Long Tau R-STDP: τ=10초로 연장 (Hunt 시냅스만)
- 생존 회로는 Static 유지 (학습 대상 아님)

---

## 이전 Phase 히스토리

### Phase 1: Slither.io 졸업 (2025-01-28)
- v40b: Best 64, Avg 37.6, Kills 0.44/ep
- Push-Pull, Disinhibition, WTA, 선천적 본능 검증
- [docs/SLITHER_GRADUATION.md](docs/SLITHER_GRADUATION.md)

### Phase 2-3: 변연계 (2025-01-28~30)
- 시상하부(항상성) + 편도체(공포) + 해마(공간기억)
- [Phase 2a](docs/PHASE2A_DESIGN.md) | [Phase 2b](docs/PHASE2B_DESIGN.md) | [Phase 3](docs/PHASE3_DESIGN.md)

> **전체 Phase 히스토리**: [docs/ROADMAP.md](docs/ROADMAP.md) 참조

### 현재 상태: Phase L15 완료 - Narrative Self (20,710 뉴런)

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase L15: Narrative Self ✓ (2026-02-21)                       ║
╠═══════════════════════════════════════════════════════════════╣
║  Learning Foundation (Phase L1→...→L14→L15):                     ║
║    - L1-L4: BG 기반 학습 (D1/D2, R-STDP, Anti-Hebbian)       ║
║    - L5-L6: 피질 학습 (perceptual R-STDP, PE)                ║
║    - L7-L8: 음식 유형별 차등 학습 (도파민 딥)                 ║
║    - L9: 피질 표상→BG 의사결정 연결                           ║
║    - L10: NAc Critic → RPE 도파민 (Schultz 1997)              ║
║    - L11: SWR Replay → 오프라인 기억 강화 (Buzsáki 2015)     ║
║    - L12: Global Workspace → 주의 기반 경쟁 (Dehaene 2011)   ║
║    - L13: Taste Aversion → 시각-내장 연합 학습 (Garcia 1966)  ║
║    - L14: Agency Detection → 전방 모델 (Frith 2005)           ║
║    - L15: Narrative Self → 에이전시 게이팅 자서전 (Damasio 2010)║
║                                                               ║
║  L15 변경 사항:                                               ║
║    - 0 new neurons, +1 DENSE Hebbian (agency→narrative)       ║
║    - Agency gate: agency_rate/0.15, [0.3, 2.0]               ║
║    - Salience gate: 1+|Δbody|×10, max 3.0                    ║
║    - Body→Narr: 2.0→14.0 (fast, ep5 saturated)               ║
║    - Agency→Narr: 1.0→4.28 (gradual, 20ep)                   ║
║                                                               ║
║  검증 결과 (20ep / 100ep):                                   ║
║    - Survival: 55% / 56% ✓                                   ║
║    - Pain Death: 0% / 0% ✓                                   ║
║    - Reward Freq: 2.79% / 2.75% ✓                            ║
║    - Selectivity: 0.70 (20ep best)                            ║
║    - Hippo: 2.12→11.84/18.0 (100ep, 계속 성장)               ║
║    - 4 synapses saturated at 100ep (Garcia/FM/Narr/Body)     ║
║                                                               ║
║  뉴런 수:       20,710 (변동 없음)                               ║
║  학습 시냅스:   41 (+1 DENSE Hebbian)                            ║
╚═══════════════════════════════════════════════════════════════╝
```

### 구현 파일

```
backend/genesis/
├── forager_gym.py     # ForagerGym 환경 (NPC + Pain Zone + 다중 음식 + GW 시각화)
├── forager_brain.py   # Forager Brain (20,710 뉴런, Phase 1-20 + L1-L15)
└── checkpoints/       # 체크포인트 저장
```

### 핵심 교훈

1. **방향성 Push-Pull이 핵심**: Pain Push-Pull(60/-40)이 생존의 기반
2. **새 회로의 Motor 간섭 방지**: 직접 0.0 + 간접 경로도 확인 필수
3. **R-STDP는 빈번하고 즉각적인 보상 필요**: 보상 빈도 >1% 필수
4. **Hebbian 학습은 안정적**: 보상 연관 학습 avg_w 2.0→7.12 (20 ep)
5. **D1/D2 MSN 분리 필수**: 단일 Striatum은 Go+NoGo 동시 강화 → 학습 상쇄
6. **R-STDP 빠른 포화**: w_max=5.0에 ep 1-2에서 도달, 점진적 개선 제한적
7. **Anti-Hebbian D2 trace는 pre-synaptic**: D1↔D2 경쟁이 D2 발화를 억제하므로 food_eye 기반 trace 사용
8. **환경 공진화**: 학습 압력(나쁜 음식)과 학습 능력(피질 STDP)을 동시 도입해야 함
9. **SensoryLIF I_input vs LIF Ioffset**: 표준 LIF의 Ioffset은 VAR가 아님, SensoryLIF I_input 사용
10. **시간적 신용 할당 노이즈**: 음식 근접 시 trace 중첩 → bad food D1도 학습 (도파민 시간적 겹침)
11. **도파민 딥이 생존율을 높임**: 음수 도파민(-0.5)이 기존 수식을 자동 반전 → 코드 변경 최소, 효과 극대 (+15pp 생존율)
12. **피질→BG 빌드 순서**: IT population은 BG보다 늦게 빌드됨 → 별도 _build_it_bg_circuit() 메서드 필요
13. **RPE는 scalar 모듈레이션**: NAc→Motor 직접 연결 없음, NAc rate로 DA magnitude만 조절 → Motor 간섭 0
14. **primary_reward 파라미터**: approach signal/bad food dip은 RPE 적용 안 함 (기존 행동 보존)
15. **IT→D2 빠른 수렴**: pre-synaptic only trace + 비측화 IT → ep5에 w_min 도달 (L4 패턴 재현)
16. **SWR Replay로 기존 Hebbian 재사용**: 새 학습 시냅스 0개, learn_food_location() 재활용 → Hebbian avg_w +76% 강화
17. **SWR Gate는 SensoryLIF I_input 전용**: 시냅스 입력 없음 → 온라인 중 절대 발화 안 함 (Motor 간섭 완벽 차단)
18. **Garcia Effect는 one-trial 학습**: eta 높게 (0.02), DENSE Hebbian으로 빠른 연합. 기존 경로(LA→CEA→Fear→Motor) 재사용
19. **음식 섭취 시 timing bug**: env.step()에서 먹으면 food 제거됨 → 다음 obs에 food_rays=0 → prev_activity 패턴 필수
20. **포식자 = 이동형 pain source**: 뇌 변경 0으로 새로운 환경 위협 추가 가능 (기존 회로 재사용)
21. **sensor_jitter는 pain_rays 제외 필수**: Push-Pull(60/-40)은 정밀한 L/R 차이에 의존 → 노이즈 적용 시 회피 기능 파괴
22. **Forward Model eta 점진적**: DENSE Hebbian uniform update는 빠르게 포화 → eta=0.005 (0.04에서 8x 감소)
23. **Agency gate로 기존 학습 조절**: 새 시냅스 최소화 (0 뉴런, 1 시냅스) — gate 곱셈으로 기존 body→narrative 학습 강화/약화
24. **100ep 중기 검증**: 해마만 계속 성장 (65%), 4개 시냅스 포화 (Garcia/FM/Agency→Narr/Body→Narr) — 500ep에서는 w_max 상향 또는 weight_decay로 동적 평형 고려

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
| R-STDP (Reward-modulated STDP) | 3-factor learning rule + eligibility trace |
| 도파민 시스템 (VTA/SNc) | Novelty → 도파민 → 학습/탐색 조절 |
| 습관화 (Habituation) | 반복 자극에 반응 감소 |
| LIF 뉴런 (Leaky Integrate-and-Fire) | 실제 뉴런 모델 |
| 항상성 가소성 (Homeostatic Plasticity) | 뉴런 활성화 안정화 |
| Dale's Law (흥분/억제 분리) | 실제 뉴런 특성 |
| WTA (Winner-Take-All) | 측면 억제를 통한 경쟁 |

---

## 레거시 코드

- Slither.io (Phase 1): `slither_pygenn_biological.py` - [졸업 보고서](docs/SLITHER_GRADUATION.md)
- FEP, Predictive Coding 등 이전 개발물: `archive/legacy` 브랜치

---

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다"
