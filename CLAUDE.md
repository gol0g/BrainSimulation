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

# WSL 실행 명령어
wsl -d Ubuntu-24.04 -e bash -c "
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=\$CUDA_PATH/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
python <PROJECT_PATH>/backend/genesis/slither_pygenn_biological.py --dev --episodes 10 --enemies 3 --render pygame
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

**실패 원인 분석:**
- Phase 12 (STS→Motor): 다감각 출력이 Pain Push-Pull과 충돌
- Phase 13 (PPC→Motor): 공간 정보가 모터에 노이즈로 작용
- Phase 14 (PMC→Motor): 운동 계획이 즉각적 회피 반응을 지연

**수정 원칙:**
- 새 회로는 기존 **생존 반사를 방해하지 않아야** 함
- Motor 직접 연결보다 **기존 회로를 통한 조절** 선호
- 가중치는 기존 Pain Push-Pull(60/-40)보다 **약하게** 설정

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

## 현재 상태: Phase 1 완료 (Slither.io 졸업)

> **상세 보고서:** [docs/SLITHER_GRADUATION.md](docs/SLITHER_GRADUATION.md)

### Phase 1: 반사하는 뇌 - 완료 (2025-01-28)

```
╔═══════════════════════════════════════════════════════════════╗
║  SLITHER.IO PROJECT - GRADUATED (v40b)                        ║
╠═══════════════════════════════════════════════════════════════╣
║  Best Length: 64    │  Avg: 37.6    │  Kills: 0.44/ep         ║
║  검증 완료: Push-Pull, Disinhibition, WTA, 선천적 본능        ║
╚═══════════════════════════════════════════════════════════════╝
```

**달성한 것:**
- 뇌간/척수 수준의 반사 회로 검증
- 생물학적 배선 원칙 (Push-Pull, 탈억제) 확인
- 선천적 본능의 시냅스 가중치 표현 검증
- PyGeNN SNN 프레임워크 확립

---

## Phase 2-3: 느끼는 뇌 (Affective Brain) - Phase 3b 완료

> **Phase 2a:** [docs/PHASE2A_DESIGN.md](docs/PHASE2A_DESIGN.md), [docs/PHASE2A_RESULTS.md](docs/PHASE2A_RESULTS.md)
> **Phase 2b:** [docs/PHASE2B_DESIGN.md](docs/PHASE2B_DESIGN.md), [docs/PHASE2B_RESULTS.md](docs/PHASE2B_RESULTS.md)
> **Phase 3:** [docs/PHASE3_DESIGN.md](docs/PHASE3_DESIGN.md), [docs/PHASE3_RESULTS.md](docs/PHASE3_RESULTS.md)

### Phase 로드맵

```
╔═══════════════════════════════════════════════════════════════╗
║  Forager Brain Project - 변연계 구현                          ║
╠═══════════════════════════════════════════════════════════════╣
║  2a: 시상하부 - 항상성 ✓ 완료 (2025-01-28)                   ║
║      └── Hunger/Satiety 드라이브, Energy 기반 행동 조절      ║
║                                                               ║
║  2b: 편도체 - 공포 회피 ✓ 완료 (2025-01-28)                  ║
║      └── Pain Zone 회피, Hunger-Fear 경쟁, 생존율 50%        ║
║                                                               ║
║  3a: 해마 - Place Cells ✓ 완료 (2025-01-28)                  ║
║      └── 20x20 Place Cells, Food Memory 기본 구조            ║
║                                                               ║
║  3b: 해마 - Hebbian 학습 ✓ 완료 (2025-01-29)                 ║
║      └── 음식 위치 기억 학습, 생존율 60%                     ║
║                                                               ║
║  3c: 해마 - 방향성 Food Memory ✓ 완료 (2025-01-30)           ║
║      └── Food Memory Left/Right 분리, 생존율 55%             ║
║                                                               ║
║  4: 기저핵 - Dopamine ✓ 완료 (2025-01-30)                    ║
║      └── Striatum + Direct/Indirect pathways, 생존율 40%     ║
║      └── 랜덤 환경에서 즉각적 향상 없음 (습관 학습 특성)     ║
║                                                               ║
║  5: 전전두엽 - Executive Function ✓ 완료 (2025-01-31)        ║
║      └── Working Memory + Goal Units + Inhibitory Control    ║
║      └── 목표 지향 행동, 충동 억제, 의사결정                 ║
║                                                               ║
║  6a: 소뇌 - Motor Coordination ✓ 완료 (2025-01-31)           ║
║      └── Granule + Purkinje + Deep Nuclei + Error Signal     ║
║      └── 운동 조정, 오류 기반 학습, 생존율 80%               ║
║                                                               ║
║  6b: 시상 - Sensory Gating & Attention ✓ 완료 (2025-01-31)   ║
║      └── Food Relay + Danger Relay + TRN + Arousal           ║
║      └── 감각 게이팅, 선택적 주의, 각성 조절                 ║
║                                                               ║
║  7: 통합 테스트 & 환경 조정 ✓ 완료 (2025-01-31)              ║
║      └── 가중치 강화 + 환경 조정 (음식↑, 시야↑, 감소율↓)    ║
║      └── 생존율: 10% → 80%, Reward Freq 2.77% ✓             ║
║                                                               ║
║  8: V1 시각 피질 ✓ 완료 (2025-02-01)                         ║
║      └── V1_Food L/R + V1_Danger L/R (400 뉴런)              ║
║      └── Lateral Inhibition, 방향 정보 보존                  ║
║                                                               ║
║  9: V2/V4 고차 시각 피질 ✓ 완료 (2025-02-01)                 ║
║      └── V2_Edge_Food/Danger (300 뉴런) + V4_Object (300 뉴런)║
║      └── 물체 분류, WTA 경쟁, Novelty Detection              ║
║      └── Top-Down 주의 조절 (Hunger/Fear/Goal → V2/V4)       ║
║                                                               ║
║  10: IT Cortex (측두엽) ✓ 완료 (2025-02-01) ★ M1 마일스톤   ║
║      └── IT_Food/Danger/Neutral (550 뉴런)                   ║
║      └── IT_Association (200) + IT_Buffer (250)              ║
║      └── "음식/위험" 범주 표상, 기억 통합                    ║
║      └── 총 10,000 뉴런 달성!                                ║
║                                                               ║
║  11: 청각 피질 (Auditory Cortex) ✓ 완료 (2026-02-01)        ║
║      └── Sound_Danger/Food L/R (400 뉴런)                    ║
║      └── A1_Danger/Food (300 뉴런) + A2_Association (200)    ║
║      └── 청각-공포 조건화, 다감각 통합 기반                  ║
║      └── 총 10,900 뉴런 달성!                                ║
║                                                               ║
║  12: 다중 감각 통합 (Multimodal STS) ✓ 완료 (2026-02-01)    ║
║      └── STS_Food/Danger (400 뉴런)                          ║
║      └── STS_Congruence/Mismatch (250 뉴런)                  ║
║      └── Multimodal_Buffer (150 뉴런)                        ║
║      └── 시청각 일치/불일치 감지, 다감각 통합                ║
║      └── 총 11,700 뉴런 달성!                                ║
╚═══════════════════════════════════════════════════════════════╝
```

### 현재 상태: Phase 12 완료 - 다중 감각 통합 (11,700 뉴런)

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 12: Multimodal Integration (STS) ✓                     ║
╠═══════════════════════════════════════════════════════════════╣
║  신규 구조:                                                   ║
║    - STS_Food: 200 뉴런 (음식 시청각 통합)                   ║
║    - STS_Danger: 200 뉴런 (위험 시청각 통합)                 ║
║    - STS_Congruence: 150 뉴런 (일치 감지)                    ║
║    - STS_Mismatch: 100 뉴런 (불일치 감지)                    ║
║    - Multimodal_Buffer: 150 뉴런 (다중 감각 기억)            ║
║                                                               ║
║  연결:                                                        ║
║    - IT → STS (시각 입력)                                    ║
║    - A1/A2 → STS (청각 입력)                                 ║
║    - STS → Hippocampus (다중 감각 기억)                      ║
║    - STS → Amygdala (통합 공포)                              ║
║    - STS → Motor (통합 행동)                                 ║
║    - STS → PFC (목표/작업 기억)                              ║
║    - Top-Down: Hunger/Fear/WM → STS                          ║
║                                                               ║
║  성능:                                                        ║
║    - Reward Freq: 3.24% ✓                                    ║
║    - Pain Avoidance: 90.7% ✓                                 ║
║    - Avg Food: 92.4                                          ║
║    - Hebbian Learning: avg_w 2.0 → 3.33 ✓                   ║
║                                                               ║
║  뉴런 수:       10,900 → 11,700 (+800)                       ║
╚═══════════════════════════════════════════════════════════════╝
```

### Phase 8 완료: V1 시각 피질

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 8: Primary Visual Cortex (V1) ✓                        ║
╠═══════════════════════════════════════════════════════════════╣
║    - V1_Food_Left/Right: 각 100 뉴런 (음식 시각 처리)        ║
║    - V1_Danger_Left/Right: 각 100 뉴런 (위험 시각 처리)      ║
║    - Lateral Inhibition: 좌우 대비 강화                      ║
╚═══════════════════════════════════════════════════════════════╝
```

### Phase 7 결과 (이전)

```
╔═══════════════════════════════════════════════════════════════╗
║  Phase 7 최종 결과                                             ║
╠═══════════════════════════════════════════════════════════════╣
║  생존율:        80% ✓ (목표 40% 초과)                         ║
║  Reward Freq:   2.77% ✓ (목표 2.5% 초과)                      ║
║  Pain Avoidance: 91.7% ✓                                      ║
║  Avg Food:      82.5 (초기 12.8, +545%)                       ║
║  뉴런 수:       8,000                                         ║
╚═══════════════════════════════════════════════════════════════╝
```

### Forager Brain 아키텍처 (5,800 뉴런)

```
┌─────────────────────────────────────────────────────────────┐
│                    Forager Brain v3a                        │
├─────────────────────────────────────────────────────────────┤
│  Sensory (1,800)                                            │
│    Food Eye L/R (800) + Wall Eye L/R (400)                  │
│    Pain Eye L/R (400) + Danger (200)                        │
│                                                             │
│  Hypothalamus (1,400) - Phase 2a                            │
│    LowEnergy (200) + HighEnergy (200)                       │
│    Hunger (500) ↔ Satiety (500) [WTA]                       │
│                                                             │
│  Amygdala (1,000) - Phase 2b                                │
│    LA (500) → CEA (300) → Fear (200)                        │
│    Pain → LA (US), Danger → LA (CS)                         │
│    Hunger ↔ Fear 경쟁                                       │
│                                                             │
│  Hippocampus (600) - Phase 3a                               │
│    Place Cells (400, 20x20) → Food Memory (200)             │
│    [학습 미구현 - Phase 3b에서]                             │
│                                                             │
│  Motor (1,000)                                              │
│    Motor L (500) ↔ Motor R (500) [WTA]                      │
└─────────────────────────────────────────────────────────────┘
```

### Food Patch 환경 (Hebbian 학습 검증용)

```
┌────────────────────────────────────────┐
│                400x400                 │
│    ┌──────┐              ┌──────┐     │
│    │Patch1│              │      │     │
│    │(100, │              │      │     │
│    │ 100) │              │      │     │
│    └──────┘              │      │     │
│              ┌──────┐    │      │     │
│              │ Nest │    │      │     │
│              └──────┘    │      │     │
│                          │Patch2│     │
│                          │(300, │     │
│                          │ 300) │     │
│                          └──────┘     │
└────────────────────────────────────────┘

- 음식의 80%가 Patch 내에 생성
- 한 번 찾으면 반복 방문 가능
- Place Cells 학습이 실제로 도움됨
```

**실행 명령어:**
```bash
# Baseline (학습 OFF, Patch 없음)
python forager_brain.py --episodes 20 --no-learning --render none

# Food Patch + Learning (검증)
python forager_brain.py --episodes 20 --food-patch --render none
```

**성공 기준:**
- Patch 방문 증가: 초반 5에피소드 vs 후반 5에피소드 → 30% 이상 증가
- avg_weight 증가: 초기 2.0 → 3.0 이상
- 생존율 유지: 80% 이상

### 핵심 발견 (교훈)

1. **방향성 Push-Pull이 핵심**: 무방향 Fear → 방향성 Pain Push-Pull로 변경 시 Pain Death 0%
2. **MOTOR DEAD 해결**: Satiety 억제 완화 (-8 → -4)
3. **학습 없는 Hippocampus는 노이즈**: Food Memory 가중치 최소화 필요
4. **랜덤 환경에서 위치 기억은 무의미**: Food Patch 환경 필요 (위치-음식 인과관계)

### 실행 명령어

```bash
# Forager Brain 테스트 (WSL)
wsl -d Ubuntu-24.04 -- bash -c "
export CUDA_PATH=/usr/local/cuda-12.6;
export PATH=\$CUDA_PATH/bin:/usr/local/bin:/usr/bin:/bin;
export LD_LIBRARY_PATH=\$CUDA_PATH/lib64;
source ~/pygenn_wsl/bin/activate;
cd ~/pygenn_test;
python /mnt/c/.../backend/genesis/forager_brain.py \
    --episodes 20 --log-level normal --render none
"

# 시각화 모드
python forager_brain.py --episodes 3 --render pygame
```

### 구현 파일

```
backend/genesis/
├── forager_gym.py     # ForagerGym 환경 (Pain Zone 포함)
├── forager_brain.py   # Forager Brain (Hypo+Amyg+Hippo)
└── checkpoints/
    └── forager_hypothalamus/  # NEW: Phase 2a 체크포인트
```

### WSL 실행 명령어 (Phase 2a)

```bash
# Phase 2a 훈련
wsl -d Ubuntu-24.04 -e bash -c "
export CUDA_PATH=/usr/local/cuda-12.6
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
python <PROJECT_PATH>/backend/genesis/forager_brain.py --episodes 20 --render pygame
"
```

---

## Phase 1 완료: Slither.io 졸업

> **상세 보고서:** [docs/SLITHER_GRADUATION.md](docs/SLITHER_GRADUATION.md)

### Phase 1 최종 결과

```
╔═══════════════════════════════════════════════════════════════╗
║  SLITHER.IO PROJECT - GRADUATED (v40b)                        ║
╠═══════════════════════════════════════════════════════════════╣
║  Best Length: 64    │  Avg: 37.6    │  Kills: 0.44/ep         ║
║  검증 완료: Push-Pull, Disinhibition, WTA, 선천적 본능        ║
╚═══════════════════════════════════════════════════════════════╝
```

### Slither.io 버전 히스토리 (요약)

> **상세 내용:** [docs/SLITHER_GRADUATION.md](docs/SLITHER_GRADUATION.md)

**핵심 마일스톤:**
- v28c: Push-Pull 반사 회로 완성 (Baseline)
- v31: Disinhibition 발견 (돌파구)
- v32c: First Kill 달성
- v37f: Defensive Kill 메커니즘 규명
- v40b: 시간 제한 해제 → 진정한 성능 발현 (졸업)

**핵심 교훈:**
1. 양쪽 모터 동시 활성화 → 신호 상쇄 (실패)
2. Fear 완전 상쇄 → 자살 공격 (실패)
3. R-STDP는 빈번하고 즉각적인 보상 필요
4. 환경의 시간 제한이 성능 병목일 수 있음

### 진화 경로

| 단계 | 환경 | 뉴런 | 성과 |
|------|------|------|------|
| 1단계 | Chrome Dino | 3,600 | High: 725 (졸업) |
| 2단계 | Slither.io snnTorch | 15,800 | High: 57 (적 3마리) |
| 3단계 | Slither.io PyGeNN | 158,000 | High: 16 (적 7마리) |
| v19 | PyGeNN + 음식추적 | 13,800 (dev) | High: 10 |
| v28c | Push-Pull 회피 | 13,800 (dev) | Best: 27~37, Avg: 15~19 |
| v32c | First Kill | 13,800 (dev) | 4 Kills (100 ep) |
| v37f | Defensive Kill | 13,800 (dev) | Best: 30, Kills: 6 (200 ep) |
| **v40b** | **Uncapped (3000)** | **13,800 (dev)** | **Best: 64, Avg: 37.6, 0.44 kills/ep!** |

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

## PyGeNN 아키텍처 (Phase 1 - Slither.io v40b)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Food Eye    │  │  Enemy Eye   │  │  Enemy Head  │  │   Body Eye   │
│  L/R 분리    │  │  L/R 분리    │  │  L/R 분리    │  │   L/R 분리   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│Hunger Circuit│  │ Fear Circuit │  │ Hunt Circuit │  │ Wall Avoid   │
│   (Food)     │  │ (Push-Pull)  │  │ (Disinhibit) │  │ (Push-Pull)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └────────────────┬┴─────────────────┴─────────────────┘
                        ▼
                   ┌─────────┐
                   │  Motor  │
                   │  L / R  │
                   └────┬────┘
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
┌───────┐          ┌───────┐          ┌───────┐
│ Left  │◄─────────│  WTA  │─────────►│ Right │
└───────┘          └───────┘          └───────┘

┌─────────────────────────────────────────────────┐
│           INNATE REFLEX v37f (Static)           │
├─────────────────────────────────────────────────┤
│ [Fear - 약화된 회피]                            │
│ Enemy_L → Motor_R (PUSH +80)                    │
│ Enemy_L → Motor_L (PULL -60)                    │
│                                                 │
│ [Hunt - 강화된 사냥]                            │
│ EnemyHead_L → Motor_L (HUNT +180)               │
│                                                 │
│ [Disinhibition - Fear 상쇄]                     │
│ EnemyHead_L → Motor_R (DISINHIBIT -100)         │
│ EnemyHead_L → Motor_L (RELEASE +80)             │
│                                                 │
│ [Wall/Food - 기존 유지]                         │
│ Body_L → Motor_R (PUSH +80)                     │
│ Food_L → Motor_L (IPSI +20)                     │
└─────────────────────────────────────────────────┘
```

### 선천적 반사 회로 (Innate Reflex v37f)

```python
# 1. Fear (약화) - 적 body 회피
Enemy_L → Motor_R (PUSH +80)   # 반대편 활성화
Enemy_L → Motor_L (PULL -60)   # 같은편 억제

# 2. Hunt - 적 head 추적 (동측)
EnemyHead_L → Motor_L (HUNT +180)  # 사냥 본능

# 3. Disinhibition - 사냥 시 Fear 상쇄
EnemyHead_L → Motor_R (DISINHIBIT -100)  # Fear Push 상쇄
EnemyHead_L → Motor_L (RELEASE +80)      # Fear Pull 상쇄

# 4. 결합 효과
# 적 body만: Fear(80) → 회피
# 적 head도: Fear(80) - Disinhibit(100) + Hunt(180) = +160 → 돌진!

# 5. 음식/벽 회피 (기존 유지)
Food_L → Motor_L (IPSI +20)
Body_L → Motor_R (PUSH +80)
```

---

## 파일 구조

```
backend/
├── genesis/
│   ├── # Phase 2a: Forager (현재 개발 중)
│   ├── forager_gym.py                # Phase 2a 환경 (NEW)
│   ├── forager_brain.py              # Phase 2a 뇌 - 시상하부 (NEW)
│   │
│   ├── # Phase 1: Slither.io (졸업)
│   ├── slither_pygenn_biological.py  # v40b PyGeNN 에이전트 (졸업)
│   ├── slither_gym.py                # Slither.io 환경
│   ├── gpu_monitor.py                # GPU 온도/메모리 모니터링
│   │
│   ├── # Legacy (참조용)
│   ├── slither_snn_agent.py          # snnTorch 에이전트
│   ├── dino_dual_channel_agent.py    # Chrome Dino (725점, 졸업)
│   │
│   └── checkpoints/
│       ├── forager_hypothalamus/     # Phase 2a 체크포인트 (NEW)
│       ├── slither_pygenn_bio/       # Phase 1 체크포인트
│       └── slither_snn/              # snnTorch 체크포인트
│
├── docs/
│   ├── PHASE2A_DESIGN.md             # Phase 2a 설계 문서 (NEW)
│   └── SLITHER_GRADUATION.md         # Phase 1 졸업 보고서
│
└── requirements.txt
```

---

## 실행 방법

### PyGeNN Slither.io (WSL 필수!)

```bash
# WSL에서 실행 (필수!)
wsl -d Ubuntu-24.04

# 환경 설정
export CUDA_PATH=/usr/local/cuda-12.6
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test

# 훈련 (headless)
python <PROJECT_PATH>/backend/genesis/slither_pygenn_biological.py \
    --dev --episodes 100 --enemies 5 --render none

# 시각화 모드
python <PROJECT_PATH>/backend/genesis/slither_pygenn_biological.py \
    --dev --episodes 10 --enemies 3 --render pygame

# 모드 옵션
# --dev   : 13,800 뉴런 (디버깅용)
# --lite  : 50,000 뉴런 (중간)
# (없음)  : 158,000 뉴런 (전체)
```

### 한 줄 실행 (PowerShell에서)

```powershell
wsl -d Ubuntu-24.04 -e bash -c "export CUDA_PATH=/usr/local/cuda-12.6 && source ~/pygenn_wsl/bin/activate && cd ~/pygenn_test && python <PROJECT_PATH>/backend/genesis/slither_pygenn_biological.py --dev --episodes 5 --enemies 3 --render pygame"
```

---

## 핵심 발견 & 교훈

### v27i 세션 교훈 (CRITICAL!)

1. **절대좌표 vs 상대각도**: `(target_x, target_y)` 절대좌표는 뱀의 현재 위치/방향과 무관하게 화면상의 점을 향함!
   - 뱀이 오른쪽에 있고 왼쪽을 향할 때 `target_x=0.9`면 뒤로 돌아서 적에게 돌진!
   - **해결**: `(angle_delta, boost)` 상대 회전 출력 사용

2. **Gym 출력 포맷**: 두 가지 지원
   - `(target_x, target_y, boost)` - 절대 화면 좌표 (문제!)
   - `(angle_delta, boost)` - 상대 회전 각도 (정답!)

3. **결과**: Best 14→27, Avg 6→12.5 (2배 향상!)

### v26 세션 교훈

1. **전압이 아닌 스파이크 누적**: 전압(V)은 스파이크 후 Vreset으로 리셋됨 → 매 스텝 RefracTime으로 스파이크 누적 필수
2. **신호 체인 전체 확인**: Enemy→Motor 경로가 작동해도, 마지막 디코딩이 잘못되면 출력 0
3. **Push-Pull 효과 확인됨**: Enemy 스파이크 시 Motor_R 100% 활성, Motor_L 강하게 억제 (-11000mV)

### v19 세션 교훈 (이전)

1. **WTA wMax 클리핑**: `wMax=0.0`이면 모든 가중치가 0으로 클리핑됨
2. **cos는 짝수함수**: `cos(-x)=cos(x)` 라서 좌우 구분 불가
3. **RefracTime 카운팅**: `>0` 대신 `>threshold`로 새 스파이크만 카운트
4. **대칭 경로 상쇄**: 대칭 경로가 교차배선 효과를 상쇄함 → 비활성화
5. **WSL 환경 사용**: Windows PyGeNN 대신 WSL PyGeNN 사용

### 생물학적 원칙

1. **빈 서판은 죽음**: 선천적 본능 = 시냅스 초기 가중치 (if문 아님)
2. **교차 배선 (Contralateral)**: 적 회피 - 반대편으로 회전
3. **동측 배선 (Ipsilateral)**: 음식 추적 - 같은 편으로 회전
4. **Fight-or-Flight**: Fear↔Attack 상호 억제로 행동 선택

### 디버깅 팁

```python
# v26: 스파이크 누적 카운팅 (CRITICAL!)
# 전압(V)으로 활성도 측정하면 안 됨 - 스파이크 후 Vreset으로 리셋!
# 반드시 시뮬레이션 루프 내에서 매 스텝마다 스파이크 누적해야 함
spike_threshold = tau_refrac - 0.5  # 1.5 (새 스파이크 감지)
for _ in range(10):
    model.step_time()
    motor_left.vars["RefracTime"].pull_from_device()
    left_spike_count += np.sum(motor_left.vars["RefracTime"].view > spike_threshold)
left_rate = left_spike_count / (n_neurons * 5)  # max 5 spikes per neuron (10ms/2ms)

# 모터 출력 확인
print(f"Motor: L={left_rate:.2f} R={right_rate:.2f} | Turn: {turn_delta:+.3f}")

# 입력 신호 확인
print(f"Enemy: L={enemy_l:.2f} R={enemy_r:.2f} | Food: L={food_l:.2f} R={food_r:.2f}")
```

---

## 레거시 코드

FEP, Predictive Coding, E6-E8 실험 등 이전 개발물은 `archive/legacy` 브랜치에 보존됨.

---

> "뇌의 근본적인 작동 원리가 모든 행위, 생각, 감정의 근원이 되어야 한다"
