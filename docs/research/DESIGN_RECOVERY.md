# 설계 복원 — 소실된 5~7월 회로 (2026-07-21)

디스크 사고로 소실된 브레인 기능(79KB 델타)의 **설계 의도**를 코드 재구현 전에 복원한다.
증거 등급을 3단계로 구분한다:

- 🟢 **증거 확실** — 생존 스크립트 주석이 메커니즘·목표·결과를 직접 서술
- 🟡 **생존 구현** — 4월판 코드에 구현이 남아 있음 (읽어서 확정)
- 🔴 **증거 희박** — CLI 플래그명 + 빈 실험 스크립트뿐. 사용자 기억 필요

---

## A. biletaxis / v3 항법 스택 🟢

biletaxis는 독립 기능이 아니라 **v3 place_pref 과제 스택 위의 최상층**이다.
선행 스택(전부 소실): `place_pref` 과제 + `zone-circle` 환경 + `v3-klino` 항법.

### A1. 과제 환경 (v3 place_pref)
- **`--task place_pref`**: 원형 goal zone으로 항법해 그 안에 체류하는 과제. 지표 = `goal-dist`(목표까지 거리), `mean_cool_dwell_ratio`(zone 내 체류율).
- **`--zone-circle`**: 원형 구역 (기본 rich zone 대신).
- **`--appetitive-place`**: 구역이 유인성(좋음). 없으면 aversive(열 구역=나쁨) → 회피 과제.
- **`--start-far`**: 목표에서 멀리 출발.
- **`--sparse-reward`**: 희소 보상.
- **`--replay-to-klino`**: SWR 리플레이가 klinotaxis로 투사.
- **`--n-food 0`**: 순수 항법 과제 (음식 없음). `--n-food 10`: forage 반사와 공존 테스트.

### A2. biletaxis 핵심 메커니즘 🟢
> "biletaxis = 고value(안전)쪽 조향. align = 명령 turn 부호가 실제 목표방향과 맞는 비율."

- **양측 비교 조향(bilateral taxis)**: 학습된 value 지도의 좌/우 차이로 조향 방향 결정. klinotaxis 계열.
- **`biletaxis-align` 지표**: 명령한 turn 부호가 실제 목표방향과 일치하는 스텝 비율. **>0.5 견고 = value 지도가 목표를 실제로 가리킴**(방향 신호가 진짜). ≈0.5 = 신호 없음(깨끗한 음성). lesson #66.
- **`--biletaxis-gain 0.5`**: 조향 이득. gain 1.0은 과조향 → orbit/dwell 붕괴(seed0). 0.5로 낮춰 과조향 제거.

### A3. 정착(settle) 메커니즘 🟢 — lesson #43/#49 돌파
목표에 도달해도 orbit(공전)/오버슈트하는 문제. 두 해법:
- **`--biletaxis-brake`**: 고value 구역에서 감속 → 정착. 결과: brake<OFF(거리) & brake>OFF(체류), 5-seed 3/3~5/5 승. "read-out 행동수준 돌파 확정".
- **`--biletaxis-settle`**: 목표 근처에서 조향 감쇠 → orbit 대신 정착. brake의 대안/보완.

### A4. Arbitration (forage vs 목표항법 공존) 🟢 — lesson #56/#59/#61
- 문제: brake 감속이 forage(음식 반사)를 방해 → step수 급감 (`biletaxis_nobrake_food.sh` 진단).
- **`--biletaxis-hunger-gate`**: 목표항법을 satiety로 게이팅(배부를 때만) → forage(반사)와 목표항법(planning)을 상태별로 분리. lesson #61 arbitration.

### A5. Factored value 🟢 — confound 제거
- 문제: place-value가 음식 DA에 오염 → 목표 gradient가 food-baseline에 묻힘.
- **`--place-value-food-exclude`**: place-value가 음식 DA 제외(장소고정 보상만) → goal-gradient 회복. align 회복하면 factoring 작동.
- **`--value-max` (vmax)**: 천장 포화 가설. vmax↑ → 값이 퍼져 goal이 food-baseline 위로. 1/2/3 스캔.

### A6. 다능력 통합 🟢 — lesson #64/#65
- **`multicap`**: 한 뇌가 냄새변별(`--v3-olf`) + 목표항법(biletaxis+factored) 동시. PI(변별)>0 + align(항법) 높음 둘 다 서면 코히어런트 다능력 뇌.
- 대조(항법만): align 0.821, PI -0.16, good 336 — 이미 확보된 수치.

### A7. 접근/회피 통합 🟢 — lesson #67
- 같은 메커니즘(지도→방향→brake)이 접근(appetitive)과 회피(aversive/열) 둘 다. aversive는 `--appetitive-place` 빼고 `--start-far`.

---

## B. M4 컨텍스트 게이트 🟡 — 4월판 생존, 프런티어의 토대

`forager_brain.py`에 온전히 구현됨. GitHub 마지막 커밋 = "M4 v9: Context hard gate — first context-dependent selectivity breakthrough".

- **KC → D1_ctx**: Kenyon Cell sparse expansion → 컨텍스트별 D1 선조체 집단. 컨텍스트 a/b 각각 별도 (`kc_to_d1ctx_a_l/r`, `kc_to_d1ctx_b_l/r`).
- **컨텍스트별 가중치 스냅샷**: `_ctx_a_*` / `_ctx_b_*` — 컨텍스트 전환 시 스왑.
- **Hard gate** (`_activate_context_hard_gate`, `context_hard_gate_enabled`): 컨텍스트-무관 food→D1 경로를 억제 → D1_ctx가 인계. M4 v9 돌파의 핵심.
- **`update_context_food_scales(food_type, da_magnitude)`** (v9b): 음식 먹는 순간 컨텍스트별 접근 가중치 갱신.
- 환경: `forager_gym.py` `context_rules_enabled` — Zone A(정상)/Zone B(good↔bad 반전), `agent_x > width/2`에서 `effective_type = 1 - food_type`.
- **한계**: 4월 시점 M4 selectivity가 0.50(우연)에 6회 막힘 → v9 hard gate로 첫 돌파. `--context-select`는 이 위에서 바로 구동(재구현 불필요, 검증 완료: 2ep PI 0.17→0.25).

---

> **설계 권한 (2026-07-21 확정)**: 이 회로들은 원래 **어시스턴트(이전 세션의 Claude)가 설계·구현**했다.
> 사용자는 목표·방향을 정하는 연구 디렉터이지 회로 내부 설계자가 아니다. 따라서 🔴 항목은
> **사용자 문답으로 메우지 않는다** — 증거(스크립트 주석·4월판 코드·생물학적 제약·최종 목표)로
> 이전 인스턴스가 했던 대로 **어시스턴트가 재유도**한다. 사용자 확인: "니가 다 설계하고 진행했는데".
> 재유도한 설계는 이 문서에 근거와 함께 기록해 다음 세션이 이어받게 한다.

## C. 시퀀스 / 조합 프런티어 🔴 — 증거 희박, 재유도 핵심

CLI 플래그명 + grep 마커 + 빈 실험 스크립트뿐. 주석 없음. **여기가 7/12 미해결로 끊긴 최전선.**

### C1. 시퀀스 학습/WM
- **`--seq-task`**: A→B 순서 학습 과제. 마커 `최종순서율`, `in-order B 비율`(초기 0.25에서 상승 = 순서 학습). ablation: `--v3-value-eta 0`으로 place→value 학습 OFF.
- **`--seq-nav`**: 시퀀스 항법. 마커 `seq-nav 정렬`.
- **`--seq-wm`**: 창발 워킹메모리. 마커 `WM latch`. ← **잃어버린 핵심**.
- **`--seq-gain`**: 시퀀스 조향 이득.
- 🔴 **미상**: WM latch의 신경 기질(어느 집단이 latch? 4월판 `working_memory` 200뉴런 재사용?), seq-nav 정렬 계산법, 순서율 정의.

### C2. 조합적 컨텍스트
- **`--context-compositional`**: M4 컨텍스트를 조합(compositional)으로 확장. `--zone-cx 0.3 --zone-cy 0.3`로 2D 구역.
- 마지막 온전한 커맨드(comp_wm2.sh): `--task integrated --context-select --seq-task --seq-wm --context-compositional --zone-cx 0.3 --zone-cy 0.3`. 주석 "조합+창발WM(seq-nav 없이)".
- 🔴 **미상**: "조합"이 정확히 무엇인가 — 컨텍스트 2개 초과? 요인 분해(zone x cue)? M4의 a/b 게이트를 어떻게 조합으로 일반화? 7/12 결과 PI~0.03(무선택)에서 무엇이 실패했나.

### C3. 중재/기타 🔴
- **`--wta-arbitration` / `--wta-cue-bid`**: WTA 기반 중재, 단서 입찰. `serial_rev`(8ep마다 cue 반전)에서 사용. 마커 `serial-cv`.
- **`--cue-reversal` / `--cue-reversal-period`**: 단서 역전.
- **`--thermal-reversal` / `--v3-recovery`**: 열 반전 후 회피 회복(value 재학습).

---

## 재유도 설계 로그 (어시스턴트, 증거 기반)

### D1. v3 place_pref 과제 스택 — 러너 레이어로 재구현 (2026-07-21)
**근거**: 4월판 gym에 task_mode 개념 없음 + 원본 run_v2_tasks가 gym 위 별도 하네스였음(엔트리포인트 증거).
→ 163KB gym을 건드리지 않고 러너에 과제 레이어를 얹는다. gym이 제공하는 것: obs `position_x/y`(정규화),
`env.agent_x/y`, `env.width/height`, `env.steps`.

**설계**:
- **goal zone**: 원형, 중심 `(zone_cx, zone_cy)`(정규화 0~1, 기본 0.5/0.5), 반경 `zone_r`(기본 0.12 = rich_zone_radius/width 근사).
- **`--start-far`**: 에피소드 시작 시 zone 반대편 코너에 배치 (reset 후 `env.agent_x/y` 오버라이드).
- **`--n-food 0`**: `env_config.n_food=0`으로 순수 항법.
- **`--appetitive-place`**: zone=목표(양성). 없으면 aversive(회피, dwell 낮을수록 좋음).
- **지표 (매 스텝 계산)**:
  - `goal-dist`: 정규화 거리 `hypot(px-cx, py-cy)`, 에피소드 평균.
  - `cool_dwell_ratio`: zone 내부 스텝 비율 (`dist < zone_r`). aversive면 "회피율" 해석.
  - `biletaxis-align`: 명령 turn 부호 vs 목표방향 부호 일치율 → **brain steering 필요**, biletaxis 구현 시.
- 출력 마커: `goal-dist:`, `mean_cool_dwell_ratio:`, `biletaxis-align:`(biletaxis ON일 때만).

이 레이어는 신경 회로가 아니라 과제 계측이므로 🟢 안전. biletaxis(조향 회로)는 이 위에서 별도.

### D1-검증. place_pref 널 baseline 확보 (2026-07-21)
항법 회로 전무(klino/biletaxis 없음) 상태 2ep 실측:
- `goal-dist: 0.3427`, `dwell 0.0745`.
- 무작위 보행 기대 체류 ≈ π·r²/맵 ≈ π·0.12² ≈ 0.045. → **0.07은 우연 수준**(항법 없음 확인).
- 이 값이 biletaxis/klino가 넘어서야 할 **널 baseline**. (원칙: [[measure-null-variance-before-separation]])
- 주: `--n-food 0`은 에피소드가 step~450에서 조기 종료(에너지 고갈). OFF-vs-ON 상대 비교엔 무방.

### D2. biletaxis 양측 조향 — 설계 (근거: 4월판 place→value 생존)
**생존 기질** (4월 forager_brain.py):
- `place_cells` 400뉴런(20×20) + `place_to_value_eta`(DA-gated 3-factor) + `place_to_value_w_max`.
  → **place→value 지도는 이미 학습된다.** 소실된 건 read-out.
**소실 = 양측 read-out**: 학습된 value 지도에서 현재 위치 기준 좌/우 방향의 value를 비교 → 높은 쪽으로 조향.
**설계 (재유도)**:
- 매 스텝 에이전트 heading 기준 좌/우 약간 회전한 지점의 place-value를 추정(place_cells 활성 또는 value pop read).
- `Δturn = gain · (V_left − V_right)`, `gain=0.5`(gain 1.0은 과조향으로 dwell 붕괴 — A2 증거).
- `align`: sign(Δturn)이 실제 목표방향 부호와 일치하는 스텝 비율. >0.5 = 지도가 목표 가리킴.
- brake: |V_gradient| 작고 V 높으면(=목표 근처) 감속. settle: 목표 근처 gain 감쇠.
- 🔴 **미상**: 좌/우 value 추정을 SNN 내부에서(별도 뉴런 집단) vs 러너에서(place_cells read-out) 했는지.
  원본 아키텍처상 조향은 뇌 기능이나, 최소 재유도는 러너 read-out으로 시작해 align>0.5 확인 후 회로화.

### D2-성공. biletaxis 재유도 완료 (2026-07-23) ✅
2겹 버그 수정 후 성공 (`a2b_on_seed0.json`):
- **align: 0.75** (last_5 0.76), 널 0.5 확실히 돌파. ep1 0.38→ep5 0.91 학습 곡선.
- **vmap_std: 0.004→0.347** 단조 상승 (value 지도 학습). align과 동반 상승 = 지도가
  zone 공간구조 인코딩 → 양측 read-out이 올바른 조향으로 변환. lesson #66 재현.
- zrew/dwell 후반에도 유지(부호 오류 땐 0으로 붕괴) = 조향이 zone 도달을 실제로 도움.
**첫 소실 신경 회로 복원 완료.** 수정 2건은 아래 D2-디버깅 참조.

### D2-디버깅 기록 (2026-07-21~23) — 재유도 함정 2건
진단 계측(zrew=zone보상수, vmap_std=지도분산)이 병목을 순차 지목:
1. **1차 실패**: align 0.00, 지도 평평. 원인=SWR 학습경로 미호출.
   → 보상시 `add_experience`, 에피소드끝 `replay_swr` 배선. 그래도 vmap_std 0.
2. **2차**: zrew>0(보상O)인데 vmap_std=0(학습X). 원인=`transition_buffer`가
   "음식 가시성"에 게이팅(forager_brain 10918)돼 n_food 0에선 안 채워짐 →
   value 역backup(replay_swr 9817, `len(transition_buffer)>0` 게이트) 스킵.
   → **v3 place_pref는 전이 트리거를 zone 근접으로 대체**. zone 보상시 `_record_transition`
   으로 복제. vmap_std 상승 시작.
3. **3차**: vmap_std↑(학습O)인데 align 0.13<<0.5(방향X). 원인=조향 부호 반전.
   gym `angle_delta>0=CCW=θ+δ`인데 왼쪽 샘플을 θ-δ서 뽑음. → CCW=θ+δ 정정.
**교훈**: 최소 러너로 13k줄 뇌를 구동할 때, 학습이 내부 기전(SWR replay+transition graph)에
숨어 있어 명시 배선 필요. 진단 계측 없이는 "align 왜 0"에서 못 벗어남.

### D2-폐기. biletaxis 1차 실험 실패 (2026-07-21)
OFF vs ON 25ep 실측 (`docs/research/rebuild_baseline/a2_{off,on}_seed0.json`):
- `biletaxis-align: 0.0000`, dwell ON 0.077 ≈ OFF 0.084 → **조향이 작동 안 함**.
- 원인: align 0.0000은 |Δturn|>1e-6 스텝이 0 = **V_L≈V_R 항상**(value 지도가 평평).
  = place→value 지도가 zone을 학습 못 함.
- 🔴 디버깅 순서(다음 세션): ① sparse-reward가 실제 DA 방출/learn_food_location을
  호출하는지(zone 진입 카운트 확인) ② place_to_value 가중치가 에피소드 경과로 변하는지
  (평평하면 학습률/보상강도 문제) ③ value read 스케일(가우시안 sigma 0.08이 20×20 격자에서
  적절한지). 널 0.07 못 넘으면 gain/look 튜닝은 무의미 — 지도 학습부터.

## 재구현 순서 (설계 확정 후)

증거 등급 = 재구현 안전도. A(🟢)부터 바텀업이 정석이나, 사용자 선택은 **설계 문서 우선 복구**.
→ 이 문서의 🔴 항목을 사용자 문답으로 메운 뒤, A2~A5 (biletaxis 코어) 부터 착수.
각 단계는 baseline 55%/2.63%에 대해 회귀 검증.
