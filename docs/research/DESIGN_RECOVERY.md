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

### C1. 시퀀스 학습/WM  🟡 — WM latch 기질 발견 (2026-07-25)
- **`--seq-task`**: A→B 순서 학습 과제. 마커 `최종순서율`, `in-order B 비율`(초기 0.25에서 상승 = 순서 학습). ablation: `--v3-value-eta 0`으로 place→value 학습 OFF.
- **`--seq-nav`**: 시퀀스 항법. 마커 `seq-nav 정렬`.
- **`--seq-wm`**: 창발 워킹메모리. 마커 `WM latch`.
- **`--seq-gain`**: 시퀀스 조향 이득.
- **WM latch 기질 = 4월판 생존** (이전 🔴 미상 → 확인): `working_memory` 200뉴런(3016) +
  `wm_recurrent` SPARSE 자기→자기 되먹임(3071, weight 8.0) + `working_memory_decay 0.98`
  = bistable 래치(상태 지속). `wm_to_goal_food`(3080)로 목표 구동. 읽기: `info["working_memory_rate"]`(11569).
  → 기억 기계는 존재. 소실된 건 그걸 쓰는 **순서 과제(A→B)**.
- **재유도 설계**: 존 2개(A/B), 정순 A→B. A 방문시 WM 래치 "A완료" 보유 → 목표 B 전환.
  러너가 working_memory_rate 읽어 목표 게이팅(biletaxis의 place_to_value read와 동형).
  지표: 최종순서율(A→B 정순 비율), WM latch(A후 wm_rate 상승·지속).

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

### D12. 최종 판정 (2026-07-25) — 순서학습 창발 실패 확정 (행동 지표로 결정)
WM 패턴 readout(막전위 상관) 시도 → 캡처 타이밍 버그(게이트가 보상 다음 스텝 열림,
캡처 순간 WM은 rest=균일, a_pat_std=0)로 pattern_corr=0. **더 안 판다.**
**이미 결정적 답이 있음**: 행동적 order_rate ≈0 (correct 3 vs wrong≈297, D11).
에이전트가 A→B 순서를 학습 못함 = 모호하지 않음. WM 측정 완벽 여부와 무관하게 결과 확정.

**seq-wm 프런티어 최종 결론**: 시도 전부 — 되먹임강도(D10b)·균형(D10c)·PBWM게이팅(D10d)·
부트스트랩(D11)·패턴측정(D12) — 일관되게 음성. 순서학습은 재유도 기전으로 **창발하지 않음.**
이유(규명됨): WM 래치가 분산 순서상태를 못 붙잡음 + explore/exploit 충돌. 7/12 원본 끊긴
지점의 실체. **억지 통과 안 시킴.** 남은 진짜 연구: dopamine-gated WM의 올바른 pattern 유지
(캡처타이밍·전입력 게이팅·explore 담금질) — 다음 세션 착수점, 정직한 열린 문제.

### D11. 부트스트랩 (2026-07-25) — 경험은 생겼으나 순서학습 창발 실패 (정직한 음성)
사용자 재지적("최종목표 이미 줬다, 니가 헤매는 답을 내가 일일이?") 수용 — 떠넘김 중단, 판단.
근본 블로커 재규정: WM 측정 아니라 **부트스트랩**(B 미방문→경험 없음→학습 없음, 최종목표 정의가 막힘).
ground-truth seek 금지 → 뇌 내재 기전: ① 존 도달=에너지 회복(생존→탐색시간, 순서=생존직결)
② curiosity_rate(신규성)로 분산 탐색.

결과 (`c1_boot2.json`, 40ep): correct(A→B) 0→**12**, steps ~450→841(ep19 3791). **경험 생성 ✅.**
그러나 order_rate 전반 0.001 → 후반 0.000, 상승 없음. ep5: correct=3인데 order=0.01 →
**wrong(B먼저)≈297** = curiosity 탐색이 무방향이라 A·B 무작위 들락, 순서 안 지킴. **순서학습 창발 ❌.**

**결론(내 판단)**: 부트스트랩으로 경험은 만들었으나 순서학습 실패. 두 원인:
① WM 래치 미작동(D10d) → 순서 강제할 기억 없음. ② explore/exploit 충돌 — 부트스트랩을
가능케 한 탐색이 동시에 순서 파괴. 둘 다 7/12 원본 끊긴 지점의 실체.
**seq-wm(프런티어)는 진짜 열린 연구문제. 억지로 통과 안 시킴.** A-트랙(🟢)은 완결 딜리버러블.

### D10. seq-task/seq-wm (2026-07-25) — WM 래치 미작동 ❌ (앞선 +0.42는 착시, 정정)
seq-wm OFF vs ON, 통합world 음식0, 30ep seed0 (`c1_{nowm,wm}.json`).

**1차(brake 트랩 있음, 오판)**: WM_latch +0.42~0.53로 "래치 작동"이라 판정. **틀림.**
그 +0.42는 brake가 에이전트를 A에 가둬 A의 place→WM 입력이 계속 들어간 **착시**였다
(래치가 아니라 지속 입력). correct 0(B 미도달)이라 순서완성은 애초에 안 됨.

**2차(brake 트랩 수정 = target-aware brake)**:

| 조건 | correct(A→B) | WM_latch | steps |
|---|---|---|---|
| seq-wm OFF(러너 플래그) | **10** | −0.245 | 513 |
| seq-wm ON(WM 래치) | **0** | −0.356 | 499 |

- **WM 래치 미작동 ❌ (정정)**: 트랩 제거로 에이전트가 A를 떠나자 WM_latch 음수(−0.25~−0.36)
  = A 떠나면 wm_rate 감쇠. **되먹임(weight 8.0)이 입력 없이 자활성 유지 못함 = bistable 아님.**
  앞선 +0.42는 트랩 착시. 창발 워킹메모리는 현 파라미터로 순서상태를 **못 붙잡는다.**
- seq-wm ON(래치 의존) 0 완성, OFF(신뢰 플래그) 10 완성 — 래치가 오히려 방해.
- **교훈**: 과장 주장(1차)을 후속 실험(트랩 수정)이 반증. 진짜 검증엔 교란요인(brake 트랩)
  제거가 필수. 이 프로젝트가 "하드코딩 금지·창발" 하드룰을 두는 이유의 실례.

### D10b. WM 되먹임 강도 스캔 (2026-07-25) — 강도 무효, 근본원인 규명
seq-wm ON, target-aware brake, 되먹임 weight ×1/4/8 (`d10b_g{4,8}.json`):

| gain | WM 되먹임 | WM_latch | correct |
|---|---|---|---|
| ×1 | 8.0 | −0.36 | 0 |
| ×4 | 32 | −0.24 | 4 |
| ×8 | 64 | −0.37 | 2 |

- **강도 무효**: ×8까지 WM_latch 계속 음수. bistable 래치 안 생김. 되먹임 강도가 원인 아님.
- **근본원인 규명**: `place_to_working_memory`(weight 10.0)가 매 스텝 **현재 위치** place cell로
  WM을 덮어씀 → WM은 "갔었던 곳"(유지)이 아니라 **"지금 있는 곳"(현재)** 추적. A 떠나면
  그곳 place cell이 A 기억 밀어냄. 되먹임이 강해도 이 현재-위치 입력에 짐.
- **다음 접근 (미구현)**: 순서 기억 = **입력 게이팅** 필요. A 보상 순간에만 WM 쓰고 이후
  place 덮어쓰기 차단(write-protect). 4월 코드에 기계 존재: `wm_gate`, `dopamine_to_wm_gate_weight`
  (도파민 게이팅 쓰기), `wm_update_gate`. 러너가 이걸 engage 안 함. → dopamine-gated WM 쓰기로
  A-state 래치·보호가 seq-wm의 진짜 재구현. **7/12 원본 끊긴 지점의 유력 후보.**

### D10c. WM 균형 스캔 + 측정 결함 발견 (2026-07-25)
되먹임×g & place드라이브÷g 양방향(g=3,6). WM_latch 여전히 음수(-0.13~-0.40).
**측정 결함 발견**: wm_pre=0.667(A 닿기 전에도 WM 높음). working_memory 집단이 탐색 내내
광범위 발화 = "A 기억"이 아니라 지나가는 place field 반응. **집단 발화율(wm_rate)로는
순서기억 격리 불가** — WM이 늘 켜져 있음. WM_latch 지표 자체가 confound.

**seq-wm 진짜 재구현에 필요한 것 (3실험 D10/b/c로 규명)**:
① **패턴 수준 WM 읽기**: A 담당 특정 뉴런들이 유지되나(집단율 아님). working_memory
   spike 벡터를 A시점 패턴과 상관으로 측정. GPU pull 필요, 큰 구현.
② **dopamine-gated write**: A 보상시만 그 패턴 쓰고 place 덮어쓰기 차단. wm_update_gate
   (도파민 구동, wm_thalamic 억제) engage. 직접 place→WM(10.0) 경로 게이팅 필요.
파라미터 튜닝(D10b/c)으론 불가. 근본적으로 다른 측정+회로 접근.

### D10d. PBWM dopamine-gated write 구현 (2026-07-25) — 기전 배선됨, 측정으로 판정불가
D10c의 "gated write" 다음접근을 실제 구현. place→WM을 뇌 자기 도파민([-1,1])으로 게이팅
(보상시 열림→적재, 감쇠→닫힘→유지). forager_brain: place_to_working_memory 핸들 저장 +
`gate_wm_input(open_frac)`. runner: seq-wm시 매 스텝 dopamine_level로 게이팅. baseline 무영향.

**구현 함정(정직 기록)**: 1차 ON==OFF 동일 → SPARSE 시냅스에 `.view` 써서 GeNN 무효
(API 오용, 과학적 실패 아님). `pull→.values[:]→push`로 수정 후 작동.

작동 후 (`c1_{nowm,wm}.json` 음식0 30ep): OFF wm_post=0.267 / ON(PBWM) 0.422 — 게이팅이
post-A WM 상승시킴(기전 작동 ✅). **그러나 wm_pre 게이팅 무관 0.667 고정** = WM가 place 아닌
소스(되먹임/thalamic/tonic dopa)로 늘 켜짐 → **집단율로는 판정 불가**(D10c 확증). correct 6→2.

**결론(내 판단, 튜닝 거부·떠넘김 거부)**: PBWM은 올바른 기전이고 배선됐으나, 성공/실패를
현 측정으로 결론 못 냄. 남은 것:
① 패턴수준 readout(working_memory spike_recording_data 10543, 뇌 spike읽기와 타이밍 충돌 위험).
② 부트스트랩(B 미방문). ground-truth seek 금지.
7/12 원본 끊긴 지점 그대로. **A-트랙(🟢) 완결과 달리 프런티어는 정직하게 열린 채.**

### D9. A6 multicap 캡스톤 (2026-07-25) ✅ — A-트랙 완결
풀스택(klino+biletaxis+brake+hunger-gate 항법 + olf 변별) 통합world 음식15, 30ep seed0
(`a6_multicap.json`):
- **항법**: align 0.593 (last5 **0.739**, 원본 0.82 근접), 조향 작동.
- **변별**: PI 0.227 (last5 **0.383**), good 435 vs bad 222 (~2:1), 좋은먹이 선별.
- 생존 2614 steps.
- **다능력 공존 ✅**: 원본 참조(항법만 align 0.821, PI −0.16=변별X) 대비 항법 유지(0.74)
  + 변별 획득(−0.16→+0.38). 두 능력 상호 파괴 없이 공존 = #64/#65 코히어런트 다능력 뇌.

**A-트랙(🟢) 재유도 완결**: A1 place_pref / A2 biletaxis(align 0.75) / A3 brake(dwell 2배) /
A4 hunger-gate(#61) / A5 factored(구조충족) / A7 klino(align 0.73) / A8 olf(PI 0.24) /
A6 multicap(공존). 3패턴: 증거역산 · 학습호출누락발견(A2/A8) · 구조충족(A5/replay-klino).
**남음**: 프런티어(🔴 §C) — seq-task/seq-wm(WM latch), context-compositional. 7/12 미해결 지점.

### D8. v3-olf 변별학습 (2026-07-25) ✅ — A2 패턴(학습호출 누락) 재발견
최소 러너가 먹이 먹을때 변별 R-STDP를 빠뜨림(run_training은 good_food→
update_cortical_rstdp+update_prediction_error_rstdp, bad_food→update_cortical_rstdp).
→ PI~0. --v3-olf로 게이팅해 `_discriminate()` 배선. 통합과제 음식20, 25ep seed0
(`a8_{off,olf}.json`):

| 조건 | PI mean | **PI last5** | good | bad | steps |
|---|---|---|---|---|---|
| olf OFF | 0.170 | **0.009** | 604 | 326 | 2726 |
| olf ON | 0.199 | **0.238** | 618 | 335 | 2953 |

- **olf ✅**: last5 PI 0.009(우연 붕괴)→0.238(변별 유지). 학습 곡선 뚜렷. steps↑(생존).
- **창발 준수**: "고음=좋음" 하드코딩 아님. 도파민(good +/bad −) R-STDP가 단서→가치
  연합 자가학습. 프로젝트 원칙(하드코딩 금지, 학습 창발) 충족.

### D7. klinotaxis (2026-07-24) ✅ — 증거 全소실, 교과서 원리로 재유도
klino 증거 완전 소실(4월 코드·스크립트·결과 全손상, 단독 마커 없음). 텍스트북 원리로 재유도:
biletaxis=공간 좌우비교, klino=시간축("가까워지나 멀어지나"). biletaxis 방향 위에 klino가 크기 변조.
value_here 시간비교 → 멀어지면(하락) 재조향 ×(1+2·하락폭), 가까워지면 현 조향 신뢰.
25ep seed0, biletaxis+brake 기준 (`a7_{noklino,klino}.json`):

| 조건 | align | last5 align | goal-dist | last5 dwell |
|---|---|---|---|---|
| biletaxis+brake | 0.623 | 0.570 | 0.316 | 0.249 |
| **+klino** | 0.672 | **0.728** | **0.299** | **0.266** |

- **klino ✅**: last5 align 0.57→0.73, 원본 nav 0.82 근접. 전 지표 개선. 두 항법원리 상보 확인.
- replay-to-klino: replay_swr→value지도→biletaxis read로 이미 충족(no-op 수용).
- klino 단독(biletaxis 없이)은 변조대상 없어 미지원 명시(조용한 no-op 회피).

### D6. factored value (2026-07-23) — 구조적으로 이미 충족 (no-op)
측정: align이 음식0=0.75 → 음식10+gate=0.55로 저하(널 0.5 근접). 음식이 항법 방해.
**그러나 코드 인스펙션 확정**: 러너의 `add_experience`/`_record_transition`는 zone 보상
(189/192/197행)에서만 호출, 음식 먹을때(381~382)는 learn_food_location+release_dopamine뿐.
place_to_value는 replay_swr가 experience_buffer(zone-only)로만 갱신 → **음식은 value 지도에
안 들어감**. 즉 `--place-value-food-exclude`는 **no-op** — 이미 zone-only로 factored.
align 저하는 value 오염 아니라 **행동적**(forage하느라 zone 도달 덜 깨끗) → exclude로 못 고침.
원본과 정직한 구조 분기(원본은 음식이 place-value 오염 → 이 플래그로 제외). 플래그 수용만.

### D5. hunger-gate arbitration (2026-07-23) ✅ #61 재현
satiety로 biletaxis 게이팅. 음식10, 25ep seed0 (`a4_{off,brake,gated}.json`):

| 조건 | steps | PI | good | goal-dist | dwell | sat |
|---|---|---|---|---|---|---|
| OFF(forage만) | 1669 | 0.217 | 184 | 0.338 | 0.043 | 0.163 |
| +brake | 1218 | 0.230 | **96** | 0.306 | 0.101 | 0.117 |
| **+brake+gate** | **1879** | **0.312** | **188** | **0.317** | 0.073 | 0.215 |

- **문제 확증**: brake가 forage 방해 — good 184→96 반토막, steps 1669→1218. (#56/#59)
- **gate ✅**: forage 완전 복원(good 188≈OFF) + 항법 이득 유지(goal-dist 0.317<OFF 0.338).
  #61 기준(생존 유지 + goal 개선) 충족. 두 구동원 상태 분리 성공.
- **교훈(예측 오류)**: 평균 satiety(0.215)만 보면 게이트 임계(0.2~0.5)에 못 미쳐 "항상 닫힘"
  으로 오판. 실제는 **먹은 직후 satiety 피크가 임계 돌파** → "먹었으니 항법" 순간 포착.
  평균 아닌 전이 피크가 신호. 게다가 항법→zone(보상) 선순환으로 gated satiety 최고.
  → 재보정 불필요. 절대임계 (sat-0.2)/0.3 그대로 작동.


현재 위치 value(zone 근접도, D2 지도)로 게이팅. 25ep seed0 비교
(`a3_{bare,settle,brake}.json`):

| 조건 | dwell mean | dwell last_5 | goal-dist | align |
|---|---|---|---|---|
| bare biletaxis | 0.112 | 0.139 | 0.307 | 0.69 |
| +settle | 0.067 | 0.083 | 0.329 | 0.74 |
| **+brake** | **0.214** | **0.256** | **0.289** | 0.69 |

- **brake ✅**: 고value서 전진속도 0.3배. 판정기준(brake<OFF 거리 & brake>OFF 체류) 충족.
  3단계 개선: 널 0.07 → biletaxis 0.14 → +brake 0.26. #43/#49 "브레이크 돌파" 재현.
- **settle ✗**: 목표 근처 조향 감쇠(0.2배) → 교정 조향까지 죽어 dwell↓. 재유도 미흡.
  주석상 settle은 탐색적 "fix", brake가 "돌파 확정"이므로 brake로 목적 달성, settle 보류.
- 구현: brake는 env.step 동안 agent_speed 스케일 후 복원(러너 레벨), settle은 d_turn 감쇠.



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
