> 📓 실험 실행로그(repo). **연구 지식·결론의 장기기억 정본 = llm_wiki 볼트 `sessions/brainsim/`.** 이 파일 = 무엇을 돌렸나·수치·런기록.

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

### D14. WM 포화 = 근본원인 직접 규명 (2026-07-31)
D13이 "래치 미작동 추정, 기계적 왜는 미검증(V readout 깨짐)"으로 남겼던 것을 **직접 측정으로 규명.**
방법(개념 forced-choice 방법론 이식): (a)깨진 막전위 V 대신 **뉴런별 스파이크카운트 벡터**로 WM 패턴
readout, (b)자유항법 "에이전트가 A 떠남" 교란 대신 **강제 프로브** — A 적재(게이트 open) 후
게이트 close+중립관측으로 입력을 순수 차단하고 recurrent 유지력만 격리. `wm_latch_probe.py`.

**측정 결과 (250ep, trials=4):**
- true-rest WM 발화율 = **0.667** (워밍업·적재 없이도). 
- WM 벡터: **200/200 뉴런 전부 활성**, std=0.36, max=4 = 매 스텝 전 뉴런 발화 = **포화.**
- A+ vs 대조 지연 발화율 차 = +0.00, 적재패턴 유지 corr ≈ 0 (구조 없어 유지할 패턴 자체가 없음).

**결론(측정 기반, 추정 아님)**: WM 집단이 rest에서 **포화**. "off" 상태가 없어 스위칭 기억으로
물리적으로 작동 불가 — 적재해도 더할 게 없고, 균일발화라 특정 패턴 보유 불가. 
→ D10 latch음수·D13 corr0·D10b 되먹임강화 무효가 전부 이 하나로 설명됨(포화 집단에 흥분 더해도 무효).
**E/I 되먹임 회로는 이미 존재**(wm→wm_inhibitory 6.0, wm_inhibitory→wm −5.0, sparsity 0.08)하나
**억제가 너무 약해 sparse coding 실패.** 다음: 억제강도를 order_rate 아닌 **WM 희소성**(독립지표,
목표 ~10-20% 활성) 목표로 스윕 = 생물학적 용량전제조건 보정(하드코딩 아님) → 그 위에서 순서창발 별도검증.

### D15. 희소화하니 A-특이 래치 출현 (예비, 2026-07-31)
D14(포화=근본원인)의 처방 검증. `wm_inhibitory→WM` 억제강도를 **order_rate 아닌 희소성(독립지표)**
목표로 스윕(용량 보정, 하드코딩 아님). 프로브 업그레이드: A+ 지연패턴/대조 지연패턴을 **같은
A-적재 기준패턴에 교차상관** → 진짜 A-특이 기억 vs 입력무관 일반 attractor 판별.

**결과 (250ep, n=12):**
| inhib | active_frac | corr(A+,A-ref) | corr(ctrl,A-ref) | 판별 |
|---|---|---|---|---|
| −5(포화) | 1.00 | 0.00 | 0.00 | +0.00 |
| −175 | 0.40 | 0.998 | 0.704 | **+0.294** |
| −200 | 0.23 | 0.999 | 0.802 | **+0.197** |
| −250 | 0.14 | 0.999 | 0.885 | +0.11 |

- 포화(−5)에선 판별 완전 0 → 희소화(−175~−200, 활성 23~40%)하니 **A-특이 지속상태 출현**(판별
  +0.2~0.29, n=12 재현). **포화가 래치 블로커였다는 직접 증거.**
- 과희소(−250+, 활성<15%)하면 대조군도 A-기준 수렴(단일 attractor 붕괴) → 판별 붕괴. sweet spot 존재.
- **약점(정직)**: 대조 corr 0.70~0.80도 높음(중립관측이 A place와 겹침 추정) = 순수 A-특이 아직 미확정.
- **한계**: 이 가중치는 포화 WM(−5)로 훈련됨 → eval-time 희소화는 "아키텍처가 A-구별 상태 담을 수
  있음"만 증명. **순서학습 창발은 희소 WM으로 재훈련해야 진짜 검증.** ← 다음 실험.

### D16. 희소 WM 재훈련 = 순서 창발 안 함 (둘째 블로커 분리, 2026-07-31)
D15 처방을 훈련에 배선(`--inhib-wm`, run_v2_tasks). 동일 seq-wm 과제, inhib −5(포화) vs −200(희소),
각 40ep seed0 재훈련. **핵심 검증: 희소화로 순서학습이 실제 창발하나.**

**결과 (d16_{sat,sparse}.json):**
| WM | 최종순서율 | last_5 | 훈련중 WM_latch |
|---|---|---|---|
| 포화 −5 | 0.0070 | 0.0055 | ~0.0000 |
| 희소 −200 | 0.0060 | 0.0089 | ~0.0000 |

- **둘 다 ~0.006 = 희소 WM으로도 순서학습 창발 안 함(음성).** 희소는 포화(D14) 필요조건 해결이나 불충분.
- **결정적 단서**: 훈련중 WM_latch 지표(발화율 기반 `wm_rate>base×1.5`) 두 조건 다 ~0. 그러나 D15에서
  희소 WM의 래치는 **발화율 아닌 패턴**(특정뉴런)에 담김. → 러너 래치감지가 rate 기반이라
  **패턴-래치를 못 읽음** → 컨트롤러가 자기 WM기억 못 봐 B전환 실패 → 순서 미완성.
- **분해된 결론**: "seq-wm 실패" = ①WM 포화(D14, 해결됨: 희소화) + ②패턴-래치가 rate기반 제어에 안읽힘
  (D16, 미해결). 첫째 기계적 해결이 실질진전. 둘째가 남은 진짜 블로커.
- **다음(정조준)**: 러너 래치감지를 **패턴 기반**(현재 WM이 A-적재 패턴과 상관?)으로 교체 —
  biletaxis가 place_to_value 읽던 것과 동형(하드코딩 아닌 창발상태 readout). 순서 창발하면 확증,
  아니면 남은 블로커=credit assignment(A래치→B선택→보상) = 심층 연구.

### D17. 패턴래치 readout 작동, 순서는 탐색 confound (2026-07-31)
D16 처방: 러너 래치감지를 rate→**패턴 상관**(A-적재 스파이크패턴)으로 교체(`--seq-pattern-latch`,
biletaxis가 place_to_value 읽던 것과 동형 readout). SeqTask._wm_v도 깨진 V→스파이크벡터로.
희소(-200)+패턴래치 40ep 재훈련(d17_patlatch.json).

**결과:** 최종순서율 0.006, last_5 0.014 (D16과 동일) — 그러나 **last_5_pattern_corr=0.9986**.
- **래치 readout은 이제 작동**(패턴 corr 0.99 = 현재 WM이 A-패턴과 강상관, 래치 감지 성공).
- 그런데 순서율 여전 ~0. ep39 correct=6인데 order 0.014 → **wrong(B먼저)≈420회.**
- **정체 규명**: 에이전트가 B탐색 위해 curiosity 무작위탐색 → A없이 B 반복 접촉 → order_rate 압도.
  **래치는 작동하나 자유행동 order_rate가 탐색노이즈에 confound**(개념 selectivity 0.64가 forage에
  가려진 것과 동형 구조 — D11 "explore/exploit 충돌"의 정체).

**decomposition 최종(D14~D17):** seq-wm실패 = ①WM포화(D14, 해결:희소화) + ②패턴래치 rate제어에
안읽힘(D16→D17, 해결:패턴readout) + ③자유 order_rate가 explore/exploit에 confound(남음). 
①②는 실질 해결. 지난세션 "래치 미작동·기계미상"에서 크게 전진.
**다음(방법론 반복):** 개념을 푼 강제선택 프로브를 순차선택에 적용 — 중립지점 curiosity OFF,
래치OFF→A로/래치ON→B로 향하나. confound 우회하고 "래치가 순차선택 구동하나" 직접 판정.

### D18. 탐색 confound 반증 + navigation value-map 갭 노출 → seq-wm 정지 (2026-08-01)
D17 가설(order confound=curiosity 탐색) 검증: `--seq-no-curiosity`로 무작위탐색 OFF, biletaxis
방향조향만. 희소+패턴래치+curiosity OFF 40ep(d18_nocur.json).

**결과:** 최종순서율 0.0076 (D17 0.006과 동일) — **탐색 confound 가설 반증.** 탐색 꺼도 순서 안 됨.
**결정적 단서(내내 있던)**: **vmap_std=0.0000, align=0.0000** 전 에피소드. biletaxis는 학습된
value map으로 조향 → **value map이 비어(std=0) 조향 자체가 작동 안 함.** "래치→target→biletaxis
조향" 사슬의 마지막 고리 부재. curiosity 꺼도 조향할 gradient 없어 무작위 배회 → order 0.

**블로커가 또 한 겹 후퇴**: WM 고쳤으나(포화·readout) 순차 존의 **navigation value-map + credit
assignment가 재구성 하니스에서 co-adapt 안 됨.** 각 수정이 다음 갭 노출 = 여러 하위시스템 동시작동
필요한 깊은 연구문제.

**seq-wm 최종(D14~D18, 정직):** 지난세션 "래치 미작동·기계 미상"을 **3 하위블로커로 완전분해**:
①WM 포화(D14 규명·해결:희소화) ②패턴래치 rate제어 미판독(D16→D17 해결:패턴readout, corr0.99)
③순차 navigation value-map 미형성+credit assignment(D18 노출, 미해결·깊은연구).
①②는 측정기반 실질해결 = 지난세션 대비 큰 전진. ③은 억지통과 안 시킴 — 진짜 열린 문제로 정확히 경계.
**여기서 정지**(막힌 데 제자리돌기 금지 원칙). 딜리버러블: 개념형성 3개✅·A트랙✅ 확정.

### D19. replay 게이팅 버그 발견·수정 → value-map 학습됨 (2026-08-02)
D18 "vmap_std=0(value-map 빔)"의 기계적 원인 규명: run_episode의 `replay_swr()` 호출이
`place is not None and place.sparse_reward`에만 게이팅 → **seq 태스크(place=None)는 replay 미실행.**
SeqTask가 A/B를 add_experience로 버퍼링해도 에피소드끝 consolidation 안 돌아 place_to_value 영영 평평.
수정: `or (seq is not None)` 추가(A트랙과 동일 기전 배선, 하드코딩 아님).

**검증(희소WM-200+패턴래치+fix, 25ep):** vmap_std ep0=0.0000→ep1=0.0199→단조상승→ep24=0.1599.
**value-map 이제 학습됨 ✅(D18 블로커 해결).** 그러나 order_rate 여전 ~0, **align=0.0000 유지.**
- value-map 생겼으나 biletaxis 조향정렬(align) 여전 0. 원인 후보: n-food 10이라 value-map이 A/B존
  아닌 음식으로 도배(순서존 신호 희석). → 다음: n-food 0(존만)로 세 fix 통합 검증.

### D19b. n-food 0 통합검증 → 부트스트랩 knot 규명, seq-wm 최종 바운드 (2026-08-02)
D19 후속(value-map이 음식 도배 추정) 검증: n-food 0(존만) + 세 fix(replay+희소WM+패턴래치) 25ep.
**결과: vmap_std=0.0000 다시 내내, order 0.0019.** n-food 0이니 value-map 학습 경험(음식) 사라짐 →
replay 돌아도 재생할 rewarded 경험 없음 + 존 거의 못닿아(order~0) 경험 소비도 없음 = **부트스트랩 knot**
(존 닿으려면 value조향 필요 ↔ value 만들려면 존 닿아야, D11 재출현).

**seq-wm 최종 decomposition (D14~D19b, 지난세션 대비 대전진):**
| # | 블로커 | 상태 |
|---|---|---|
| ① | WM 포화(200/200) | ✅ 해결(D14규명·D15희소화 -200) |
| ② | 패턴래치 rate제어 미판독 | ✅ 해결(D17 패턴readout, corr0.99) |
| ③ | seq replay 미실행→value-map 빔 | ✅ 해결(D19 게이팅버그 수정, vmap 0→0.16) |
| ④ | A→B 순서구조가 steerable value/credit로 미인코딩 | ❌ 남음(부트스트랩+credit knot) |

①②③ 전부 **구체적 버그/원인 규명·수정**(측정기반). ④ = value-map이 음식으로 학습되거나(순서 아님)
비거나(경험없음), **순서 credit이 zone value에 배정 안 됨** = 부트스트랩/credit 매듭. 깊은 연구.
지난세션 "래치 미작동·기계 미상"에서 **3 블로커 실제 수정**까지 전진. ④는 억지통과 안 시킴 — 정밀 바운드.
**seq-wm 여기서 확정 정지.** 딜리버러블: 개념형성 4층위✅(SCORECARD.md).

### D19c~e. order 지표 버그 수정 + chance floor 대조 = 우연 확정 (2026-08-04)
**측정 공백/버그 2개 규명·수정 (둘 다 계측, 행동 불변):**
- **align 미계측**: align이 `place`에만 계산돼 seq는 항상 0(아티팩트). seq target 기준 계측 추가 →
  **align 0.95~0.97** = biletaxis가 A/B로 실제 정확히 조향함 규명(D19c).
- **order_rate 버그**: wrong++가 B 체류 매 스텝 집계(correct는 이벤트당) → correct 완성 후 B 체류가
  wrong 폭증시켜 order_rate 무의미. 진입 이벤트 기반으로 수정 → order_rate 0.006→**0.25**(D19c).

**성급결론 직전 chance floor 대조로 반증(중요):**
| 조건 | overall | last_5 |
|---|---|---|
| floor (무작위 목표, 순서정책 없음) | 0.20 | 0.36 |
| 뇌 WM (패턴래치) | 0.21~0.28 | 0.25~0.29 |
| 상한 (bookkeeping 완벽 target 전환) | 0.32 | 0.52 |

- **뇌 WM(0.25) ≈ 무작위 floor(0.20~0.36) = 우연 넘지 못함.** 지표수정으로 드러난 0.25는 학습 아닌 우연.
- bookkeeping(0.32~0.52)만 floor 위 = **환경은 완벽 target전환 시 above-chance 순서 지원**하나,
  뇌 WM 래치는(corr 0.99로 A 읽어도) target전환을 bookkeeping만큼 못 구동 → 순서 우연 머묾.
- **결론(정직)**: 지표 아티팩트 제거는 real·필수였으나, seq-wm 순서는 **뇌 WM으로 above-chance 창발 안 함.**
  갭 정밀 규명: 래치가 A를 읽지만 **timely·효과적 target전환으로 번역 안 됨**(bookkeeping과의 차이).
- **교훈(5번째 성급결론 차단)**: 지표수정 0.006→0.25에서 "해결" 선언할 뻔 → floor 대조가 우연임을 폭로.
  단일 지표 개선을 성과로 오인 금지, chance floor 필수. 개념 selectivity 규율의 연장.

**seq-wm 최종(D14~D19e):** ①WM포화(해결) ②패턴readout(해결) ③replay버그(해결) ④order지표버그(해결)
+ **⑤남음: 래치→timely target전환 미번역 = 순서 우연.** 4개 구체 수정 + 정밀 바운드. 억지통과 안 함.
러너 플래그: --seq-random-target(floor 대조), --inhib-wm/--seq-pattern-latch/--seq-no-curiosity.

### D13. WM 계측 재시도 실패 + 최종 확정 (2026-07-25)
D12 결론을 자가의심 → WM 패턴 제대로 측정하려 재시도(전 감각→WM 게이팅 + 캡처타이밍 수정).
결과: pattern_corr 여전히 0/20 — WM 막전위 readout(`vars["V"].view`)이 항상 균일(std=0).
실제 뉴런이면 분산이 있어야 하므로 **내 막전위 계측 자체가 live값을 못 읽음**(GeNN 상태접근 문제).

**교훈(자책)**: WM 내부 계측 토끼굴에 사이클 낭비 = 헤맴. 계측은 애초에 불필요했음.
**답은 행동에 있었고 처음부터 확정적**: order_rate ≈0 = 순서학습 창발 안 함.

**seq-wm 최종 확정**: 순서학습은 재유도 기전으로 **창발하지 않음(행동 지표, 확정)**.
WM 내부 상태는 신뢰성 있게 계측 못 함 → 기계적 "왜"는 불확실(래치 미작동 추정이나 미검증).
7/12 원본 끊긴 지점. 억지 통과 안 시킴. 진짜 열린 연구문제.
남은 착수점: WM 상태 신뢰 계측(GeNN V 접근 재확인) → 그 위에서 dopamine-gated 래치 재설계.

### D12. 순서학습 창발 실패 (행동 지표로 결정)
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

### C21. "미구현" 항목 종결 — dopamine-gated WM 게이트 회로는 죽어 있음 (2026-08-26)
333행에 "다음 접근(미구현): 4월 코드에 wm_gate/dopamine_to_wm_gate/wm_update_gate 기계 존재,
러너가 engage 안 함 → **7/12 원본 끊긴 지점 유력 후보**"라 적어두고 검증하지 않았던 항목을 측정
(`wm_gate_probe.py`). 배선 확인: `dopamine → wm_update_gate → wm_thalamic → working_memory` 전부 존재.

**측정(도파민 OFF→ON):** dopamine_neurons 325→350(+25), **wm_update_gate 162→162(+0)**,
wm_thalamic 126→134(+8), working_memory 690→650(−40). **VERDICT: 게이트 안 열림·전달 없음·WM쓰기 없음.**

- **= 회로는 구조만 있고 기능하지 않음.** 도파민이 발화해도 게이트가 0만큼 반응(가중치 6.0·sparsity 0.05로
  약하고, `wm_update_gate` 자체가 C14 포화 목록에 포함 = 162로 고정 발화).
- **함의**: 내가 이 회로 대신 만든 우회(`gate_wm_input`, 시냅스 가중치 스케일)는 결과적으로 옳은 선택이었음
  — 원 회로를 engage했어도 작동하지 않았을 것. 단 **검증 없이 "유력 후보"로 방치한 것은 잘못**이었고 이제 닫음.
- **"7/12 원본 끊긴 지점" 후보에서 제외.**

### C22. 맥락 의존 개념 — 🔴 미상(97행) 착수 + **둘째 죽은 스위치 발견** (2026-08-26)
97행 "C2 조합적 컨텍스트 🔴 미상"은 이번 재건 세션에서 **한 번도 측정 안 된 채** 남아 있었음.
`--context-select`(Zone A 정상 / Zone B에서 good↔bad 반전)는 4월 "M4 v9 hard gate 첫 돌파(PI 0.17→0.25)"
로 보고된 능력. 프로브 신설(`context_probe.py`): **동일 시각자극**을 A/B 존에서 제시해 조향이 뒤집히는지.

**(a) 죽은 스위치 (C21과 동형)**: `--context-select`는 `context_gate_enabled`만 켜고
**v9 돌파의 핵심 `context_hard_gate_enabled`는 켜지 않음**(기본 False, 러너 어디서도 미설정.
유일한 설정처=별도 `test_context_100ep.py`). → 이 세션에서 `--context-select`를 썼다면 **돌파 이전 설정**.

**(b) 내 지표가 ill-posed였음(자기정정)**: combined=(A정답+B정답)/2는 고정매핑만 있어도 A가 천장이라
68.3%로 "PASS"가 뜸 — D24에서 내가 비판한 2존 order_rate와 **같은 오류**. 올바른 지표는
**CSI = P(good쪽|A) − P(good쪽|B)** (맥락무관 0, 완전반전 +1).

**(c) 무학습 기준선 (brain_concepts_550ep, 맥락규칙 없이 훈련됨, n=60):**
| 설정 | ZoneA(정상) | ZoneB(반전) | CSI | 반전 |
|---|---|---|---|---|
| hard gate OFF | 98.3% | 38.3% | **+0.366** | NO |
| hard gate ON | 100.0% | 10.0% | **+0.100** | NO |

- 둘 다 **반전 없음**(ZoneB<50%) = 맥락 개념 미형성. 단 OFF에서 36.6%p 부분 변조 존재(위치 교란 포함).
- **hard gate ON이 오히려 악화**(+0.366→+0.100): v9 게이트는 맥락-무관 D1을 억제해 D1_ctx에 정책을
  넘기는 설계인데, 이 뇌는 맥락규칙 없이 훈련돼 **D1_ctx가 비어 있음** → 넘겨받을 정책이 없음.
- → **맥락 훈련이 전제**. `--context-select 100ep` 훈련 가동(brain_ctx_100ep.npz), 훈련 후 CSI 재측정 예정.

### C23. 죽은 스위치 전수 감사 — 남은 건 없음 (2026-08-26)
C21(도파민→WM 게이트)·C22(v9 맥락 hard gate)가 **같은 병리**(회로는 배선됐는데 러너가 켜지지 않음)여서
전수 점검(`dead_switch_audit.py`). 판정식: ForagerBrainConfig의 bool 필드가 기본 False + 어디서도
True 설정 안 됨 + 뇌 코드가 실제 참조 = 영원히 꺼진 회로.

**결과: bool 설정필드 43개 중 죽은 스위치 0 / 정상 43.** C21·C22 두 건이 남아 있던 전부였고 둘 다 닫음.
- 중간에 내 감사 자체의 거짓음성 수정: `getattr(self.config,"X",...)` 접근을 못 잡아 `typed_sound_enabled`를
  "뇌 미사용 껍데기"로 오분류 → 패턴 추가. 설정처도 러너뿐 아니라 평가 스크립트까지 확장.
- **남는 구별(중요)**: 이 감사는 "어딘가에서 켜지는가"만 봄. `sound_food_flip`처럼 **평가에서만 켜지고
  훈련에서는 못 켜는** 스위치는 여전히 "훈련 중 검증 불가". C22의 hard gate가 정확히 그 경우였고
  (평가전용 `test_context_100ep.py`에만 존재) 그래서 러너에 `--context-hard-gate`로 배선함.
- **부수 발견**: 기존 러너는 soft gate만 켜면서 로그에 `"context hard gate ON"`이라고 **거짓 출력**.
  이게 죽은 스위치를 오래 못 본 직접 원인. 실제 상태를 출력하도록 수정.

### C23b. 평가전용 스위치 = 소리개념 83.8%의 등급 하향 (2026-08-26)
C23의 남은 구별("평가에서만 켜지고 훈련엔 배선 없음")을 별도 감사(`evalonly_switch_audit.py`).
**결과 2개, 둘 다 소리 경로**: `typed_sound_enabled`, `sound_food_flip`.

- **함의(자기정정)**: 확인된 개념 4층 중 **소리 83.8%는 `--typed-sound`를 평가시에만 켜고 잰 값**.
  뇌는 그 회로를 켠 채 **훈련된 적이 없음**. → D15에서 내가 스스로 지적한 약점("eval-time 처방은
  아키텍처가 담을 수 있음만 증명, 학습 창발은 재훈련 필요")과 **같은 등급의 한계**.
  소리개념 = "형성됐다"가 아니라 **"평가시 배선하면 나온다"**로 하향 표기.
- 러너에 `--typed-sound`/`--sound-flip` 배선 후 100ep 재훈련 가동(brain_typedsound_100ep.npz) → 훈련된
  소리개념이 실제로 형성되는지 재측정 예정.

**동시 가동 3런**: ctx(soft) / ctxhard(v9 hard gate, 4월 돌파 첫 재현시도) / typedsound.
ctxhard는 ep10에서 steps 4500→329 붕괴 — hard gate가 맥락무관 D1을 억제하는데 D1_ctx가 미학습이라
먹이접근 실패(예상된 초기현상). **학습으로 회복하는지가 v9 돌파의 실체 여부를 가름**(회복 못하면 재현불가 판정).

### C22b. hard gate 학습 후 CSI — **반전 없음(반복이 첫 측정을 반박)** + 내 프로브 결함 발견 (2026-08-26)
v9 hard gate 100ep 훈련 완주(초기붕괴 steps 329 → ep97 steps 4500 만점 회복 = D1_ctx 단독정책으로
먹이접근 재학습 성공). **첫 CSI 측정 ZoneB 56.7%(CSI +0.567) = 반전 넘음**처럼 보였으나,
n=60에서 우연 대비 +6.7%p는 이항 SE 6.45%p의 1 SE 이내 → **결론 보류하고 사전기준 선언 후 반복.**

**반복(각 n=200, 3회):** ZoneB 49.5 / 49.5 / 0.0 (평균 33.0). **사전기준(3회 모두 >50%) 불충족 = 반전 없음.**
첫 56.7%는 잡음. (C12 "56% 개선"→C13 반박, C15/C16 "악화"→노이즈와 **같은 계열의 실패를 사전기준으로 차단**.)

**부수 발견 — 프로브 자체가 불안정(내 코드 결함)**: ZoneA가 49.5~99.5%로 요동, `49.5%`(=99/200) 반복 등장
= "항상 한쪽 조향" 퇴화행동 서명. 원인: `run()`이 시작 때만 reset하고 **시행·존 사이 상태 초기화 없음**
→ 앞 시행이 뒤를 오염. **따라서 이 프로브로 잰 기준선(CSI +0.366/+0.100)도 무효.**
수정: 시행·존마다 `brain.reset()`+중립 안정화, 존 순서 카운터밸런싱. 학습/무학습 전부 재측정 중.

### ★C24. 체크포인트 로드가 학습의 63.6%를 파괴하고 있었음 — 세션 최대 발견 (2026-08-26)
C22b의 "런 수준 불안정"(같은 뇌·같은 코드인데 결과가 두 체제로 갈림)을 추적하다 발견.

**기계**: `_load_sparse_weights`(forager_brain.py:5996)는 SPARSE 연결 개수가 저장본과 다르면
저장 가중치를 **평균 스칼라 하나로 브로드캐스트**(6004~6006). 그런데 `GeNNModel`에 **시드가 없어**
(1566행) SPARSE 연결이 **매 런 새로 무작위 생성** → 개수가 항상 어긋남 → 학습 구조가 상수로 치환.
로그는 `Weights loaded`라고 정상 보고하므로 **조용히** 일어남.

**정량(brain_concepts_550ep, `load_damage_audit.py`):**
- 로더 보고 55 시냅스 중 **35개(63.6%) 구조 파괴**, 20개만 정상 복원.
- 파괴 목록이 하필 **학습 경로 그 자체**: `rstdp_left/right`(food_eye→D1 R-STDP), `good/bad_food_d1_*`,
  `it_food_d1_*`, `*_d2_*`.
- 치환된 평균값 4.988·4.991·0.103·0.101 = **초기값(5.0, 0.1)과 사실상 동일** → 로드된 "학습된 뇌"는
  해당 경로에서 **초기화 상태로 되돌아간 뇌**.

**함의(중대, 자기정정)**: 이 세션에서 체크포인트를 로드해 잰 **모든 수치**가 재해석 대상.
개념 4층(공간 31.8 / 시각 79.2 / 소리 83.8 / 일반화 78.0 / 조합 98.0), seq-WM, 사회성 13접근 전부.
우연 이상으로 나온 부분은 **학습이 아니라 선천 배선(아키텍처)에서 온 것**일 수 있음.
또한 이것이 **노이즈 바닥 ±6~7의 정체**이자, 사회성이 13접근 모두 우연에 머문 이유 후보
(D1 경로가 매번 상수로 초기화되면 학습이 남을 수 없음).

**수리**: `genn_seed=12345` 도입해 SPARSE 연결 생성을 결정론화(연결이 재현되면 개수가 맞아 시냅스별
구조가 그대로 복원). 검증 중 — 파괴 35→0 확인이 수리 성공 기준.

**C24 수리 검증 (왕복시험 `seed_roundtrip_test.py`)**: 시드 적용 뇌를 저장→새 뇌에 재로드
→ **구조 파괴 35 → 0 (54/54 정상 복원). 수리 성공.**

**단, 성공이 곧 나쁜 소식**: 기존 체크포인트는 무작위 연결 시절 저장분이고 그 연결 구조는
어디에도 없어 **복구 불가**. 따라서
- `brain_concepts_550ep.npz`(정본 개념 뇌) 포함 **모든 기존 체크포인트는 로드시 63.6% 손상 확정**
- 이 세션의 로드 기반 측정 전부 무효 → **재훈련이 유일한 길**
- 수리 이전 코드로 돌던 런 3건(ctx soft ep80, typedsound ep75)도 폐기(저장해도 로드시 손상)

**재훈련 가동(수리 코드)**: concepts 250ep(정본 재건, C24 이후 유효한 첫 뇌) /
ctxhard 150ep(C22 재실행) / typedsound 150ep(C23b 검증).
→ 완료 후 개념 4층·CSI·소리개념을 **처음으로 손상 없는 뇌에서** 측정.

### ★C26. 소리 개념에 **학습 기여 없음** — 선천 배선 성능이었다 (2026-08-26)
C24 수리 후 처음으로 손상 없는 뇌에서 측정. 프로토콜대로 **무학습 대조 동반**(`save_untrained.py`로
훈련 0회 뇌를 같은 로드 경로로 저장, `brain_seeded_untrained.npz`).

| 조건 | 반복 3회 | 평균 |
|---|---|---|
| 학습(typedsound 150ep) | 76.0 / 80.0 / 92.0 | **82.7** |
| **무학습(훈련 0회)** | 90.0 / 77.0 / 70.0 | **79.0** |

**차이 +3.7%p = 반복 SD(~9~10) 이내 → 학습 기여 없음.**
- 훈련하지 않은 뇌가 우연(50%) 대비 **79%**를 낸다. 소리 "개념"은 학습 산물이 아니라 **선천 배선 성능**.
- C24 이전 83.8%(63.6% 파괴된 뇌) ≈ 지금 82.7%(손상0 학습뇌) ≈ 79.0%(무학습뇌) — **셋이 구분되지 않음**.
  손상 여부·학습 여부와 무관하게 같은 점수 = 점수의 출처가 학습이 아님을 삼중으로 확인.
- 로드 손상 감사: 새 시드 체크포인트 **파괴 0/54** (C24 수리가 훈련→저장→로드 전 과정에서 작동 확인).

**→ 나머지 개념도 동일 의심.** 특히 조합 98.0%. 무학습 뇌에 시각·일반화·조합·공간 전 항목 측정 착수.
동시에 C25 학습델타 프로브(로드 무관, 한 프로세스 내 훈련 전후 가중치 직접 비교) 가동 —
"학습이 일어나기는 하는가"를 독립 판정.

### ★C27. 맥락 개념도 학습 기여 없음 + 내 C24 귀인 일부 정정 (2026-08-26)
손상 0 확인(54/54) 후 학습(ctxhard 150ep) vs 무학습 각 3회, **사전 기준 선언 후** 측정.

| 조건 | ZoneB(반전) 3회 | 평균 |
|---|---|---|
| 학습(v9 hard gate 150ep) | 46.7 / 49.3 / 1.3 | **32.4** |
| 무학습(훈련 0회) | 52.0 / 44.0 / 52.0 | **49.3** |

- 사전기준(3회 모두 >50% & 평균−SD>50) **명백 미달 → 맥락 개념 미형성.**
- 무학습이 우연(50%)에 있고 **학습이 그보다 낮음** → v9 hard gate 훈련은 개념을 만들지 못했을 뿐 아니라
  기존 행동을 **교란**. 4월 "M4 v9 첫 맥락 선택성 돌파"는 **현재 코드로 재현 불가**로 판정.
- **자기정정**: C22b에서 런간 이체제 요동(ZoneA 100% vs ~50%)의 원인을 C24 로드 손상으로 지목했는데,
  **손상 0인 지금도 동일하게 나타남** → 그 귀인은 틀렸다. 요동은 뇌 자체의 행동 쌍안정성.

**소리(C26)와 맥락(C27) 두 축 모두에서 "무학습 ≥ 학습".** 이는 C24와 별개 문제 —
손상을 고쳤는데도 학습이 성능에 기여하지 않는다. → C25 학습델타(로드 무관, 훈련 전후 가중치 직접 비교)로
"가중치가 변하는가" vs "변해도 행동에 안 닿는가"를 분리한다.

### ★★C28. 개념 프로브가 재던 것은 **하드코딩된 반사**였다 (2026-08-26) — 프로젝트 핵심 주장 붕괴
C26(소리)·C27(맥락)에서 "무학습 ≥ 학습"이 반복되자, **무학습 뇌가 왜 79~98%를 내는가**를 직접 물음
(`innate_pathway_probe.py`).

**(a) 선천 직결 배선 실재:**
```
good_food_to_motor_l   n=15097  평균g = 25.000
good_food_to_motor_r   n=15013  평균g = 25.000
food_explore_motor_l/r n≈30000  평균g = 10.000
```
`good_food_rays → motor`가 **가중치 25.0의 직결 경로**로 배선돼 있다(학습 시냅스 초기값 5.0의 5배).

**(b) 무학습 뇌 조향(학습 0회):**
| 자극 | 조향(음수=좌) |
|---|---|
| good만 좌 | **−0.605** (좌=good쪽) |
| good만 우 | **+0.133** (우=good쪽) |
| good좌+bad우 | −0.518 (good쪽) |
| good우+bad좌 | +0.124 (good쪽) |
| bad만 좌 | −0.369 (bad쪽으로 감) |
| bad만 우 | −0.272 (여전히 좌) |

**판정: good→접근 반사 YES / 경합시 good 우선 YES / bad→회피 NO.**

**→ 결론: "좋은 음식 쪽으로 조향"은 이 뇌에서 학습 대상이 아니라 반사다.**
내 개념 프로브는 거의 전부 이 판별을 요구한다(시각변별·조합(good vs pain)·소리·일반화 모두
최종적으로 "good 쪽으로 가는가"를 묻는다). 따라서 **개념 4층 점수는 가중치 25짜리 배선 하나의 성능**이며,
"개념 형성"의 증거가 아니다. 이것이 무학습 79%, 조합 98%, 학습기여 0, C24 손상 무관 동일점수를
**하나로 설명**한다.

**부수 결함 2건:**
- **좌편향**: good만 좌 −0.605 vs good만 우 +0.133 (4.5배 비대칭) → 좌/우 강제선택 프로브에 계통오차.
  런간 이체제 요동(C22b/C27)의 유력 원인이기도 하다.
- **bad→회피 반사 없음**: bad 단독 제시시 오히려 bad쪽/좌쪽으로 감 → 회피는 배선돼 있지 않다.

**함의**: 개념 형성을 주장하려면 **선천 반사로 풀리지 않는 과제**로 재설계해야 한다.
반사는 "good 태그가 붙은 자극 → 접근"이므로, 개념 과제는 (i)good 태그 없이 임의 자극↔가치를
학습으로 연결하거나 (ii)반사와 **반대** 응답을 요구해야 한다(맥락반전 과제가 그 시도였고 실패).

### ★C31. 무학습 뇌 개념 전수 측정 — 조합·공간·소리는 선천, 시각·일반화는 미해결 (2026-08-26)
무학습 뇌(`brain_seeded_untrained.npz`, 훈련 0회)에 개념 전 항목 3회씩.

| 개념 | 이전 주장(손상뇌) | **무학습 뇌 (3회)** | 차이 | 판정 |
|---|---|---|---|---|
| 조합 | 98.0±2.1 | 95.8 / 97.5 / 96.7 → **96.7** | **+1.3%p** | **선천 배선** |
| 공간 | 31.8±1.1 | 31.2 / 30.8 / 28.3 → **30.1** | **+1.7%p** | **선천 배선** |
| 소리 | 83.8±8 | 79.0 (C26 직접비교) | +4.8%p | **선천 배선** |
| 시각 | 79.2±6.5 | 60.0 / 63.0 / 54.0 → **59.0** | +20.2%p | **미해결** |
| 일반화 | 78.0±9 | 53.3 / 62.5 / 56.7 → **57.5** | +20.5%p | **미해결** |

- **내가 최강 증거로 삼았던 조합 98%가 학습과 1.3%p 차이** = 전적으로 배선(C28의 good→motor 25.0으로 설명).
- 공간 31.8%도 무학습 30.1% → 배선.
- **단, 전부가 배선은 아니다**: 시각·일반화에서 20%p 격차. **아직 공정한 비교가 아님** —
  79.2/78.0은 63.6% 손상된 뇌 값이고 무학습은 손상 0. 손상된 뇌가 더 높다는 건 학습분이 손상을 견디고
  남았을 가능성을 시사하나, 확정하려면 **손상 없는 학습뇌**가 필요 → `concepts_run` 250ep 대기.
- **결론 유보(중요)**: "전부 선천"으로 성급히 닫지 않는다. 두 항목은 추적 가치가 있다.

### C30. 가소성 전수 감사 — 예비결과와 **내 프로브 결함**(발표 직전 차단) (2026-08-26)
C25 수정판(러너와 동일 학습호출, 리플레이 `updates=740~950` 실행 확인)에도 지정 12개 시냅스가
|Δ|=0 → 전 시냅스로 확대 조사.

**예비결과(결함 수정 전, 참고용):** 변화 9개 / 무변화 60개.
- **진짜 구조학습 2곳**: `place_to_food_memory_left/right` (|Δ|7.9~8.9, **std 0.0000→10.3~11.3**),
  `place_to_value` (|Δ|0.119, **std 0→0.1156**). 둘 다 해마/장소 계열.
- **균일 스케일만**: `good_food_to_it_food_*`, `pe_food_to_it_*`, `good_food_to_it_danger_*`
  (|Δ|1.4~2.4인데 **std 0 유지** = 변별 구조 없이 전역 이득만 변함).
- 무변화 60개에 기저핵 전체(`*_to_d1_*`, `*_to_d2_*`, `kc_to_d1*`)와 운동경로 포함.

**★그러나 이 결론을 발표하기 전 내 프로브 결함 발견**: SPARSE 시냅스는
`pull_connectivity_from_device()`를 **먼저 호출해야** `vars["g"].values`가 채워진다. 빠뜨리면 빈 배열이
반환되고 `np.mean(빈)`=nan → `nan > 1e-9`이 False → **조용히 "변화 없음"으로 오분류**된다.
출력의 `RuntimeWarning: Mean of empty slice`가 그 증거였다.
→ **"기저핵 전체 동결"은 빈 배열의 산물일 수 있음.** 로더 `_load_sparse_weights`는 이 호출을 하고 있었는데
내 프로브들만 빠뜨렸다. C28의 `food_memory_*_to_motor n=0`도 같은 원인(실제로는 prob=0.15로 생성됨).

**수정 후 재측정 중.** C21·C22·C25에 이어 **네 번째로 "코드에 있다 ≠ 실제로 그렇다"를 발표 전에 차단.**

### ★C31b/C32. 시각은 학습기여 없음, **일반화만 후보로 생존** (2026-08-26)
손상 0인 학습뇌 2종(typedsound/ctxhard 150ep) vs 무학습뇌, 각 3회.

| 개념 | 무학습 | 학습(TS) | 학습(CH) | 판정 |
|---|---|---|---|---|
| 시각변별 | 74/56/67 → **65.7** | 59/65/64 → 62.7 | 54/53/60 → 55.7 | **학습기여 없음**(학습≤무학습) |
| 일반화 | 55.0/59.2/50.0 → **54.7** | 63.3/63.3/77.5 → **68.0** | 60.0/62.5/66.7 → 63.1 | **후보 생존** |

- **C31의 "시각 20%p 격차"는 착시로 판명**: 그건 손상뇌(79.2) vs 무손상 무학습(59.0)의 부당한 비교였고,
  공정 비교하니 무학습이 65.7로 올라 학습뇌보다 높다. → 시각변별도 선천.
- **일반화만 유일 후보**: 학습 6런 전부 ≥60.0, 무학습 3런 전부 ≤59.2 (겹침 없음). 단 간격 0.8pp에 n=3.
  → **C32로 각 8회 반복.** 사전기준: 평균차 >8%p **그리고** 두 분포 min/max 비겹침.

### ★C28b. 선천 반사 재측정 — 절대부호는 런마다 뒤집히나 **변조폭은 재현** (2026-08-26)
연결 pull 수정 후 재실행하니 판정이 뒤집힘(1차 "접근반사 YES" → 2차 "NO"). 원인 분석:

| 측정 | 1차 | 2차 | 재현성 |
|---|---|---|---|
| good만 좌 / 우 (절대) | −0.605 / +0.133 | +0.256 / +0.963 | **부호 뒤집힘** |
| **good 좌↔우 차이** | **0.738** | **0.707** | **재현됨** |
| **bad 좌↔우 차이** | 0.097 | 0.130 | 재현됨(good의 1/7) |

- **런마다 변하는 상수 오프셋**이 절대부호를 지배하고, 단서 변조폭(≈0.72)만 재현된다.
  내 판정식이 절대부호 임계값을 써서 같은 현상을 두 번 다르게 읽었다(**내가 비판했던 ill-posed 지표와 동류**).
- **`food_memory_*_to_motor` n=0은 내 결함이었음** — 연결 pull 후 n=7562/7586, g=5.0으로 **실재**.
  "학습된 장소기억에 출력경로 없음" 가설 **철회**.
- **여전히 유효**: `good_food_to_motor` n≈15000, **g=25.0** 직결 배선 실재. good 변조폭이 bad의 7배.
- **중대 함의**: 좌/우 강제선택 프로브는 전부 절대부호로 정답을 센다 → 오프셋이 런마다 부호를 바꾸면
  같은 뇌가 100%도 0%도 낸다(C22b·C27의 이체제 요동 정체). **오프셋 보정 없이는 모든 강제선택 수치가
  불안정.** 무학습 대조로도 못 거른다(대조군도 자기 런 오프셋을 가짐).
