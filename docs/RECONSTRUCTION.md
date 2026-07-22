# 재건 명세 (2026-07-20)

2026-07 디스크 사고로 5~7월 3개월치 작업이 소실됐다. 이 문서는 **살아남은 증거로부터 역산한 재구현 명세**다.

## 소실 범위 (실측)

| 대상 | 상태 |
|---|---|
| origin/main `6762572` | **2026-04-18** 커밋. 5~7월 미push |
| 로컬 `.git` | index·objects 손상, 복구 불가 |
| `run_v2_tasks.py` | **완전 소실** (GitHub에 존재한 적 없음, 로컬본 손상) |
| `test_context_m5_smoke.py` | **완전 소실** |
| `forager_brain.py` | 4월판 687,671B 온전 / 7월판 766,689B 손상 → **79KB 차이가 소실분** |
| genesis `*.sh` | 169개 중 30개 생존 |
| `v2_results/*.json` | 132개 중 1개만 부분 생존 (21/30 ep에서 절단) |

## 1순위: `run_v2_tasks.py` 재구현

7월 실험 24개가 전부 이 러너를 호출한다. 4월 `forager_brain.py`와 **플래그 공유가 `--episodes` 하나뿐**이므로,
이것은 forager_brain 위에 얹힌 별도 실험 하네스 레이어다.

### CLI 인터페이스 (생존 스크립트에서 추출, 34개)

```
--task {integrated|place_pref|olfactory|reversal}
--episodes N   --seed N   --output PATH   --n-food N

# 환경/과제
--zone-circle          --zone-cx F  --zone-cy F
--appetitive-place     --start-far
--sparse-reward        --thermal-reversal
--cue-reversal         --cue-reversal-period N

# 주행(biletaxis) 계열
--biletaxis            --biletaxis-gain F
--biletaxis-brake      --biletaxis-hunger-gate
--biletaxis-settle PATH

# v3 회로
--v3-klino   --v3-olf   --v3-recovery
--v3-value-eta F   --v3-cue-eta F
--replay-to-klino      --place-value-food-exclude
--value-max F

# 시퀀스 / 컨텍스트 (7월 최전선)
--seq-task   --seq-nav   --seq-wm   --seq-gain F
--context-select         --context-compositional

# 중재
--wta-arbitration        --wta-cue-bid

# 계측
--traj-dump PATH
```

### 표준 베이스 플래그 조합 (스크립트에서 그대로 복원)

```bash
# 통합 항법 베이스 (biletaxis_integ_perf5.sh)
B="--task integrated --zone-circle --appetitive-place --v3-klino --sparse-reward \
   --start-far --replay-to-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake \
   --biletaxis-hunger-gate --episodes 25"

# 장소선호 베이스 (v3 계열)
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward \
   --start-far --replay-to-klino --n-food 0 --episodes 20 --seed 0"

# 항법 풀스택 (multicap.sh)
NAV="$B --place-value-food-exclude --episodes 30"
```

### 필수 stdout 마커 (grep 대상 — 재구현 시 반드시 동일 문자열로 출력)

러너의 출력은 `grep -E`로 소비된다. 아래 마커가 없으면 기존 실험 스크립트가 전부 무음이 된다.

| 마커 | 의미 |
|---|---|
| `biletaxis-align` | 양측주행 정렬도 |
| `goal-dist` | 목표까지 거리 |
| `plan-value` | 계획 가치 |
| `mean_cool_dwell_ratio:` | 쿨존 체류 비율 평균 |
| `last_5_mean_dwell:` | 말기 5ep 체류 |
| `mean_steps:` / `total_good:` / `mean_pi` | 요약 지표 |
| `traj-dump` | 궤적 덤프 확인 |
| `WM latch` | 창발 워킹메모리 래치 |
| `[compositional]` / `comp ctx` | 조합적 컨텍스트 |
| `seq-nav 정렬` / `최종순서율` / `방향성` / `시퀀스` | 시퀀스 학습 |
| `serial-cv` / `serial-reversal` | 연속 역전 |

### 결과 JSON 스키마 (생존본에서 추출)

```json
{
  "v2_runner_version": "1.0",
  "timestamp": "2026-07-12T01:05:30.341742",
  "task": "integrated",
  "seed": 1,
  "n_episodes": 30,
  "ablation": "full_baseline",
  "ablation_flags": {},
  "elapsed_sec": 2665.84,
  "episodes": [
    {
      "task_mode": "integrated", "steps": 4500, "steps_taken": 4500,
      "cool_dwell_ratio": 1.0, "performance_index": 0.0621,
      "good_eaten": 94, "bad_eaten": 83, "n_choices": 177,
      "thermal_entries": 0, "episode": 0
    }
  ]
}
```

파일명 규칙: `v2_{task}_{ablation}_seed{N}.json` → `docs/research/v2_results/`

### 지표 정의 (생존 데이터에서 역산, 21/21 에피소드 검증 완료)

```python
n_choices         = good_eaten + bad_eaten
performance_index = (good_eaten - bad_eaten) / n_choices
```

생존본 21개 에피소드 전부에서 부동소수점 오차 1e-12 이내로 일치 확인.
예: ep0 (94-83)/177 = 0.062146892655367235 (기록값과 완전 일치),
ep20 (97-98)/195 = -0.005128205128205128 (일치).

→ PI는 **좋은/나쁜 음식 선택의 정규화된 편향도**. 0 = 무선택(우연), 1 = 완벽 변별, 음수 = 역선택.
7월 최종 결과가 ~0.03이었다는 것은 사실상 변별 실패를 뜻한다.

`cool_dwell_ratio`, `thermal_entries`는 thermal/zone 과제 지표이며 생존본에서는 각각 1.0, 0으로
상수라 공식 역산 불가 — 재구현 시 정의를 새로 정해야 한다.

## 2순위: `test_context_m5_smoke.py`

M5 스모크 테스트. 출력 마커: `=== M5 smoke`, `sel=`, `shunt_good`, `shunt_bad`, `shunt:`, `DONE`.
`sel=`은 컨텍스트 선택성 — M4에서 0.50(우연 수준)에 막혀 있던 그 지표다.

## 마지막 작업 지점 (7/11 타임스탬프 순)

```
comp_wm → comp_wm2 → comp_wm_3seed → comp3_smoke → comp3_3seed
→ comp_rev_smoke → comp_rev_3seed → comp_robust_3seed
→ comp3_distractor_3seed → comp3_distractor_wm_smoke  (7/11 22:39, 마지막)
```

`comp_wm2.sh` 이후는 전부 손상 — 정확한 플래그 소실. 마지막 온전한 커맨드:

```bash
python -u run_v2_tasks.py --task integrated --context-select --seq-task --seq-wm \
  --context-compositional --zone-cx 0.3 --zone-cy 0.3 --episodes 20 --seed $SEED
```

주석: "조합+창발WM(seq-nav 없이)". 즉 **M4에서 뚫은 컨텍스트 의존 선택성을
조합적 개념 + 창발 워킹메모리로 확장**하던 중이었다 — 최종 목표("개념을 형성하는 인공 뇌")의 정면.

**단, 성공하지 못한 상태에서 끊겼다.** 유일 생존 결과(7/12 01:05, integrated, 30ep 중 21ep):
`performance_index` 평균 ~0.03, good 94 / bad 83 → 사실상 무선택. 재건 후 이 지점부터 다시 붙어야 한다.

## 소실 규모 정정: 러너만이 아니다

4월판 `ForagerBrainConfig`(L77~1414)를 실측한 결과 **`zone_circle`·`thermal`·`biletaxis`·
`klino`·`seq_*`·`context_compositional` 필드가 하나도 없다.**
즉 소실분은 `run_v2_tasks.py` 하나가 아니라 **뇌 본체의 5~7월 기능 전체**(687KB→766KB, 79KB 델타)다.
러너만 복원해도 구동할 대상이 없다. 브레인 측 기능을 함께 재구현해야 한다.

## 좋은 소식: M4 토대는 온전하다

4월판을 실측한 결과 조합적 컨텍스트가 딛고 설 기반이 살아 있다:

| 위치 | 내용 |
|---|---|
| `forager_gym.py:179` | `context_rules_enabled` — Zone A(정상)/Zone B(good↔bad 반전) |
| `forager_gym.py:778` | `agent_x > width/2` 이면 `effective_type = 1 - food_type` |
| `forager_brain.py:1266` | `context_gate_enabled` |
| `forager_brain.py:2031` | `_build_context_gate_circuit()` |
| `forager_brain.py:12225` | `_ctxval_w[ctx_side]` 보상시점 컨텍스트별 가치 갱신 (eta 0.15) |

주석에 명시: *"같은 시각 자극이지만 위치에 따라 다른 행동 — WM + PFC + hippocampal context가 필수인 과제."*
즉 `--context-select`는 재구현 불필요, 4월 코드로 바로 구동 가능하다.
소실분은 이 위에 얹혔던 **조합(compositional) 확장 + 창발 WM + biletaxis/v3 계열**이다.

## ⚠️ 실행 환경 소실 (BLOCKER)

2026-07-20 실측:

| 항목 | 상태 |
|---|---|
| GPU | ✓ RTX 3070, driver 591.86, 8GB |
| WSL 배포판 | ✗ **0개** — Ubuntu-24.04 소실 |
| `~/pygenn_wsl` venv | ✗ 소실 |
| CUDA 12.3 (WSL 내) | ✗ 소실 |

CLAUDE.md 명시: **Windows Python으로 PyGeNN 실행 불가.** WSL 재구축 전에는 단 한 줄도 실행 검증할 수 없다.

## 재건 기준점 (2026-07-21 확보)

재건 환경에서 4월판 baseline 20ep 실행 → **CLAUDE.md 기준 통과**:

| 지표 | 결과 | 기준 |
|---|---|---|
| Survival Rate | **55.0%** ✓ | >40% |
| Reward Freq | **2.63%** ✓ | >2.5% |
| Pain Composite | ✓ PASS | 전 항목 |
| 뇌 규모 | 28,323 뉴런 | — |

README 기록(400ep 61%/2.62%)과 정합적 → **재건 환경이 4월판을 정확히 재현.**
로그: `docs/research/rebuild_baseline/baseline_20ep_20260721.log`.
이후 브레인 기능 재구현 시 이 수치가 회귀 판정 기준.

환경 구축에서 걸린 것 4가지(전부 Ubuntu 24.04발, 커밋 메시지에 상술):
cuda-toolkit 메타패키지 libtinfo5 / PyGeNN PyPI 부재(소스빌드) /
pkg-config·libffi-dev 누락 / **CUDA 12.3 vs gcc 13/14** → `-ccbin g++-12` 필수.

## 재건 순서

0. ~~환경 재구축~~ ✓ (Ubuntu 24.04 + CUDA 12.3 + PyGeNN 5.4.0 소스빌드)
1. ~~GitHub 4월판 clone~~ ✓
2. ~~생존 자산 38개 통합~~ ✓
3. ~~인터페이스 명세 역산~~ ✓ (이 문서)
4. ~~4월판 baseline 재검증~~ ✓ (55%/2.63%, 기준점 확보)
5. ~~`run_v2_tasks.py` 재작성~~ ✓ (CLI/마커/스키마, 미구현 플래그 exit 2)
6. **[진행중]** `--context-select` 첫 실행 검증 (4월 M4 토대로 구동)
7. 브레인 기능 재구현 — 스크립트 타임스탬프가 알려주는 실제 개발 순서대로:
   `biletaxis`(7/1~4) → `multicap`/`v3-olf`(7/5~6) → `seq_*`(7/7~10) → `context_compositional`+`seq_wm`(7/10~11)
8. `test_context_m5_smoke.py` 재작성 → `m5_smoke.sh`로 `sel=` 회복 확인
9. `comp_wm2.sh` 재현 → 7/11 지점 복귀
10. 조합 컨텍스트 + 창발 WM 재도전 (7/12 시점 미해결 상태에서 이어감)
