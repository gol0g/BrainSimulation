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

## 재건 순서

1. ~~GitHub 4월판 clone~~ ✓
2. ~~생존 자산 38개 통합~~ ✓
3. ~~인터페이스 명세 역산~~ ✓ (이 문서)
4. `run_v2_tasks.py` 재구현 — 위 CLI/마커/스키마 준수
5. `m5_smoke.sh`로 M5 경로 검증 (`sel=` 회복 확인)
6. `comp_wm2.sh` 재현 → 7/11 지점 복귀
7. 조합 컨텍스트 + 창발 WM 재도전
