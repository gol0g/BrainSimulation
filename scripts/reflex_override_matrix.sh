#!/bin/bash
# C38: 반사역전 학습 — 차단요인 3개의 필요성 판정 (귀속 가능한 대조 설계)
#
# 이 세션에서 찾은 차단 요인:
#   ① 학습 기계 부재  (food_to_d1이 정적, 피질 R-STDP는 전역 스칼라)  → --real-rstdp
#   ② 위상적 감금      (food_eye→d1→direct→motor가 전부 동측, 교차 없음) → --crossed
#   ③ 탐색 부재        (결정론적 정책 → 정답 표본 0개)                  → --epsilon/--bias
#
# 각 조건을 빼고 돌려 **무엇이 필요조건인지** 귀속한다. 사전 기준:
#   성공 = 사후 정답률이 사전 대비 +10%p 이상 상승(전 조건 사전은 0.0%였음).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
BIAS="${1:-60}"     # 예비점검에서 보상>0을 만든 세기를 인자로 받음
EP="${2:-40}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

run () {
  echo "=== $1 ==="
  shift
  timeout 3600 python reflex_override_task.py --episodes "$EP" "$@" 2>&1 \
    | grep -E "사전|학습\]|D1가중치|사후|=>"
}

run "①②③ 전부 (기계+교차+탐색)"      --real-rstdp --crossed --epsilon 0.6 --bias "$BIAS"
run "②③ (기계 없음=정적) → ①필요?"    --crossed --epsilon 0.6 --bias "$BIAS"
run "①③ (교차 없음) → ②필요?"          --real-rstdp --epsilon 0.6 --bias "$BIAS"
run "①② (탐색 없음) → ③필요?"          --real-rstdp --crossed --epsilon 0.0
