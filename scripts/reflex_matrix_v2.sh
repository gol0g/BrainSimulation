#!/bin/bash
# C38b: 반사역전 학습 판정 — ε-탐욕 수정판(무작위 방향 탐색) + 조건별 3회 반복.
#
# 1차(C38) 결함 2건을 고침:
#   (a) 탐색이 **항상 정답 방향**이라 보상률 98% → 도파민 상수 → 전 시냅스 균일 포화(std→0, 변별 소멸).
#       → 무작위 방향 탐색으로 교체(우연히 맞을 때만 보상 = 대비 생성).
#   (b) 사전 정답률이 런마다 0~49%로 흔들려(오프셋 불안정) 조건 간 비교 불가.
#       → 각 조건 3회 반복해 **런 내 전후 차이**의 분포로 판정.
#
# 사전 기준: 3회 평균 (사후−사전) > +10%p 이면 그 조건에서 학습 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
BIAS="${1:-25}"
EP="${2:-40}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

run3 () {
  local label="$1"; shift
  echo "=== $label ==="
  for i in 1 2 3; do
    timeout 3600 python reflex_override_task.py --episodes "$EP" "$@" 2>&1 \
      | grep -E "사전|학습\]|D1가중치|사후|=>"
    echo "  ---"
  done
}

run3 "①②③ 전부"          --real-rstdp --crossed --epsilon 0.6 --bias "$BIAS"
run3 "①③ (교차 없음)"     --real-rstdp --epsilon 0.6 --bias "$BIAS"
run3 "②③ (정적=기계없음)" --crossed --epsilon 0.6 --bias "$BIAS"
