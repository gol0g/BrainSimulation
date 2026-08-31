#!/bin/bash
# C69: 효과 포화의 원인 — 가중치가 상한에 붙었는가.
#
# C67: 학습률 20배(0.02→0.4)에도 효과가 -0.005 근처로 불변 = 포화. 병목이 학습률이 아니다.
# 후보: w_max=30에 이미 도달해 더 커질 수 없음. 학습 후 D1 가중치 분포를 직접 본다.
#
# 판정: 평균이 w_max(30) 근처이고 std가 작으면 상한 포화 → w_max를 올려 재시험.
#       평균이 낮고 std가 크면 상한 문제 아님 → 다른 병목(영향력 자체의 한계).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for WM in 30 100; do
  echo "########## w_max = $WM (60ep) ##########"
  for S in 0 1; do
    printf "  seed=%s: " "$S"
    timeout 3600 python reflex_override_task.py --episodes 60 --seed "$S" \
      --real-rstdp --crossed --epsilon 0.6 --bias 25 --w-max "$WM" \
      --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 \
      | grep -E "D1가중치|^=>" | tr '\n' ' '
    echo
  done
done
