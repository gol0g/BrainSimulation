#!/bin/bash
# E070 본 실험: 대역폭 비율과 학습 효과의 관계 (H004 판별)
# 사전등록: research/experiments/E070.md
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -400 --reflex-w 3 --episodes 60"

run_pair () {
  local label="$1" ls="$2" rs="$3" seed="$4"
  printf "  %-12s seed=%s 수리: " "$label" "$seed"
  timeout 3600 python reflex_override_task.py $COMMON --seed "$seed" \
    --learn-sparsity "$ls" --reflex-sparsity "$rs" 2>&1 | grep -E "^=>"
  printf "  %-12s seed=%s 정적: " "$label" "$seed"
  timeout 3600 python reflex_override_task.py --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 --episodes 60 --seed "$seed" \
    --learn-sparsity "$ls" --reflex-sparsity "$rs" 2>&1 | grep -E "^=>"
}

for S in 0 1 2; do
  echo "--- seed=$S ---"
  run_pair "기준1.0x"   0.08 0.15 "$S"
  run_pair "학습3.75x"  0.30 0.15 "$S"
  run_pair "반사3.0x"   0.08 0.05 "$S"
  run_pair "양쪽11.25x" 0.30 0.05 "$S"
done
