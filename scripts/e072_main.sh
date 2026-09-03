#!/bin/bash
# E072: 전달이 살아 있는 조건(direct_inhib -100)에서 대역폭 재검증.
# 사전등록: research/experiments/E072.md
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

echo "=== 조작검증: 전달 생존(보상>0 이어야) ==="
timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 \
  --bias 300 --bias-at-d1 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 \
  --episodes 6 --seed 0 2>&1 | grep -E "학습\]"

echo "=== 본 실험 ==="
COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"
for S in 0 1 2; do
  echo "--- seed=$S ---"
  for CFG in "기준1.0x 0.08 0.15" "양쪽11.25x 0.30 0.05"; do
    set -- $CFG
    printf "  %-12s 수리: " "$1"
    timeout 3600 python reflex_override_task.py $COMMON --seed "$S" \
      --learn-sparsity "$2" --reflex-sparsity "$3" 2>&1 | grep -E "^=>"
    printf "  %-12s 정적: " "$1"
    timeout 3600 python reflex_override_task.py --epsilon 0.6 --bias 25 \
      --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60 --seed "$S" \
      --learn-sparsity "$2" --reflex-sparsity "$3" 2>&1 | grep -E "^=>"
  done
done
