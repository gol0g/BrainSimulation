#!/bin/bash
# E075: H007 판별 — 환경 8개 x 뇌 5반복. 환경 특성과 효과의 상관을 잰다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

echo "=== 조작검증: 전달 생존 ==="
timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 \
  --bias 300 --bias-at-d1 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 \
  --episodes 6 --brain-seed 0 --env-seed 0 2>&1 | grep -E "학습\]"

COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"
STATIC="--epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"

for E in 0 1 2 3 4 5 6 7; do
  echo "########## env=$E ##########"
  for B in 0 1 2 3 4; do
    for CFG in "기준1.0x 0.08 0.15" "11.25x 0.30 0.05"; do
      set -- $CFG
      printf "  env%s b%s %-9s 수리: " "$E" "$B" "$1"
      OUT=$(timeout 3600 python reflex_override_task.py $COMMON --brain-seed "$B" --env-seed "$E" \
        --learn-sparsity "$2" --reflex-sparsity "$3" 2>&1)
      echo "$OUT" | grep -E "^=>" || { echo "[실패]"; echo "$OUT" | tail -3; }
      printf "  env%s b%s %-9s 정적: " "$E" "$B" "$1"
      OUT=$(timeout 3600 python reflex_override_task.py $STATIC --brain-seed "$B" --env-seed "$E" \
        --learn-sparsity "$2" --reflex-sparsity "$3" 2>&1)
      echo "$OUT" | grep -E "^=>" || { echo "[실패]"; echo "$OUT" | tail -3; }
    done
  done
done
