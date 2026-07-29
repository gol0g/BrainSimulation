#!/bin/bash
# selectivity 분포 확인: 250ep 가중치로 eval 5회 (내부 무작위) → 0.64가 캡인지 draw인지.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_concepts_250ep.npz
for i in 1 2 3 4 5; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test selectivity 2>&1 | grep -iE "Selectivity" | sed "s/^/[run $i] /"
done
echo "=== DONE ==="
