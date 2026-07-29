#!/bin/bash
# 시각 강제선택 프로브: good/bad 명확 배치, 좋은쪽 조향률. capacity vs behavioral 판별. 3회.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_concepts_250ep.npz
for i in 1 2 3; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test visual_discrim 2>&1 | grep -iE "Visual Discrim" | sed "s/^/[run $i] /"
done
echo "=== DONE ==="
