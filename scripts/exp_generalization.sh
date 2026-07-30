#!/bin/bash
# Test2 범주 일반화: 훈련 안 본 변형(강도무작위/노이즈/부분마스크) good/bad 변별 유지 확인. 암기 vs 개념. 3회.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_concepts_250ep.npz
for i in 1 2 3; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test generalization 2>&1 | grep -iE "Generalization" | sed "s/^/[run $i] /"
done
echo "=== DONE ==="
