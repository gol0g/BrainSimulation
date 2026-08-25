#!/bin/bash
# C18: 정본 스코어카드를 다중런(5회) 평균±범위로 재측정 — C17 노이즈(±6~7) 반영.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"; source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=${1:-$R/checkpoints/brain_concepts_350ep.npz}
for t in spatial generalization compositional; do
  for i in 1 2 3 4 5; do
    cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
    V=$(python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test $t 2>/dev/null \
        | grep -oE "(Spatial Memory|Generalization|Compositional): [0-9.]+" | grep -oE "[0-9.]+$")
    echo "MULTI $t run$i: $V"
  done
done
