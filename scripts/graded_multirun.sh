#!/bin/bash
# C19: 정본 550ep의 graded 합성을 다중런으로 — 스코어카드 마지막 미측정 칸.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"; source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=${1:-$R/checkpoints/brain_concepts_550ep.npz}
for i in 1 2 3 4; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  V=$(python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test compositional_graded 2>/dev/null \
      | grep -oE "MAX diff across danger = \+[0-9.]+" | grep -oE "[0-9.]+")
  echo "GRADED run$i: $V"
done
