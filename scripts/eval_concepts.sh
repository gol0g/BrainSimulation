#!/bin/bash
# 개념 형성 평가: 훈련된 뇌 로드 → 4 개념 테스트 점수.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$REPO/checkpoints/brain_concepts_50ep.npz
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== 개념 평가 (baseline) ==="
python -u $REPO/backend/genesis/evaluate_concepts.py --load-weights "$W" --test all 2>&1 | tail -40
echo "=== DONE ==="
