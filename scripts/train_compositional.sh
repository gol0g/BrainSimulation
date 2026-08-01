#!/bin/bash
# C3 강화: danger_food_ratio 0.3→0.6 (위험-음식 결합 2배) 120ep 훈련 + 저장.
# 목표: graded value×danger 합성 강화(brave-for-good 다수화). 이후 compositional_graded 재프로브.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$REPO/checkpoints/brain_compositional_120ep.npz
mkdir -p $REPO/checkpoints
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== C3 강화훈련 120ep (danger_food_ratio=0.6) → $W ==="
python -u $REPO/backend/genesis/forager_brain.py --episodes 120 --render none \
  --danger-food-ratio 0.6 --save-weights "$W" 2>&1 \
  | grep -iE "Survival|Reward Freq|danger_food_ratio|saved|저장|Traceback|Error" | tail -10
echo "=== DONE: $W ==="
