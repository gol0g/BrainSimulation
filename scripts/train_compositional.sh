#!/bin/bash
# C3 강화: danger_food_ratio 0.3→0.6 (위험-음식 결합 2배) 60ep 훈련 + 저장.
# 목표: graded value×danger 합성 강화(brave-for-good 다수화). 이후 compositional_graded 재프로브.
# 전체 출력을 로그파일에 tee(진행 episode 모니터 가능).
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$REPO/checkpoints/brain_compositional_60ep.npz
LOG=/tmp/c3_train.log
mkdir -p $REPO/checkpoints
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== C3 강화훈련 60ep (danger_food_ratio=0.6) → $W ==="
python -u $REPO/backend/genesis/forager_brain.py --episodes 60 --render none \
  --danger-food-ratio 0.6 --save-weights "$W" 2>&1 | tee "$LOG" | grep -iE "Episode|Survival|saved|저장|Traceback|Error"
echo "=== DONE: $W ==="
