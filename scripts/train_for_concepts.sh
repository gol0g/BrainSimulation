#!/bin/bash
# 개념 형성용 훈련 + 가중치 저장. 이후 evaluate_concepts.py로 개념 테스트.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$REPO/checkpoints/brain_concepts_${1:-30}ep.npz
mkdir -p $REPO/checkpoints
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== 훈련 ${1:-30}ep + 저장 → $W ==="
python -u $REPO/backend/genesis/forager_brain.py --episodes ${1:-30} --render none --save-weights "$W" 2>&1 | grep -iE "Survival|Reward Freq|saved|저장|Traceback|Error" | tail -8
echo "=== DONE: $W ==="
