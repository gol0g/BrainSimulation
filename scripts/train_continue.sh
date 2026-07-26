#!/bin/bash
# 가중치 이어서 훈련 (효율). 인자: 로드ep 추가ep 총ep
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOAD=$REPO/checkpoints/brain_concepts_${1}ep.npz
ADD=${2}; TOTAL=${3}
SAVE=$REPO/checkpoints/brain_concepts_${TOTAL}ep.npz
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== 이어훈련: ${1}ep 로드 + ${ADD}ep → ${TOTAL}ep ==="
python -u $REPO/backend/genesis/forager_brain.py --episodes $ADD --render none \
  --load-weights "$LOAD" --persist-learning --save-weights "$SAVE" 2>&1 | grep -iE "Survival|Reward Freq|SAVE|Loaded|Traceback|Error" | tail -8
echo "=== DONE: $SAVE ==="
