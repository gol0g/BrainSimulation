#!/bin/bash
# C4 커리큘럼 사회 훈련: food_hidden 점진 램프(초반 가시→후반 은닉) 150ep + 저장.
# food_hidden 단독(60ep) 형성 실패 후속 — 커리큘럼이 부트스트랩 넘나. 스모크: 4ep GOOD FOOD 170.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$REPO/checkpoints/brain_social_curriculum_150ep.npz
LOG=/tmp/c4_curriculum150.log
mkdir -p $REPO/checkpoints
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== C4 커리큘럼 사회 훈련 150ep → $W ==="
python -u $REPO/backend/genesis/forager_brain.py --episodes 150 --render none \
  --food-hidden-curriculum --save-weights "$W" 2>&1 | tee "$LOG" \
  | grep -iE "C4|Episode [0-9]+/80 Summary|Survival|saved|저장|Traceback|Error"
echo "=== DONE: $W ==="
