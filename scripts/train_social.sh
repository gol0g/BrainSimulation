#!/bin/bash
# C4 사회 개념 훈련: food_hidden(음식 직접시각 차단) → NPC 단서로만 음식 찾기.
# 관찰학습(NPC 따라가 먹음→사회경로 학습) 유도. 이후 npc_call/npc_social_rich 재프로브.
# 전체 로그 tee(진행 모니터 가능). 스모크 통과: 학습가능(good_eaten>0, 생존 66%).
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$REPO/checkpoints/brain_social_60ep.npz
LOG=/tmp/c4_social.log
mkdir -p $REPO/checkpoints
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== C4 사회 개념 훈련 60ep (food_hidden) → $W ==="
python -u $REPO/backend/genesis/forager_brain.py --episodes 60 --render none \
  --food-hidden --save-weights "$W" 2>&1 | tee "$LOG" \
  | grep -iE "C4|Episode [0-9]+/60 Summary|Survival|saved|저장|Traceback|Error"
echo "=== DONE: $W ==="
