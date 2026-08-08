#!/bin/bash
# C4 사회 드라이브 훈련: curriculum(부트스트랩 절벽 처리) + social_drive(NPC 근접 내재보상) 60ep.
# 보상기반 훈련 4가지 실패 후 근본 다른 접근 — 사회 attention을 음식과 무관히 부트스트랩.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$REPO/checkpoints/brain_social_drive_60ep.npz
LOG=/tmp/c4_social_drive.log
mkdir -p $REPO/checkpoints
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== C4 사회 드라이브 훈련 60ep (curriculum+social_drive) → $W ==="
python -u $REPO/backend/genesis/forager_brain.py --episodes 60 --render none \
  --food-hidden-curriculum --social-drive --save-weights "$W" 2>&1 | tee "$LOG" \
  | grep -iE "C4|Episode [0-9]+/60 Summary|Survival|saved|저장|Traceback|Error"
echo "=== DONE: $W ==="
