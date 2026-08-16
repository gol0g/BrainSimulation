#!/bin/bash
# C5: 사회 과제 재설계 훈련(NPC 먹은자리 리스폰+food_hidden) 60ep → npc_call 프로브.
# 이전 6접근은 우회로 있었음. 이번엔 NPC 관찰이 음식 발견 유일 경로. 스모크: 3ep GOOD FOOD 87.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_social_task_60ep.npz
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== C5 사회과제 훈련 60ep ==="
python -u $R/backend/genesis/forager_brain.py --episodes 60 --render none \
  --social-task --save-weights "$W" 2>&1 | grep -iE "C5|Survival|saved|Traceback|Error" | tail -4
echo "=== 사회 프로브 ==="
bash $R/scripts/probe_social.sh "$W" 2>&1 | grep -iE "call [0-9]|rich [0-9]"
echo "=== DONE ==="
