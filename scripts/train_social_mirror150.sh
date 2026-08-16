#!/bin/bash
# C5: 사회 과제 재설계 훈련(NPC 먹은자리 리스폰+food_hidden) 60ep → npc_call 프로브.
# 이전 6접근은 우회로 있었음. 이번엔 NPC 관찰이 음식 발견 유일 경로. 스모크: 3ep GOOD FOOD 87.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_social_mirror_150ep.npz
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== C7 사회과제+관찰회로 150ep(늦은 형성 검증) ==="
python -u $R/backend/genesis/forager_brain.py --episodes 150 --render none \
  --social-task --mirror-motor 8.0 --save-weights "$W" 2>&1 | grep -iE "C5|Survival|saved|Traceback|Error" | tail -4
echo "=== 사회 프로브 ==="
bash $R/scripts/probe_social.sh "$W" 2>&1 | grep -iE "call [0-9]|rich [0-9]"
echo "=== DONE ==="
