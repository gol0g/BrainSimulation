#!/bin/bash
# 사회 개념 재프로브: social-60ep 가중치로 npc_call + npc_social_rich 각 5런.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=${1:-$R/checkpoints/brain_social_60ep.npz}
echo "### npc_call (weights=$(basename $W)) ###"
for i in 1 2 3 4 5; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test npc_call 2>&1 \
    | grep -iE "NPC Call Response" | sed "s/^/[call $i] /"
done
echo "### npc_social_rich ###"
for i in 1 2 3 4 5; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test npc_social_rich 2>&1 \
    | grep -iE "NPC Social" | sed "s/^/[rich $i] /"
done
echo "### DONE ###"
