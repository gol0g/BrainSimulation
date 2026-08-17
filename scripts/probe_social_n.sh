#!/bin/bash
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=${1:?weights}
OUT=/tmp/social_probe_n.txt; : > $OUT
for i in 1 2 3 4 5 6 7 8; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test npc_call 2>&1 \
    | grep -i "NPC Call Response" >> $OUT
done
echo "=== npc_call 8런 ==="; cat $OUT
awk -F'[: %]+' '{s+=$4; n++} END{if(n)print "평균: " s/n "% (n=" n ")"}' $OUT
