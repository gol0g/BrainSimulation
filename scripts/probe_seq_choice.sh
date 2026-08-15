#!/bin/bash
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=${1:-$R/checkpoints/brain_seq_penalty_40ep.npz}
echo "### seq-choice: $(basename $W) ###"
for i in 1 2 3 4 5; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/seq_choice_probe.py --load-weights "$W" --inhib-wm -200 --trials 100 2>&1 \
    | grep -iE "SEQ-CHOICE|Traceback|Error" | sed "s/^/[run $i] /"
done
echo "### DONE ###"
