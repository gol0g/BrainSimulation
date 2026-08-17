#!/bin/bash
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"; source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
for inh in -30 -100 -300; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/sts_inhib_one.py $inh 2>&1 | grep -E "INHRESULT|RuntimeError|Error"
done
