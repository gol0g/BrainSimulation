#!/bin/bash
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"; source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
for inh in 0 -20 -60; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/d1_inhib_test.py $inh 2>&1 | grep -E "D1RESULT|Error|Traceback"
done
