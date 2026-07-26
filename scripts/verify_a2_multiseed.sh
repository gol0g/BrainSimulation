#!/bin/bash
# A2 biletaxis align 다중시드 검증. seed0의 align 0.75가 robust한가.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
BASE="--task place_pref --zone-circle --appetitive-place --start-far --sparse-reward --n-food 0 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 20"
for S in 0 1 2; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  echo "########## seed $S ##########"
  python -u $R $BASE --seed $S --output "$D/verify_a2_s$S.json" 2>&1 | grep -iE "biletaxis-align:|last_5_align|last_5_mean_dwell|Traceback|Error" | tail -4
done
echo "########## DONE ##########"
