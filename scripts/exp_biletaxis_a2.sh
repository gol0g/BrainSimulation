#!/bin/bash
# A2 검증: biletaxis OFF vs ON. align이 널(0.5) 넘고 dwell이 널(0.07) 넘나.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
mkdir -p "$D"
BASE="--task place_pref --zone-circle --appetitive-place --start-far --sparse-reward --n-food 0 --episodes 25 --seed 0"
cd ~/pygenn_test

echo "########## OFF (biletaxis 없음) ##########"
rm -rf forager_brain_CODE CODE
python -u $R $BASE --output "$D/a2_off_seed0.json" 2>&1 | grep -iE "\[ep |goal-dist:|mean_cool_dwell|Traceback|Error" | tail -30

echo "########## ON (biletaxis gain 0.5) ##########"
rm -rf forager_brain_CODE CODE
python -u $R $BASE --biletaxis --biletaxis-gain 0.5 --output "$D/a2_on_seed0.json" 2>&1 | grep -iE "\[ep |goal-dist:|biletaxis-align:|last_5_align|mean_cool_dwell|Traceback|Error" | tail -35
echo "########## DONE ##########"
