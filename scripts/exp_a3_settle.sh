#!/bin/bash
# A3 검증: biletaxis 정착. dwell↑ & goal-dist↓ 가 brake/settle로 개선되나.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
BASE="--task place_pref --zone-circle --appetitive-place --start-far --sparse-reward --n-food 0 --episodes 25 --seed 0 --biletaxis --biletaxis-gain 0.5"
run () { cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; echo "########## $1 ##########"; python -u $R $BASE $2 --output "$D/$3" 2>&1 | grep -iE "biletaxis-align:|last_5_align|goal-dist:|last_5_mean_dwell|mean_cool_dwell|Traceback|Error" | tail -8; }
run "biletaxis only (A2 대조)" "" "a3_bare.json"
run "biletaxis + settle" "--biletaxis-settle" "a3_settle.json"
run "biletaxis + brake" "--biletaxis-brake" "a3_brake.json"
echo "########## DONE ##########"
