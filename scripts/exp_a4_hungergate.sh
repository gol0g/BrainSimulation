#!/bin/bash
# A4 검증: hunger-gate arbitration. 음식10 존재.
# 성공기준(#61): hunger-gate가 생존(steps) OFF수준 유지 + goal-dist 개선.
# brake만이면 forage 방해로 steps 급감(진단 #56/#59).
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
BASE="--task place_pref --zone-circle --appetitive-place --start-far --sparse-reward --n-food 10 --episodes 25 --seed 0"
run () { cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; echo "########## $1 ##########"; python -u $R $BASE $2 --output "$D/$3" 2>&1 | grep -iE "mean_steps:|mean_pi|total_good|goal-dist:|biletaxis-align:|mean_cool_dwell|last_5_mean_dwell|Traceback|Error" | tail -9; }
run "OFF (forage만, biletaxis 없음)" "" "a4_off.json"
run "biletaxis+brake (forage 방해 예상)" "--biletaxis --biletaxis-gain 0.5 --biletaxis-brake" "a4_brake.json"
run "biletaxis+brake+hunger-gate (arbitration)" "--biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate" "a4_gated.json"
echo "########## DONE ##########"
