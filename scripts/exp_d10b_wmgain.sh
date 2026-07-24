#!/bin/bash
# D10b: WM 되먹임 강도 스캔. bistable 래치 생기나(WM_latch 양수 유지) + correct 오르나.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
BASE="--task integrated --seq-task --seq-nav --seq-wm --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 0 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 25 --seed 0"
run () { cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; echo "########## WM gain ×$1 ##########"; python -u $R $BASE --seq-gain $1 --output "$D/d10b_g$2.json" 2>&1 | grep -iE "seq-gain\]|WM latch|최종순서율|Traceback|Error" | tail -4; }
run 4 "4"
run 8 "8"
echo "########## DONE ##########"
