#!/bin/bash
# D10c: WM 균형(되먹임↑ + place드라이브↓) 스캔. WM_latch 양수(A 떠나도 유지) 되나.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
BASE="--task integrated --seq-task --seq-nav --seq-wm --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 0 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 25 --seed 0"
run () { cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; echo "########## 균형 ×$1 ##########"; python -u $R $BASE --seq-gain $1 --output "$D/d10c_g$2.json" 2>&1 | grep -iE "seq-gain|WM latch|최종순서율|Traceback|Error" | tail -4; }
run 3 "3"
run 6 "6"
echo "########## DONE ##########"
