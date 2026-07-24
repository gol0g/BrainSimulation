#!/bin/bash
# C1/D10 seq 검증: WM 래치가 A후 상승하나(창발), --seq-wm이 순서율 올리나.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
BASE="--task integrated --seq-task --seq-nav --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 0 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 30 --seed 0"
run () { cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; echo "########## $1 ##########"; python -u $R $BASE $2 --output "$D/$3" 2>&1 | grep -iE "최종순서율|last_5_최종순서율|WM latch|Traceback|Error" | tail -5; }
run "seq-wm OFF (래치 미사용)" "" "c1_nowm.json"
run "seq-wm ON (래치로 목표전환)" "--seq-wm" "c1_wm.json"
echo "########## DONE ##########"
