#!/bin/bash
# C1 부트스트랩: 음식으로 생존시간 늘려 A→B 우연완성→value backup 부트스트랩되나. 40ep.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "########## seq-wm ON, 음식10, 40ep (부트스트랩) ##########"
python -u $R --task integrated --seq-task --seq-nav --seq-wm --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 10 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 40 --seed 0 --output "$D/c1_boot.json" 2>&1 | grep -iE "\[ep  0\]|\[ep 1[05]\]|\[ep 2[05]\]|\[ep 3[05]\]|\[ep 39\]|최종순서율|WM latch|Traceback|Error" | tail -12
echo "########## DONE ##########"
