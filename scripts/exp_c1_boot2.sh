#!/bin/bash
# C1 부트스트랩2: 존=에너지회복(생존) + curiosity 탐색. B 도달·순서학습 창발하나. 40ep.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "########## seq-wm+탐색+생존, 40ep ##########"
python -u $R --task integrated --seq-task --seq-nav --seq-wm --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 0 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 40 --seed 0 --output "$D/c1_boot2.json" 2>&1 | grep -iE "\[ep  [05]\]|\[ep 1[05]\]|\[ep 2[05]\]|\[ep 3[05]\]|\[ep 39\]|최종순서율|Traceback|Error" | tail -14
echo "########## DONE ##########"
