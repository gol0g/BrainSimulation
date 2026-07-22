#!/bin/bash
# A2 재실험: SWR 학습경로(add_experience→replay_swr) 배선 후.
# zrew>0(zone 보상 발생) + vmap_std 상승(지도 학습) + align 널0.5 돌파 확인.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
BASE="--task place_pref --zone-circle --appetitive-place --start-far --sparse-reward --n-food 0 --episodes 25 --seed 0"
echo "########## biletaxis ON (SWR 학습경로 배선) ##########"
python -u $R $BASE --biletaxis --biletaxis-gain 0.5 --output "$D/a2b_on_seed0.json" 2>&1 | grep -iE "\[ep |biletaxis-align:|last_5_align|goal-dist:|Traceback|Error" | tail -32
echo "########## DONE ##########"
