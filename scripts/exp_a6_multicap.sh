#!/bin/bash
# A6 multicap 캡스톤: 한 뇌가 항법(align)+변별(PI) 동시. 둘 다 서면 다능력 공존.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
# 풀스택: klino+biletaxis+brake+hunger-gate 항법 + olf 변별, 통합world 음식有
FULL="--task integrated --zone-circle --appetitive-place --start-far --sparse-reward --n-food 15 --v3-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --v3-olf --episodes 30 --seed 0"
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "########## multicap: 항법+변별 동시 ##########"
python -u $R $FULL --output "$D/a6_multicap.json" 2>&1 | grep -iE "biletaxis-align:|last_5_align|mean_pi|total_good|mean_steps:|mean_cool_dwell|Traceback|Error" | tail -9
echo "########## DONE ##########"
