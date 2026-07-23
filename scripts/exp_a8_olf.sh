#!/bin/bash
# D8 v3-olf 검증: 피질 R-STDP 변별학습. PI가 음수(변별X)→양수(변별O)로 오르나.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
BASE="--task integrated --n-food 20 --episodes 25 --seed 0"
run () { cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; echo "########## $1 ##########"; python -u $R $BASE $2 --output "$D/$3" 2>&1 | grep -iE "mean_pi|total_good|mean_steps:|Traceback|Error" | tail -5; }
run "olf OFF (변별 없음)" "" "a8_off.json"
run "olf ON (피질 R-STDP 변별)" "--v3-olf" "a8_olf.json"
echo "########## DONE ##########"
