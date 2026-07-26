#!/bin/bash
# C1 청각 부호 경험적 수리: flip OFF vs ON, call_semantics 점수 비교. 데이터가 방향 결정.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_concepts_250ep.npz
E=$R/backend/genesis/evaluate_concepts.py
run(){ cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; echo "########## $1 ##########"; python -u $E --load-weights "$W" --test call_semantics $2 2>&1 | grep -iE "Call Semantics|Call Response|PASS|FAIL|flip" | tail -3; }
run "flip OFF (원래 배선)" ""
run "flip ON (좌/우 반전)" "--flip-audio"
echo "########## DONE ##########"
