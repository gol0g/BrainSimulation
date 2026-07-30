#!/bin/bash
# 소리 강제선택: typed_sound OFF(대조) vs ON. 시각훈련 가중치가 소리에도 적용되나(재훈련X).
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_concepts_250ep.npz
E=$R/backend/genesis/evaluate_concepts.py
run(){ cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; echo "########## $1 ##########"; for i in 1 2; do python -u $E --load-weights "$W" --test sound_discrim $2 2>&1 | grep -iE "Sound Discrim|typed_sound" | sed "s/^/[run $i] /"; done; }
run "typed_sound OFF (대조)" ""
run "typed_sound ON (타입×방향 소리)" "--typed-sound"
echo "########## DONE ##########"
