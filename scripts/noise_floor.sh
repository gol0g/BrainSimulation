#!/bin/bash
# C17: 동일 가중치 반복 측정으로 개념 스위트의 런간 변동폭(노이즈 플로어) 확정.
# 세션 내내 84 vs 77 같은 비교를 해왔는데 변동폭을 모르면 해석 불가.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"; source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_concepts_350ep.npz
for i in 1 2 3 4 5; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  V=$(python -u $R/backend/genesis/evaluate_concepts.py --load-weights $W --test visual_discrim 2>/dev/null | grep -oE "Visual Discrim: [0-9.]+" | grep -oE "[0-9.]+")
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  S=$(python -u $R/backend/genesis/evaluate_concepts.py --load-weights $W --test sound_discrim --typed-sound 2>/dev/null | grep -oE "Sound Discrim: [0-9.]+" | grep -oE "[0-9.]+")
  echo "NOISE run$i: visual=$V sound=$S"
done
