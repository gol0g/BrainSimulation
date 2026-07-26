#!/bin/bash
# C2 파일럿(단일시드 스크린): 회피 강화 훈련 → selectivity가 baseline 0.64 넘나.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_avoid_50ep.npz
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== 회피강화 훈련 50ep (d2_eta 0.0006, dip 0.8) ==="
python -u $R/backend/genesis/forager_brain.py --episodes 50 --render none \
  --d2-eta 0.0006 --dip-mag 0.8 --save-weights "$W" 2>&1 | grep -iE "C2\]|Survival|Reward Freq|SAVE" | tail -6
echo "=== selectivity 평가 ==="
python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test selectivity 2>&1 | grep -iE "Selectivity|PASS|FAIL" | tail -2
echo "=== DONE ==="
