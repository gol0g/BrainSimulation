#!/bin/bash
# C2 변별학습 강화: cortical_rstdp_eta 2배(0.0016) 훈련 50ep → selectivity 3회 분포.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_discrim_50ep.npz
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== 변별학습 강화 훈련 50ep (cortical_eta 0.0016) ==="
python -u $R/backend/genesis/forager_brain.py --episodes 50 --render none \
  --cortical-eta 0.0016 --save-weights "$W" 2>&1 | grep -iE "C2\]|Survival|Reward Freq|SAVE" | tail -5
echo "=== selectivity 3회 (분포) ==="
for i in 1 2 3; do cd ~/pygenn_test && rm -rf forager_brain_CODE CODE; python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test selectivity 2>&1 | grep -iE "Selectivity" | sed "s/^/[run $i] /"; done
echo "=== DONE ==="
