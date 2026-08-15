#!/bin/bash
# D29: 일반 개념학습 더(250ep→+100=350ep) → seq-WM 부산물 커지나. D28 lead(일반학습이 seq-WM 만듦).
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_concepts_350ep.npz
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== D29 일반 이어훈련 250ep→+100 ==="
python -u $R/backend/genesis/forager_brain.py --episodes 100 --render none \
  --load-weights $R/checkpoints/brain_concepts_250ep.npz --save-weights "$W" 2>&1 \
  | grep -iE "Survival|saved|Traceback|Error" | tail -3
echo "=== seq-choice 프로브(350ep) ==="
for i in 1 2 3; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/seq_choice_probe.py --load-weights "$W" --inhib-wm -200 --trials 100 2>&1 \
    | grep -iE "SEQ-CHOICE" | sed "s/^/[run $i] /"
done
echo "=== DONE ==="
