#!/bin/bash
# D26: seq-특화 훈련(save) → forced-choice 재프로브. 약한 순차-WM(~62%) 강화되나.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_seq_penalty_40ep.npz
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== D27 역순페널티 seq 훈련 40ep (save) ==="
python -u $R/backend/genesis/run_v2_tasks.py --task integrated --seq-task --seq-nav --seq-wm \
  --seq-pattern-latch --seq-brain-nav --seq-penalize-wrong --easy-survival --inhib-wm -200 --zone-cx 0.3 --zone-cy 0.3 \
  --sparse-reward --n-food 4 --episodes 40 --seed 0 --save-weights "$W" 2>&1 \
  | grep -iE "최종순서율:|save-weights|Traceback|Error" | tail -4
echo "=== forced-choice 재프로브 (seq-훈련 브레인) ==="
for i in 1 2 3 4 5; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  python -u $R/backend/genesis/seq_choice_probe.py --load-weights "$W" --inhib-wm -200 --trials 100 2>&1 \
    | grep -iE "SEQ-CHOICE" | sed "s/^/[run $i] /"
done
echo "=== DONE ==="
