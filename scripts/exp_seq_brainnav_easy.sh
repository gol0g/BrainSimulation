#!/bin/bash
# D23: 뇌 내부 목표항법(학습 value map→motor V편향). biletaxis 내부화. floor 대조.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
RUN=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== D24 뇌항법+easy-survival 25ep ==="
python -u "$RUN" --task integrated --seq-task --seq-nav --seq-wm --seq-pattern-latch --seq-brain-nav --easy-survival \
  --inhib-wm -200 --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 4 --episodes 25 --seed 0 \
  --output "$D/d24_brainnav_easy.json" 2>&1 | tee /tmp/d24_seq.log \
  | grep -iE "\[ep|최종순서율|last_5"
echo "=== DONE ==="
