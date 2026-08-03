#!/bin/bash
# 무작위 목표 chance floor: 항법은 하되 A→B 순서정책 없음(목표 무작위). order_rate 우연 기준선.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
RUN=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
LOG=/tmp/d19e_seq.log
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== D19e 무작위목표 chance floor 25ep ==="
python -u "$RUN" --task integrated --seq-task --seq-nav --seq-random-target \
  --inhib-wm -200 --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 4 \
  --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 25 --seed 0 \
  --output "$D/d19e_randfloor.json" 2>&1 | tee "$LOG" \
  | grep -iE "\[ep|최종순서율|last_5|Traceback|Error"
echo "=== DONE ==="
