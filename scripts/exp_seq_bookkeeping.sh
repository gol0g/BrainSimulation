#!/bin/bash
# D19: seq에 replay_swr 배선 fix 검증. vmap_std 0→>0(value-map 학습)되고 order_rate 움직이나.
# 희소WM(-200)+패턴래치+replay fix. 전체 로그 tee.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
RUN=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
LOG=/tmp/d19d_seq.log
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "=== D19d bookkeeping 상한 25ep (뇌WM 없이 visited_a로 목표전환) ==="
python -u "$RUN" --task integrated --seq-task --seq-nav \
  --inhib-wm -200 --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 4 \
  --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 25 --seed 0 \
  --output "$D/d19d_bookkeeping.json" 2>&1 | tee "$LOG" \
  | grep -iE "\[ep|최종순서율|last_5|vmap_std|Traceback|Error"
echo "=== DONE ==="
