#!/bin/bash
# V3-A2 reversal smoke — thermal 반전 후 회피 회복(value 재학습)되나. recovery ON.
set -u
SD="/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis"
cd ~/pygenn_test && rm -rf forager_brain_CODE
echo "=== V3-A2 smoke (place_pref, 20ep, flip@10, recovery=0.02) === $(date)"
PYTHONUNBUFFERED=1 stdbuf -oL python -u "$SD/run_v2_tasks.py" \
  --task place_pref --episodes 20 --seed 0 --v3-klino \
  --v3-recovery 0.02 --thermal-reversal \
  --output /tmp/v3_a2_smoke.json
echo "=== DONE === $(date)"
