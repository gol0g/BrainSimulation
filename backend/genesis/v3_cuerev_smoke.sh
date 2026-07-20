#!/bin/bash
set -u
SD="/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis"
cd ~/pygenn_test && rm -rf forager_brain_CODE
echo "=== cue-reversal smoke (olfactory, flip@10) ==="
PYTHONUNBUFFERED=1 stdbuf -oL python -u "$SD/run_v2_tasks.py" \
  --task olfactory --episodes 20 --seed 0 --v3-olf --v3-cue-eta 0.7 --cue-reversal \
  --output /tmp/cuerev.json 2>&1 | grep -E "cue-reversal|pre-flip|post-flip|적응"
echo DONE
