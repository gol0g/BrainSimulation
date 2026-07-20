#!/bin/bash
# serial reversal: cue 8ep마다 반전(~5회). cv_high가 매번 부호 뒤집으면 유연성 유지(반복 규칙변화 견딤).
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
echo "=== serial reversal (8ep마다) 40ep seed0 ==="
python -u $S --task integrated --zone-circle --appetitive-place --v3-klino --sparse-reward --start-far --replay-to-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --place-value-food-exclude --v3-olf --wta-arbitration --wta-cue-bid --cue-reversal-period 8 --episodes 40 --seed 0 2>&1 | grep -iE "serial-cv|serial-reversal"
echo "=== DONE ==="
