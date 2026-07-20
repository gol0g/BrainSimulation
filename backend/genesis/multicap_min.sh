#!/bin/bash
# 최소: 항법+냄새변별 seed0 하나. PI>0(변별)+align 높음(항법)이면 다능력 공존.
# (대조 항법만: align 0.821, PI -0.16, good 336 — 이미 얻음)
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
echo "=== 항법+냄새변별 seed0 ==="
python -u $S --task integrated --zone-circle --appetitive-place --v3-klino --sparse-reward --start-far --replay-to-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --place-value-food-exclude --v3-olf --episodes 25 --seed 0 2>&1 | grep -E "biletaxis-align|mean_steps:|total_good:|mean_pi"
echo "=== DONE ==="
