#!/bin/bash
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 10 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --episodes 40 --seed 0"
rm -rf forager_brain_CODE CODE
echo "=== exclude-OFF ==="
python -u $S $B 2>&1 | grep -E "biletaxis-align|goal-dist|plan-value"
rm -rf forager_brain_CODE CODE
echo "=== factored(exclude-ON) ==="
python -u $S $B --place-value-food-exclude 2>&1 | grep -E "biletaxis-align|goal-dist|plan-value"
echo "=== DONE ==="
