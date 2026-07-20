#!/bin/bash
# 통합 world 궤적: OFF vs factored. factored가 목표로 항법하나 눈으로(capstone).
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research
B="--task integrated --zone-circle --appetitive-place --v3-klino --sparse-reward --start-far --replay-to-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --episodes 20 --seed 1"
rm -rf forager_brain_CODE CODE
echo "=== integ OFF ==="
python -u $S $B --traj-dump $D/integ_traj_off.npz 2>&1 | grep -E "biletaxis-align|traj-dump"
rm -rf forager_brain_CODE CODE
echo "=== integ factored ==="
python -u $S $B --place-value-food-exclude --traj-dump $D/integ_traj_factored.npz 2>&1 | grep -E "biletaxis-align|traj-dump"
echo "=== DONE ==="
