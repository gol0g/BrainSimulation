#!/bin/bash
# 궤적 덤프: biletaxis 켠/끈 마지막 3ep 궤적 저장 → orbit/오버슈트 눈으로 관찰.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 0 --episodes 20 --seed 0"
rm -rf forager_brain_CODE CODE
echo "=== OFF ==="
python -u $S $B --traj-dump $D/biletaxis_traj_off.npz 2>&1 | grep -E "traj-dump|goal-dist"
rm -rf forager_brain_CODE CODE
echo "=== biletaxis0.5 ==="
python -u $S $B --biletaxis --biletaxis-gain 0.5 --traj-dump $D/biletaxis_traj_on.npz 2>&1 | grep -E "traj-dump|goal-dist|biletaxis-align"
echo "=== DONE ==="
