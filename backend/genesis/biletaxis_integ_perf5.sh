#!/bin/bash
# 통합 성능 5-seed: seed 2,3,4. 0,1은 이미 2/2 압승(생존+50%, 먹이+90%, 열회피↑, align↑).
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
B="--task integrated --zone-circle --appetitive-place --v3-klino --sparse-reward --start-far --replay-to-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --episodes 25"
for seed in 2 3 4; do
  rm -rf forager_brain_CODE CODE
  echo "=== integ OFF seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "mean_cool_dwell_ratio:|mean_steps:|total_good:|biletaxis-align"
  rm -rf forager_brain_CODE CODE
  echo "=== integ factored seed=$seed ==="
  python -u $S $B --place-value-food-exclude --seed $seed 2>&1 | grep -E "mean_cool_dwell_ratio:|mean_steps:|total_good:|biletaxis-align"
done
echo "=== DONE ==="
