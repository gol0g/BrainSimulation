#!/bin/bash
# 브레이크 돌파 5-seed 확정(#43/#49): seed 3,4 추가. 0,1,2는 이미 3/3 승(거리+체류).
# 5-seed 모두 brake<OFF(거리) & brake>OFF(체류)면 read-out 행동수준 돌파 확정.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 0 --episodes 40"
for seed in 3 4; do
  rm -rf forager_brain_CODE CODE
  echo "=== OFF seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "goal-dist|mean_cool_dwell_ratio:"
  rm -rf forager_brain_CODE CODE
  echo "=== brake seed=$seed ==="
  python -u $S $B --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --seed $seed 2>&1 | grep -E "goal-dist|biletaxis-align|mean_cool_dwell_ratio:"
done
echo "=== DONE ==="
