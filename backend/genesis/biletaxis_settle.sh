#!/bin/bash
# settle fix: 목표 근처서 조향 감쇠 → orbit 대신 정착하나. OFF vs biletaxis0.5+settle 3seed 40ep.
# 거리 ON<OFF 3/3 견고 + 궤적서 orbit 감소면 = read-out 행동수준 닫힘. seed0 궤적 덤프.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 0 --episodes 40"
for seed in 0 1 2; do
  rm -rf forager_brain_CODE CODE
  echo "=== OFF seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "goal-dist|mean_cool_dwell_ratio:"
  rm -rf forager_brain_CODE CODE
  echo "=== settle seed=$seed ==="
  if [ "$seed" = "0" ]; then TD="--traj-dump $D/biletaxis_traj_settle.npz"; else TD=""; fi
  python -u $S $B --biletaxis --biletaxis-gain 0.5 --biletaxis-settle $TD --seed $seed 2>&1 | grep -E "goal-dist|biletaxis-align|mean_cool_dwell_ratio:|traj-dump"
done
echo "=== DONE ==="
