#!/bin/bash
# 안정 제어기: 신호 확정 정답(80% 목표방향)이나 gain 1.0서 seed0 dwell 붕괴(과조향).
# gain 0.5로 낮춰 과조향 제거하면 정답 신호가 robust dwell로 이어지나. 3-seed 40ep 음식0.
# OFF baseline in-zone ~0.21. align도 같이 재서 신호 유지 확인.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 0 --biletaxis --biletaxis-gain 0.5 --episodes 40"
for seed in 0 1 2; do
  rm -rf forager_brain_CODE CODE
  echo "=== gain0.5 seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "biletaxis-align|mean_cool_dwell_ratio:|last_5_mean_dwell:"
done
echo "=== DONE ==="
