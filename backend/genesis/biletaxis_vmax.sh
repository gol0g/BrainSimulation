#!/bin/bash
# confound=천장포화 가설: vmax↑→값 퍼져 goal이 food-baseline 위로→align 회복하나(모듈0, 튜닝만).
# 음식10 + biletaxis+brake+gate, vmax 1/2/3 스캔 seed0. align 오르면 포화가 병목 확인.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 10 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --episodes 30 --seed 0"
for vm in 1.0 2.0 3.0; do
  rm -rf forager_brain_CODE CODE
  echo "=== vmax=$vm ==="
  python -u $S $B --value-max $vm 2>&1 | grep -E "biletaxis-align|goal-dist|plan-value"
done
echo "=== DONE ==="
