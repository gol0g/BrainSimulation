#!/bin/bash
# factored value: place-value가 음식 DA 제외(장소고정 보상만) → goal-gradient 회복하나.
# 음식10 + biletaxis+brake+gate. exclude OFF vs ON, align+goal-dist. align 회복하면 factoring 작동.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 10 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --episodes 40"
for seed in 0 1 2; do
  rm -rf forager_brain_CODE CODE
  echo "=== exclude-OFF seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "biletaxis-align|goal-dist|plan-value|mean_cool_dwell_ratio:"
  rm -rf forager_brain_CODE CODE
  echo "=== exclude-ON(factored) seed=$seed ==="
  python -u $S $B --place-value-food-exclude --seed $seed 2>&1 | grep -E "biletaxis-align|goal-dist|plan-value|mean_cool_dwell_ratio:"
done
echo "=== DONE ==="
