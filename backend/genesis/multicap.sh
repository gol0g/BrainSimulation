#!/bin/bash
# 다능력 통합: 한 뇌가 냄새-변별(--v3-olf, #60/#63) + 목표항법(biletaxis+factored) 동시?
# 통합 world. PI(변별)+align(항법)+생존 다 봄. 둘 다 서면 코히어런트 다능력 뇌(#64/#65 다능력판).
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
NAV="--task integrated --zone-circle --appetitive-place --v3-klino --sparse-reward --start-far --replay-to-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --place-value-food-exclude --episodes 30"
for seed in 0 1 2; do
  rm -rf forager_brain_CODE CODE
  echo "=== 항법만 seed=$seed ==="
  python -u $S $NAV --seed $seed 2>&1 | grep -E "biletaxis-align|mean_steps:|total_good:|mean_pi"
  rm -rf forager_brain_CODE CODE
  echo "=== 항법+냄새변별 seed=$seed ==="
  python -u $S $NAV --v3-olf --seed $seed 2>&1 | grep -E "biletaxis-align|mean_steps:|total_good:|mean_pi"
done
echo "=== DONE ==="
