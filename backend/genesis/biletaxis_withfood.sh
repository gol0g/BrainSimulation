#!/bin/bash
# richer 일반화: 음식 존재(경쟁 반사 구동원)서 접근+brake가 여전히 goal 항법 돕나.
# 음식 forage(반사)와 goal navigation(planning)이 한 뇌서 공존하나 (lesson #56/#59).
# appetitive goal zone + 음식 산재. OFF vs biletaxis+brake 3seed 40ep.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
# 음식 존재(n_food 기본 10). goal zone은 appetitive.
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 10 --episodes 40"
for seed in 0 1 2; do
  rm -rf forager_brain_CODE CODE
  echo "=== OFF seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "goal-dist|mean_cool_dwell_ratio:"
  rm -rf forager_brain_CODE CODE
  echo "=== brake seed=$seed ==="
  python -u $S $B --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --seed $seed 2>&1 | grep -E "goal-dist|biletaxis-align|mean_cool_dwell_ratio:"
done
echo "=== DONE ==="
