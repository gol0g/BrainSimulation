#!/bin/bash
# arbitration: 목표항법을 satiety 게이팅(배부를때만) → forage와 상태별 분리. 음식 존재.
# 게이팅이 생존(step수) OFF 수준 유지 + goal거리 개선이면 = 경쟁 구동원 arbitration 성공.
# OFF vs biletaxis+brake+hunger-gate 3seed 40ep.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 10 --episodes 40"
for seed in 0 1 2; do
  rm -rf forager_brain_CODE CODE
  echo "=== OFF seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "goal-dist|mean_cool_dwell_ratio:"
  rm -rf forager_brain_CODE CODE
  echo "=== gated seed=$seed ==="
  python -u $S $B --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --biletaxis-hunger-gate --seed $seed 2>&1 | grep -E "goal-dist|biletaxis-align|mean_cool_dwell_ratio:"
done
echo "=== DONE ==="
