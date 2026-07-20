#!/bin/bash
# 진단: 음식공존 실패 원인=brake 감속이 forage 방해(step수 급감). brake 없이 조향만이면
# forage 방해 없이 goal 항법 돕나. 도우면 arbitration 해법=포만도로 brake 게이팅(#61).
# 음식 존재. OFF vs biletaxis(no brake) 3seed. step수(n) + goal거리 비교.
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
  echo "=== biletaxis-only(no brake) seed=$seed ==="
  python -u $S $B --biletaxis --biletaxis-gain 0.5 --seed $seed 2>&1 | grep -E "goal-dist|biletaxis-align|mean_cool_dwell_ratio:"
done
echo "=== DONE ==="
