#!/bin/bash
# biletaxis 내부 신호 직접 측정(lesson #66): 명령 turn 부호가 실제 목표방향과 맞는 비율.
# >0.5 견고 = 학습 value 지도가 목표를 실제로 가리킴(방향 신호 진짜, 행동 wash는 노이즈 탓).
# ≈0.5 = 신호 없음(깨끗한 음성). 3-seed 40ep 음식0.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
B="--task place_pref --appetitive-place --zone-circle --v3-klino --sparse-reward --start-far --replay-to-klino --n-food 0 --biletaxis --episodes 40"
for seed in 0 1 2; do
  rm -rf forager_brain_CODE CODE
  echo "=== seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "biletaxis-align|mean_cool_dwell_ratio:"
done
echo "=== DONE ==="
