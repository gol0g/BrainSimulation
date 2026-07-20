#!/bin/bash
# 일반화(반대 valence): 같은 지도→방향→brake가 *회피*도 하나. 원형 aversive(열) 구역.
# biletaxis=고value(안전)쪽 조향=회피, brake=안전서 감속=정착. cool_dwell 높을수록 회피 잘함.
# #67(한 메커니즘 접근/회피 둘 다)의 항법판. OFF vs biletaxis+brake 3seed 40ep 음식0.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
# aversive: --appetitive-place 없음(열 구역=나쁨). --start-far로 멀리서 출발.
B="--task place_pref --zone-circle --v3-klino --start-far --replay-to-klino --n-food 0 --episodes 40"
for seed in 0 1 2; do
  rm -rf forager_brain_CODE CODE
  echo "=== OFF seed=$seed ==="
  python -u $S $B --seed $seed 2>&1 | grep -E "mean_cool_dwell_ratio:|goal-dist"
  rm -rf forager_brain_CODE CODE
  echo "=== biletaxis+brake seed=$seed ==="
  python -u $S $B --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --seed $seed 2>&1 | grep -E "mean_cool_dwell_ratio:|goal-dist"
done
echo "=== DONE ==="
