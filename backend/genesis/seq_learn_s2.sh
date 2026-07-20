#!/bin/bash
# 시퀀스 학습: 40ep 동안 in-order B 비율이 0.25(초기)서 오르면 A→B 순서 학습.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
echo "=== 시퀀스 학습 ON 40ep seed2 ==="
python -u $S --task integrated --seq-task --v3-klino --sparse-reward --replay-to-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --place-value-food-exclude --episodes 40 --seed 2 2>&1 | grep -iE "시퀀스" | head
echo "=== DONE ==="
