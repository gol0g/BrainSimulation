#!/bin/bash
# 시퀀스 ablation: place→value 학습 OFF(v3-value-eta 0). 완성 횟수 안 오르면 상승=학습 덕분.
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/run_v2_tasks.py
echo "=== 시퀀스 학습 OFF(ablation) 40ep off_s1 ==="
python -u $S --task integrated --seq-task --v3-klino --v3-value-eta 0 --sparse-reward --replay-to-klino --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --place-value-food-exclude --episodes 40 --seed 1 2>&1 | grep -iE "시퀀스" | head
echo "=== DONE ==="
