#!/bin/bash
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
S=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
SEED=${1:-0}
echo "=== compositional context seed$SEED ==="
python -u $S --task integrated --context-select --seq-task --context-compositional --zone-cx 0.3 --zone-cy 0.3 --episodes 20 --seed $SEED 2>&1 | grep -iE "compositional|comp ctx|Traceback|Error" | tail -5
echo "=== END ==="
