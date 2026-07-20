#!/bin/bash
unset PATH && unset LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export PYTHONUNBUFFERED=1
source ~/pygenn_wsl/bin/activate
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
export M5_SHUNT=1 M5_SEED=0 M5_EP=20
python -u /mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/test_context_m5_smoke.py 2>&1 | grep -iE "M5 smoke|sel=|shunt:|DONE|Error|Traceback" | head -20
echo "=== END ==="
