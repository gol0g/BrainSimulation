#!/bin/bash
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
echo "=== 기준 (learn 0.08 / reflex 0.15) ==="
timeout 2400 python synapse_count_probe.py 2>&1 | grep -E "SPARSITY|n="
echo "=== 학습경로 상향 (learn 0.30) ==="
timeout 2400 python synapse_count_probe.py --learn-sparsity 0.30 2>&1 | grep -E "SPARSITY|n="
echo "=== 반사 하향 (reflex 0.05) ==="
timeout 2400 python synapse_count_probe.py --reflex-sparsity 0.05 2>&1 | grep -E "SPARSITY|n="
