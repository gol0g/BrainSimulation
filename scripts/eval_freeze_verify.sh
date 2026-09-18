#!/bin/bash
# reset() 수정 후 검증: 평가 구간에서 **모든 학습 시냅스**가 동결되는가.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/freeze_run && cd /root/freeze_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
echo "##### 수정된 reset() — 동결돼야 정상 #####"
timeout 2400 python eval_freeze_probe.py --seed 0 --d1-inhib -400 --direct-inhib -100 2>&1 | tail -26
