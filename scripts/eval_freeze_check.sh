#!/bin/bash
# 외부 검토 지적 #1 검증: 평가 중 학습이 멈추는가.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/freeze_run && cd /root/freeze_run
cp $R/backend/genesis/*.py . 2>/dev/null
echo "##### 현행 코드 #####"
timeout 2400 python eval_freeze_probe.py --seed 0 --d1-inhib -400 --direct-inhib -100 2>&1 | tail -22
echo
echo "##### 양성대조: 평가 전 도파민을 장치까지 0으로 #####"
timeout 2400 python eval_freeze_probe.py --seed 0 --d1-inhib -400 --direct-inhib -100 --fix 2>&1 | tail -22
