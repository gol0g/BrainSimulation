#!/bin/bash
# 검토 제안 0단계: 같은 뇌를 5번 평가해 재현성을 잰다 (규약 P11).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/freeze_run && cd /root/freeze_run
cp $R/backend/genesis/*.py . 2>/dev/null
echo "##### 기본 이득 (kc_d1_w=0.5) #####"
timeout 2400 python eval_stability_probe.py --seed 0 --repeats 5 --kc-d1-w 0.5 --d1-inhib -400 --direct-inhib -100 2>&1 | tail -14
echo
echo "##### 고이득 (kc_d1_w=150) #####"
timeout 2400 python eval_stability_probe.py --seed 0 --repeats 5 --kc-d1-w 150 --d1-inhib -400 --direct-inhib -100 2>&1 | tail -14
