#!/bin/bash
# 안정화 길이 용량-반응(P7): 몇 스텝이면 1회차 편향이 사라지는가.
# 기준선: 30스텝에서 최대-최소 0.0084 (판정 기준 0.008보다 크다 = 검출 불가).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/freeze_run && cd /root/freeze_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
for S in 30 100 300 1000; do
  echo "##### 안정화 ${S}스텝 #####"
  timeout 2400 python eval_stability_probe.py --seed 0 --repeats 5 --stab "$S" \
    --kc-d1-w 0.5 --d1-inhib -400 --direct-inhib -100 2>&1 \
    | grep -E "회차|평균|최대-최소" | tail -7
done
