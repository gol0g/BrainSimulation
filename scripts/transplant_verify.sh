#!/bin/bash
# 가중치 이식 평가 검증 (A 항등성 / B 이력 무관성).
# 통과 기준: 이식평가 최대-최소 < 0.008(검출 목표). A는 차이 < 1e-9.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/freeze_run && cd /root/freeze_run
cp $R/backend/genesis/*.py . 2>/dev/null
timeout 3000 python transplant_eval.py --seed 0 --churn 0 200 600 \
  --d1-inhib -400 --direct-inhib -100 2>&1 \
  | grep -E "^===|갓 만든|초기 가중치|churn|최대-최소|판정|일치"
