#!/bin/bash
# E113 도구 확인(탐색): 뉴런 수준 분석 모드가 E112 뇌 0 가중치에서 동작하는가. 학습 없음.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e113_run && cd /root/e113_run
cp $R/backend/genesis/*.py . 2>/dev/null
mkdir -p $R/research/experiments/logs/E113
python reflex_override_task.py --real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 30 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0 \
  --episodes 0 --brain-seed 0 --decomp-weights $R/research/experiments/traces/E112/w_b0.npz --decomp-mode neuron > $R/research/experiments/logs/E113/toolcheck_e112b0.log 2>&1
grep -E "^=> (NEURON|KCPRE)|Error|error" $R/research/experiments/logs/E113/toolcheck_e112b0.log | tail -8
