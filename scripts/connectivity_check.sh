#!/bin/bash
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/freeze_run && cd /root/freeze_run
cp $R/backend/genesis/*.py . 2>/dev/null
timeout 3000 python connectivity_check.py --seed 0 --d1-inhib -400 --direct-inhib -100 2>&1 \
  | grep -E "^\[완료\]|^시냅스|^-|^kc_|^food_|^good_|^판정"
