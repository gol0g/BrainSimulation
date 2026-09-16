#!/bin/bash
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/kc_run
cp $R/backend/genesis/*.py . 2>/dev/null
timeout 2400 python kc_fire_probe.py --kc-rstdp --seed 1 2>&1 | tail -15
