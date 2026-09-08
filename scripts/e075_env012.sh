#!/bin/bash
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
for E in 0 1 2; do
  timeout 600 python env_characteristics.py --env-seed "$E" 2>&1 | grep -E "ENV|n_good|nearest_good|spread|asym"
done
