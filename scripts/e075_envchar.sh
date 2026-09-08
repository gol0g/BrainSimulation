#!/bin/bash
# E075 조작검증 겸 사전측정: 환경 특성이 시드별로 실제 다른가.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
for E in 0 1 2 3 4 5 6 7; do
  timeout 600 python env_characteristics.py --env-seed "$E" 2>&1 | grep -E "ENV|n_good|nearest|spread|asym|agent_start"
done
