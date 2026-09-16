#!/bin/bash
# E077 조작검증: n_food/food_ratio가 실제로 특성을 바꾸는가, 다른 특성은 얼마나 흔들리는가.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
printf "기준        : "; timeout 600 python env_characteristics.py --env-seed 0 2>&1 | grep -E "n_good=|nearest_good|spread" | tr '\n' ' '; echo
printf "개수70      : "; timeout 600 python env_characteristics.py --env-seed 0 --n-food 70 2>&1 | grep -E "n_good=|nearest_good|spread" | tr '\n' ' '; echo
printf "품질0.85    : "; timeout 600 python env_characteristics.py --env-seed 0 --food-ratio 0.85 2>&1 | grep -E "n_good=|nearest_good|spread" | tr '\n' ' '; echo
printf "둘다        : "; timeout 600 python env_characteristics.py --env-seed 0 --n-food 70 --food-ratio 0.85 2>&1 | grep -E "n_good=|nearest_good|spread" | tr '\n' ' '; echo
printf "env6(목표)  : "; timeout 600 python env_characteristics.py --env-seed 6 2>&1 | grep -E "n_good=|nearest_good|spread" | tr '\n' ' '; echo
