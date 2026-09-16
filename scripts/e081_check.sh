#!/bin/bash
# E081 조작검증: 감마분포가 D1 다양성(CV)을 만드는가.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/kc_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
for S in 0 1 2; do
  printf "기본  seed=%s: " "$S"
  timeout 2400 python d1_diversity_probe.py --kc-rstdp --seed "$S" 2>&1 | grep "d1_left" | tr -s ' '
done
rm -rf forager_brain_CODE CODE
for S in 0 1 2; do
  printf "감마  seed=%s: " "$S"
  timeout 2400 python d1_diversity_probe.py --kc-rstdp --kc-gamma --seed "$S" 2>&1 | grep -E "d1_left|E081" | tr -s ' ' | tr '\n' ' '
  echo
done
