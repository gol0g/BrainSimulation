#!/bin/bash
# E080 조작검증: D1 측면억제가 변별을 만드는가.
# 기준: d1 측성차이가 3시드 모두 기존(0) 대비 2배 이상 / 절대발화 50 미만이면 붕괴
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
for LAT in 0 -20 -60 -150; do
  echo "########## 측면억제 $LAT ##########"
  for S in 0 1 2; do
    printf "  seed=%s: " "$S"
    timeout 2400 python pathway_transfer_probe.py --real-rstdp --kc-rstdp --set-d1-weight 30 \
      --d1-inhib -400 --direct-inhib -100 --d1-lateral "$LAT" --seed "$S" 2>&1 \
      | grep -E "^d1" | tr -s ' '
  done
done
