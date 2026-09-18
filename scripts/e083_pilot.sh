#!/bin/bash
# E083 pilot: 20런 전에 배선이 실제로 도는지 확인 (--bias-at-d1, --dump-kc-weights, 이득/학습 조작).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 3 --real-rstdp --crossed --kc-w-max 750"
for CELL in "A w0.5 KCoff 0.5 " "B w0.5 KCon 0.5 --kc-rstdp" "C w150 KCoff 150 " "D w150 KCon 150 --kc-rstdp"; do
  set -- $CELL
  echo "### $1 ($2 $3)"
  timeout 1800 python reflex_override_task.py $BASE --kc-d1-w "$4" ${5:-} \
    --brain-seed 0 --env-seed 0 --dump-kc-weights 2>&1 | grep -E "^=>|kc_to_d1|사전|사후" | tail -5
done
