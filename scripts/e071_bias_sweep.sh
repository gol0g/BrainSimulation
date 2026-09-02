#!/bin/bash
# E071 조작검증 2차: d1_inhib -400 아래에서 D1 주입이 작동하는 편향 세기를 찾는다.
# 배경: 편향 300 + 억제 -400 → 보상 0회. C55(억제 -200)에서는 300이 413회를 만들었다.
#       불변식(INV-A4 -400)과 조작(D1 주입)이 충돌하는지, 세기로 넘을 수 있는지 판별.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
for B in 1000 3000 10000; do
  printf "D1편향 %-6s: " "$B"
  timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 \
    --bias "$B" --bias-at-d1 --d1-inhib -400 --direct-inhib -400 --reflex-w 3 \
    --episodes 6 --seed 0 2>&1 | grep -E "학습\]"
done
echo "--- 대조: 억제 -200에서 편향 300 (C55 재현) ---"
printf "D1편향 300/억제-200: "
timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 \
  --bias 300 --bias-at-d1 --d1-inhib -200 --direct-inhib -400 --reflex-w 3 \
  --episodes 6 --seed 0 2>&1 | grep -E "학습\]"
