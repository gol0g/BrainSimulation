#!/bin/bash
# C67: 효과를 키우면 반사를 뒤집을 수 있는가 — 학습률 상향.
#
# C66 확정: 용량-반응 단조(10ep -0.0007 → 30ep -0.0037 → 60ep -0.0050) = 누적 학습.
# 그러나 크기가 작아 절대 변조폭(~0.09)을 뒤집지 못한다.
# 학습률(eta)을 올리면 커지는지, 아니면 포화/불안정해지는지 본다.
#
# 판정: eta에 따라 (수리−정적)이 계속 음수로 커지면 스케일 가능.
#       커지다 뒤집히거나 발산하면 안정성 한계에 도달한 것.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for ETA in 0.02 0.1 0.4; do
  echo "########## rstdp-eta $ETA (60ep) ##########"
  for S in 0 1 2; do
    printf "  seed=%s 수리: " "$S"
    timeout 3600 python reflex_override_task.py --episodes 60 --seed "$S" \
      --real-rstdp --rstdp-eta "$ETA" --crossed --epsilon 0.6 --bias 25 \
      --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
  done
done
echo "########## 기준: 정적(60ep) ##########"
for S in 0 1 2; do
  printf "  seed=%s 정적: " "$S"
  timeout 3600 python reflex_override_task.py --episodes 60 --seed "$S" \
    --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
done
