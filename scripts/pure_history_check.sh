#!/bin/bash
# 순수 이력 효과: **모든 학습을 끄고** 에피소드 수만 바꾼다.
# 앞선 측정(학습 일부 켬)은 0.0711 흔들렸으나 가중치도 변했으므로 이력 단독 효과가 아니다.
# 여기서 가중치가 전혀 안 변하는데도 흔들리면, 원인은 **뇌의 동역학 상태**로 확정된다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --brain-seed 0 --env-seed 0 --no-reward"
for E in 3 4 5 6; do
  printf "  에피소드 %d: " "$E"
  timeout 1800 python reflex_override_task.py $BASE --episodes "$E" --dump-kc-weights 2>&1 \
    | grep -E "^\[사후\]|kc_to_d1_l" | tr '\n' ' '
  echo
done
