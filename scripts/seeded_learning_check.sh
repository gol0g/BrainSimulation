#!/bin/bash
# C48: 학습 시험을 **결정론적으로** 재실행 — 잡음이 학습 효과를 가렸는지 확인.
#
# 이 세션의 학습 시험(C34~C43)도 전부 무시드였다. 사전 정답률이 런마다 0%~72%로 흔들렸으므로
# (C43 실측) 작은 학습 효과는 묻혔을 수 있다. 시드를 고정하고 같은 시드에서
# **네 수리 전부** vs **정적 대조**를 짝지어 비교한다.
#
# 사전 기준: 5시드 중 4개 이상에서 (수리조건 사후−사전) − (정적 사후−사전) > +8%p 이면 학습 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2 3 4; do
  echo "--- seed=$S ---"
  printf "  수리전부: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -200 --reflex-w 3 2>&1 \
    | grep -E "=>" | tail -1
  printf "  정적대조: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --epsilon 0.6 --bias 25 --d1-inhib -200 --reflex-w 3 2>&1 \
    | grep -E "=>" | tail -1
done
