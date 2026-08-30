#!/bin/bash
# C66: 용량-반응 — 효과가 진짜면 훈련량에 따라 커져야 한다.
#
# C65 확정: 귀무 차이 정확히 0, (수리−정적) 4/5 음수 평균 -0.0049 = 학습이 행동을 바꿈.
# 그러나 크기가 작다(기저 변조폭 0.09 대비 5%). 진짜 학습이면 **에피소드를 늘릴수록 커져야** 한다.
# 커지지 않으면 일회성 상태 변화이지 누적 학습이 아니다.
#
# 판정: (수리−정적) 평균이 10ep < 30ep < 60ep 순으로 음수 방향 증가하면 용량-반응 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for EP in 10 60; do
  echo "########## 에피소드 $EP ##########"
  for S in 0 1 2; do
    printf "  seed=%s 수리: " "$S"
    timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
      --real-rstdp --crossed --epsilon 0.6 --bias 25 \
      --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
    printf "  seed=%s 정적: " "$S"
    timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
      --epsilon 0.6 --bias 25 \
      --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
  done
done
