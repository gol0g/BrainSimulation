#!/bin/bash
# C49: 반사를 **완전히 제거**하고 학습 가능성 시험 — "이 뇌가 어떤 행동이든 학습할 수 있는가".
#
# C48: 결정론 적용 후 10런 전부 정확히 +0.0%p. 수리조건과 정적대조가 완전 동일.
#      반사역전은 15000×3 반사를 뒤집어야 하는 과제라 첫 시연으로 과도할 수 있다.
# 반사(food_approach_init_w)를 0으로 두면 조향을 만들 주체가 기저핵뿐이므로,
# 학습이 조금이라도 작동하면 여기서 드러난다. 안 드러나면 이 뇌는 행동 학습이 불가능하다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2; do
  echo "--- seed=$S (반사 0) ---"
  printf "  수리전부: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -200 \
    --reflex-w 0 --d1-direct-w 1 2>&1 | grep -E "사전|=>" | tr '\n' ' '
  echo
  printf "  정적대조: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --epsilon 0.6 --bias 25 --d1-inhib -200 --reflex-w 0 2>&1 | grep -E "사전|=>" | tr '\n' ' '
  echo
done
