#!/bin/bash
# C54: C50의 학습 신호(+0.013)가 **해마에서 온 것인가** — 출처 귀속.
#
# C53 확정: D1을 300까지 밀어도 행동 변화 0 = **기저핵은 행동을 제어할 수 없다.**
# 따라서 C50에서 관측한 "수리한 뇌가 정적 대조보다 더 변한다(+0.013, 3/3)"의 출처가
# 기저핵일 수 없다. 남는 후보는 **해마 경로**:
#   place_to_food_memory는 실제로 학습되고(std 0→11), food_memory→motor는 n=7562·g=5.0으로 실재.
#
# 해마 학습률을 0으로 끄면 그 신호가 사라지는지 본다. 사라지면 출처 = 해마.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2; do
  echo "--- seed=$S ---"
  printf "  해마ON (기본0.15): "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 \
    --d1-inhib -200 --reflex-w 3 --d1-direct-w 1 2>&1 | grep -E "^=>"
  printf "  해마OFF(eta 0)   : "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 --hippo-eta 0 \
    --d1-inhib -200 --reflex-w 3 --d1-direct-w 1 2>&1 | grep -E "^=>"
done
