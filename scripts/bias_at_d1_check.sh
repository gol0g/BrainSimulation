#!/bin/bash
# C52: 탐색 주입 지점을 D1로 옮겨 학습 방향이 바뀌는가.
#
# C50 실측: 반사 반대쪽을 보상했는데 변조폭이 **반사 방향으로** +0.036 이동(수리) / +0.023(정적).
#   짝지은 차이 +0.004/+0.016/+0.019 (3/3 양수) = 학습은 행동에 도달하나 방향이 반대.
# 원인: 탐색 편향을 motor에 주입했는데 학습 시냅스(food_eye→D1)는 그보다 **상류**.
#   강제된 행동이 D1을 안 거치므로 자격흔적에는 자극이 만든 원래(반사정렬) 패턴이 기록되고
#   보상이 그걸 강화한다.
# 수정: 편향을 D1에 주입 → 흔적이 탐색한 상태를 담게 한다.
#
# 판정: 변조폭 변화가 D1주입에서 motor주입보다 **음수 방향**으로 가면 수정이 맞다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2; do
  echo "--- seed=$S ---"
  printf "  D1주입  : "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 --bias-at-d1 \
    --d1-inhib -200 --reflex-w 3 --d1-direct-w 1 2>&1 | grep -E "^=>"
  printf "  motor주입: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 \
    --d1-inhib -200 --reflex-w 3 --d1-direct-w 1 2>&1 | grep -E "^=>"
done
