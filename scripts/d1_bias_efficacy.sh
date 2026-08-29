#!/bin/bash
# C53: D1 편향 주입이 **실제로 작동하는지** 먼저 확인 (해석 전 조작 검증).
#
# C52에서 D1주입과 motor주입이 seed0에서 완전히 같은 값(+0.0256)이 나왔다 → D1 주입이 무효일 가능성.
# 조작이 듣는지 확인하지 않고 결과를 해석하면 안 된다(C37에서 motor 편향은 보상 1469회로 확인했었다).
# 보상 횟수를 보면 탐색이 실제로 행동을 바꿨는지 알 수 있다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for B in 25 100 300; do
  echo "--- D1편향 세기 $B ---"
  timeout 3600 python reflex_override_task.py --episodes 6 --seed 0 \
    --real-rstdp --crossed --epsilon 0.6 --bias "$B" --bias-at-d1 \
    --d1-inhib -200 --reflex-w 3 --d1-direct-w 1 2>&1 | grep -E "학습\]|^=>"
done
echo "--- 대조: motor편향 25 ---"
timeout 3600 python reflex_override_task.py --episodes 6 --seed 0 \
  --real-rstdp --crossed --epsilon 0.6 --bias 25 \
  --d1-inhib -200 --reflex-w 3 --d1-direct-w 1 2>&1 | grep -E "학습\]|^=>"
echo "--- 대조: 탐색 없음 ---"
timeout 3600 python reflex_override_task.py --episodes 6 --seed 0 \
  --real-rstdp --crossed --epsilon 0.0 \
  --d1-inhib -200 --reflex-w 3 --d1-direct-w 1 2>&1 | grep -E "학습\]|^=>"
