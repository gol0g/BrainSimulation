#!/bin/bash
# C55: C53 결론의 허점 점검 — D1 편향 실패가 내가 켠 억제(-200) 때문인가.
#
# C53: D1 편향 25/100/300 전부 보상 0회 → "기저핵은 행동 제어 불가"로 결론.
# 그런데 그 시험은 `--d1-inhib -200`을 켠 채였다. **강한 E/I 억제가 주입한 흥분을 상쇄**했다면
# 그 결론은 내가 만든 조건의 산물이다.
# 억제 없이(기본 0) 같은 시험을 돌려 구분한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

echo "--- D1편향 100 / 억제 없음 ---"
timeout 3600 python reflex_override_task.py --episodes 6 --seed 0 \
  --real-rstdp --crossed --epsilon 0.6 --bias 100 --bias-at-d1 \
  --reflex-w 3 2>&1 | grep -E "학습\]|^=>"
echo "--- D1편향 300 / 억제 없음 ---"
timeout 3600 python reflex_override_task.py --episodes 6 --seed 0 \
  --real-rstdp --crossed --epsilon 0.6 --bias 300 --bias-at-d1 \
  --reflex-w 3 2>&1 | grep -E "학습\]|^=>"
echo "--- 대조: D1편향 100 / 억제 -200 (C53 재현) ---"
timeout 3600 python reflex_override_task.py --episodes 6 --seed 0 \
  --real-rstdp --crossed --epsilon 0.6 --bias 100 --bias-at-d1 \
  --d1-inhib -200 --reflex-w 3 2>&1 | grep -E "학습\]|^=>"
