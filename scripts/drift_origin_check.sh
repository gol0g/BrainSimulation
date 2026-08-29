#!/bin/bash
# C63: +0.02~0.04 변조폭 표류의 정체 — 학습인가 단순 처리인가.
#
# C60~C62 관측: 수리/정적/피질OFF 어느 조건이든 변조폭이 +0.02~0.04 증가하고,
#   피질 학습을 꺼도(cortical-eta 0) **소수점까지 동일**했다 → 표류원이 피질 학습이 아니다.
# 남은 설명: 사전·사후 측정 사이 수천 번의 process()로 뇌의 동역학 상태(적응·잔류전류)가 변한 것.
#
# 결정적 대조: **도파민을 한 번도 주지 않고** 같은 횟수만큼 처리만 한다.
#   표류가 그대로면 → 학습과 무관한 측정 아티팩트. 이 지표로는 학습을 볼 수 없다는 뜻.
#   표류가 사라지면 → 표류는 보상 유래이고, 그 안에서 R-STDP 기여를 찾아야 한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2; do
  echo "--- seed=$S ---"
  printf "  보상없음(처리만): "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 --no-reward \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
  printf "  보상있음(정상)  : "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
done
