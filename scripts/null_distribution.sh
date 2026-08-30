#!/bin/bash
# C65: 귀무 분포 측정 — 아무 차이 없는 두 조건의 짝지은 차이는 얼마나 큰가.
#
# C64에서 (수리−정적)이 4/5 시드에서 음수, 평균 -0.0049로 사전기준을 충족했다.
# 그러나 잔여 아티팩트(보상없음 조건에서 최대 +0.0177)가 효과보다 크므로,
# **차이가 없어야 하는 두 런**을 같은 방식으로 비교해 귀무 분포를 알아야 한다.
#
# 방법: 정적 조건을 같은 시드로 **두 번** 돌려 차이를 잰다(실행 간 비결정성만 남음).
#   귀무 차이가 |0.005| 이상으로 흔들리면 C64의 -0.0049는 잡음과 구분되지 않는다.
#   귀무 차이가 0에 가까우면 C64 결과는 실제 효과다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2 3 4; do
  echo "--- seed=$S (정적 vs 정적, 차이 0이어야 정상) ---"
  for R2 in A B; do
    printf "  정적%s: " "$R2"
    timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
      --epsilon 0.6 --bias 25 \
      --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
  done
done
