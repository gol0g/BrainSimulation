#!/bin/bash
# 이력 민감도: 훈련 스텝 수가 조금 달라지면 사후 변조폭이 얼마나 흔들리는가.
#
# 결정론은 확인됐다(같은 커맨드 3회 = 소수점까지 동일). 런 간 잡음은 0이다.
# 그러나 안정화 스윕에서 **측정 시작 상태 의존성**(0.028 폭)이 드러났으므로,
# 조건마다 에피소드 길이가 달라 스텝 수가 어긋나면 그 의존성이 조건 차이로 둔갑할 수 있다.
# 같은 조건에서 에피소드 수만 바꿔 사후 변조폭의 흔들림을 잰다.
# 이 흔들림이 검출 목표(0.008)보다 크면 **가중치 이식 평가**로 가야 한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --real-rstdp --crossed --kc-w-max 750 --brain-seed 0 --env-seed 0"
echo "### KC학습 끔 (가중치 변화 없음 = 순수 이력 효과) ###"
for E in 3 4 5 6; do
  printf "  에피소드 %d: " "$E"
  timeout 1800 python reflex_override_task.py $BASE --episodes "$E" 2>&1 | grep -E "^\[사후\]"
done
