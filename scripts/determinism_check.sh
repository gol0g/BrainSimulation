#!/bin/bash
# 같은 시드로 같은 커맨드를 두 번 돌리면 같은 값이 나오는가.
#
# 왜: 안정화 스윕에서 평가가 **시작 상태에 의존**함이 드러났다(30/300 -> 0.553, 100/1000 -> 0.581).
# "사전 vs 사후"는 두 측정의 뇌 이력이 다르므로 이 의존성이 통째로 섞인다.
# 대안은 **같은 시드로 조건을 짝지어 사후만 비교**하는 것(INV-B3). 두 조건의 이력이 같아지기 때문이다.
# 그 축이 쓸 만하려면 먼저 **같은 조건 반복이 결정론적**이어야 한다. 그것을 잰다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
ARGS="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 3 --real-rstdp --crossed --kc-w-max 750 --kc-rstdp --kc-d1-w 150 --brain-seed 0 --env-seed 0"
for i in 1 2 3; do
  printf "%d회: " "$i"
  timeout 1800 python reflex_override_task.py $ARGS 2>&1 | grep -E "^\[사후\]" 
done
