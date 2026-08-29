#!/bin/bash
# C62: 공통 표류를 제거하고 R-STDP 효과를 본다.
#
# C60/C61에서 두 조건 모두 변조폭이 +0.02~0.04 표류했다. 원인은 `update_cortical_rstdp`의
# **전역 스칼라 증가**(C35: `w[:] += eta*trace`, 시냅스별 구분 없음)로, 수리/정적 양쪽에서 동일하게 일어난다.
# 측정하려는 효과(~0.003)가 이 공통항의 1/10이라 묻힌다.
# cortical_rstdp_eta=0으로 공통항을 제거하고 같은 비교를 반복한다.
#
# 사전 기준: 공통 표류가 사라지고(양 조건 |변조폭 변화| < 0.01),
#   (수리−정적) 차이가 5시드 중 4개에서 음수면 R-STDP 학습 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2 3 4; do
  echo "--- seed=$S ---"
  printf "  수리(피질OFF): "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 --cortical-eta 0 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
  printf "  정적(피질OFF): "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --epsilon 0.6 --bias 25 --cortical-eta 0 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
done
