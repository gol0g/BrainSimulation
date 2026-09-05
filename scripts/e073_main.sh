#!/bin/bash
# E073 본 실험: 분산 출처 분리. 조건별 (수리-정적) 표준편차를 잰다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"
STATIC="--epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"

pair () {  # $1=label $2=brain $3=env
  printf "  %-16s 수리: " "$1"
  timeout 3600 python reflex_override_task.py $COMMON --brain-seed "$2" --env-seed "$3" 2>&1 | grep -E "^=>"
  printf "  %-16s 정적: " "$1"
  timeout 3600 python reflex_override_task.py $STATIC --brain-seed "$2" --env-seed "$3" 2>&1 | grep -E "^=>"
}

echo "### A. 뇌 고정(0), 환경 0~4 → 환경 분산 ###"
for E in 0 1 2 3 4; do pair "A_env$E" 0 "$E"; done
echo "### B. 환경 고정(0), 뇌 0~4 → 뇌 분산 ###"
for B in 0 1 2 3 4; do pair "B_brain$B" "$B" 0; done
