#!/bin/bash
# E072 확장: 사전등록의 "방향이 갈리면 5시드" 조항에 따라 seed 3,4 추가.
# 3시드 결과가 지지(-0.005)와 기각(-0.002) 사이(-0.0046)에 떨어졌고 방향이 2/3로 갈렸다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"
for S in 3 4; do
  echo "--- seed=$S ---"
  for CFG in "기준1.0x 0.08 0.15" "양쪽11.25x 0.30 0.05"; do
    set -- $CFG
    printf "  %-12s 수리: " "$1"
    timeout 3600 python reflex_override_task.py $COMMON --seed "$S" \
      --learn-sparsity "$2" --reflex-sparsity "$3" 2>&1 | grep -E "^=>"
    printf "  %-12s 정적: " "$1"
    timeout 3600 python reflex_override_task.py --epsilon 0.6 --bias 25 \
      --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60 --seed "$S" \
      --learn-sparsity "$2" --reflex-sparsity "$3" 2>&1 | grep -E "^=>"
  done
done
