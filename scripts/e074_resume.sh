#!/bin/bash
# E074 재개: env2의 brain 2,3,4 (12런). 48/60에서 종료코드 1로 중단됨.
# 이번엔 stderr를 버리지 않는다 — 이전 실행은 grep 필터로 오류를 놓쳤다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"
STATIC="--epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"
for B in 2 3 4; do
  for CFG in "기준1.0x 0.08 0.15" "11.25x 0.30 0.05"; do
    set -- $CFG
    printf "  env2 b%s %-9s 수리: " "$B" "$1"
    OUT=$(timeout 3600 python reflex_override_task.py $COMMON --brain-seed "$B" --env-seed 2 \
      --learn-sparsity "$2" --reflex-sparsity "$3" 2>&1)
    echo "$OUT" | grep -E "^=>" || { echo "[실패] 마지막 5줄:"; echo "$OUT" | tail -5; }
    printf "  env2 b%s %-9s 정적: " "$B" "$1"
    OUT=$(timeout 3600 python reflex_override_task.py $STATIC --brain-seed "$B" --env-seed 2 \
      --learn-sparsity "$2" --reflex-sparsity "$3" 2>&1)
    echo "$OUT" | grep -E "^=>" || { echo "[실패] 마지막 5줄:"; echo "$OUT" | tail -5; }
  done
done
