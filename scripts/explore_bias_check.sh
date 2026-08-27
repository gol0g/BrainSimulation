#!/bin/bash
# C37: 탐색 주입 세기 예비 점검 — 보상이 0을 벗어나야 본 실험이 의미를 갖는다.
# (인라인 WSL 명령에서 $변수가 Git Bash에 소비되는 문제가 반복돼 스크립트 파일로 고정)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for B in 25 60 120; do
  echo "--- bias=$B ---"
  timeout 3000 python reflex_override_task.py --episodes 8 --real-rstdp --crossed \
    --epsilon 0.6 --bias "$B" 2>&1 | grep -E "학습\]|=>"
done
