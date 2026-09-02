#!/bin/bash
# E071 조작검증 3차: direct_inhibition이 D1 영향력을 차단하는가.
# C55(보상 413회) 명령에는 --direct-inhib가 없었다. 지금은 -400이 걸려 있고 보상 0회다.
# INV-A5(direct 포화 해소)가 INV-A3와 같은 계열의 부작용을 내는지 판별한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
BASE="--real-rstdp --crossed --epsilon 0.6 --bias 300 --bias-at-d1 --reflex-w 3 --episodes 6 --seed 0"
printf "%-38s: " "C55 정확재현(d1-200, direct 없음)"
timeout 2400 python reflex_override_task.py $BASE --d1-inhib -200 2>&1 | grep -E "학습\]"
printf "%-38s: " "d1-400, direct 없음"
timeout 2400 python reflex_override_task.py $BASE --d1-inhib -400 2>&1 | grep -E "학습\]"
printf "%-38s: " "d1-200, direct -100"
timeout 2400 python reflex_override_task.py $BASE --d1-inhib -200 --direct-inhib -100 2>&1 | grep -E "학습\]"
printf "%-38s: " "d1-400, direct -400 (현 불변식)"
timeout 2400 python reflex_override_task.py $BASE --d1-inhib -400 --direct-inhib -400 2>&1 | grep -E "학습\]"
