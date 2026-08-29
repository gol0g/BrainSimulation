#!/bin/bash
# C59: d1 탈포화를 **안정적으로** 만들 수 있는가 — 억제 강도 상향 스윕.
#
# C58 규명: 억제 -200은 일부 시드에서만 d1을 탈포화시킨다.
#   탈포화된 시드(절대 456/377, 381/350) → 변별 107.1 / 38.3
#   포화된 시드(절대 ~594)              → 변별 4.6 / 3.5
# 즉 **탈포화가 되면 변별이 생긴다.** 문제는 탈포화가 상태 의존적이라는 것.
# 억제를 올려 5시드 전부에서 절대발화가 내려가고 변별이 서는지 본다.
#
# 판정: 어떤 강도에서 5시드 모두 절대<500 & 변별>30 이면 안정적 탈포화 달성.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for INH in -400 -800 -1500; do
  echo "### d1_inhibition = $INH ###"
  for S in 0 1 2 3 4; do
    printf "  seed=%s : " "$S"
    timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
      --d1-inhib "$INH" --seed "$S" 2>&1 | grep -E "^d1" | tr -s ' '
  done
done
