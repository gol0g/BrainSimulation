#!/bin/bash
# C58: C41을 **원래 조건 그대로** 시드만 바꿔 재현 시도.
#
# C57에서 d1 측성차이가 3.8~36.6으로 나와 C41의 318/596이 재현되지 않았다.
# 그러나 C57에는 `--reflex-w 0.5`가 들어 있었고 **C41에는 없었다**(기본 25).
# 조건이 다르면 재현 실패를 C41 탓으로 돌릴 수 없다. 원래 플래그 그대로 시드만 바꾼다.
#
# C41 원본 명령: pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 --d1-inhib -200
# 판정: 5시드 중 다수에서 >100이면 C41은 재현되는 현상. 대부분 두 자릿수 이하면 기록 정정.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2 3 4; do
  printf "seed=%s : " "$S"
  timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
    --d1-inhib -200 --seed "$S" 2>&1 | grep -E "^d1" | tr -s ' '
done
