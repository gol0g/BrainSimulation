#!/bin/bash
# C57: C41 "돌파"가 재현되는가 — d1 변별을 시드별로 확인.
#
# C41(무시드): d1억제 -200에서 d1 측성차이 318 / 6.8 / 596 → "기저핵이 처음으로 정보 전달"로 기록.
# C56(seed 0 고정): 같은 조건에서 d1 측성차이가 **4.3~5.6에 불과**.
# → C41의 큰 값이 특정 환경 상태에서만 나온 것일 수 있다. 시드 5개로 재현성을 본다.
#
# 판정: 5시드 중 다수에서 d1 측성차이가 >50이면 재현되는 현상,
#       대부분 한 자릿수면 C41은 우연한 환경 상태였고 기록을 정정해야 한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2 3 4; do
  printf "seed=%s : " "$S"
  timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
    --d1-inhib -200 --reflex-w 0.5 --seed "$S" 2>&1 | grep -E "^d1" | tr -s ' '
done
