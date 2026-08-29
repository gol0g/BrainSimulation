#!/bin/bash
# C46: 측정 결정론 검증 — 같은 시드면 같은 결과가 나오는가.
#
# C45에서 같은 설정의 두 런이 direct 차이 +10.1 vs -10.3으로 **부호까지 반대**였다.
# GeNN 시드는 고정돼 있으므로 원인은 환경·워밍업 난수. 이제 그것도 고정했다.
# 같은 시드 2회가 일치하고 다른 시드가 갈리면 → 결정론화 성공 + 런 오프셋의 정체 확정.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 0 1 1; do
  echo "### seed=$S ###"
  timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
    --d1-inhib -200 --d1-direct-w 1 --reflex-w 0.5 --seed "$S" 2>&1 \
    | grep -E "^d1|^direct|^조향"
done
