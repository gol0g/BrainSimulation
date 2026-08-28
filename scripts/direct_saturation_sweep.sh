#!/bin/bash
# C40: direct 포화 해소 스윕 — D1의 변별 신호가 direct를 통과하게 만들 수 있는가.
#
# C39 측정: d1 측성차이 31.3(변별O) → direct 2.0(활동O·변별X, 절대 668로 포화) → 신호 소멸.
# D1→direct가 DENSE·g=20.0이라 어느 쪽 D1이 활동하든 direct 전체가 최대 발화한다.
# 가중치를 낮추면 변별이 통과하는지 본다. 통과하면 학습→행동 경로가 처음으로 열린다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for W in 20 8 3 1 0.3; do
  echo "### d1_to_direct_weight = $W ###"
  timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
    --d1-direct-w "$W" 2>&1 | grep -E "^d1|^direct|^indirect|^motor|^조향"
done
