#!/bin/bash
# C39: 전달 병목 격리 검사 — 학습을 건너뛰고 food_to_d1 가중치를 직접 박아
# "D1이 강해지면 행동이 바뀌는가"만 순수하게 묻는다.
# 안 바뀌면 학습을 아무리 고쳐도 소용없다(병목이 전달에 있음).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

echo "### 기준(가중치 조작 없음) ###"
timeout 3000 python pathway_transfer_probe.py --real-rstdp 2>&1 | tail -10
for W in 1 10 30 100; do
  echo "### food_to_d1 = $W ###"
  timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight "$W" 2>&1 | tail -9
done
