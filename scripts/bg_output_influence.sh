#!/bin/bash
# C44: 기저핵 출력이 운동에 **영향을 주기는 하는가** — 다섯 번째 후보 검증.
#
# C42 관찰: 학습 전후 오프셋이 소수점 셋째자리까지 동일(+0.001→+0.001).
# 반사가 세기만 한 거라면 기저핵 기여가 조금이라도 있어 오프셋이 미세하게라도 움직여야 한다.
# → 기저핵 출력이 운동에 도달하는 기여가 **사실상 0**일 가능성.
#
# 검증: 반사를 0.5로 낮춰 무대를 비우고, direct→motor 가중치를 25→200으로 올려
# 조향이 반응하는지 본다. 반응 없으면 기저핵→운동 경로 자체가 무력한 것.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for DM in 25 100 200; do
  echo "### direct_to_motor = $DM (반사 0.5, d1억제 -200) ###"
  timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
    --d1-inhib -200 --reflex-w 0.5 --direct-motor-w "$DM" 2>&1 \
    | grep -E "^d1|^direct|^indirect|^motor|^조향"
done
